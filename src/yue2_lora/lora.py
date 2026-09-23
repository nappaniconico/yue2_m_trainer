from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .assets import sha256


def target_names(layers: int, branch: str = "ar") -> Iterable[str]:
    branch_prefix = "nar_" if branch == "nar" else ""
    for index in range(layers):
        for module, names in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        ):
            for name in names:
                yield f"model.layers.{index}.{branch_prefix}{module}.{name}"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.A = nn.Parameter(
            torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features)
        )
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + ((value.float() @ self.A.T) @ self.B.T).to(value.dtype)


def install_lora(model: nn.Module, rank: int) -> None:
    for key in target_names(model.config.num_hidden_layers):
        parent_name, name = key.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, name, LoRALinear(getattr(parent, name), rank))


def trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: value for name, value in model.named_parameters() if name.endswith((".A", ".B"))}


def _save(path: Path, values: dict[str, torch.Tensor], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {key: value.detach().float().cpu().contiguous() for key, value in values.items()},
        str(temporary),
        metadata={key: str(value) for key, value in metadata.items()},
    )
    os.replace(temporary, path)


def _pair(module: LoRALinear) -> tuple[torch.Tensor, torch.Tensor]:
    return module.A.detach(), module.B.detach()


def fuse_pairs(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent vertically concatenated linear deltas as one exact block-rank LoRA."""
    if not pairs:
        raise ValueError("At least one LoRA pair is required")
    input_sizes = {pair[0].shape[1] for pair in pairs}
    if len(input_sizes) != 1:
        raise ValueError("Fused projections must have the same input size")
    down = torch.cat([pair[0] for pair in pairs], dim=0)
    output_size = sum(pair[1].shape[0] for pair in pairs)
    rank = sum(pair[0].shape[0] for pair in pairs)
    up = pairs[0][1].new_zeros(output_size, rank)
    output_offset = rank_offset = 0
    for a, b in pairs:
        out, current_rank = b.shape
        up[output_offset : output_offset + out, rank_offset : rank_offset + current_rank] = b
        output_offset += out
        rank_offset += current_rank
    return down, up


def export_fl_ar(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    values: dict[str, torch.Tensor] = {}
    for key in target_names(model.config.num_hidden_layers):
        down, up = _pair(model.get_submodule(key))
        values[key + ".lora_down.weight"] = down
        values[key + ".lora_up.weight"] = up
    _save(path, values, {"format": "fl-yue2-lora-v1", "branch": "ar", **metadata})


def export_native_ar(model: nn.Module, path: Path, metadata: dict[str, object]) -> None:
    values: dict[str, torch.Tensor] = {}
    for index in range(model.config.num_hidden_layers):
        layer = model.model.layers[index]
        for target, modules in (
            ("self_attn.qkv_proj", [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]),
            ("mlp.gate_up_proj", [layer.mlp.gate_proj, layer.mlp.up_proj]),
        ):
            down, up = fuse_pairs([_pair(module) for module in modules])
            key = f"text_encoders.model.layers.{index}.{target}"
            values[key + ".lora_down.weight"] = down
            values[key + ".lora_up.weight"] = up
        for target, module in (("self_attn.o_proj", layer.self_attn.o_proj), ("mlp.down_proj", layer.mlp.down_proj)):
            down, up = _pair(module)
            key = f"text_encoders.model.layers.{index}.{target}"
            values[key + ".lora_down.weight"] = down
            values[key + ".lora_up.weight"] = up
    _save(path, values, {"format": "comfyui-yue2-lora-v1", "branch": "ar", **metadata})


def _kit_pairs(checkpoint: dict, layers: int, branch: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    keys = list(target_names(layers, branch))
    values = checkpoint["lora"]
    if len(values) != len(keys) * 2:
        raise ValueError(f"NAR adapter has {len(values)} tensors; expected {len(keys) * 2}")
    return {key: (values[index * 2], values[index * 2 + 1]) for index, key in enumerate(keys)}


def export_nar_companions(source: Path, base_model: nn.Module, output: Path) -> tuple[Path, Path]:
    layers = base_model.config.num_hidden_layers
    checkpoint = load_nar_checkpoint(source, layers)
    pairs = _kit_pairs(checkpoint, layers, "nar")
    for name, (down, up) in pairs.items():
        base = base_model.get_submodule(name)
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] != up.shape[1]
            or (up.shape[0], down.shape[1]) != tuple(base.weight.shape)
        ):
            raise ValueError(f"NAR projection shape mismatch: {name}")
    fl_values: dict[str, torch.Tensor] = {}
    for key, (down, up) in pairs.items():
        fl_values[key + ".lora_down.weight"] = down
        fl_values[key + ".lora_up.weight"] = up
    for module_name in ("vae2llm", "llm2vae"):
        source_module = checkpoint["io"][module_name]
        base_module = getattr(base_model, module_name)
        fl_values[module_name + ".diff"] = source_module["weight"].float() - base_module.weight.detach().cpu().float()
        fl_values[module_name + ".diff_b"] = source_module["bias"].float() - base_module.bias.detach().cpu().float()
    metadata = {"branch": "nar", "rank": checkpoint["rank"], "source_sha256": sha256(source)}
    fl_path = output / f"{source.stem}.fl_yue2.safetensors"
    _save(fl_path, fl_values, {"format": "fl-yue2-lora-v1", **metadata})

    native_values: dict[str, torch.Tensor] = {}
    for index in range(layers):
        prefix = f"model.layers.{index}.nar_"
        for target, source_names in (
            ("self_attn.qkv_proj", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("mlp.gate_up_proj", ("mlp.gate_proj", "mlp.up_proj")),
        ):
            down, up = fuse_pairs([pairs[prefix + name] for name in source_names])
            key = f"model.layers.{index}.{target}"
            native_values[key + ".lora_down.weight"] = down
            native_values[key + ".lora_up.weight"] = up
        for target, source_name in (("self_attn.o_proj", "self_attn.o_proj"), ("mlp.down_proj", "mlp.down_proj")):
            down, up = pairs[prefix + source_name]
            key = f"model.layers.{index}.{target}"
            native_values[key + ".lora_down.weight"] = down
            native_values[key + ".lora_up.weight"] = up
    for key in ("vae2llm.diff", "vae2llm.diff_b", "llm2vae.diff", "llm2vae.diff_b"):
        native_values[key] = fl_values[key]
    native_path = output / f"{source.stem}.comfyui.safetensors"
    _save(native_path, native_values, {"format": "comfyui-yue2-lora-v1", **metadata})
    return native_path, fl_path


def load_nar_checkpoint(source: Path, layers: int) -> dict:
    if source.suffix != ".safetensors":
        return torch.load(source, map_location="cpu", weights_only=True)
    state = load_file(str(source))
    values = []
    for name in target_names(layers, "nar"):
        key = name.removeprefix("model.")
        values.extend([state.pop(key + ".lora_A"), state.pop(key + ".lora_B")])
    io = {
        module: {key: state.pop(f"{module}.{key}") for key in ("weight", "bias")} for module in ("vae2llm", "llm2vae")
    }
    if state:
        raise ValueError(f"Unexpected NAR checkpoint tensors: {sorted(state)}")
    rank = values[0].shape[0]
    for a, b in zip(values[::2], values[1::2], strict=True):
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError("Inconsistent NAR LoRA ranks or dimensions")
    return {"lora": values, "io": io, "rank": rank}
