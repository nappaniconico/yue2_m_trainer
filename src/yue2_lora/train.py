from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.nar import attention
from yue2.protocol import CODEC_OFFSET, MUSIC_END, SongRequest, token_prefixes
from yue2.tokenization_yue2 import YuE2TextTokenizer

from .assets import Assets, sha256
from .config import Config
from .dataset import fingerprint, read_json, verify_manifest, write_json
from .lora import export_fl_ar, export_nar_companions, export_native_ar, install_lora, trainable_parameters


@dataclass(frozen=True)
class SupervisedSequence:
    ids: torch.Tensor
    abc_targets: tuple[int, int] | None
    audio_targets: tuple[int, int]


@dataclass(frozen=True)
class LossParts:
    total: torch.Tensor
    abc: torch.Tensor | None
    audio: torch.Tensor


def _load_regularizer(path: Path) -> list[dict[str, Any]]:
    # This file is pinned and SHA-256 verified before loading.
    records = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(records, list) or not records:
        raise ValueError("Regularizer must be a nonempty list")
    for item in records:
        codec = np.asarray(item.get("codec"))
        if (
            item.get("src") not in {"minted", "minted_val"}
            or codec.ndim != 1
            or not np.issubdtype(codec.dtype, np.integer)
            or not len(codec)
            or codec.min() < 0
            or codec.max() >= 32768
        ):
            raise ValueError("Invalid minted regularizer record")
    return records


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _ar_layer(layer, value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    query, key, val = layer.self_attn.project_qkv(layer.input_layernorm(value), cos, sin)
    hidden = attention(query[0], key[0], val[0], causal=True)
    value = value + layer.self_attn.o_proj(hidden.flatten(1)[None])
    return value + layer.mlp(layer.post_attention_layernorm(value))


def _hidden(model: YuE2ForCausalLM, ids: torch.Tensor, checkpoint_layers: bool) -> torch.Tensor:
    backbone = model.model
    value = backbone.embed_tokens(ids)
    positions = torch.arange(value.shape[1], device=value.device)[None]
    cos, sin = backbone.rotary_emb(positions)
    for layer in backbone.layers:
        if checkpoint_layers:
            value = checkpoint(_ar_layer, layer, value, cos, sin, use_reentrant=False)
        else:
            value = _ar_layer(layer, value, cos, sin)
    return backbone.norm(value)


def _target_loss(
    model: YuE2ForCausalLM,
    hidden: torch.Tensor,
    ids: torch.Tensor,
    target_range: tuple[int, int],
    checkpoint_layers: bool,
) -> torch.Tensor:
    start, stop = target_range
    target_hidden = hidden[start - 1 : stop - 1]
    targets = ids[0, start:stop]
    if not len(targets):
        raise ValueError("A supervised target range must not be empty")
    total = torch.zeros((), device=ids.device)
    for start in range(0, len(targets), 256):
        end = start + 256

        def cross_entropy(value: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(model.lm_head(value).float(), labels, reduction="sum")

        if checkpoint_layers:
            total = total + checkpoint(cross_entropy, target_hidden[start:end], targets[start:end], use_reentrant=False)
        else:
            total = total + cross_entropy(target_hidden[start:end], targets[start:end])
    return total / len(targets)


def _loss(
    model: YuE2ForCausalLM,
    sequence: SupervisedSequence,
    checkpoint_layers: bool,
    abc_weight: float,
    audio_weight: float,
) -> LossParts:
    hidden = _hidden(model, sequence.ids, checkpoint_layers)[0]
    audio = _target_loss(model, hidden, sequence.ids, sequence.audio_targets, checkpoint_layers)
    if sequence.abc_targets is None:
        return LossParts(total=audio, abc=None, audio=audio)
    abc = _target_loss(model, hidden, sequence.ids, sequence.abc_targets, checkpoint_layers)
    total = (abc_weight * abc + audio_weight * audio) / (abc_weight + audio_weight)
    return LossParts(total=total, abc=abc, audio=audio)


def _sequence(
    item: dict[str, Any],
    tokenizer: YuE2TextTokenizer,
    budget: int,
    allow_truncation: bool,
    device,
    cot: str = "off",
) -> SupervisedSequence:
    abc = item.get("abc")
    if cot != "off" and (not isinstance(abc, str) or not abc.strip()):
        raise ValueError(f"{item['name']}: {cot} training requires a nonempty ABC score")
    request = SongRequest(style=item["style"], lyrics=item["lyrics"], cot=cot, abc=abc if cot != "off" else None)
    abc_ids = tokenizer.encode(abc) if cot != "off" else None
    prefix = token_prefixes(request, tokenizer, abc_ids=abc_ids)
    abc_targets = None
    if abc_ids is not None:
        # Predict the ABC text and ABC_END after the model has seen ABC_START.
        # MUSIC_START is inserted by the protocol, so it is intentionally not a target.
        abc_targets = (len(prefix) - len(abc_ids) - 2, len(prefix) - 1)
    room = budget - len(prefix) - 1
    if room < 1:
        raise ValueError(f"{item['name']}: style/lyrics/ABC leave no audio room in the sequence budget")
    codec = np.asarray(item["codec"])
    if len(codec) > room and not allow_truncation:
        raise ValueError(
            f"{item['name']}: {len(codec)} audio tokens exceed the {budget}-token budget; "
            "increase sequence_tokens or explicitly enable allow_truncation"
        )
    body = [int(value) + CODEC_OFFSET for value in codec[:room]]
    if len(codec) <= room:
        body.append(MUSIC_END)
    ids = torch.tensor([prefix + body], device=device, dtype=torch.long)
    return SupervisedSequence(ids=ids, abc_targets=abc_targets, audio_targets=(len(prefix), len(prefix) + len(body)))


def _latest_prepared(config: Config) -> Path:
    paths = sorted(
        config.prepare.cache_directory.glob("prepared-*.json"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if config.train.cot == "off":
        if paths:
            return paths[0]
    else:
        for path in paths:
            manifest = read_json(path)
            if manifest.get("songs") and all(
                isinstance(song.get("abc"), str) and song["abc"].strip() for song in manifest["songs"]
            ):
                return path
    suffix = " containing ABC scores" if config.train.cot != "off" else ""
    raise FileNotFoundError(f"No prepared manifest{suffix}; run yue2-lora prepare with this config")


def _save_resume(
    path: Path,
    signature: str,
    step: int,
    parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> None:
    state = {
        "signature": signature,
        "step": step,
        "model": {name: value.detach().cpu() for name, value in parameters.items()},
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, ensure_ascii=False, allow_nan=False) + "\n")


def train(config: Config, assets: Assets, prepared_path: Path | None = None, resume: bool = False) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YuE2 LoRA training")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    prepared_path = prepared_path or _latest_prepared(config)
    prepared = read_json(prepared_path)
    verify_manifest(prepared)
    if config.train.cot != "off":
        missing = [song["name"] for song in prepared["songs"] if not str(song.get("abc", "")).strip()]
        if missing:
            raise ValueError(f"Prepared manifest has no ABC for: {', '.join(missing)}; run prepare with the ABC config")
    if prepared["head_hash"] != sha256(assets.head):
        raise ValueError("Tokenizer head changed; run prepare again")

    songs = [{**song, "codec": np.load(song["tokens"], allow_pickle=False)} for song in prepared["songs"]]
    real_train = [song for song in songs if song["split"] == "train"]
    real_validation = [song for song in songs if song["split"] == "validation"][:6]
    regularizer = _load_regularizer(assets.regularizer)
    minted_train = [song for song in regularizer if song["src"] == "minted"]
    minted_validation = [song for song in regularizer if song["src"] == "minted_val"][:6]
    if not real_train or not real_validation or not minted_train or not minted_validation:
        raise ValueError("Training requires both real and generated train/validation examples")

    output = config.train.output_directory
    adapters = output / "adapters"
    output.mkdir(parents=True, exist_ok=True)
    adapters.mkdir(parents=True, exist_ok=True)
    resume_path = output / "resume.pt"
    metrics_path = output / "metrics.jsonl"
    signature = fingerprint(
        {
            "prepared": prepared["fingerprint"],
            "train": asdict(config.train),
            "model_revision": assets.model_revision,
            "head": sha256(assets.head),
            "regularizer": sha256(assets.regularizer),
            "nar": sha256(assets.nar),
        }
    )
    if not resume:
        if resume_path.exists():
            raise FileExistsError(f"{resume_path} already exists; pass --resume or choose another output_directory")
        metrics_path.unlink(missing_ok=True)

    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)
    print(f"Loading {assets.model} in BF16...", flush=True)
    model = (
        YuE2ForCausalLM.from_pretrained(
            assets.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("cuda")
    )
    model.requires_grad_(False)
    export_nar_companions(assets.nar, model, adapters)
    install_lora(model, config.train.rank)
    named = trainable_parameters(model)
    parameters = list(named.values())
    optimizer = torch.optim.AdamW(parameters, lr=config.train.learning_rate, betas=(0.9, 0.95), weight_decay=0)
    tokenizer = YuE2TextTokenizer(assets.model / "qwen.tiktoken")
    start_step = 0
    if resume:
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume config, dataset, or assets do not match this run")
        for name, parameter in named.items():
            parameter.data.copy_(saved["model"][name].to(parameter.device))
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"])
        start_step = int(saved["step"])
        print(f"Resuming at step {start_step}", flush=True)

    base_lr = config.train.learning_rate
    started = time.monotonic()

    def loss_for(item: dict[str, Any], training: bool) -> LossParts:
        # The minted regularizer has no score and retains the original cot=off
        # semantic-token objective. Real examples use the configured score mode.
        cot = config.train.cot if "abc" in item else "off"
        sequence = _sequence(
            item,
            tokenizer,
            config.train.sequence_tokens,
            config.train.allow_truncation,
            next(model.parameters()).device,
            cot=cot,
        )
        return _loss(
            model,
            sequence,
            checkpoint_layers=training,
            abc_weight=config.train.abc_loss_weight,
            audio_weight=config.train.audio_loss_weight,
        )

    def validation_losses(items: list[dict[str, Any]]) -> dict[str, float]:
        total_values: list[float] = []
        abc_values: list[float] = []
        audio_values: list[float] = []
        with torch.no_grad():
            for item in items:
                parts = loss_for(item, training=False)
                total_values.append(float(parts.total))
                audio_values.append(float(parts.audio))
                if parts.abc is not None:
                    abc_values.append(float(parts.abc))
        result = {
            "total": sum(total_values) / len(total_values),
            "audio": sum(audio_values) / len(audio_values),
        }
        if abc_values:
            result["abc"] = sum(abc_values) / len(abc_values)
        return result

    for step in range(start_step + 1, config.train.steps + 1):
        multiplier = min(1.0, step / max(1, config.train.warmup_steps))
        multiplier *= 0.2 + 0.8 * 0.5 * (
            1 + math.cos(math.pi * min(step, config.train.schedule_steps) / config.train.schedule_steps)
        )
        optimizer.param_groups[0]["lr"] = base_lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        audio_losses: list[float] = []
        abc_losses: list[float] = []
        for _ in range(config.train.gradient_accumulation):
            if random.random() < config.train.generated_fraction:
                item = random.choice(minted_train)
                source = "minted"
            else:
                item = random.choice(real_train)
                source = "real"
            parts = loss_for(item, training=True)
            loss = parts.total
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss is not finite")
            audio_losses.append(float(parts.audio.detach()))
            if parts.abc is not None:
                abc_losses.append(float(parts.abc.detach()))
            (loss / config.train.gradient_accumulation).backward()
            mean_loss += float(loss.detach()) / config.train.gradient_accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        metric: dict[str, Any] = {
            "step": step,
            "loss": mean_loss,
            "last_source": source,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": float(gradient_norm),
            "elapsed_seconds": time.monotonic() - started,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "audio_loss": sum(audio_losses) / len(audio_losses),
        }
        if abc_losses:
            metric["abc_loss"] = sum(abc_losses) / len(abc_losses)
        if step <= 3 or step % 10 == 0:
            print(
                f"step {step}/{config.train.steps} loss={mean_loss:.4f} lr={metric['learning_rate']:.2e} "
                f"vram={metric['peak_vram_gb']:.1f} GiB",
                flush=True,
            )
        if step % config.train.save_every == 0 or step == config.train.steps:
            rng = _rng_state()
            try:
                real_metrics = validation_losses(real_validation)
                minted_metrics = validation_losses(minted_validation)
                metric["real_validation"] = real_metrics["total"]
                metric["real_audio_validation"] = real_metrics["audio"]
                if "abc" in real_metrics:
                    metric["real_abc_validation"] = real_metrics["abc"]
                metric["minted_validation"] = minted_metrics["total"]
            finally:
                _restore_rng(rng)
            metadata = {
                "rank": config.train.rank,
                "step": step,
                "base_model": config.assets.model_repo,
                "base_revision": assets.model_revision,
                "trigger": config.dataset.trigger,
                "cot": config.train.cot,
                "abc_loss_weight": config.train.abc_loss_weight,
                "audio_loss_weight": config.train.audio_loss_weight,
                "head_hash": prepared["head_hash"],
                "acoustic_adapter": f"{output.name}/nar_lora_joint_v4.fl_yue2.safetensors",
            }
            native_path = adapters / f"step-{step:06d}.comfyui.safetensors"
            fl_path = adapters / f"step-{step:06d}.fl_yue2.safetensors"
            export_native_ar(model, native_path, metadata)
            export_fl_ar(model, fl_path, metadata)
            _save_resume(resume_path, signature, step, named, optimizer)
            metric["native_adapter"] = str(native_path)
            metric["fl_adapter"] = str(fl_path)
            abc_validation = f" real_abc={metric['real_abc_validation']:.4f}" if "real_abc_validation" in metric else ""
            print(
                f"checkpoint {step}: real_val={metric['real_validation']:.4f} "
                f"minted_val={metric['minted_validation']:.4f}{abc_validation}",
                flush=True,
            )
        _append_metric(metrics_path, metric)
    write_json(
        output / "run.json",
        {
            "status": "complete",
            "step": config.train.steps,
            "trigger": config.dataset.trigger,
            "cot": config.train.cot,
            "prepared": str(prepared_path),
            "signature": signature,
            "adapters": str(adapters),
        },
    )
    return output
