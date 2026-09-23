from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from yue2_lora import assets
from yue2_lora.config import load_config
from yue2_lora.lora import export_nar_companions, load_nar_checkpoint, target_names


@pytest.mark.parametrize("pair", ["v4", "v5", "v8", "v9"])
def test_pair_resolves_both_files_at_same_revision(monkeypatch, pair):
    config = load_config(Path(__file__).parents[1] / "configs/examples/v9_off.toml")
    calls = []
    monkeypatch.setattr(assets, "_snapshot", lambda repo, revision, path, offline: path)

    def fake_file(repo, revision, filename, directory, expected, offline):
        calls.append((revision, filename, expected))
        return directory / filename

    monkeypatch.setattr(assets, "_file", fake_file)
    resolved = assets.resolve_assets(replace(config.assets, pair=pair), offline=True)
    assert resolved.pair == pair
    assert calls[0][0] == calls[1][0] == assets.PAIRS[pair]["revision"]
    assert calls[0][1].startswith(f"tokenizer_head_joint_{pair}.")
    assert calls[1][1].startswith(f"nar_lora_joint_{pair}.")
    assert all(len(call[2]) == 64 for call in calls)


def test_nar_safetensors_preserve_order_and_full_io(tmp_path):
    state = {}
    for index, name in enumerate(target_names(1, "nar")):
        key = name.removeprefix("model.")
        state[key + ".lora_A"] = torch.full((2, 4), float(index))
        state[key + ".lora_B"] = torch.full((4, 2), float(index + 1))
    for module in ("vae2llm", "llm2vae"):
        state[module + ".weight"] = torch.ones(4, 4)
        state[module + ".bias"] = torch.ones(4)
    path = tmp_path / "nar_lora_joint_v9.safetensors"
    save_file(state, str(path))
    loaded = load_nar_checkpoint(path, 1)
    assert loaded["rank"] == 2
    assert len(loaded["lora"]) == 14
    assert loaded["lora"][12][0, 0] == 6
    torch.testing.assert_close(loaded["io"]["vae2llm"]["weight"], state["vae2llm.weight"])

    base = torch.nn.Module()
    base.config = type("Config", (), {"num_hidden_layers": 1})()
    for name in target_names(1, "nar"):
        current = base
        for part in name.split(".")[:-1]:
            if not hasattr(current, part):
                current.add_module(part, torch.nn.Module())
            current = getattr(current, part)
        current.add_module(name.split(".")[-1], torch.nn.Linear(4, 4))
    for module in ("vae2llm", "llm2vae"):
        base.add_module(module, torch.nn.Linear(4, 4))
    native, fl = export_nar_companions(path, base, tmp_path / "export")
    assert native.name == "nar_lora_joint_v9.comfyui.safetensors"
    assert fl.name == "nar_lora_joint_v9.fl_yue2.safetensors"
    exported = load_file(str(native))
    prefix = "model.layers.0.self_attn.qkv_proj"
    actual = exported[prefix + ".lora_up.weight"] @ exported[prefix + ".lora_down.weight"]
    expected = torch.cat([loaded["lora"][index + 1] @ loaded["lora"][index] for index in (0, 2, 4)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(exported["vae2llm.diff"] + base.vae2llm.weight, state["vae2llm.weight"])
    with safe_open(str(fl), framework="pt") as handle:
        assert handle.metadata()["source_sha256"] == assets.sha256(path)
