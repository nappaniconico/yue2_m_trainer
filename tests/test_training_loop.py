"""CPU smoke of orchestration, splits, all objectives, repeated eval and report."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from yue2_lora.assets import Assets, sha256
from yue2_lora.comparison import compare_runs, variants
from yue2_lora.config import load_config
from yue2_lora.dataset import build_manifest, read_json, write_json


def test_three_training_loops_use_common_evaluation(tmp_path, monkeypatch):
    import yue2_lora.train as training

    audio = tmp_path / "audio"
    audio.mkdir()
    for index in range(2):
        path = audio / f"song{index}.wav"
        sf.write(path, np.full(24000, index * 0.1), 24000)
        path.with_suffix(".lyrics.txt").write_text("")
        path.with_suffix(".abc").write_text("X:1\nK:C\nCDEF|")
    config = load_config(Path(__file__).parents[1] / "configs/examples/v9_comparison.toml")
    config = replace(
        config,
        dataset=replace(config.dataset, audio_directory=audio),
        prepare=replace(config.prepare, cache_directory=tmp_path / "cache"),
        train=replace(
            config.train,
            output_directory=tmp_path / "runs",
            steps=2,
            save_every=1,
            abc_regularizer=tmp_path / "abc.json",
            rank=2,
            gradient_accumulation=4,
        ),
    )
    rows = [
        {"name": name, "src": src, "style": "music", "lyrics": "", "abc": score, "cot": "full"}
        for name, src, score in [("a", "minted", "K:C\nCDEF|"), ("b", "minted_val", "K:C\nGABc|")]
    ]
    write_json(config.train.abc_regularizer, {"version": 1, "records": rows})
    files = [tmp_path / name for name in ("head", "nar", "regularizer")]
    for file in files:
        file.write_bytes(b"fixture")
    assets = Assets(tmp_path, tmp_path, *files, pair="v9")
    prepared_path = build_manifest(config.dataset, tmp_path / "prepared.json")
    prepared = read_json(prepared_path)
    prepared["head_hash"] = sha256(assets.head)
    for song in prepared["songs"]:
        tokens = tmp_path / (song["name"] + ".npy")
        np.save(tokens, np.array([1, 2, 3]))
        song["tokens"] = str(tokens)
    write_json(prepared_path, prepared)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A = torch.nn.Parameter(torch.ones(1))

        def to(self, *args, **kwargs):
            return self

    class Tokenizer:
        def __init__(self, *args):
            pass

        def encode(self, text):
            return [ord(value) for value in text]

    observed = []

    def loss(model, sequence, checkpoint_layers, abc_weight, audio_weight):
        value = model.A.square().sum()
        if checkpoint_layers:
            observed.append(sequence.audio_targets is not None)
        return training.LossParts(
            value, value if sequence.abc_targets else None, value if sequence.audio_targets else None
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(training, "_rng_state", dict)
    monkeypatch.setattr(training, "_restore_rng", lambda state: None)
    monkeypatch.setattr(training.YuE2ForCausalLM, "from_pretrained", lambda *args, **kwargs: Model())
    monkeypatch.setattr(training, "YuE2TextTokenizer", Tokenizer)
    monkeypatch.setattr(training, "install_lora", lambda model, rank: model.A.requires_grad_(True))
    monkeypatch.setattr(training, "trainable_parameters", lambda model: {"A": model.A})
    monkeypatch.setattr(training, "export_nar_companions", lambda *args: (tmp_path / "nar", tmp_path / "nar.fl"))
    monkeypatch.setattr(training, "export_native_ar", lambda *args: None)
    monkeypatch.setattr(training, "export_fl_ar", lambda *args: None)
    monkeypatch.setattr(training, "_loss", loss)
    monkeypatch.setattr(
        training,
        "_load_regularizer",
        lambda path: [
            {k: v for k, v in row.items() if k not in {"abc", "cot"}} | {"codec": np.array([1, 2, 3])} for row in rows
        ],
    )
    runs = variants(config)
    for name, run in runs.items():
        observed.clear()
        training.train(run, assets, prepared_path)
        if name == "abc_only":
            assert observed and not any(observed)
        elif name == "cot_off":
            assert all(observed)
        metrics = [json.loads(line) for line in (run.train.output_directory / "metrics.jsonl").read_text().splitlines()]
        assert [row["step"] for row in metrics] == [0, 1, 2]
        assert all("minted_abc_validation" in row and "real_conditioned_audio_validation" in row for row in metrics)
    report = compare_runs([run.train.output_directory for run in runs.values()], tmp_path / "report.json")
    assert report["common_steps"] == [0, 1, 2]
