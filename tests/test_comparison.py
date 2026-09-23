import json
from dataclasses import replace
from pathlib import Path

import pytest

from yue2_lora.comparison import compare_runs, variants
from yue2_lora.config import load_config, validate_config
from yue2_lora.dataset import write_json
from yue2_lora.regularizer import build_abc_regularizer, load_abc_regularizer


def test_variants_share_dataset_and_schedule():
    config = load_config(Path(__file__).parents[1] / "configs/examples/v9_comparison.toml")
    runs = variants(config)
    assert len({run.train.output_directory for run in runs.values()}) == 3
    for run in runs.values():
        assert run.dataset == config.dataset
        assert run.assets == config.assets
        assert run.train.steps == config.train.steps
        assert run.train.seed == config.train.seed
        assert run.train.sequence_tokens == config.train.sequence_tokens
    assert runs["abc_only"].train.abc_regularizer_fraction == 1
    assert runs["cot_off"].train.cot == "off"
    with pytest.raises(ValueError, match="assets.pair"):
        validate_config(replace(config, assets=replace(config.assets, pair="v7")))
    with pytest.raises(ValueError, match="avoid semantic"):
        validate_config(replace(config, train=replace(config.train, objective="abc-only")))


def test_abc_pack_inherits_splits_and_rejects_score_leakage(tmp_path):
    records = [
        {"name": name, "src": src, "style": "music", "lyrics": ""}
        for name, src in [("a", "minted"), ("b", "minted_val")]
    ]
    for item in records:
        folder = tmp_path / item["name"]
        folder.mkdir()
        (folder / "score.abc").write_text("K:C\n" + item["name"])
        write_json(folder / "request.json", {"cot": "full"})
    output = build_abc_regularizer(tmp_path, records, tmp_path / "pack.json")
    loaded = load_abc_regularizer(output)
    assert [row["src"] for row in loaded] == ["minted", "minted_val"]
    assert all(row["abc_only"] for row in loaded)
    loaded[1]["abc"] = loaded[0]["abc"]
    write_json(output, {"version": 1, "records": loaded})
    with pytest.raises(ValueError, match="leaks"):
        load_abc_regularizer(output)


def test_report_only_compares_same_protocol_and_steps(tmp_path):
    runs = [tmp_path / "off", tmp_path / "abc"]
    for run, steps in zip(runs, [[0, 2, 4], [0, 2]], strict=True):
        write_json(run / "validation.json", {"signature": "same"})
        (run / "metrics.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "step": step,
                        "validation_signature": "same",
                        "real_audio_validation": 2.0,
                    }
                )
                for step in steps
            )
        )
    report = compare_runs(runs, tmp_path / "comparison.json")
    assert report["common_steps"] == [0, 2]
    write_json(runs[1] / "validation.json", {"signature": "different"})
    with pytest.raises(ValueError, match="signature"):
        compare_runs(runs, tmp_path / "comparison.json")
