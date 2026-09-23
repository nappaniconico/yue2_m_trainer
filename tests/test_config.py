from pathlib import Path

import pytest

from yue2_lora.config import load_config


def test_example_config_resolves_paths() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "examples" / "v9_off.toml")
    assert config.dataset.audio_directory.name == "YouTube_1"
    assert config.dataset.trigger == "kfbass_v1"
    assert config.train.rank == 64
    assert config.train.steps == 1500
    assert config.train.cot == "off"


def test_abc_config_enables_full_symbolic_training() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "examples" / "v9_comparison.toml")
    assert config.dataset.require_abc
    assert config.train.cot == "full"
    assert config.train.output_directory.name == "kfb_v9_comparison"


def test_invalid_generated_fraction(tmp_path: Path) -> None:
    text = (Path(__file__).parents[1] / "configs" / "examples" / "v9_off.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(text.replace("generated_fraction = 0.5", "generated_fraction = 1.0"))
    with pytest.raises(ValueError, match="generated_fraction"):
        load_config(path)
