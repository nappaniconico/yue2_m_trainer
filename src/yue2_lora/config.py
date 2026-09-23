from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    audio_directory: Path
    trigger: str
    default_style: str
    validation_fraction: float
    split_seed: int
    require_abc: bool = False


@dataclass(frozen=True)
class AssetConfig:
    directory: Path
    model_repo: str
    mert_repo: str
    tokenizer_repo: str
    regularizer_repo: str
    pair: str = "v4"


@dataclass(frozen=True)
class PrepareConfig:
    cache_directory: Path


@dataclass(frozen=True)
class TrainConfig:
    output_directory: Path
    rank: int
    learning_rate: float
    generated_fraction: float
    sequence_tokens: int
    allow_truncation: bool
    steps: int
    save_every: int
    schedule_steps: int
    warmup_steps: int
    gradient_accumulation: int
    seed: int
    cot: str = "off"
    abc_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    objective: str = "abc+semantic"
    abc_regularizer: Path | None = None
    abc_regularizer_fraction: float = 0.0


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    assets: AssetConfig
    prepare: PrepareConfig
    train: TrainConfig
    source: Path


def _path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required(section: dict[str, Any], name: str) -> Any:
    if name not in section:
        raise ValueError(f"Missing config value: {name}")
    return section[name]


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve(strict=True)
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    base = source.parent
    dataset = raw.get("dataset", {})
    assets = raw.get("assets", {})
    prepare = raw.get("prepare", {})
    train = raw.get("train", {})
    result = Config(
        dataset=DatasetConfig(
            audio_directory=_path(base, _required(dataset, "audio_directory")),
            trigger=str(dataset.get("trigger", "")).strip(),
            default_style=str(_required(dataset, "default_style")).strip(),
            validation_fraction=float(dataset.get("validation_fraction", 0.1)),
            split_seed=int(dataset.get("split_seed", 42)),
            require_abc=bool(dataset.get("require_abc", False)),
        ),
        assets=AssetConfig(
            directory=_path(base, assets.get("directory", "../models")),
            model_repo=str(assets.get("model_repo", "m-a-p/YuE2-3B")),
            mert_repo=str(assets.get("mert_repo", "m-a-p/MERT-v2-FullSong")),
            tokenizer_repo=str(
                assets.get("tokenizer_repo", "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4")
            ),
            regularizer_repo=str(assets.get("regularizer_repo", "Mothersuperior/yue2-minted-corpus")),
            pair=str(assets.get("pair", "v4")),
        ),
        prepare=PrepareConfig(cache_directory=_path(base, prepare.get("cache_directory", "../cache/prepared"))),
        train=TrainConfig(
            output_directory=_path(base, train.get("output_directory", "../outputs/yue2_lora")),
            rank=int(train.get("rank", 64)),
            learning_rate=float(train.get("learning_rate", 1e-4)),
            generated_fraction=float(train.get("generated_fraction", 0.5)),
            sequence_tokens=int(train.get("sequence_tokens", 12288)),
            allow_truncation=bool(train.get("allow_truncation", False)),
            steps=int(train.get("steps", 1500)),
            save_every=int(train.get("save_every", 200)),
            schedule_steps=int(train.get("schedule_steps", 3000)),
            warmup_steps=int(train.get("warmup_steps", 50)),
            gradient_accumulation=int(train.get("gradient_accumulation", 2)),
            seed=int(train.get("seed", 42)),
            cot=str(train.get("cot", "off")).strip(),
            abc_loss_weight=float(train.get("abc_loss_weight", 1.0)),
            audio_loss_weight=float(train.get("audio_loss_weight", 1.0)),
            objective=str(train.get("objective", "abc+semantic")),
            abc_regularizer=_path(base, train["abc_regularizer"]) if train.get("abc_regularizer") else None,
            abc_regularizer_fraction=float(train.get("abc_regularizer_fraction", 0.0)),
        ),
        source=source,
    )
    validate_config(result)
    return result


def validate_config(config: Config) -> None:
    if config.assets.pair not in {"v4", "v5", "v8", "v9"}:
        raise ValueError("assets.pair must be v4, v5, v8 or v9")
    if config.train.objective not in {"abc+semantic", "abc-only"}:
        raise ValueError("train.objective must be abc+semantic or abc-only")
    if config.train.objective == "abc-only" and config.train.cot == "off":
        raise ValueError("abc-only requires cot=melody/full")
    if not 0 <= config.train.abc_regularizer_fraction <= 1:
        raise ValueError("train.abc_regularizer_fraction must be in [0, 1]")
    if config.train.abc_regularizer_fraction and config.train.abc_regularizer is None:
        raise ValueError("train.abc_regularizer is required when its fraction is nonzero")
    if config.train.objective == "abc-only" and config.train.abc_regularizer_fraction != 1:
        raise ValueError("abc-only requires abc_regularizer_fraction=1 to avoid semantic supervision")
    if not config.dataset.default_style:
        raise ValueError("dataset.default_style must not be empty")
    if not 0 <= config.dataset.validation_fraction < 1:
        raise ValueError("dataset.validation_fraction must be in [0, 1)")
    if not 1 <= config.train.rank <= 128:
        raise ValueError("train.rank must be in 1..128")
    if not 0 < config.train.generated_fraction < 1:
        raise ValueError("train.generated_fraction must be between 0 and 1")
    if config.train.sequence_tokens < 256:
        raise ValueError("train.sequence_tokens must be at least 256")
    if config.train.cot not in {"off", "melody", "full"}:
        raise ValueError("train.cot must be off, melody or full")
    if config.train.cot != "off" and not config.dataset.require_abc:
        raise ValueError("train.cot=melody/full requires dataset.require_abc=true")
    if not math.isfinite(config.train.audio_loss_weight) or config.train.audio_loss_weight <= 0:
        raise ValueError("train.audio_loss_weight must be positive")
    if not math.isfinite(config.train.abc_loss_weight) or config.train.abc_loss_weight < 0:
        raise ValueError("train.abc_loss_weight must be nonnegative")
    if config.train.cot != "off" and config.train.abc_loss_weight == 0:
        raise ValueError("train.abc_loss_weight must be positive for melody/full training")
    for name in ("steps", "save_every", "schedule_steps", "gradient_accumulation"):
        if getattr(config.train, name) < 1:
            raise ValueError(f"train.{name} must be positive")
