from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .config import Config, load_config


def _config(path: str) -> Config:
    config = load_config(path)
    print(f"Config: {config.source}")
    return config


def _assets(config: Config, offline: bool):
    from .assets import resolve_assets

    return resolve_assets(config.assets, offline=offline)


def command_sidecars(args: argparse.Namespace) -> None:
    from .dataset import initialize_sidecars

    config = _config(args.config)
    written = initialize_sidecars(config.dataset, instrumental=args.instrumental, overwrite=args.overwrite)
    print(f"Wrote {len(written)} sidecar files in {config.dataset.audio_directory}")
    suffixes = ".caption.txt, .lyrics.txt and .abc" if config.dataset.require_abc else ".caption.txt and .lyrics.txt"
    print(f"Review every {suffixes} before running prepare.")


def command_download(args: argparse.Namespace) -> None:
    config = _config(args.config)
    assets = _assets(config, offline=False)
    for name, path in assets.json().items():
        print(f"{name}: {path}")


def command_prepare(args: argparse.Namespace) -> None:
    from .prepare import prepare_dataset

    config = _config(args.config)
    result = prepare_dataset(config, _assets(config, args.offline), args.device)
    print(result)


def command_train(args: argparse.Namespace) -> None:
    from .train import train

    config = _config(args.config)
    prepared = Path(args.prepared).resolve(strict=True) if args.prepared else None
    result = train(config, _assets(config, args.offline), prepared_path=prepared, resume=args.resume)
    print(f"Training output: {result}")


def _copy(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite if intentional")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def command_install(args: argparse.Namespace) -> None:
    config = _config(args.config)
    source = config.train.output_directory / "adapters"
    pattern = f"step-{args.step:06d}.comfyui.safetensors" if args.step else "step-*.comfyui.safetensors"
    matches = sorted(source.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trained adapters matching {pattern} in {source}")
    native_ar = matches[-1]
    step = native_ar.name.split(".")[0]
    files = [
        native_ar,
        source / "nar_lora_joint_v4.comfyui.safetensors",
        source / f"{step}.fl_yue2.safetensors",
        source / "nar_lora_joint_v4.fl_yue2.safetensors",
    ]
    destination = (
        Path(args.comfyui).expanduser().resolve(strict=True)
        / "models"
        / "loras"
        / "YuE2"
        / config.train.output_directory.name
    )
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        _copy(path, destination / path.name, args.overwrite)
    print(f"Installed checkpoint {step.removeprefix('step-')} into {destination}")
    print("Native ComfyUI: load the .comfyui AR and NAR files with two Load LoRA nodes.")
    print("ComfyUI-FL-YuE2: select the .fl_yue2 AR file; its paired NAR is loaded from metadata.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YuE2 real-audio genre LoRA trainer")
    parser.set_defaults(function=None)
    subparsers = parser.add_subparsers(dest="command")

    sidecars = subparsers.add_parser("init-sidecars", help="Create configured sidecar templates next to audio")
    sidecars.add_argument("--config", default="configs/kawaii_future_bass.toml")
    sidecars.add_argument("--instrumental", action=argparse.BooleanOptionalAction, default=True)
    sidecars.add_argument("--overwrite", action="store_true")
    sidecars.set_defaults(function=command_sidecars)

    download = subparsers.add_parser("download", help="Download and verify pinned model assets")
    download.add_argument("--config", default="configs/kawaii_future_bass.toml")
    download.set_defaults(function=command_download)

    prepare = subparsers.add_parser("prepare", help="Tokenize the reviewed real-audio dataset")
    prepare.add_argument("--config", default="configs/kawaii_future_bass.toml")
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--offline", action="store_true")
    prepare.set_defaults(function=command_prepare)

    trainer = subparsers.add_parser("train", help="Train AR LoRA and export ComfyUI adapters")
    trainer.add_argument("--config", default="configs/kawaii_future_bass.toml")
    trainer.add_argument("--prepared", help="Prepared manifest; newest cache manifest is used when omitted")
    trainer.add_argument("--resume", action="store_true")
    trainer.add_argument("--offline", action="store_true")
    trainer.set_defaults(function=command_train)

    install = subparsers.add_parser("install-comfyui", help="Copy one checkpoint into a ComfyUI installation")
    install.add_argument("--config", default="configs/kawaii_future_bass.toml")
    install.add_argument("--comfyui", required=True, help="Path to the ComfyUI root")
    install.add_argument("--step", type=int, default=0, help="Checkpoint step; 0 selects the latest")
    install.add_argument("--overwrite", action="store_true")
    install.set_defaults(function=command_install)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.function is None:
        parser.print_help()
        return
    args.function(args)


if __name__ == "__main__":
    main()
