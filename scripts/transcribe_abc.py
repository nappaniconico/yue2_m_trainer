#!/usr/bin/env python3
"""Batch-transcribe audio into YuE2-compatible SheetSage2 ABC sidecars.

Run this script in the separate Python environment recommended by SheetSage2.
It deliberately has no dependency on the yue2_lora package.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}
ABC_TEMPLATE_MARKER = "REPLACE WITH REVIEWED SHEETSAGE2 ABC"


def _audio_files(directory: Path) -> list[Path]:
    root = directory.expanduser().resolve(strict=True)
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise ValueError(f"No WAV, FLAC or MP3 files in {root}")
    if len({path.stem.casefold() for path in files}) != len(files):
        raise ValueError("Audio filenames must have unique stems")
    return files


def _target_path(audio: Path, output_directory: Path | None) -> Path:
    return (output_directory if output_directory is not None else audio.parent) / f"{audio.stem}.abc"


def _needs_transcription(path: Path, overwrite: bool) -> bool:
    if overwrite or not path.exists():
        return True
    return ABC_TEMPLATE_MARKER in path.read_text(encoding="utf-8")


def _extract_abc(result: Any) -> str:
    abc = result.get("abc") if isinstance(result, dict) else getattr(result, "abc", None)
    if not isinstance(abc, str) or not abc.strip():
        raise ValueError("SheetSage2 returned no nonempty ABC text")
    abc = abc.strip()
    if "\x00" in abc:
        raise ValueError("SheetSage2 returned ABC containing a NUL character")
    return abc + "\n"


def _write_abc(path: Path, abc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(abc, encoding="utf-8")
    os.replace(temporary, path)


def _load_model(
    model_name: str,
    base_model_name: str | None,
    revision: str | None,
    device: str,
    dtype: str,
    offline: bool,
):
    import torch
    from transformers import AutoModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    selected_dtype = (
        dtypes[dtype] if dtype != "auto" else (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": offline,
        "torch_dtype": selected_dtype,
    }
    if revision:
        kwargs["revision"] = revision
    if base_model_name:
        kwargs["base_model_path"] = base_model_name
    print(f"Loading SheetSage2 from {model_name} on {device} ({selected_dtype})...", flush=True)
    model = AutoModel.from_pretrained(model_name, **kwargs).eval().to(device)
    return model, device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-transcribe audio to SheetSage2 ABC sidecars")
    parser.add_argument("--input-directory", default="YouTube", help="Directory containing WAV/FLAC/MP3 files")
    parser.add_argument(
        "--output-directory",
        help="Optional ABC output directory; by default each .abc is written beside its audio file",
    )
    parser.add_argument("--model", default="m-a-p/SheetSage2", help="Hugging Face repo ID or local model directory")
    parser.add_argument(
        "--base-model",
        help="Optional local MERT-v2-FullSong parent directory; avoids downloading the parent model again",
    )
    parser.add_argument("--revision", help="Optional pinned Hugging Face model revision")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--melody-only",
        action="store_true",
        help="Omit chord symbols for YuE2 cot=melody; default output is melody plus chords for cot=full",
    )
    parser.add_argument("--offline", action="store_true", help="Use only an already downloaded local model")
    parser.add_argument("--overwrite", action="store_true", help="Replace reviewed ABC files as well as templates")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later tracks after a transcription error and exit nonzero at the end",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = Path(args.input_directory)
    output = Path(args.output_directory).expanduser().resolve() if args.output_directory else None
    pairs = [(audio, _target_path(audio, output)) for audio in _audio_files(source)]
    pending = [(audio, target) for audio, target in pairs if _needs_transcription(target, args.overwrite)]
    skipped = len(pairs) - len(pending)
    if not pending:
        print(f"Nothing to transcribe: protected {skipped} existing reviewed ABC files.")
        return 0

    model, device = _load_model(
        args.model,
        args.base_model,
        args.revision,
        args.device,
        args.dtype,
        args.offline,
    )
    failures: list[tuple[Path, Exception]] = []
    written = 0
    for index, (audio, target) in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] {audio.name} -> {target}", flush=True)
        try:
            result = model.transcribe(str(audio), melody_only=args.melody_only)
            _write_abc(target, _extract_abc(result))
            written += 1
        except Exception as error:
            failures.append((audio, error))
            print(f"ERROR {audio.name}: {error}", flush=True)
            if not args.continue_on_error:
                raise

    print(f"ABC transcription complete: wrote={written}, protected={skipped}, failed={len(failures)}")
    if failures:
        for audio, error in failures:
            print(f"  {audio.name}: {error}")
        return 1
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
