from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import DatasetConfig

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}


def fingerprint(value: Any) -> str:
    def encode_path(item: Any) -> str:
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Object of type {item.__class__.__name__} is not JSON serializable")

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=encode_path,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audio_files(directory: Path) -> list[Path]:
    root = directory.resolve(strict=True)
    files = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise ValueError(f"No WAV, FLAC or MP3 files in {root}")
    if len({path.stem.casefold() for path in files}) != len(files):
        raise ValueError("Audio filenames must have unique stems")
    return files


def sidecar(audio: Path, kind: str) -> Path:
    return audio.with_suffix(f".{kind}.txt")


def abc_sidecar(audio: Path) -> Path:
    return audio.with_suffix(".abc")


def initialize_sidecars(config: DatasetConfig, instrumental: bool, overwrite: bool = False) -> list[Path]:
    written: list[Path] = []
    for audio in audio_files(config.audio_directory):
        values = {
            sidecar(audio, "caption"): config.default_style + "\n",
            sidecar(audio, "lyrics"): "" if instrumental else "[Verse]\nREPLACE WITH REVIEWED LYRICS\n",
        }
        if config.require_abc:
            values[abc_sidecar(audio)] = "X:1\nT:REPLACE WITH REVIEWED SHEETSAGE2 ABC\n"
        for path, text in values.items():
            if overwrite or not path.exists():
                path.write_text(text, encoding="utf-8")
                written.append(path)
    return written


def build_manifest(config: DatasetConfig, output: Path) -> Path:
    import soundfile as sf

    errors: list[str] = []
    songs: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for audio in audio_files(config.audio_directory):
        caption_path = sidecar(audio, "caption")
        lyrics_path = sidecar(audio, "lyrics")
        abc_path = abc_sidecar(audio)
        if not lyrics_path.is_file():
            errors.append(f"{audio.name}: missing {lyrics_path.name}; use an empty file for instrumental audio")
            continue
        style = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else config.default_style
        lyrics = lyrics_path.read_text(encoding="utf-8").strip()
        if not style:
            errors.append(f"{audio.name}: style caption is empty")
            continue
        if "REPLACE WITH REVIEWED LYRICS" in lyrics:
            errors.append(f"{audio.name}: lyrics template has not been reviewed")
            continue
        abc: str | None = None
        if config.require_abc:
            if not abc_path.is_file():
                errors.append(f"{audio.name}: missing {abc_path.name}; export this track's SheetSage2 score.abc here")
                continue
            abc = abc_path.read_text(encoding="utf-8").strip()
            if not abc:
                errors.append(f"{audio.name}: {abc_path.name} is empty")
                continue
            if "REPLACE WITH REVIEWED SHEETSAGE2 ABC" in abc:
                errors.append(f"{audio.name}: ABC template has not been replaced with reviewed SheetSage2 output")
                continue
            if "\x00" in abc:
                errors.append(f"{audio.name}: {abc_path.name} contains a NUL character")
                continue
        info = sf.info(audio)
        if info.frames < info.samplerate or info.channels not in (1, 2):
            errors.append(f"{audio.name}: expected at least one second of mono/stereo audio")
            continue
        digest = file_sha256(audio)
        if digest in hashes:
            errors.append(f"{audio.name}: duplicate audio content")
            continue
        hashes.add(digest)
        identity_path = sidecar(audio, "song")
        identity = identity_path.read_text(encoding="utf-8").strip() if identity_path.is_file() else audio.stem
        song = {
            "name": audio.stem,
            "audio": str(audio.resolve()),
            "sha256": digest,
            "song": identity,
            "style": f"{config.trigger}, {style}" if config.trigger else style,
            "lyrics": lyrics,
            "seconds": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "instrumental": not lyrics,
        }
        if abc is not None:
            song["abc"] = abc
            song["abc_path"] = str(abc_path.resolve())
        songs.append(song)
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))
    groups = sorted({song["song"] for song in songs}, key=lambda name: fingerprint([config.split_seed, name]))
    held_count = 0
    if config.validation_fraction and len(groups) > 1:
        held_count = min(len(groups) - 1, max(1, round(len(groups) * config.validation_fraction)))
    held = set(groups[:held_count])
    for song in songs:
        song["split"] = "validation" if song["song"] in held else "train"
    manifest: dict[str, Any] = {
        "version": 1,
        "trigger": config.trigger,
        "directory": str(config.audio_directory),
        "split_seed": config.split_seed,
        "require_abc": config.require_abc,
        "songs": songs,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(output, manifest)
    return output


def verify_manifest(manifest: dict[str, Any]) -> None:
    for song in manifest["songs"]:
        path = Path(song["audio"])
        if not path.is_file() or file_sha256(path) != song["sha256"]:
            raise ValueError(f"{song['name']}: audio changed or is missing; run prepare again")
