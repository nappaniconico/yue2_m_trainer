import wave
from pathlib import Path

from yue2_lora.config import DatasetConfig
from yue2_lora.dataset import abc_sidecar, build_manifest, fingerprint, initialize_sidecars, read_json


def _wav(path: Path, seconds: int = 1) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(b"\0\0\0\0" * 48_000 * seconds)


def test_sidecars_and_grouped_split(tmp_path: Path) -> None:
    for seconds, name in enumerate(("one", "two", "three"), 1):
        _wav(tmp_path / f"{name}.wav", seconds)
    config = DatasetConfig(tmp_path, "genre_token", "Kawaii Future Bass", 0.34, 42)
    assert len(initialize_sidecars(config, instrumental=True)) == 6
    path = build_manifest(config, tmp_path / "dataset.json")
    manifest = read_json(path)
    assert len(manifest["songs"]) == 3
    assert {song["split"] for song in manifest["songs"]} == {"train", "validation"}
    assert all(song["style"].startswith("genre_token, ") for song in manifest["songs"])


def test_fingerprint_serializes_paths_stably(tmp_path: Path) -> None:
    assert fingerprint({"output": tmp_path}) == fingerprint({"output": str(tmp_path)})


def test_required_abc_is_stored_in_manifest(tmp_path: Path) -> None:
    _wav(tmp_path / "song.wav")
    config = DatasetConfig(tmp_path, "genre_token", "Kawaii Future Bass", 0.0, 42, require_abc=True)
    initialize_sidecars(config, instrumental=True)
    abc_sidecar(tmp_path / "song.wav").write_text("X:1\nM:4/4\nK:C\nCDEF|", encoding="utf-8")
    manifest = read_json(build_manifest(config, tmp_path / "dataset.json"))
    assert manifest["require_abc"] is True
    assert manifest["songs"][0]["abc"] == "X:1\nM:4/4\nK:C\nCDEF|"
