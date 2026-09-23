"""Portable, non-pickle ABC grammar regularizer built from minted score artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import fingerprint, read_json, write_json


def load_abc_regularizer(path: Path) -> list[dict[str, Any]]:
    pack = read_json(path)
    records = pack.get("records", [])
    if pack.get("version") != 1 or not records:
        raise ValueError("ABC regularizer must be a version-1 pack with records")
    seen = set()
    scores: dict[str, str] = {}
    for item in records:
        for key in ("name", "style", "abc"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"ABC regularizer requires nonempty {key}")
        if (
            not isinstance(item.get("lyrics"), str)
            or item.get("src") not in {"minted", "minted_val"}
            or item.get("cot") not in {"melody", "full"}
        ):
            raise ValueError("ABC regularizer requires lyrics, split and cot=melody/full")
        if item["name"] in seen:
            raise ValueError("Duplicate ABC regularizer identity")
        seen.add(item["name"])
        score_hash = fingerprint(item["abc"].strip())
        if score_hash in scores and scores[score_hash] != item["src"]:
            raise ValueError("ABC regularizer score leaks between train and validation")
        scores[score_hash] = item["src"]
        item["abc_only"] = True
    if {item["src"] for item in records} != {"minted", "minted_val"}:
        raise ValueError("ABC regularizer needs both minted and minted_val records")
    return records


def build_abc_regularizer(tracks: Path, semantic_records: list[dict], output: Path) -> Path:
    """Join by exact minted ID; inherit held-out membership from the pinned pack."""
    if output.exists():
        raise FileExistsError(output)
    records = []
    for item in semantic_records:
        name = item["name"]
        if Path(name).name != name:
            raise ValueError("Invalid minted track ID")
        folder = tracks / name
        score = folder / "score.abc"
        request_path = folder / "request.json"
        if not score.exists() or not request_path.exists():
            continue
        request = read_json(request_path)
        if request.get("cot") not in {"melody", "full"}:
            continue
        records.append(
            {
                "name": name,
                "src": item["src"],
                "style": item["style"],
                "lyrics": item["lyrics"],
                "abc": score.read_text(encoding="utf-8").strip(),
                "cot": request["cot"],
            }
        )
    pack = {"version": 1, "records": records, "source": str(tracks.resolve())}
    # Validate before publishing, including exact duplicate score leakage.
    temporary = output.with_suffix(".checking.json")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        write_json(temporary, pack)
        load_abc_regularizer(temporary)
        write_json(output, pack)
    finally:
        temporary.unlink(missing_ok=True)
    return output
