from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import AssetConfig

# Revisions and digests are the reviewed values used by ComfyUI-FL-YuE2 0.2.3.
MODEL_REVISION = "1a96eca688d6ae5d7f0feb88573fec89920fcd19"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
TOKENIZER_REVISION = "f2278a2e005dc4ecc421c53a0929f62b3aeb2280"
REGULARIZER_REVISION = "5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35"
HEAD_SHA256 = "d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e"
NAR_SHA256 = "df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe"
REGULARIZER_SHA256 = "bdd9b9780de46bb0752c3e3bc101869759493443c6e35b1b2d4a2c5033eadc4e"


@dataclass(frozen=True)
class Assets:
    model: Path
    mert: Path
    head: Path
    nar: Path
    regularizer: Path
    model_revision: str = MODEL_REVISION
    mert_revision: str = MERT_REVISION

    def json(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> Path:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return path


def _snapshot(repo: str, revision: str, local_dir: Path, offline: bool) -> Path:
    if offline:
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Offline asset directory not found: {local_dir}")
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    )


def _file(repo: str, revision: str, filename: str, directory: Path, expected: str, offline: bool) -> Path:
    path = directory / filename
    if not path.is_file():
        if offline:
            raise FileNotFoundError(f"Offline asset not found: {path}")
        directory.mkdir(parents=True, exist_ok=True)
        path = Path(
            hf_hub_download(
                repo_id=repo,
                revision=revision,
                filename=filename,
                local_dir=directory,
                local_dir_use_symlinks=False,
                repo_type="dataset" if repo == "Mothersuperior/yue2-minted-corpus" else None,
            )
        )
    return _verify(path, expected)


def resolve_assets(config: AssetConfig, offline: bool = False) -> Assets:
    root = config.directory
    model = _snapshot(config.model_repo, MODEL_REVISION, root / "YuE2-3B", offline)
    mert = _snapshot(config.mert_repo, MERT_REVISION, root / "MERT-v2-FullSong", offline)
    training = root / "training_assets"
    head = _file(
        config.tokenizer_repo,
        TOKENIZER_REVISION,
        "tokenizer_head_joint_v4.pt",
        training,
        HEAD_SHA256,
        offline,
    )
    nar = _file(
        config.tokenizer_repo,
        TOKENIZER_REVISION,
        "nar_lora_joint_v4.pt",
        training,
        NAR_SHA256,
        offline,
    )
    regularizer = _file(
        config.regularizer_repo,
        REGULARIZER_REVISION,
        "regularizer/minted_regularizer_pack.pt",
        training,
        REGULARIZER_SHA256,
        offline,
    )
    return Assets(model=model, mert=mert, head=head, nar=nar, regularizer=regularizer)
