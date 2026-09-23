from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from transformers import AutoFeatureExtractor, AutoModel

from .assets import Assets, sha256
from .config import Config
from .dataset import build_manifest, fingerprint, read_json, verify_manifest, write_json


class TokenHead(nn.Module):
    """MERT layer-20 features to YuE2's 32,768 semantic codes."""

    def __init__(
        self,
        width: int = 512,
        layers: int = 8,
        heads: int = 8,
        window: int = 512,
        input_dim: int = 1024,
        vocab: int = 32768,
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(input_dim, width)
        self.pos = nn.Parameter(torch.empty(1, window, width))
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.enc(self.inp(value) + self.pos[:, : value.shape[1]])))


def load_head(path: Path, device: torch.device) -> TokenHead:
    state = (
        load_file(str(path))
        if path.suffix == ".safetensors"
        else torch.load(path, map_location="cpu", weights_only=True)["model"]
    )
    with torch.device("meta"):
        head = TokenHead()
    head.load_state_dict(state, strict=True, assign=True)
    return head.to(device).eval()


def normalize(features: np.ndarray) -> np.ndarray:
    value = features.astype(np.float32)
    return (value - value.mean(axis=0)) / (value.std(axis=0) + 1e-5)


@torch.inference_mode()
def predict(head: TokenHead, features: np.ndarray, progress: Callable[[], None] = lambda: None) -> np.ndarray:
    features = normalize(features)
    total, window = len(features), head.pos.shape[1]
    output = np.zeros(total, dtype=np.int32)
    starts = list(range(0, max(1, total - window + 1), window // 2))
    if starts[-1] + window < total:
        starts.append(max(0, total - window))
    device = next(head.parameters()).device
    for start in starts:
        progress()
        value = features[start : start + window]
        count = len(value)
        value = np.pad(value, ((0, window - count), (0, 0)))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ids = head(torch.tensor(value[None], device=device))[0, :count].float().argmax(-1).cpu().numpy()
        low = start + (0 if start == 0 else window // 4)
        high = start + count - (0 if start + count >= total else window // 4)
        output[low:high] = ids[low - start : high - start]
    return output


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    divisor = math.gcd(source, target)
    if source == target:
        return audio.astype(np.float32, copy=False)
    return resample_poly(audio, target // divisor, source // divisor, axis=0).astype(np.float32)


@torch.inference_mode()
def mert_features(model: nn.Module, processor, mono: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    chunks = [mono[start : start + 24_000 * 30] for start in range(0, len(mono), 24_000 * 30)]
    chunks = [chunk for chunk in chunks if len(chunk) >= 24_000]
    if not chunks:
        raise ValueError("Training audio must be at least one second")
    outputs = []
    for chunk in chunks:
        inputs = {
            key: value.to(device)
            for key, value in processor([chunk], sampling_rate=24_000, return_tensors="pt").items()
        }
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            value = model(**inputs, output_hidden_states=True).hidden_states[20][0]
        outputs.append(value.detach().float().cpu())
    joined = torch.cat(outputs)
    frame_count = round(len(mono) / 24_000 * 25)
    return F.interpolate(joined.T[None], size=frame_count, mode="linear", align_corners=False)[0].T.half().numpy()


def prepare_dataset(config: Config, assets: Assets, device_name: str = "cuda") -> Path:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for YuE2 dataset preparation")
    device = torch.device(device_name)
    cache = config.prepare.cache_directory
    cache.mkdir(parents=True, exist_ok=True)
    source_path = build_manifest(config.dataset, cache / "dataset.json")
    source = read_json(source_path)
    verify_manifest(source)
    head_hash = sha256(assets.head)
    feature_version = fingerprint([assets.mert_revision, "layer20-kit30s-v1"])
    result = {**source, "head_hash": head_hash, "pair": assets.pair, "mode": "ar", "songs": []}
    processor = AutoFeatureExtractor.from_pretrained(assets.mert, local_files_only=True)
    mert = AutoModel.from_pretrained(assets.mert, trust_remote_code=True, local_files_only=True).to(device).eval()
    mert.requires_grad_(False)
    head = load_head(assets.head, device)
    head.requires_grad_(False)
    try:
        for index, song in enumerate(source["songs"], 1):
            folder = cache / song["sha256"]
            folder.mkdir(exist_ok=True)
            features_path = folder / f"features-{feature_version[:16]}.npy"
            tokens_path = folder / f"tokens-{feature_version[:16]}-{head_hash[:16]}.npy"
            print(f"[{index}/{len(source['songs'])}] prepare {song['name']}", flush=True)
            if not features_path.exists():
                audio, sample_rate = sf.read(song["audio"], dtype="float32", always_2d=True)
                if not np.isfinite(audio).all():
                    raise ValueError(f"{song['name']}: audio has NaN or infinite samples")
                mono = resample(audio.mean(axis=1), sample_rate, 24_000)
                np.save(features_path, mert_features(mert, processor, mono))
            if not tokens_path.exists():
                np.save(tokens_path, predict(head, np.load(features_path, allow_pickle=False)))
            result["songs"].append({**song, "features": str(features_path), "tokens": str(tokens_path)})
    finally:
        del mert, processor, head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["fingerprint"] = fingerprint(result)
    output = cache / f"prepared-{result['fingerprint'][:16]}.json"
    write_json(output, result)
    print(f"Prepared manifest: {output}", flush=True)
    return output
