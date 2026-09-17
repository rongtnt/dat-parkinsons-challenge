"""Lane 3: small 3D CNN over the full (2, 48, 64, 64) crop.

Shared by scripts/train_3d.py and by main.py at inference time.
Weights live in models/3d/*.pt as int8 packs (see quant.py).
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SCALE = 4.0
BATCH = 16
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "3d")


def make_input(crop) -> np.ndarray:
    """crop (2,48,64,64) -> float32 (2,48,64,64).

    ch0 = peak-normalised - 0.5, ch1 = background-normalised / 4 - 0.5.
    """
    x = np.asarray(crop, dtype=np.float32).copy()
    x[0] -= 0.5
    x[1] = x[1] / SCALE - 0.5
    return x


class Block(nn.Module):
    """conv-BN-ReLU-conv-BN + (1x1 conv) shortcut -> ReLU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm3d(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm3d(cout)
            )

    def forward(self, x):
        idt = x if self.short is None else self.short(x)
        h = self.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.relu(h + idt)


class Net3D(nn.Module):
    """(B,2,48,64,64) -> (B,1) logit."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(2, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.stages = nn.Sequential(Block(16, 32, 2), Block(32, 64, 2), Block(64, 128, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = self.stages(self.stem(x))
        h = torch.flatten(self.pool(h), 1)
        return self.fc(self.drop(h))


def n_params(model=None) -> int:
    model = model if model is not None else Net3D()
    return sum(p.numel() for p in model.parameters())


def _dequantize_sd():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from quant import dequantize_sd  # noqa: E402  (same directory at inference)

    return dequantize_sd


def load_models(device="cpu", model_dir=None):
    """Every models/3d/*.pt, dequantised from int8, in eval mode on `device`."""
    dequantize_sd = _dequantize_sd()
    paths = sorted(glob.glob(os.path.join(model_dir or MODEL_DIR, "*.pt")))
    models = []
    for p in paths:
        m = Net3D()
        m.load_state_dict(dequantize_sd(torch.load(p, map_location="cpu")))
        models.append(m.to(device).eval())
    return models


_CACHE = {}


@torch.no_grad()
def predict_3d(crops, device="cpu") -> np.ndarray:
    """Raw mean-TTA ({identity, X-flip}) mean-model logits. np.nan where crop is None."""
    key = str(device)
    if key not in _CACHE:
        _CACHE[key] = load_models(device)
    models = _CACHE[key]
    out = np.full(len(crops), np.nan, dtype=np.float64)
    idx = [i for i, c in enumerate(crops) if c is not None]
    if not idx or not models:
        return out
    for s in range(0, len(idx), BATCH):
        chunk = idx[s:s + BATCH]
        xb = torch.from_numpy(np.stack([make_input(crops[i]) for i in chunk])).to(device)
        xf = torch.flip(xb, dims=[-1])
        acc = torch.zeros(len(chunk), dtype=torch.float32, device=device)
        for m in models:
            acc += (m(xb).squeeze(1) + m(xf).squeeze(1)) / 2.0
        out[chunk] = (acc / len(models)).float().cpu().numpy()
    return out
