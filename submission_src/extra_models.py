"""Lane 2 inference module: handcrafted features (LightGBM + logistic regression) and a coronal 2.5D CNN.

Single source of truth for feature extraction and the coronal-CNN input so training and inference share
one definition (scripts/ import this file via sys.path).  Runtime deps: numpy, scipy, lightgbm, torch,
torchvision only.  Models are loaded from Path(__file__).parent/"models".
"""
import json
import os
from pathlib import Path

# MUST precede the lightgbm/torch imports.  torch and lightgbm each ship their own libomp; on
# macOS/arm64 loading both in one process segfaults (exit 139) whichever order they are imported in
# -- verified here on torch 2.14 + lightgbm 4.7.  Capping OpenMP to one thread before either library
# initialises fixes it.  main.py must therefore either `import extra_models` BEFORE `import torch`,
# or set OMP_NUM_THREADS itself before any import.  Not reproducible on Linux here -> unverified for
# the A100 container, but the setting is harmless there (inference is CUDA-bound, prep is by process).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
# lightgbm MUST load before torch: torch initialises its own libomp first and a LightGBM Booster
# then segfaults (verified on macOS/arm64 here, exit 139).  This module keeps torch lazy (_torch()),
# so importing extra_models before torch makes every caller safe -> in main.py put
# `import extra_models` ABOVE `import torch`.
import lightgbm as lgb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from quant import dequantize_sd

MODELS = Path(__file__).resolve().parent / "models"

# ---------------------------------------------------------------- Step A: handcrafted features

# striatal box inside the (48, 64, 64) crop -> (24, 32, 48) in (Z, Y, X)
BOX = (slice(12, 36), slice(16, 48), slice(8, 56))
X_MID = 24          # midline inside the box (global x = 32)


def _asym(a, b):
    return abs(a - b) / (a + b) if (a + b) > 0 else 0.0


def compute_features(crop: np.ndarray, sbr: float) -> dict:
    """crop: (2, 48, 64, 64) in (C, Z, Y, X).  Uses channel 1 (background-normalised)."""
    c = np.asarray(crop)[1].astype(np.float32)
    B = c[BOX]
    flat = B.ravel()
    k = max(int(round(0.005 * flat.size)), 1)
    peak = float(np.mean(np.partition(flat, -k)[-k:]))
    M = B > 0.5 * peak
    M70 = B > 0.7 * peak

    f = {}
    f["peak"] = peak
    f["sbr"] = float(sbr)
    f["vol50"] = float(M.sum())
    f["vol70"] = float(M70.sum())
    f["mean_B"] = float(B.mean())
    f["p90_B"] = float(np.percentile(B, 90))
    f["ring"] = float((c.sum() - B.sum()) / (c.size - B.size))

    for s, xs in (("L", slice(None, X_MID)), ("R", slice(X_MID, None))):
        Bs, Ms, M70s = B[..., xs], M[..., xs], M70[..., xs]
        f["max_" + s] = float(Bs.max())
        fl = Bs.ravel()
        k1 = max(int(round(0.01 * fl.size)), 1)
        f["top1_" + s] = float(np.mean(np.partition(fl, -k1)[-k1:]))
        f["vol50_" + s] = float(Ms.sum())
        f["vol70_" + s] = float(M70s.sum())
        zz, yy, _ = np.nonzero(Ms)
        if zz.size:
            w = Bs[Ms]
            f["extent_y_" + s] = float(yy.max() - yy.min())
            f["extent_z_" + s] = float(zz.max() - zz.min())
            ycen = float((yy * w).sum() / w.sum()) if w.sum() > 0 else 0.0
            f["ycen_" + s] = ycen
            post, ant = w[yy < ycen], w[yy >= ycen]
            ok = post.size > 0 and ant.size > 0 and float(ant.mean()) > 0
            f["pa_ratio_" + s] = float(post.mean() / ant.mean()) if ok else 1.0
        else:
            f["extent_y_" + s] = 0.0
            f["extent_z_" + s] = 0.0
            f["ycen_" + s] = 0.0
            f["pa_ratio_" + s] = 1.0

    f["asym_max"] = _asym(f["max_L"], f["max_R"])
    f["asym_top1"] = _asym(f["top1_L"], f["top1_R"])
    f["asym_vol50"] = _asym(f["vol50_L"], f["vol50_R"])
    hi = max(f["top1_L"], f["top1_R"])
    f["min_over_max_top1"] = min(f["top1_L"], f["top1_R"]) / hi if hi > 0 else 0.0
    f["min_pa"] = min(f["pa_ratio_L"], f["pa_ratio_R"])
    f["min_extent_y"] = min(f["extent_y_L"], f["extent_y_R"])

    return {k_: (0.0 if not np.isfinite(v) else float(v)) for k_, v in f.items()}


def demo():
    rng = np.random.default_rng(0)
    crop = rng.random((2, 48, 64, 64)).astype(np.float32) * 3.0
    f = compute_features(crop, 4.0)
    assert len(f) == 29, len(f)
    assert all(np.isfinite(v) for v in f.values())
    z = compute_features(np.zeros((2, 48, 64, 64), np.float32), 0.0)
    assert all(np.isfinite(v) for v in z.values()), z
    print("demo ok", len(f), "features")


if __name__ == "__main__":
    demo()


# ---------------------------------------------------------------- Step C: coronal 2.5D input + net

Y_RANGES = ((23, 29), (29, 35), (35, 41))   # coronal slabs over Y (posterior -> anterior)
Z_PAD = 8                                   # 48 -> 64
SEEDS = (0, 1)


def make_input_cor(crop: np.ndarray, yshift: int = 0) -> np.ndarray:
    """crop (2,48,64,64) -> (3,64,64) float32: 3 coronal slabs of channel 1, scaled, zero-padded in Z."""
    c = np.asarray(crop)[1].astype(np.float32)
    slabs = [c[:, a + yshift:b + yshift, :].mean(1) for a, b in Y_RANGES]   # each (Z=48, X=64)
    x = np.stack(slabs) / 4.0 - 0.5
    out = np.zeros((3, 64, 64), np.float32)
    out[:, Z_PAD:Z_PAD + x.shape[1], :] = x
    return out


X_RANGES = ((20, 26), (29, 35), (38, 44))   # sagittal slabs over X (left striatum, midline, right)


def make_input_sag(crop: np.ndarray, xshift: int = 0) -> np.ndarray:
    """crop (2,48,64,64) -> (3,64,64) float32: 3 sagittal slabs of channel 1, scaled, zero-padded in Z.
    L/R mirror for this view is a channel reversal (x[[2,1,0]] == flip(-3)), not a spatial flip."""
    c = np.asarray(crop)[1].astype(np.float32)
    slabs = [c[:, :, a + xshift:b + xshift].mean(2) for a, b in X_RANGES]   # each (Z=48, Y=64)
    x = np.stack(slabs) / 4.0 - 0.5
    out = np.zeros((3, 64, 64), np.float32)
    out[:, Z_PAD:Z_PAD + x.shape[1], :] = x
    return out


def mirror_cor(t):
    """L/R mirror for the coronal view: flip the X axis (last dim).  Works on (3,H,W) and (N,3,H,W)."""
    return torch.flip(t, dims=[-1])


def mirror_sag(t):
    """L/R mirror for the sagittal view: reverse the slab order (3 channels -> flip dim -3)."""
    return torch.flip(t, dims=[-3])


class NetCorT(nn.Module):
    """torchvision resnet18 truncated after layer3 (256 ch) -> GAP -> Linear(256, 1)."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.interpolate(x, size=128, mode="bilinear", align_corners=False)
        return self.fc(self.pool(self.stem(x)).flatten(1))


# ---------------------------------------------------------------- Step D: inference entry point

_CACHE = {}


def _feature_matrix(crops, sbrs, valid):
    names = json.loads((MODELS / "feat/features.json").read_text())
    rows = [compute_features(crops[i], float(sbrs[i])) for i in valid]
    return np.array([[r[n] for n in names] for r in rows], np.float64) if rows else np.zeros((0, len(names)))


def _lgb_fold(k, X):
    key = ("lgb", k)
    if key not in _CACHE:
        _CACHE[key] = lgb.Booster(model_file=str(MODELS / f"feat/lgb_fold{k}.txt"))
    return _CACHE[key].predict(X, raw_score=True)


def _lr_fold(k, X):
    if "lr" not in _CACHE:
        _CACHE["lr"] = json.loads((MODELS / "feat/lr.json").read_text())
    p = _CACHE["lr"][str(k)]
    z = (X - np.array(p["mean"])) / np.array(p["scale"])
    return z @ np.array(p["coef"]) + p["intercept"]


VIEWS = {                       # key -> (weights subdir, input builder, mirror for TTA)
    "cnn_cor":  ("2dcor",  make_input_cor, mirror_cor),
    "cnn_corc": ("2dcorc", make_input_cor, mirror_cor),
    "cnn_sag":  ("2dsag",  make_input_sag, mirror_sag),
}


def _seeds_for(view, k):
    return [s for s in SEEDS if (MODELS / f"{VIEWS[view][0]}/f{k}_s{s}.pt").exists()]


def _cnn_fold(view, k, xb, device):
    """xb: float tensor (N,3,64,64) on `device`.  Mean logit over this fold's seeds, TTA id + mirror."""
    sub, _, mirror = VIEWS[view]
    outs = []
    for s in _seeds_for(view, k):
        key = (view, k, s, str(device))
        if key not in _CACHE:
            net = NetCorT(pretrained=False)
            net.load_state_dict(dequantize_sd(torch.load(MODELS / f"{sub}/f{k}_s{s}.pt", map_location="cpu")))
            _CACHE[key] = net.to(device).eval()
        net = _CACHE[key]
        with torch.no_grad():
            chunks = []
            for i in range(0, len(xb), 64):
                b = xb[i:i + 64]
                chunks.append(((net(b)[:, 0] + net(mirror(b))[:, 0]) / 2).float().cpu().numpy())
        outs.append(np.concatenate(chunks).astype(np.float64))
    return np.mean(outs, axis=0)


def available_views():
    return [v for v in VIEWS if all(_seeds_for(v, k) for k in range(5))]


def predict_extra(crops, sbrs, device) -> dict:
    """crops: list of (2,48,64,64) arrays or None.  Returns raw logits per model, np.nan for None crops."""
    n = len(crops)
    valid = [i for i, c in enumerate(crops) if c is not None]
    views = available_views()
    out = {k: np.full(n, np.nan) for k in ("feat_lgb", "feat_lr", *views)}
    if not valid:
        return out

    X = _feature_matrix(crops, sbrs, valid)
    out["feat_lgb"][valid] = np.mean([_lgb_fold(k, X) for k in range(5)], axis=0)
    out["feat_lr"][valid] = np.mean([_lr_fold(k, X) for k in range(5)], axis=0)

    for v in views:
        build = VIEWS[v][1]
        xb = torch.from_numpy(np.stack([build(crops[i]) for i in valid])).to(device)
        out[v][valid] = np.mean([_cnn_fold(v, k, xb, device) for k in range(5)], axis=0)
    return out
