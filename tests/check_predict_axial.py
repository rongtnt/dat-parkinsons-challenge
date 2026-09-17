"""predict_axial logits must match the OOF-time logits (int8 round-trip included)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_src"
sys.path.insert(0, str(SRC))
import cnn2d
from quant import dequantize_sd

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
uids_c = np.load(ROOT / "data/cache/uids.npy", allow_pickle=True)
pos = {u: i for i, u in enumerate(uids_c)}
crops_all = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")

VARIANTS = [
    ("cnn_2d", "2d", "results/oof_2d.csv", lambda: cnn2d.Net(weights=None), cnn2d.make_input),
    ("cnn_2dt", "2dt", "results/oof_2dt.csv", lambda: cnn2d.NetT(pretrained=False), cnn2d.make_input),
    ("cnn_2dtc", "2dtc", "results/oof_2dtc.csv", lambda: cnn2d.NetT(pretrained=False), cnn2d.make_input),
    ("cnn_2dtc5", "2dtc5", "results/oof_2dtc5.csv", lambda: cnn2d.NetT(pretrained=False), cnn2d.make_input),
    ("cnn_2dt6", "2dt6", "results/oof_2dt6.csv", lambda: cnn2d.NetT6(pretrained=False), cnn2d.make_input6),
    ("cnn_2dt6n", "2dt6n", "results/oof_2dt6n.csv", lambda: cnn2d.NetT6(pretrained=False), cnn2d.make_input6),
    ("cnn_2dt6c5", "2dt6c5", "results/oof_2dt6c5.csv", lambda: cnn2d.NetT6(pretrained=False), cnn2d.make_input6),
]
crops = None
for key, mdir, oof_csv, ctor, mkin in VARIANTS:
    d = SRC / "models" / mdir
    n_all = len(list(d.glob("*.pt")))
    if not (ROOT / oof_csv).exists() or n_all == 0:
        print(f"{key}: SKIP (models {n_all}, oof {(ROOT / oof_csv).exists()})")
        continue
    oof = pd.read_csv(ROOT / oof_csv)
    f0 = oof[oof.fold == 0].head(20)
    if crops is None:
        crops = [np.asarray(crops_all[pos[u]], np.float32) for u in f0.uid]
    x = np.stack([mkin(c) for c in crops]).astype(np.float32)
    models = []
    for mp in sorted(d.glob("f0_s*.pt")):
        m = ctor()
        m.load_state_dict(dequantize_sd(torch.load(mp, map_location="cpu")))
        models.append(m.to(dev).eval())
    diff = np.abs(cnn2d._mean_tta(models, x, dev) - f0.logit.to_numpy()).max()
    print(f"{key:11s} dir has {n_all:2d} .pt | {len(models)} fold-0 models | max|dlogit| vs OOF = {diff:.5f}")

print("--- predict_axial() ---")
out = cnn2d.predict_axial(crops, dev)
for k in sorted(out):
    print(f"  {k:11s} shape {out[k].shape} mean {out[k].mean():+.4f}")
