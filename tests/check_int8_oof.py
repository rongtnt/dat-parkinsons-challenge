"""Does int8 quantisation actually cost log loss? Recompute the whole OOF with the saved int8 models."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_src"
sys.path.insert(0, str(SRC))
import cnn2d
from quant import dequantize_sd

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
oof = pd.read_csv(ROOT / "results/oof_2dt.csv")
uids_c = np.load(ROOT / "data/cache/uids.npy", allow_pickle=True)
pos = {u: i for i, u in enumerate(uids_c)}
crops = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")
x = np.stack([cnn2d.make_input(np.asarray(crops[pos[u]], np.float32)) for u in oof.uid]).astype(np.float32)

lg8 = np.full(len(oof), np.nan)
for k in sorted(oof.fold.unique()):
    va = np.where(oof.fold.to_numpy() == k)[0]
    ms = []
    for mp in sorted((SRC / "models" / "2dt").glob(f"f{k}_s*.pt")):
        m = cnn2d.NetT(pretrained=False)
        m.load_state_dict(dequantize_sd(torch.load(mp, map_location="cpu")))
        ms.append(m.to(dev).eval())
    lg8[va] = cnn2d._mean_tta(ms, x[va], dev)

y = oof.y.to_numpy(np.float64)
lg32 = oof.logit.to_numpy()
ll = lambda p: float(log_loss(y, np.clip(p, 0.02, 0.98), labels=[0, 1]))
sig = lambda z: 1.0 / (1.0 + np.exp(-z))
print(f"max|dlogit| fp32 vs int8 over all {len(y)} OOF scans: {np.abs(lg32 - lg8).max():.5f} "
      f"(mean {np.abs(lg32 - lg8).mean():.5f})")
for tag, lg in [("fp32", lg32), ("int8", lg8)]:
    a, b = LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), y).coef_[0][0], \
           LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), y).intercept_[0]
    print(f"{tag}: OOF raw {ll(sig(lg)):.5f}  platt {ll(sig(a * lg + b)):.5f}  "
          f"auc {roc_auc_score(y, lg):.5f}  (a={a:.4f}, b={b:.4f})")
