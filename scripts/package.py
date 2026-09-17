"""Fable packaging: fit blend on chosen models' OOF, stage a minimal submission dir, zip it, split into 2 MB chunks.
usage: python scripts/package.py <tag> key1 key2 ...   keys in {feat_lgb, feat_lr, cnn_cor, cnn_3d, cnn_2dt, cnn_2d}"""
import sys, json, shutil, subprocess, itertools
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]
OOF = {"feat_lgb": "oof_feat", "feat_lr": "oof_lr", "cnn_cor": "oof_2dcor", "cnn_3d": "oof_3d", "cnn_2dt": "oof_2dt", "cnn_2d": "oof_2d", "cnn_2dtc": "oof_2dtc", "cnn_2dt6": "oof_2dt6", "cnn_sag": "oof_2dsag", "cnn_corc": "oof_2dcorc", "cnn_2dtc5": "oof_2dtc5", "cnn_2dt6n": "oof_2dt6n", "cnn_2dt6c5": "oof_2dt6c5"}
MDIR = {"feat_lgb": "feat", "feat_lr": "feat", "cnn_cor": "2dcor", "cnn_3d": "3d", "cnn_2dt": "2dt", "cnn_2d": "2d", "cnn_2dtc": "2dtc", "cnn_2dt6": "2dt6", "cnn_sag": "2dsag", "cnn_corc": "2dcorc", "cnn_2dtc5": "2dtc5", "cnn_2dt6n": "2dt6n", "cnn_2dt6c5": "2dt6c5"}
tag, keys = sys.argv[1], sys.argv[2:]
base = pd.read_csv(ROOT / "data/folds.csv")
y = base.y.values; fam = base.family.values
L, platt = {}, {}
for k in keys:
    d = pd.read_csv(ROOT / f"results/{OOF[k]}.csv").set_index("uid").loc[base.uid]
    lg = d.logit.values.astype(float); assert np.isfinite(lg).all(), k
    lr = LogisticRegression(C=1e6).fit(lg[:, None], y)
    platt[k] = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    L[k] = platt[k]["a"] * lg + platt[k]["b"]
    p = 1 / (1 + np.exp(-L[k])); print(f"{k:9s} platt OOF logloss {log_loss(y, np.clip(p, .02, .98)):.4f}  auc {roc_auc_score(y, p):.4f}")
# ---- stacking: non-negative weights on Platt-scaled logits (+ intercept), cross-fitted by fold, then refit on all
from scipy.optimize import minimize
Lm = np.stack([L[k] for k in keys], 1)
def fit_w(idx):
    def obj(v):
        z = Lm[idx] @ v[:-1] + v[-1]; pz = 1 / (1 + np.exp(-z))
        return -np.mean(y[idx] * np.log(np.clip(pz, 1e-9, 1)) + (1 - y[idx]) * np.log(np.clip(1 - pz, 1e-9, 1))) + 1e-3 * np.sum(v[:-1] ** 2)
    v0 = np.r_[np.ones(len(keys)) / len(keys), 0.0]
    return minimize(obj, v0, method="L-BFGS-B", bounds=[(0, None)] * len(keys) + [(None, None)]).x
cf = np.zeros(len(y)); folds = base.fold.values
for k in range(5):
    v = fit_w(np.where(folds != k)[0]); te = folds == k
    cf[te] = 1 / (1 + np.exp(-(Lm[te] @ v[:-1] + v[-1])))
v = fit_w(np.arange(len(y))); w = v[:-1]; b0 = float(v[-1])
zb = Lm @ w + b0; p = np.clip(1 / (1 + np.exp(-zb)), .02, .98)
weights = {k: float(round(wi, 4)) for k, wi in zip(keys, w)}
print("STACK", weights, "b0", round(b0, 3), f"in-sample {log_loss(y, p):.4f}  CROSS-FIT {log_loss(y, np.clip(cf, .02, .98)):.4f}  auc {roc_auc_score(y, p):.4f}")
eq = Lm.mean(1); print(f"equal-weight OOF {log_loss(y, np.clip(1/(1+np.exp(-eq)), .02, .98)):.4f}")
print(pd.DataFrame({"fam": fam, "y": y, "p": p}).groupby("fam").apply(lambda s: round(log_loss(s.y, s.p, labels=[0, 1]), 3)).to_dict())
print("per-fold", {k: round(log_loss(y[folds == k], p[folds == k]), 3) for k in range(5)})
post = {"a": 1.0, "b": b0}   # intercept folded into the post step; weights already on the right scale
keys = [k for k in keys if weights[k] > 1e-4]; weights = {k: weights[k] for k in keys}
print("SHIPPED", keys)
# ---- stage
src = ROOT / "submission_src"; stage = ROOT / f"tmp/stage_{tag}"
if stage.exists(): shutil.rmtree(stage)
stage.mkdir(parents=True)
for f in ["prep.py", "quant.py", "cnn2d.py", "extra_models.py", "cnn3d.py"]:
    if (src / f).exists(): shutil.copy(src / f, stage / f)
shutil.copy(ROOT / "src/main_blend.py", stage / "main.py")
json.dump({"weights": weights, "platt": platt, "post": post}, open(stage / "blend.json", "w"), indent=1)
for d in sorted({MDIR[k] for k in keys}):
    shutil.copytree(src / "models" / d, stage / "models" / d, ignore=shutil.ignore_patterns("__pycache__"))
(ROOT / "upload").mkdir(exist_ok=True); zpath = ROOT / f"upload/submission_{tag}.zip"; zpath.unlink(missing_ok=True)
subprocess.run(["zip", "-r", "-X", "-q", str(zpath), "."], cwd=stage, check=True)
for old in (ROOT / "upload").glob(f"{tag}.part.*"): old.unlink()
subprocess.run(["split", "-b", "2000000", "-d", "-a", "2", str(zpath), str(ROOT / "upload" / f"{tag}.part.")], check=True)
parts = sorted((ROOT / "upload").glob(f"{tag}.part.*"))
print(f"ZIP {zpath.name} {zpath.stat().st_size/1e6:.1f} MB -> {len(parts)} chunks; sha256 {subprocess.run(['shasum','-a','256',str(zpath)],capture_output=True,text=True).stdout[:16]}")
print(subprocess.run(["unzip", "-l", str(zpath)], capture_output=True, text=True).stdout[-1200:])
