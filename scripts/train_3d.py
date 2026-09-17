"""Lane 3: small 3D CNN on the full crop, 5 folds. Writes OOF, Platt calibration, int8 state_dicts.

DEADLINE_UTC (unix seconds, optional): no new fold is started past it; the running fold finishes.
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "src"))
from cnn3d import Net3D, n_params  # noqa: E402
from quant import quantize_sd, dequantize_sd  # noqa: E402

SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0").split(","))
EPOCHS = 30
BATCH = 16
LR = 1e-3
WD = 1e-4
CLIP = (0.02, 0.98)
MODELS = SRC / "models" / "3d"
DEADLINE = float(os.environ.get("DEADLINE_UTC", "0")) or None


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(16)
    return torch.device("cpu")


def ll(y, p):
    return float(log_loss(y, np.clip(p, *CLIP), labels=[0, 1]))


def load_crops(folds):
    """float16 (N,2,48,64,64) in folds.csv row order."""
    uids = np.load(ROOT / "data/cache/uids.npy", allow_pickle=True)
    pos = {u: i for i, u in enumerate(uids)}
    rows = np.array([pos[u] for u in folds.uid], dtype=np.int64)
    crops = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")
    return np.ascontiguousarray(crops[rows])


def to_x(crops16, idx, dev):
    """rows -> normalised float32 tensor on dev."""
    xb = torch.from_numpy(np.ascontiguousarray(crops16[idx])).to(dev).float()
    xb[:, 0] -= 0.5
    xb[:, 1] = xb[:, 1] / 4.0 - 0.5
    return xb


def augment(xb, rng):
    """flip X -> integer shifts +-3 per axis -> intensity scale -> Gaussian noise."""
    b = xb.shape[0]
    flip = torch.from_numpy(rng.random(b) < 0.5).to(xb.device)
    if bool(flip.any()):
        xb[flip] = torch.flip(xb[flip], dims=[-1])
    sh = rng.integers(-3, 4, (b, 3))
    for i in range(b):
        s = sh[i]
        if s.any():
            xb[i] = torch.roll(xb[i], shifts=(int(s[0]), int(s[1]), int(s[2])), dims=(-3, -2, -1))
    sc = torch.from_numpy(rng.uniform(0.9, 1.1, b).astype(np.float32)).to(xb.device).view(b, 1, 1, 1, 1)
    xb = (xb + 0.5) * sc - 0.5
    noise = torch.from_numpy(rng.random(b) < 0.5).to(xb.device)
    if bool(noise.any()):
        xb[noise] += torch.randn_like(xb[noise]) * 0.02
    return xb


@torch.no_grad()
def predict(model, crops16, idx, dev, tta=False, batch=32):
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        xb = to_x(crops16, idx[i:i + batch], dev)
        lg = model(xb).squeeze(1)
        if tta:
            lg = (lg + model(torch.flip(xb, dims=[-1])).squeeze(1)) / 2.0
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_one(crops16, y, tr, va, seed, fold, dev, batch, epochs, probe=False):
    """Returns the trained model, or None if `probe` and epoch 1 took > 25 s."""
    torch.manual_seed(seed * 100 + fold)
    rng = np.random.default_rng(seed * 1000 + fold)
    model = Net3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    nstep = int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * nstep, eta_min=0.0)
    lossf = nn.BCEWithLogitsLoss()
    t_smooth = torch.from_numpy((y * 0.90 + 0.05).astype(np.float32))
    for ep in range(epochs):
        t_ep = time.time()
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for s in range(nstep):
            idx = perm[s * batch:(s + 1) * batch]
            if len(idx) == 0:
                continue
            xb = augment(to_x(crops16, idx, dev), rng)
            yb = t_smooth[idx].to(dev)
            loss = lossf(model(xb).squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        dt = time.time() - t_ep
        if probe and ep == 0 and dt > 25.0:
            print(f"  PROBE f{fold} s{seed} epoch1 {dt:.1f}s > 25s (batch {batch})", flush=True)
            return None
        vlg = predict(model, crops16, va, dev)
        vp = 1.0 / (1.0 + np.exp(-vlg))
        print(f"  f{fold} s{seed} ep{ep + 1:02d} {dt:5.1f}s train {tot / len(perm):.4f} "
              f"val {ll(y[va], vp):.4f} auc {roc_auc_score(y[va], vp):.4f}", flush=True)
    return model


def save_int8(model, crops16, va, fold, seed, dev):
    """int8 pack + fp32-vs-int8 max |logit| diff on 32 val crops."""
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"f{fold}_s{seed}.pt"
    torch.save(quantize_sd(model.state_dict()), path)
    chk = va[:32]
    lg32 = predict(model, crops16, chk, dev)
    m8 = Net3D()
    m8.load_state_dict(dequantize_sd(torch.load(path, map_location="cpu")))
    lg8 = predict(m8.to(dev).eval(), crops16, chk, dev)
    d = float(np.abs(lg32 - lg8).max())
    print(f"  int8 f{fold} s{seed} max|dlogit| on {len(chk)} val crops = {d:.5f} "
          f"({'OK' if d < 0.05 else 'FAIL'}) size {path.stat().st_size // 1024}KB", flush=True)
    return d


def summarise(folds, oof, done, dev, secs, batch, epochs, int8_diffs, params):
    y = folds.y.to_numpy(np.float64)
    m = ~np.isnan(oof)
    lg, yy = oof[m], y[m]
    praw = 1.0 / (1.0 + np.exp(-lg))
    clf = LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), yy)
    a, b = float(clf.coef_[0][0]), float(clf.intercept_[0])
    pcal = 1.0 / (1.0 + np.exp(-(a * lg + b)))
    MODELS.mkdir(parents=True, exist_ok=True)
    json.dump({"a": a, "b": b}, open(MODELS / "calib.json", "w"))

    sub = folds[m]
    per_fold = {}
    for k in sorted(sub.fold.unique()):
        msk = (sub.fold == k).to_numpy()
        per_fold[int(k)] = {"n": int(msk.sum()), "raw": ll(yy[msk], praw[msk]),
                            "platt": ll(yy[msk], pcal[msk]),
                            "auc": float(roc_auc_score(yy[msk], praw[msk]))}
    per_fam = {}
    for fam in sorted(sub.family.unique()):
        msk = (sub.family == fam).to_numpy()
        one = len(set(yy[msk])) < 2
        per_fam[fam] = {"n": int(msk.sum()), "raw": ll(yy[msk], praw[msk]),
                        "platt": ll(yy[msk], pcal[msk]),
                        "auc": None if one else float(roc_auc_score(yy[msk], praw[msk]))}
    summ = {"n": int(m.sum()), "folds_done": sorted(done), "device": str(dev), "epochs": epochs,
            "batch": batch, "seeds": list(SEEDS), "params": params,
            "logloss_raw": ll(yy, praw), "logloss_platt": ll(yy, pcal),
            "auc": float(roc_auc_score(yy, praw)), "platt": {"a": a, "b": b},
            "per_fold": per_fold, "per_family": per_fam, "seconds_per_fold": secs,
            "int8_max_abs_logit_diff": int8_diffs}
    json.dump(summ, open(ROOT / "results/3d_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_family"}, indent=1), flush=True)
    print("per_family " + json.dumps(per_fam, indent=1), flush=True)


def resume(folds, oof):
    """Seed OOF from a previous run of this script; return (folds already scored, secs, int8 diffs,
    batch, epochs). The recipe of the earlier run is reused so every fold is trained identically."""
    p, s = ROOT / "results/oof_3d.csv", ROOT / "results/3d_summary.json"
    if os.environ.get("RESUME", "1") != "1" or not p.exists():
        return [], {}, {}, BATCH, EPOCHS
    prev = pd.read_csv(p)
    if len(prev) != len(folds) or not (prev.uid.to_numpy() == folds.uid.to_numpy()).all():
        print("resume: oof_3d.csv does not match folds.csv; starting fresh", flush=True)
        return [], {}, {}, BATCH, EPOCHS
    oof[:] = prev.logit.to_numpy()
    fa = folds.fold.to_numpy()
    done = [int(k) for k in sorted(folds.fold.unique()) if not np.isnan(oof[fa == int(k)]).any()]
    secs, diffs, batch, epochs = {}, {}, BATCH, EPOCHS
    if s.exists():
        prv = json.load(open(s))
        secs = {int(k): v for k, v in prv.get("seconds_per_fold", {}).items()}
        diffs = prv.get("int8_max_abs_logit_diff", {})
        batch, epochs = prv.get("batch", BATCH), prv.get("epochs", EPOCHS)
    print(f"resume: folds {done} already scored; reusing batch {batch} / {epochs} epochs", flush=True)
    return done, secs, diffs, batch, epochs


def write_oof(folds, oof):
    out = folds.copy()
    out["logit"] = oof
    out["prob"] = 1.0 / (1.0 + np.exp(-oof))
    out[["uid", "y", "fold", "family", "logit", "prob"]].to_csv(ROOT / "results/oof_3d.csv", index=False)


def main():
    dev = device()
    params = n_params()
    print(f"device {dev}  params {params}  seeds {SEEDS}", flush=True)
    folds = pd.read_csv(ROOT / "data/folds.csv")
    y = folds.y.to_numpy(np.float64)
    t0 = time.time()
    crops16 = load_crops(folds)
    print(f"crops {crops16.shape} {crops16.dtype} loaded in {time.time() - t0:.1f}s", flush=True)

    fa = folds.fold.to_numpy()
    oof = np.full(len(folds), np.nan)
    done, secs, int8_diffs, batch, epochs = resume(folds, oof)
    probe = not done
    for k in sorted(folds.fold.unique()):
        k = int(k)
        if k in done:
            continue
        if DEADLINE and time.time() > DEADLINE:
            print(f"DEADLINE hit before fold {k}; stopping with folds {done}", flush=True)
            break
        tf = time.time()
        tr, va = np.where(fa != k)[0], np.where(fa == k)[0]
        lgs = []
        for seed in SEEDS:
            while True:
                m = train_one(crops16, y, tr, va, seed, k, dev, batch, epochs, probe=probe)
                if m is not None:
                    break
                if batch == BATCH:
                    batch = 32
                    print(f"SWITCH batch -> {batch} (epoch > 25s)", flush=True)
                elif epochs == EPOCHS:
                    epochs = 20
                    print(f"SWITCH epochs -> {epochs} (epoch still > 25s at batch {batch})", flush=True)
                else:
                    probe = False
                    print("epoch still > 25s; continuing at batch 32 / 20 epochs", flush=True)
            probe = False
            int8_diffs[f"f{k}_s{seed}"] = save_int8(m, crops16, va, k, seed, dev)
            lgs.append(predict(m, crops16, va, dev, tta=True))
            del m
        oof[va] = np.mean(lgs, axis=0)
        secs[k] = round(time.time() - tf, 1)
        done.append(k)
        p = 1.0 / (1.0 + np.exp(-oof[va]))
        print(f"FOLD {k} done {secs[k]}s  oof_ll {ll(y[va], p):.4f} auc {roc_auc_score(y[va], p):.4f}",
              flush=True)
        write_oof(folds, oof)
    summarise(folds, oof, done, dev, secs, batch, epochs, int8_diffs, params)


if __name__ == "__main__":
    main()
