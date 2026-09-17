"""Lane 1d: NetT with the top-5% (p95) noisiest blend-OOF scans dropped from TRAIN only."""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_src"
sys.path.insert(0, str(SRC))
from cnn2d import make_input, NetT  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from quant import quantize_sd, dequantize_sd  # noqa: E402

SEEDS = tuple(int(v) for v in os.environ.get("SEEDS", "0,1").split(","))
EPOCHS = 24
BATCH = 32
LR = 3e-4
WD = 1e-4
ZSHIFTS = (-2, -1, 0, 1, 2)
CLIP = (0.02, 0.98)
MODELS = SRC / "models" / "2dtc5"
NOISY_PCT = 95.0


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(16)
    return torch.device("cpu")


def ll(y, p):
    return float(log_loss(y, np.clip(p, *CLIP), labels=[0, 1]))


def build_inputs(folds):
    """(3,64,64) tensors for every zshift, in folds.csv row order."""
    uids = np.load(ROOT / "data/cache/uids.npy", allow_pickle=True)
    pos = {u: i for i, u in enumerate(uids)}
    rows = np.array([pos[u] for u in folds.uid], dtype=np.int64)
    crops = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")
    xz = {z: np.empty((len(rows), 3, 64, 64), np.float32) for z in ZSHIFTS}
    for i, r in enumerate(rows):
        crop = np.asarray(crops[r], np.float32)
        for z in ZSHIFTS:
            xz[z][i] = make_input(crop, z)
    return xz


def augment(xb, rng):
    """xb: CPU float tensor (B,3,64,64). Flip -> shift -> rotate -> intensity."""
    b = xb.shape[0]
    flip = torch.from_numpy(rng.random(b) < 0.5)
    if flip.any():
        xb[flip] = torch.flip(xb[flip], dims=[-1])
    dy, dx = rng.integers(-4, 5, b), rng.integers(-4, 5, b)
    ang = rng.uniform(-10.0, 10.0, b)
    for i in range(b):
        if dy[i] or dx[i]:
            xb[i] = torch.roll(xb[i], shifts=(int(dy[i]), int(dx[i])), dims=(-2, -1))
        xb[i] = TF.rotate(xb[i], float(ang[i]), interpolation=TF.InterpolationMode.BILINEAR)
    sc = torch.from_numpy(rng.uniform(0.9, 1.1, b).astype(np.float32)).view(b, 1, 1, 1)
    return xb * sc


@torch.no_grad()
def predict(model, x, dev, tta=False):
    model.eval()
    out = []
    for i in range(0, len(x), 64):
        xb = torch.from_numpy(x[i:i + 64]).to(dev)
        lg = model(xb).squeeze(1)
        if tta:
            lg = (lg + model(torch.flip(xb, dims=[-1])).squeeze(1)) / 2.0
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


def train_one(xz, y, tr, va, seed, fold, dev):
    torch.manual_seed(seed * 100 + fold)
    rng = np.random.default_rng(seed * 1000 + fold)
    model = NetT(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    nstep = int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * nstep, eta_min=0.0)
    lossf = nn.BCEWithLogitsLoss()
    t_smooth = torch.from_numpy((y * 0.90 + 0.05).astype(np.float32))
    xval, yval = xz[0][va], y[va]
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(tr)
        zpick = rng.choice(ZSHIFTS, size=len(perm))
        tot = 0.0
        for s in range(nstep):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            if len(idx) == 0:
                continue
            zb = zpick[s * BATCH:(s + 1) * BATCH]
            xb = np.empty((len(idx), 3, 64, 64), np.float32)
            for z in ZSHIFTS:
                m = zb == z
                if m.any():
                    xb[m] = xz[z][idx[m]]
            xb = augment(torch.from_numpy(xb), rng).to(dev)
            yb = t_smooth[idx].to(dev)
            loss = lossf(model(xb).squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        vlg = predict(model, xval, dev)
        vp = 1.0 / (1.0 + np.exp(-vlg))
        print(f"  f{fold} s{seed} ep{ep + 1:02d} train {tot / len(perm):.4f} "
              f"val {ll(yval, vp):.4f} auc {roc_auc_score(yval, vp):.4f}", flush=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"f{fold}_s{seed}.pt"
    torch.save(quantize_sd(model.state_dict()), path)
    mq = NetT(pretrained=False)
    mq.load_state_dict(dequantize_sd(torch.load(path, map_location="cpu")))
    mq = mq.to(dev).eval()
    n = min(32, len(xval))
    d = float(np.abs(predict(model, xval[:n], dev) - predict(mq, xval[:n], dev)).max())
    print(f"  f{fold} s{seed} int8 max|dlogit| {d:.5f} on {n} val crops "
          f"({'OK' if d < 0.05 else 'FAIL'}, {path.stat().st_size / 1e6:.2f} MB)", flush=True)
    return model, d


def main():
    dev = device()
    print("device", dev, flush=True)
    folds = pd.read_csv(ROOT / "data/folds.csv")
    blend = pd.read_csv(ROOT / "results/oof_blend_s1.csv")
    thresh = float(np.percentile(blend.loss, NOISY_PCT))
    noisy = set(blend.loc[blend.loss > thresh, "uid"])
    clean = ~folds.uid.isin(noisy).to_numpy()
    print(f"p{NOISY_PCT:g} blend loss threshold = {thresh:.6f}", flush=True)
    print(f"noisy uids dropped from train: {len(noisy)} "
          f"({100 * len(noisy) / len(blend):.2f}% of {len(blend)})", flush=True)
    y = folds.y.to_numpy(np.float64)
    t0 = time.time()
    xz = build_inputs(folds)
    print(f"inputs built in {time.time() - t0:.1f}s", flush=True)

    oof = np.full(len(folds), np.nan)
    secs, done, int8_diffs = {}, [], {}
    for k in sorted(folds.fold.unique()):
        tf = time.time()
        tr_all = folds.fold.to_numpy() != k
        tr = np.where(tr_all & clean)[0]
        print(f"fold {k}: train {len(tr)} of {int(tr_all.sum())} "
              f"(dropped {int(tr_all.sum()) - len(tr)})", flush=True)
        va = np.where(folds.fold.to_numpy() == k)[0]
        lgs = []
        for seed in SEEDS:
            m, d8 = train_one(xz, y, tr, va, seed, int(k), dev)
            int8_diffs[f"f{k}_s{seed}"] = d8
            lgs.append(predict(m, xz[0][va], dev, tta=True))
            del m
        oof[va] = np.mean(lgs, axis=0)
        secs[int(k)] = round(time.time() - tf, 1)
        done.append(int(k))
        p = 1.0 / (1.0 + np.exp(-oof[va]))
        print(f"FOLD {k} done {secs[int(k)]}s  oof_ll {ll(y[va], p):.4f} "
              f"auc {roc_auc_score(y[va], p):.4f}", flush=True)
        out = folds.copy()
        out["logit"] = oof
        out["prob"] = 1.0 / (1.0 + np.exp(-oof))
        out[["uid", "y", "fold", "family", "logit", "prob"]].to_csv(ROOT / "results/oof_2dtc5.csv", index=False)

    m = ~np.isnan(oof)
    lg, yy = oof[m], y[m]
    praw = 1.0 / (1.0 + np.exp(-lg))
    clf = LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), yy)
    a, b = float(clf.coef_[0][0]), float(clf.intercept_[0])
    pcal = 1.0 / (1.0 + np.exp(-(a * lg + b)))
    json.dump({"a": a, "b": b}, open(SRC / "calib_2dtc5.json", "w"))

    sub = folds[m]
    per_fold = {int(k): {"n": int((sub.fold == k).sum()),
                         "raw": ll(yy[(sub.fold == k).to_numpy()], praw[(sub.fold == k).to_numpy()]),
                         "platt": ll(yy[(sub.fold == k).to_numpy()], pcal[(sub.fold == k).to_numpy()])}
                for k in sorted(sub.fold.unique())}
    per_fam = {}
    for fam in sorted(sub.family.unique()):
        msk = (sub.family == fam).to_numpy()
        if len(set(yy[msk])) < 2:
            per_fam[fam] = {"n": int(msk.sum()), "raw": ll(yy[msk], praw[msk]),
                            "platt": ll(yy[msk], pcal[msk]), "auc": None}
        else:
            per_fam[fam] = {"n": int(msk.sum()), "raw": ll(yy[msk], praw[msk]),
                            "platt": ll(yy[msk], pcal[msk]), "auc": float(roc_auc_score(yy[msk], pcal[msk]))}
    summ = {"n": int(m.sum()), "folds_done": done, "device": str(dev), "epochs": EPOCHS, "seeds": list(SEEDS),
            "logloss_raw": ll(yy, praw), "logloss_platt": ll(yy, pcal), "auc": float(roc_auc_score(yy, praw)),
            "platt": {"a": a, "b": b}, "per_fold": per_fold, "per_family": per_fam, "seconds_per_fold": secs, "int8_max_dlogit": int8_diffs,
            "arch": "NetT(resnet18 truncated@layer3)", "label_smoothing_eps": 0.10,
            "noisy_loss_thresh": thresh, "noisy_pct": NOISY_PCT, "n_noisy_dropped": len(noisy)}
    json.dump(summ, open(ROOT / "results/2dtc5_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_family"}, indent=1), flush=True)
    print("per_family", json.dumps(per_fam, indent=1), flush=True)


if __name__ == "__main__":
    main()
