"""V3b: coronal-clean.  train_2d_cor.py with ONE change -- scans whose OOF blend loss exceeds
NOISY_LOSS (top 3% of results/oof_blend_s1.csv) are dropped from the TRAINING set only.
Validation / OOF still covers every scan of the held-out fold."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))   # extra_models sets OMP_NUM_THREADS + imports lgb first
sys.path.insert(0, str(ROOT / "src"))
from extra_models import make_input_cor, NetCorT  # noqa: E402
from quant import quantize_sd, dequantize_sd      # noqa: E402

import torch                                      # noqa: E402
import torchvision.transforms.functional as TF    # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

OUT = ROOT / "submission_src/models/2dcorc"
NOISY_LOSS = 1.672          # top 3% of the current blend OOF loss
YSHIFTS = (-2, -1, 0, 1, 2)
SEEDS = (0, 1)
EPOCHS, BATCH, CLIP, EPS = 24, 32, (0.02, 0.98), 0.10
FOLD_SECONDS_LIMIT = 300      # > 5 min/fold on MPS -> CPU fallback (brief)
STEP_C_CAP = 40 * 60


def augment(x, rng):
    """x: (3,64,64) CPU tensor.  L/R flip, +-4 px roll in Z/X, +-10 deg rotate, intensity scale."""
    if rng.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    x = torch.roll(x, shifts=(int(rng.integers(-4, 5)), int(rng.integers(-4, 5))), dims=(-2, -1))
    x = TF.rotate(x, float(rng.uniform(-10, 10)), interpolation=TF.InterpolationMode.BILINEAR)
    return x * float(rng.uniform(0.9, 1.1))


def predict(net, X, device):
    """TTA = mean logit over {identity, L/R flip}."""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            b = torch.from_numpy(X[i:i + 64]).to(device)
            out.append(((net(b)[:, 0] + net(torch.flip(b, dims=[-1]))[:, 0]) / 2).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def summarise(y, p, fold, family):
    p = np.clip(p, *CLIP)
    return dict(logloss=float(log_loss(y, p, labels=[0, 1])), auc=float(roc_auc_score(y, p)),
                per_fold={int(k): float(log_loss(y[fold == k], p[fold == k], labels=[0, 1])) for k in np.unique(fold)},
                per_family={str(f): (float(log_loss(y[family == f], p[family == f], labels=[0, 1]))
                                     if len(np.unique(y[family == f])) > 1 else None) for f in np.unique(family)},
                per_family_n={str(f): int((family == f).sum()) for f in np.unique(family)})


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    crops = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")
    uids = list(np.load(ROOT / "data/cache/uids.npy", allow_pickle=True))
    folds = pd.read_csv(ROOT / "data/folds.csv").set_index("uid").loc[uids].reset_index()
    y, fold, family = folds.y.to_numpy(np.float32), folds.fold.to_numpy(), folds.family.to_numpy()

    bank = {s: np.stack([make_input_cor(np.asarray(crops[i], np.float32), s) for i in range(len(uids))])
            for s in YSHIFTS}
    drop = set(pd.read_csv(ROOT / "results/oof_blend_s1.csv").query(f"loss > {NOISY_LOSS}").uid)
    noisy = np.array([u in drop for u in uids])
    print(f"device={device}  banks {bank[0].shape}  seeds={SEEDS}  eps={EPS}", flush=True)
    print(f"noisy-label drop: {noisy.sum()} scans with blend OOF loss > {NOISY_LOSS}; "
          f"by fold {dict(zip(*np.unique(fold[noisy], return_counts=True)))}", flush=True)

    oof, oof32, secs, done, qdiff = np.zeros(len(uids)), np.zeros(len(uids)), {}, [], []
    for k in range(5):
        tf0 = time.time()
        tr, va = np.where((fold != k) & ~noisy)[0], np.where(fold == k)[0]
        print(f"fold {k}: train {len(tr)} (dropped {int((fold != k).sum()) - len(tr)} noisy), val {len(va)} (all)",
              flush=True)
        Xva = bank[0][va]
        seed_logits, seed_logits32 = [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(1000 * seed + k)
            net = NetCorT(pretrained=True).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
            nstep = int(np.ceil(len(tr) / BATCH))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nstep * EPOCHS, eta_min=0.0)
            lossf = torch.nn.BCEWithLogitsLoss()
            t_sm = torch.from_numpy(y * (1 - EPS) + EPS / 2)      # y*0.90 + 0.05

            for ep in range(EPOCHS):
                net.train()
                perm, tot = rng.permutation(tr), 0.0
                for i in range(0, len(perm), BATCH):
                    idx = perm[i:i + BATCH]
                    xb = torch.stack([augment(torch.from_numpy(bank[YSHIFTS[rng.integers(5)]][j]).clone(), rng)
                                      for j in idx]).to(device)
                    opt.zero_grad()
                    loss = lossf(net(xb)[:, 0], t_sm[idx].to(device))
                    loss.backward(); opt.step(); sched.step()
                    tot += float(loss.detach()) * len(idx)
                if ep % 2 == 1 or ep == EPOCHS - 1:
                    pv = np.clip(1 / (1 + np.exp(-predict(net, Xva, device))), *CLIP)
                    print(f"  f{k}s{seed} ep{ep:02d} train {tot/len(perm):.4f}  val ll "
                          f"{log_loss(y[va], pv, labels=[0,1]):.4f}  auc {roc_auc_score(y[va], pv):.4f}", flush=True)

            lg_fp32 = predict(net, Xva, device)
            torch.save(quantize_sd(net.state_dict()), OUT / f"f{k}_s{seed}.pt")
            q = NetCorT(pretrained=False)
            q.load_state_dict(dequantize_sd(torch.load(OUT / f"f{k}_s{seed}.pt", map_location="cpu")))
            lg_int8 = predict(q.to(device), Xva, device)
            d = float(np.abs(lg_int8[:20] - lg_fp32[:20]).max())
            qdiff.append(d)
            print(f"  f{k}s{seed} int8-vs-fp32 max|dlogit| on 20 crops = {d:.4f}"
                  f"  ({'OK' if d <= 0.05 else 'OVER 0.05'})", flush=True)
            seed_logits.append(lg_int8); seed_logits32.append(lg_fp32)
            del net, q

        oof[va] = np.mean(seed_logits, axis=0)
        oof32[va] = np.mean(seed_logits32, axis=0)
        secs[k] = round(time.time() - tf0, 1); done.append(k)
        print(f"fold {k} done in {secs[k]}s  (total {time.time()-t0:.0f}s)", flush=True)
        if secs[k] > FOLD_SECONDS_LIMIT and device == "mps":
            device = "cpu"; torch.set_num_threads(12)
            print(f"fold > {FOLD_SECONDS_LIMIT}s -> switching to CPU", flush=True)
        if k < 4 and (time.time() - t0) + secs[k] > STEP_C_CAP:
            print(f"STOP: projected overrun of the {STEP_C_CAP}s cap after fold {k}", flush=True)
            break

    summary = dict(device=device, arch="resnet18-trunc-layer3+Linear(256,1) CORONAL-CLEAN", noisy_loss_thresh=NOISY_LOSS,
                   n_dropped=int(noisy.sum()), folds_done=done, seeds=list(SEEDS),
                   epochs=EPOCHS, label_smoothing_eps=EPS, seconds_per_fold=secs,
                   seconds_total=round(time.time() - t0, 1), int8_max_abs_logit_diff=max(qdiff) if qdiff else None,
                   weight_bytes_per_model=int((OUT / "f0_s0.pt").stat().st_size) if (OUT / "f0_s0.pt").exists() else None)
    if len(done) == 5:
        out = folds[["uid", "y", "fold", "family"]].copy()
        out["logit"] = oof
        out["prob"] = 1 / (1 + np.exp(-oof))
        out.to_csv(ROOT / "results/oof_2dcorc.csv", index=False)
        platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)
        a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
        (OUT / "calib.json").write_text(json.dumps({"a": a, "b": b}))
        summary["platt"] = {"a": a, "b": b}
        summary["raw"] = summarise(y, 1 / (1 + np.exp(-oof)), fold, family)
        summary["fp32_raw"] = summarise(y, 1 / (1 + np.exp(-oof32)), fold, family)
        summary["int8_vs_fp32_oof"] = dict(
            max_abs_logit_diff=float(np.abs(oof - oof32).max()),
            mean_abs_logit_diff=float(np.abs(oof - oof32).mean()),
            delta_logloss=float(summary["raw"]["logloss"] - summary["fp32_raw"]["logloss"]))
        print("\nint8 vs fp32 OOF: max|dlogit| {max_abs_logit_diff:.4f}  mean {mean_abs_logit_diff:.4f}"
              "  d(logloss) {delta_logloss:+.5f}".format(**summary["int8_vs_fp32_oof"]), flush=True)
        summary["platt_scaled"] = summarise(y, 1 / (1 + np.exp(-(a * oof + b))), fold, family)
        for tag in ("raw", "platt_scaled"):
            s = summary[tag]
            print(f"\n[cnn_corc {tag}] OOF logloss {s['logloss']:.5f}  AUC {s['auc']:.4f}")
            print("  per-fold : " + "  ".join(f"{k}:{v:.4f}" for k, v in s["per_fold"].items()))
            print("  per-family: " + "  ".join(f"{k}:{v:.4f}(n={s['per_family_n'][k]})"
                                               for k, v in s["per_family"].items() if v is not None))
    else:
        print("INCOMPLETE: not all folds trained, no oof_2dcor.csv written", flush=True)
    (ROOT / "results/2dcorc_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nint8 max|dlogit| over all models = {summary['int8_max_abs_logit_diff']}, "
          f"{summary['weight_bytes_per_model']} bytes/model -> results/2dcor_summary.json  "
          f"total {summary['seconds_total']}s", flush=True)


if __name__ == "__main__":
    main()
