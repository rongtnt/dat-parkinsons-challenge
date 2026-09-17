"""Step E: Platt-scale every results/oof_*.csv and search simplex weights minimising OOF log loss."""
import json, itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CLIP = (0.02, 0.98)
STEP = 0.05
NAMES = {"oof_feat": "feat_lgb", "oof_lr": "feat_lr", "oof_2dcor": "cnn_cor", "oof_2d": "cnn_2d",
         "oof_2dt": "cnn_2dt", "oof_2dtc": "cnn_2dtc", "oof_2dt6": "cnn_2dt6", "oof_2dsag": "cnn_sag",
         "oof_2dcorc": "cnn_corc", "oof_3d": "cnn_3d"}


def simplex(k, step=STEP):
    m = int(round(1 / step))
    for cut in itertools.combinations(range(1, m + k), k - 1):          # compositions of m into k parts
        prev, w = 0, []
        for c in cut:
            w.append(c - prev - 1); prev = c
        w.append(m + k - 1 - prev)
        yield np.array(w, float) / m


def scores(y, p, family):
    p = np.clip(p, *CLIP)
    return dict(logloss=float(log_loss(y, p, labels=[0, 1])), auc=float(roc_auc_score(y, p)),
                per_family={str(f): (float(log_loss(y[family == f], p[family == f], labels=[0, 1]))
                                     if len(np.unique(y[family == f])) > 1 else None)
                            for f in np.unique(family)})


def main():
    files = sorted((ROOT / "results").glob("oof_*.csv"))
    assert files, "no results/oof_*.csv found"
    base, names = None, []
    logits, platt, single = {}, {}, {}
    for f in files:
        name = NAMES.get(f.stem, f.stem)
        d = pd.read_csv(f)[["uid", "y", "fold", "family", "logit"]].rename(columns={"logit": name})
        bad = int(d[name].isna().sum())
        if bad:                       # another lane's file mid-write -> skip, do not crash
            print(f"skipping {f.name}: {bad}/{len(d)} NaN logits (incomplete)")
            continue
        names.append(name)
        base = d if base is None else base.merge(d[["uid", name]], on="uid", how="inner", validate="1:1")
    assert names, "no complete oof_*.csv"
    y, family, fold = base.y.to_numpy(float), base.family.to_numpy(), base.fold.to_numpy()
    print(f"models: {names}   n={len(base)}")

    for n in names:
        lg = base[n].to_numpy(float)
        logits[n] = lg
        lr = LogisticRegression(C=1e6).fit(lg.reshape(-1, 1), y)
        a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
        platt[n] = {"a": a, "b": b}
        single[n] = dict(raw=scores(y, 1 / (1 + np.exp(-lg)), family),
                         platt=scores(y, 1 / (1 + np.exp(-(a * lg + b))), family), **platt[n])

    P = np.stack([platt[n]["a"] * logits[n] + platt[n]["b"] for n in names], 1)      # (n, k) platt logits
    k = len(names)
    W = np.array(list(simplex(k)))
    Z = P @ W.T
    L = np.array([log_loss(y, np.clip(1 / (1 + np.exp(-Z[:, j])), *CLIP), labels=[0, 1]) for j in range(Z.shape[1])])
    j = int(np.argmin(L))
    wbest = W[j]
    pbest = 1 / (1 + np.exp(-(P @ wbest)))
    weq = np.full(k, 1.0 / k)
    peq = 1 / (1 + np.exp(-(P @ weq)))

    res = dict(models=names, platt=platt, grid_step=STEP, n_grid=len(W),
               best=dict(weights={n: float(w) for n, w in zip(names, wbest)}, **scores(y, pbest, family)),
               equal=dict(weights={n: float(w) for n, w in zip(names, weq)}, **scores(y, peq, family)),
               single=single)
    res["best"]["per_fold"] = {int(f): float(log_loss(y[fold == f], np.clip(pbest[fold == f], *CLIP), labels=[0, 1]))
                               for f in np.unique(fold)}
    (ROOT / "results/blend.json").write_text(json.dumps(res, indent=1))

    print(f"\n{'model':<28s} {'logloss':>9s} {'AUC':>7s}   (platt a,b)")
    for n in names:
        s = single[n]
        print(f"  {n+' (raw)':<26s} {s['raw']['logloss']:9.5f} {s['raw']['auc']:7.4f}")
        print(f"  {n+' (platt)':<26s} {s['platt']['logloss']:9.5f} {s['platt']['auc']:7.4f}   "
              f"a={s['a']:.3f} b={s['b']:.3f}")
    print(f"  {'equal-weight blend':<26s} {res['equal']['logloss']:9.5f} {res['equal']['auc']:7.4f}   "
          + " ".join(f"{n}={w:.2f}" for n, w in res["equal"]["weights"].items()))
    print(f"  {'best blend (grid)':<26s} {res['best']['logloss']:9.5f} {res['best']['auc']:7.4f}   "
          + " ".join(f"{n}={w:.2f}" for n, w in res["best"]["weights"].items()))
    print("\nbest-blend per-fold : " + "  ".join(f"{f}:{v:.4f}" for f, v in res["best"]["per_fold"].items()))
    print("best-blend per-family: " + "  ".join(f"{f}:{v:.4f}" for f, v in res["best"]["per_family"].items()
                                                if v is not None))
    print(f"\ngrid points {len(W)} over {k} models -> results/blend.json")


if __name__ == "__main__":
    main()
