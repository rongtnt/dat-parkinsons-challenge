"""Step B: LightGBM + logistic regression on the handcrafted features (5 folds from data/folds.csv)."""
import json, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_src/models/feat"
CLIP = (0.02, 0.98)

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=0)
NUM_ROUND = 400


def summarise(df, name):
    p = np.clip(df.prob.to_numpy(), *CLIP)
    y = df.y.to_numpy()
    s = dict(model=name, n=len(df), logloss=float(log_loss(y, p, labels=[0, 1])),
             auc=float(roc_auc_score(y, p)),
             per_fold={int(k): float(log_loss(g.y, np.clip(g.prob, *CLIP), labels=[0, 1]))
                       for k, g in df.groupby("fold")},
             per_fold_n={int(k): int(len(g)) for k, g in df.groupby("fold")},
             per_family={str(k): float(log_loss(g.y, np.clip(g.prob, *CLIP), labels=[0, 1]))
                         if g.y.nunique() > 1 else None for k, g in df.groupby("family")},
             per_family_n={str(k): int(len(g)) for k, g in df.groupby("family")})
    print(f"\n[{name}] OOF logloss {s['logloss']:.5f}  AUC {s['auc']:.4f}")
    print("  per-fold : " + "  ".join(f"{k}:{v:.4f}(n={s['per_fold_n'][k]})" for k, v in s["per_fold"].items()))
    print("  per-family:")
    for k, v in sorted(s["per_family"].items(), key=lambda kv: -s["per_family_n"][kv[0]]):
        print(f"    {k:<14s} n={s['per_family_n'][k]:<4d} " + (f"{v:.4f}" if v is not None else "single-class"))
    return s


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(ROOT / "results/feats.csv")
    folds = pd.read_csv(ROOT / "data/folds.csv")
    df = folds.merge(feats, on="uid", how="left")
    cols = [c for c in feats.columns if c != "uid"]
    assert len(cols) == 29, len(cols)
    X, y, fold = df[cols].to_numpy(np.float64), df.y.to_numpy(np.float64), df.fold.to_numpy()
    (OUT / "features.json").write_text(json.dumps(cols))

    oof = {"lgb": np.zeros(len(df)), "lr": np.zeros(len(df))}
    lr_json = {}
    for k in range(5):
        tr, va = fold != k, fold == k
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X[tr], label=y[tr]), num_boost_round=NUM_ROUND)
        booster.save_model(str(OUT / f"lgb_fold{k}.txt"))
        oof["lgb"][va] = booster.predict(X[va], raw_score=True)

        mean, scale = X[tr].mean(0), X[tr].std(0)
        scale[scale == 0] = 1.0
        lr = LogisticRegression(C=0.3, max_iter=2000).fit((X[tr] - mean) / scale, y[tr])
        oof["lr"][va] = lr.decision_function((X[va] - mean) / scale)
        lr_json[str(k)] = dict(mean=mean.tolist(), scale=scale.tolist(),
                               coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    (OUT / "lr.json").write_text(json.dumps(lr_json))

    summary = {}
    for key, fname, name in (("lgb", "oof_feat.csv", "feat_lgb"), ("lr", "oof_lr.csv", "feat_lr")):
        logit = oof[key]
        out = df[["uid", "y", "fold", "family"]].copy()
        out["logit"] = logit
        out["prob"] = 1.0 / (1.0 + np.exp(-logit))
        out.to_csv(ROOT / "results" / fname, index=False)
        summary[name] = summarise(out, name)

    imp = sorted(zip(cols, lgb.Booster(model_file=str(OUT / "lgb_fold0.txt")).feature_importance("gain")),
                 key=lambda kv: -kv[1])[:10]
    print("\ntop-10 LGB gain (fold 0): " + ", ".join(f"{n}:{g:.0f}" for n, g in imp))
    summary["seconds"] = round(time.time() - t0, 1)
    summary["n_features"] = len(cols)
    (ROOT / "results/feat_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote results/feat_summary.json in {summary['seconds']}s")


if __name__ == "__main__":
    main()
