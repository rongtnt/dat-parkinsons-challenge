"""Step A: compute the handcrafted feature table for all 1362 cached crops -> results/feats.csv."""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission_src"))
from extra_models import compute_features  # noqa: E402  (single definition, shared with inference)


def main():
    t0 = time.time()
    crops = np.load(ROOT / "data/cache/crops_2mm.npy", mmap_mode="r")
    uids = np.load(ROOT / "data/cache/uids.npy", allow_pickle=True)
    info = pd.read_csv(ROOT / "data/cache/prep_info.csv").set_index("uid")
    sbr = info.loc[list(uids), "sbr"].to_numpy()

    rows = [dict(uid=u, **compute_features(crops[i], float(sbr[i]))) for i, u in enumerate(uids)]
    feats = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    feats.to_csv(ROOT / "results/feats.csv", index=False)

    folds = pd.read_csv(ROOT / "data/folds.csv")
    df = folds.merge(feats, on="uid", how="left")
    assert df[feats.columns[1:]].notna().all().all(), "missing features after merge"
    cols = [c for c in feats.columns if c != "uid"]
    auc = {c: roc_auc_score(df.y, df[c]) for c in cols if df[c].nunique() > 1}
    top = sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:10]
    print(f"features {feats.shape} in {time.time()-t0:.1f}s -> results/feats.csv")
    print("top 10 single-feature AUC (|auc-0.5|):")
    for name, a in top:
        print(f"  {name:<20s} {a:.4f}")


if __name__ == "__main__":
    main()
