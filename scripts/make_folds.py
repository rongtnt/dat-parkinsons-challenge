"""Step 0b: 5-fold split stratified by label x acquisition family -> data/folds.csv.
Family = voxel size + matrix shape from data/meta_train.csv (scripts/census.py). Run after census.py.
usage: python scripts/make_folds.py [out.csv]"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SEED, K = 42, 5


def family(sx, nx, nz):
    if sx < 1.95: return "f15hi"
    if sx < 2.2: return "f20"
    if sx < 2.35: return "f23"
    if sx < 2.43: return "f2398_87" if nz == 87 else "f2398_other"
    if sx < 3.0: return "f246_256" if nx == 256 else "f246_crop"
    if sx < 3.5: return "f33"
    if sx < 4.0: return "f3895_128" if nz == 128 else "f3895_thin"
    return "f442"


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/folds.csv"
    if out.exists(): sys.exit(f"{out} exists; delete it first")
    meta = pd.read_csv(ROOT / "data/meta_train.csv")
    lab = pd.read_csv(ROOT / "data/train_labels.csv").set_index("uid")["is_pathologic"]
    df = pd.DataFrame({"uid": meta.uid, "y": lab.loc[meta.uid].astype(int).values,
                       "family": [family(a, b, c) for a, b, c in zip(meta.sx, meta.nx, meta.nz)]})
    df["fold"] = -1
    strata = df.y.astype(str) + "_" + df.family
    for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED).split(df, strata)):
        df.loc[te, "fold"] = k
    df.to_csv(out, index=False)
    print(out, len(df), df.fold.value_counts().sort_index().tolist(), df.family.value_counts().to_dict())
