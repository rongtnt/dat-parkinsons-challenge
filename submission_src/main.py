"""DaT Parkinson's — inference entrypoint. Reads submission_format.csv, writes submission.csv."""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from cnn2d import make_input, load_model  # noqa: E402
from prep import preprocess  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
FALLBACK = 0.55
LO, HI = 0.02, 0.98
BATCH = 64


def prep_one(path):
    """Never raise: a single bad scan must not kill the run."""
    try:
        crop, info = preprocess(path)
        if not info.get("ok", False):
            return None
        x = make_input(crop)
        if not np.isfinite(x).all():
            return None
        return x.astype(np.float32)
    except Exception:
        return None


@torch.no_grad()
def predict(models, x, dev):
    """Mean logit over models and over TTA {identity, L/R flip}."""
    out = np.zeros(len(x), np.float64)
    for model in models:
        chunks = []
        for i in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[i:i + BATCH]).to(dev)
            lg = model(xb).squeeze(1) + model(torch.flip(xb, dims=[-1])).squeeze(1)
            chunks.append((lg / 2.0).float().cpu().numpy())
        out += np.concatenate(chunks)
    return out / len(models)


def main():
    t0 = time.time()
    print("start", flush=True)
    fmt = pd.read_csv(DATA_DIR / "submission_format.csv")
    uids = [str(u) for u in fmt["uid"]]
    paths = [str(DATA_DIR / "niftis" / f"{u}.nii.gz") for u in uids]
    print(f"n scans {len(uids)}", flush=True)

    t1 = time.time()
    nproc = min(12, os.cpu_count() or 1)
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=nproc) as pool:
        feats = list(pool.imap(prep_one, paths, chunksize=4))
    ok = [i for i, f in enumerate(feats) if f is not None]
    print(f"prep seconds {time.time() - t1:.1f}", flush=True)
    print(f"n failed {len(uids) - len(ok)}", flush=True)

    probs = np.full(len(uids), FALLBACK, np.float64)
    t2 = time.time()
    if ok:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device {dev}", flush=True)
        paths_pt = sorted((SRC / "models" / "2d").glob("*.pt"))
        models = [load_model(p, dev) for p in paths_pt]
        print(f"models {len(models)}", flush=True)
        logit = predict(models, np.stack([feats[i] for i in ok]), dev)
        cal = json.load(open(SRC / "calib_2d.json"))
        probs[ok] = 1.0 / (1.0 + np.exp(-(cal["a"] * logit + cal["b"])))
    print(f"inference seconds {time.time() - t2:.1f}", flush=True)

    probs = np.clip(np.nan_to_num(probs, nan=FALLBACK, posinf=FALLBACK, neginf=FALLBACK), LO, HI)
    out = pd.DataFrame({"uid": uids, "is_pathologic": probs})
    out.to_csv("submission.csv", index=False)

    assert len(out) == len(fmt), "row count mismatch"
    assert list(out["uid"]) == [str(u) for u in fmt["uid"]], "uid order mismatch"
    assert np.isfinite(probs).all() and probs.min() >= LO and probs.max() <= HI, "bad probabilities"
    print(f"written {len(out)} rows in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
