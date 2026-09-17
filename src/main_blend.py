"""DaT Parkinson's — blended inference entrypoint (Fable). Reads submission_format.csv, writes submission.csv.
Models present under models/ are auto-detected; blend weights + per-model Platt params come from blend.json."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # torch+lightgbm libomp clash on macOS; harmless on Linux
import json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import extra_models  # noqa: E402  (must precede torch: loads lightgbm first)
import torch  # noqa: E402
from prep import preprocess  # noqa: E402
from quant import dequantize_sd  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
FALLBACK, LO, HI, BATCH = 0.55, 0.02, 0.98, 64


def prep_one(path):
    try:
        crop, info = preprocess(path)
        if not info.get("ok", False) or not np.isfinite(crop).all():
            return None
        return crop.astype(np.float32), float(info.get("sbr", 0.0))
    except Exception:
        return None


def run_axial(crops, valid, dev):
    """Axial 2.5D models: cnn2d.predict_axial handles models/2dt (int8 truncated) and models/2d (full) if present."""
    from cnn2d import predict_axial
    return predict_axial([crops[i] for i in valid], dev)


def main():
    t0 = time.time(); print("start", flush=True)
    fmt = pd.read_csv(DATA_DIR / "submission_format.csv")
    uids = [str(u) for u in fmt["uid"]]
    paths = [str(DATA_DIR / "niftis" / f"{u}.nii.gz") for u in uids]
    print(f"n scans {len(uids)}", flush=True)
    t1 = time.time()
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=min(12, os.cpu_count() or 1)) as pool:
        res = list(pool.imap(prep_one, paths, chunksize=4))
    crops = [r[0] if r else None for r in res]
    sbrs = [r[1] if r else 0.0 for r in res]
    valid = [i for i, c in enumerate(crops) if c is not None]
    print(f"prep seconds {time.time() - t1:.1f}  n failed {len(uids) - len(valid)}", flush=True)

    probs = np.full(len(uids), FALLBACK, np.float64)
    if valid:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blend = json.load(open(SRC / "blend.json"))
        weights, platt = blend["weights"], blend["platt"]
        logits = {}                                   # name -> array over `valid`
        t2 = time.time()
        vcrops = [crops[i] for i in valid]; vsbrs = [sbrs[i] for i in valid]
        try:
            extra = extra_models.predict_extra(vcrops, vsbrs, dev)   # feat_lgb, feat_lr, cnn_cor (if present)
            logits.update({k: np.asarray(v, np.float64) for k, v in extra.items()})
        except Exception as e:
            print(f"extra_models failed: {type(e).__name__}", flush=True)
        try:
            logits.update({k: np.asarray(v, np.float64) for k, v in run_axial(crops, valid, dev).items()})
        except Exception as e:
            print(f"axial failed: {type(e).__name__}", flush=True)
        if (SRC / "models" / "3d").exists():
            try:
                from cnn3d import predict_3d
                logits["cnn_3d"] = np.asarray(predict_3d(vcrops, dev), np.float64)
            except Exception as e:
                print(f"cnn_3d failed: {type(e).__name__}", flush=True)
        used = [k for k in weights if k in logits and weights[k] > 0]
        print(f"models used {used}  inference seconds {time.time() - t2:.1f}", flush=True)
        z = np.zeros(len(valid)); wsum = np.zeros(len(valid))
        for k in used:
            lk = logits[k]; ok = np.isfinite(lk)
            z[ok] += weights[k] * (platt[k]["a"] * lk[ok] + platt[k]["b"]); wsum[ok] += weights[k]
        post = blend.get("post", {"a": 1.0, "b": 0.0})
        wtot = float(sum(weights[k] for k in used)) if used else 1.0
        zb = post["a"] * (z * (wtot / np.where(wsum > 0, wsum, 1.0))) + post["b"]   # rescale only if a model is missing for a scan
        p = np.where(wsum > 0, 1.0 / (1.0 + np.exp(-zb)), FALLBACK)
        probs[valid] = p
    probs = np.clip(np.nan_to_num(probs, nan=FALLBACK, posinf=FALLBACK, neginf=FALLBACK), LO, HI)
    out = pd.DataFrame({"uid": uids, "is_pathologic": probs})
    out.to_csv("submission.csv", index=False)
    assert len(out) == len(fmt) and list(out["uid"]) == [str(u) for u in fmt["uid"]], "uid order mismatch"
    assert np.isfinite(probs).all() and probs.min() >= LO and probs.max() <= HI, "bad probabilities"
    print(f"written {len(out)} rows in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
