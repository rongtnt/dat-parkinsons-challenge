"""Canonical preprocessing shared by training and inference (runtime has scipy/nibabel/numpy).
preprocess(path) -> (crop float32 (2, 48, 64, 64) in (C, Z, Y, X) order, info dict)
Crop = 128x128x96 mm box at 2 mm isotropic, centred on the striatal complex.
ch0 = crop / peak (mean of top 0.5% voxels)   ch1 = crop / brain background median, clipped to [0, 12].
Each scan is processed independently (no test-set statistics)."""
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

VOX = 2.0
SHAPE = (48, 64, 64)  # Z, Y, X voxels  -> 96 x 128 x 128 mm
PRIOR = 0.55

def load_2mm(path):
    im = nib.as_closest_canonical(nib.load(path))  # RAS
    vol = np.asarray(im.dataobj).astype(np.float32)
    if vol.ndim == 4: vol = vol[..., 0]
    vox = np.sqrt((im.affine[:3, :3] ** 2).sum(0))
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    vol[vol < 0] = 0
    z = (vox / VOX).astype(np.float64)
    vol = ndi.zoom(vol, z, order=1)          # RAS, 2 mm
    return np.ascontiguousarray(vol.transpose(2, 1, 0))  # (Z, Y, X)

def crop_at(vol, center, shape=SHAPE):
    out = np.zeros(shape, np.float32)
    for_src, for_dst = [], []
    for c, n, s in zip(center, vol.shape, shape):
        lo = int(round(c)) - s // 2
        src_lo, src_hi = max(lo, 0), min(lo + s, n)
        for_src.append(slice(src_lo, src_hi)); for_dst.append(slice(src_lo - lo, src_hi - lo))
    out[tuple(for_dst)] = vol[tuple(for_src)]
    return out

def preprocess(path):
    vol = load_2mm(path)
    s = ndi.gaussian_filter(vol, 2.0)
    top = np.percentile(s, 99.9)
    info = {"ok": True}
    if top <= 0:
        info["ok"] = False
        return np.zeros((2,) + SHAPE, np.float32), info
    mask = s > 0.25 * top
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    c0 = np.array(ndi.center_of_mass(mask))
    # hotspot search restricted to 50 mm of the brain centroid (rejects ring/scalp artefacts)
    zz, yy, xx = np.ogrid[:s.shape[0], :s.shape[1], :s.shape[2]]
    near = ((zz - c0[0]) ** 2 + (yy - c0[1]) ** 2 + (xx - c0[2]) ** 2) <= 25 ** 2
    sm = np.where(near, s, -1.0)
    h = np.array(np.unravel_index(int(np.argmax(sm)), s.shape), np.float64)
    peak_s = s[tuple(h.astype(int))]
    # striatal complex centre = intensity-weighted centroid of bright voxels within 40 mm of hotspot
    near_h = ((zz - h[0]) ** 2 + (yy - h[1]) ** 2 + (xx - h[2]) ** 2) <= 20 ** 2
    bright = near_h & (s > 0.5 * peak_s)
    w = np.where(bright, s, 0.0)
    center = np.array(ndi.center_of_mass(w)) if w.sum() > 0 else h
    crop = crop_at(vol, center)
    cmask = crop_at(mask.astype(np.float32), center) > 0.5
    flat = crop[cmask] if cmask.sum() > 100 else crop[crop > 0]
    if flat.size < 100: flat = crop.ravel()
    k = max(int(0.005 * flat.size), 5)
    peak = float(np.mean(np.sort(flat)[-k:])) if flat.size else 1.0
    bg = float(np.median(flat)) if flat.size else 1.0
    peak = max(peak, 1e-6); bg = max(bg, 1e-6)
    ch0 = crop / peak
    ch1 = np.clip(crop / bg, 0, 12.0)
    info.update(center=center.tolist(), hot=h.tolist(), c0=c0.tolist(), peak=peak, bg=bg, sbr=peak / bg,
                brain_vox=int(mask.sum()), shape2mm=list(vol.shape))
    return np.stack([ch0, ch1]).astype(np.float32), info

if __name__ == "__main__":
    import sys, glob, os, json, pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(f"{D}/cache", exist_ok=True)
    lab = pd.read_csv(f"{D}/train_labels.csv")
    uids = lab.uid.tolist()
    paths = [f"{D}/niftis/{u}.nii.gz" for u in uids]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(preprocess, paths, chunksize=4))
    arr = np.stack([r[0] for r in res]).astype(np.float16)
    np.save(f"{D}/cache/crops_2mm.npy", arr)
    np.save(f"{D}/cache/uids.npy", np.array(uids))
    rows = []
    for u, (_, info) in zip(uids, res):
        rows.append(dict(uid=u, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in info.items()}))
    pd.DataFrame(rows).to_csv(f"{D}/cache/prep_info.csv", index=False)
    print("cache", arr.shape, arr.dtype, "bad", sum(1 for r in res if not r[1]["ok"]))
