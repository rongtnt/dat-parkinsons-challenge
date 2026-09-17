"""Step 5b: build tmp/garbage with 6 pathological NIfTIs + 1 missing-file uid."""
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp" / "garbage"
(D / "niftis").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
A = np.diag([2.0, 2.0, 2.0, 1.0])
cases = {
    "g_noise": (rng.integers(0, 4000, (64, 64, 64)).astype(np.uint16), A),
    "g_zeros": (np.zeros((64, 64, 64), np.uint16), A),
    "g_tiny": (rng.integers(0, 100, (8, 8, 8)).astype(np.uint16), A),
    "g_4d": (rng.integers(0, 4000, (64, 64, 64, 2)).astype(np.uint16), A),
    "g_lps": (rng.integers(0, 4000, (64, 64, 64)).astype(np.uint16), np.diag([-2.0, -2.0, 2.0, 1.0])),
    "g_huge": (np.full((64, 64, 64), 1e9, np.float32), A),
}
for name, (data, aff) in cases.items():
    nib.save(nib.Nifti1Image(data, aff), D / "niftis" / f"{name}.nii.gz")
uids = list(cases) + ["g_missing"]
pd.DataFrame({"uid": uids, "is_pathologic": 0.5}).to_csv(D / "submission_format.csv", index=False)
print("wrote", len(cases), "niftis +", len(uids), "format rows ->", D)
