"""Step 5a: verify submission row order and score the smoke set."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sub = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "submission_src/submission.csv")
fmt = pd.read_csv(ROOT / "data/smoke/submission_format.csv")
lab = pd.read_csv(ROOT / "data/smoke/test_labels.csv")
assert list(sub.uid) == list(fmt.uid), "ORDER MISMATCH"
print("order check: PASS (%d rows, identical to submission_format)" % len(sub))
y = lab.set_index("uid").loc[sub.uid, "is_pathologic"].to_numpy()
p = np.clip(sub.is_pathologic.to_numpy(), 0.02, 0.98)
print("smoke log loss %.4f  (n=%d, mean p %.3f, min %.3f, max %.3f)"
      % (log_loss(y, p, labels=[0, 1]), len(y), p.mean(), p.min(), p.max()))
print("expect < 0.35:", "PASS" if log_loss(y, p, labels=[0, 1]) < 0.35 else "FAIL")
