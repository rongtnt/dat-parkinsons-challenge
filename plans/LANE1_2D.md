# LANE 1 — 2.5D ResNet18 (critical path) + inference main.py    HARD CAP: finish by 20:50 UTC (check `date -u`)

You OWN: scripts/train_2d.py, results/oof_2d.csv, results/2d_summary.json, runtime/submission_src/main.py,
runtime/submission_src/cnn2d.py, runtime/submission_src/models/2d/*.pt, runtime/submission_src/calib_2d.json,
runtime/submission_src/prep.py (byte copy of src/prep.py), runtime/submission/submission.zip, tests/ (your test scripts).
Do NOT touch anything else (lane 2 owns scripts/features.py, scripts/train_feat.py, scripts/train_2d_cor.py, extra_models.py, models/feat, models/2dcor).

## Step 1 — input tensor (fixed design, no alternatives)
From crop (2,48,64,64) use channel 1 (background-normalised). Three axial slabs = mean over z-ranges [18:22), [22:26), [26:30)
-> tensor (3,64,64). Scale: x = slab/4.0 - 0.5. In the model forward, upsample to 128x128 with F.interpolate(bilinear, align_corners=False).
Write this as a function `make_input(crop: np.ndarray, zshift: int = 0) -> np.ndarray (3,64,64)` in runtime/submission_src/cnn2d.py
(zshift shifts all three z-ranges by the same offset; used only for train-time augmentation). Training imports cnn2d.py from
runtime/submission_src via sys.path so train and inference share one definition.

## Step 2 — model (fixed)
`torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)`; replace fc with nn.Linear(512, 1).
Class `Net(nn.Module)` in cnn2d.py: forward(x) = resnet(F.interpolate(x, size=128)). Inference builds `Net()` with weights=None
(no download in the container!) and loads a state_dict. Save weights as fp16 state_dicts: {k: v.half()} -> torch.save; load with
map_location and .float().

## Step 3 — training (fixed) : scripts/train_2d.py
- 5 folds from data/folds.csv, seeds 0 and 1 (10 models). Per fold: AdamW(lr=3e-4, weight_decay=1e-4), batch 32, 24 epochs,
  cosine schedule to 0 (per step), BCEWithLogitsLoss on smoothed targets t = y*0.95 + 0.025. Use the LAST epoch (no best-epoch picking).
- Augmentation (train only, applied on the (3,64,64) tensor): L/R flip p=0.5 (flip the X axis = last dim), z-jitter zshift in
  {-2,-1,0,1,2} (recompute slabs), random shift by up to ±4 pixels in y/x (torch.roll), random rotation ±10° (torchvision.transforms.functional.rotate on the tensor, bilinear), intensity scale ×U(0.9,1.1).
- Precompute per-fold arrays in RAM (they are small: 1362x3x64x64 float32 = 67 MB per zshift; precompute all 5 zshifts).
- Validation each epoch: print fold, epoch, train loss, val log loss (raw sigmoid, clipped) and AUC. ≤ 30 lines per fold.
- OOF: for fold k models (both seeds), predict fold k with TTA = mean of logits over {identity, L/R flip}; average logits over seeds.
  Save results/oof_2d.csv with columns uid,y,fold,family,logit,prob (prob = sigmoid(logit)).
- Calibration: fit Platt scaling on the OOF logits: p = sigmoid(a*logit + b) via sklearn LogisticRegression(C=1e6) on the 1-D logit.
  Save runtime/submission_src/calib_2d.json = {"a":..., "b":...}. Report OOF log loss raw vs Platt, AUC, per-fold, per-family.
- Time budget: if one fold takes > 4 min on MPS, stop after that fold and report timing; do not silently reduce epochs.
- Save results/2d_summary.json with all numbers (raw/platt log loss, auc, per-fold, per-family, seconds per fold, device).

## Step 4 — inference: runtime/submission_src/main.py (fixed contract)
- DATA_DIR = Path(os.environ.get("DATA_DIR", "/code_execution/data")); SRC = Path(__file__).resolve().parent.
- Read DATA_DIR/"submission_format.csv" with pandas; keep its EXACT row order; paths = DATA_DIR/"niftis"/f"{uid}.nii.gz" (no glob).
- Preprocess with multiprocessing.Pool(processes=min(12, os.cpu_count())) using imap over paths (chunksize 4); each worker calls a
  wrapper that returns None on ANY exception (try/except Exception) — never let one scan kill the run. Use the "fork" context if available.
- Device = "cuda" if torch.cuda.is_available() else "cpu". Load the 10 state_dicts once. Batch inference (batch 64), TTA identity + L/R flip,
  mean logit over TTA and models -> Platt -> clip to [0.02, 0.98]. Scans that failed preprocessing -> 0.55.
- Write "submission.csv" in the working dir with columns uid,is_pathologic, same order as submission_format, index=False.
- Logging: at most 15 print lines in total (start, n scans, prep seconds, n failed, inference seconds, written). No per-scan prints.
- Assert at the end: len(out)==len(format), all probs finite, in [0.02,0.98], uids identical in order.

## Step 5 — tests (all must pass; paste outputs)
a) `cd runtime/submission_src && DATA_DIR=./data/smoke python main.py` -> then a script that checks
   uid order == submission_format order and prints log loss vs data/smoke/test_labels.csv (these 20 are training scans; expect < 0.35).
b) Garbage-robustness: create tmp/garbage/niftis with 6 NIfTIs written by nibabel: random noise 64^3 (uint16), all-zeros 64^3,
   tiny 8x8x8, 4-D (64,64,64,2), one with an LPS/flipped affine (diag(-2,-2,2)), one with huge values (1e9) — plus a
   submission_format.csv listing those 6 uids AND one uid whose file does not exist. Run main.py with DATA_DIR=tmp/garbage:
   must exit 0 and write 7 rows (missing/failed -> 0.55).
c) Timing on the smoke set: report seconds per scan for preprocessing and total wall time.
d) Package: `cd runtime/submission_src && rm -f ../submission/submission.zip && zip -r -q ../submission/submission.zip . -x '*.DS_Store' -x '__pycache__/*' -x '*/__pycache__/*'`
   then `unzip -l ../submission/submission.zip | head -20` and `ls -la ../submission/submission.zip` (main.py must be at the root; size < 400 MB).

## Deliverable
status=complete with: all metrics from Step 3, test outputs from Step 5, zip size, and a list of every file you created.
If blocked, status=blocked + discrepancies.
