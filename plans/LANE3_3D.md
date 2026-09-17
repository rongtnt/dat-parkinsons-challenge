# LANE 3 — small 3D CNN on the full crop (both channels)     HARD CAP: 21:00 UTC for seed 0 (check `date -u`); seed 1 only if < 21:20

You OWN: scripts/train_3d.py, results/oof_3d.csv, results/3d_summary.json, runtime/submission_src/cnn3d.py,
runtime/submission_src/models/3d/*.pt, runtime/submission_src/models/3d/calib.json, runtime/submission_src/quant.py (byte copy of src/quant.py).
Do NOT touch main.py, cnn2d.py, extra_models.py, prep.py, models/2d, models/2dcor, models/feat, submission.zip.

## Input (fixed)
crop (2,48,64,64) float16 from data/cache/crops_2mm.npy -> float32 tensor x with x[0] = crop[0] - 0.5 (peak-normalised, peak ~1),
x[1] = crop[1]/4.0 - 0.5 (background-normalised). Shape (2, 48, 64, 64) = (C, Z, Y, X). X axis (last) is left-right.

## Model (fixed): class Net3D(nn.Module) in runtime/submission_src/cnn3d.py
stem: Conv3d(2,16,3,padding=1) -> BatchNorm3d -> ReLU -> MaxPool3d(2)            # -> (16, 24, 32, 32)
block(cin,cout,stride): [Conv3d(cin,cout,3,stride=stride,padding=1)-BN-ReLU-Conv3d(cout,cout,3,padding=1)-BN] + shortcut
(1x1 conv stride s when shape changes) -> ReLU. Stages: block(16,32,2) -> block(32,64,2) -> block(64,128,2)  # -> (128, 3, 4, 4)
head: AdaptiveAvgPool3d(1) -> Flatten -> Dropout(0.3) -> Linear(128,1). Print the parameter count (expect ~1.0M).

## Training (fixed): scripts/train_3d.py
5 folds from data/folds.csv (train folds != k, OOF on fold k), seed 0 first (then seed 1 if time). AdamW(lr=1e-3, weight_decay=1e-4),
batch 16, 30 epochs, cosine to 0 per step, BCEWithLogitsLoss on smoothed targets t = y*0.90 + 0.05.
Augmentation (train only, on the tensor): flip X axis p=0.5; random integer shifts up to ±3 voxels in each axis via torch.roll;
intensity scale ×U(0.9,1.1) applied to both channels' (x+0.5) then re-centred; Gaussian noise N(0,0.02) p=0.5.
Device: MPS if available. TIME CHECK: measure the first epoch; if one epoch > 25 s, switch to batch 32 and report; if still > 25 s,
reduce to 20 epochs and report. Never silently change anything else. Print per-epoch val log loss (clipped [0.02,0.98]) + AUC, ≤ 35 lines/fold.
Use the LAST epoch. TTA at prediction: mean logit over {identity, X-flip}. Average logits over seeds if seed 1 exists.
OOF -> results/oof_3d.csv (uid,y,fold,family,logit,prob). Platt (sklearn LogisticRegression C=1e6 on the 1-D logit) -> models/3d/calib.json {"a","b"}.
Summary -> results/3d_summary.json: raw/platt log loss, AUC, per-fold, per-family, seconds per fold, params, device.

## Weights (fixed)
Save int8: sys.path.insert(0,'src'); from quant import quantize_sd, dequantize_sd; torch.save(quantize_sd(model.state_dict()),
'runtime/submission_src/models/3d/f{k}_s{seed}.pt'). cp src/quant.py runtime/submission_src/quant.py. After each fold, reload with
dequantize_sd and check max |logit_fp32 - logit_int8| on 32 val crops < 0.05 (print it).

## Inference module (fixed): runtime/submission_src/cnn3d.py must also expose
`predict_3d(crops: list[np.ndarray|None], device) -> np.ndarray` of raw mean-TTA mean-model logits (np.nan for None crops), batch 16,
loading every models/3d/*.pt once via dequantize_sd (import quant from the same directory). Test on 20 cached crops: matches OOF-time logits within 0.05.

## Deliverable
status complete/blocked; all metrics; timings; files. Never submit anything anywhere.
