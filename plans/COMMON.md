# COMMON CONTEXT (read fully before any step)

Competition: DrivenData "DaT Parkinson's Challenge" (id 311). Binary classification of DaT SPECT volumes, metric = log loss
(lower better), code-execution submission (main.py in a zip, runs offline on 1x A100, 24 vCPU, Python 3.12, 3 h limit).
Deadline 23:59 UTC TODAY (2026-09-16). Fable (strategist) submits; you never submit anything.

## Environment (mandatory)
- Working dir: . ; interpreter: `source .venv/bin/activate` (Python 3.12, torch with MPS, nibabel, scipy,
  sklearn, lightgbm, pandas, numpy, matplotlib installed). Check `python -c "import torch;print(torch.backends.mps.is_available())"`.
  If MPS is unavailable or a training step is >5x slower than expected, fall back to CPU (20 cores; set torch.set_num_threads(16)).
- First Bash call of your session is blocked by a "GateGuard" hook: reply with (1) the user request in one sentence, (2) what the
  command produces, then re-run the same command. Do not fight it, do not disable it.
- Never write files into the home directory. Scratch -> ./tmp/ (create it).
- Runtime package versions on the platform (do NOT rely on anything else): torch 2.12.1, torchvision 0.27.1, timm 1.0.27,
  numpy 2.2.6, scipy 1.15.3, nibabel 5.4.2, pandas 3.0.3, scikit-learn 1.8.0, lightgbm 4.6.0. No internet at inference.
  Ship only torch state_dicts (torch.save(model.state_dict())), JSON, and lightgbm text models. NEVER pickle sklearn objects.

## Data (already prepared; do not rebuild, do not modify)
- data/cache/crops_2mm.npy : float16 array (1362, 2, 48, 64, 64) in (N, C, Z, Y, X). Load with np.load(..., mmap_mode='r').
  Crop = 96 x 128 x 128 mm at 2 mm isotropic, RAS, centred on the striatal complex (z=24, y=32, x=32 is the striatal centre).
  X axis = left-right (index 32 = midline), Y axis = posterior->anterior, Z = inferior->superior.
  channel 0 = intensity / striatal peak (peak ~1.0); channel 1 = intensity / brain-background median, clipped [0, 12] (peak ~3-6).
- data/cache/uids.npy : uid per row (same order as crops). data/cache/prep_info.csv : uid, peak, bg, sbr, brain_vox, centre coords.
- data/folds.csv : uid, y (label 0/1), family (acquisition family string), fold (0..4). Stratified by label x family, seed 42.
  EVERYONE uses exactly these folds. Train on folds != k, predict fold k -> out-of-fold (OOF) predictions for all 1362 scans.
- data/train_labels.csv : uid, is_pathologic. data/smoke/ : 20 training scans + submission_format.csv + test_labels.csv
  (the platform smoke-test layout). data/niftis/*.nii.gz : the raw training volumes.
- src/prep.py : the ONLY preprocessing. preprocess(path) -> (crop float32 (2,48,64,64), info dict). Do not edit it.
  runtime/submission_src/prep.py must stay a byte-identical copy (cp src/prep.py runtime/submission_src/prep.py).

## Reporting rules
- Every metric you report must come from a command you ran; paste the printed numbers. OOF log loss uses
  sklearn.metrics.log_loss on probabilities clipped to [0.02, 0.98]. Also report ROC AUC, per-fold log loss, per-family log loss.
- Label noise is known to be large (hardest 5% of scans carry ~half the loss). Do NOT chase individual outliers.
- Stop and report `status=blocked` (with exact discrepancies) if: a step is ambiguous, contradicts reality, the same error
  happens twice, or a time cap is hit. Never improvise a different design. Never modify files owned by the other lane.
