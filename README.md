# DaT Parkinson's Challenge: 70th on the private leaderboard

Solution code for DrivenData's [DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/): binary classification of DaT-SPECT brain volumes (pathologic vs normal), scored by log loss, code-execution submissions (offline container, 1x A100, 3 h limit).

Built and submitted in one day (2026-09-16) on a MacBook with PyTorch MPS.

## Result

| | log loss |
|---|---|
| Private leaderboard | **rank 70** |
| Best public score (submission 1) | 0.2696 |
| Public top-3 line | 0.2263 |
| Local 5-fold OOF of the shipped stack | 0.2585 |

## Approach

**Preprocessing** (`src/prep.py`): resample to 2 mm isotropic RAS, locate the striatal complex (smoothed hotspot near the brain centroid, then intensity-weighted centroid), crop a 96 x 128 x 128 mm box, two channels: intensity / striatal peak, and intensity / brain-background median clipped to [0, 12]. No test-set statistics.

**Models**, each 5-fold (stratified by label x acquisition family) x 2 seeds, last epoch, label smoothing 0.1, TTA = left/right flip:

| key | input | model | OOF log loss |
|---|---|---|---|
| `2dt` | 3 axial slabs (2.5D) | ResNet18 truncated after layer3, int8 | 0.291 |
| `2d` | 3 axial slabs | ResNet18, fp16 | 0.290 |
| `2dcor` | 3 coronal slabs | truncated ResNet18, int8 | 0.303 |
| `3d` | full crop, both channels | small 3D CNN, 0.9M params | 0.312 |
| `2dsag` | 3 sagittal slabs | truncated ResNet18, int8 | 0.415 |
| `feat` | 29 striatal features (SBR, volumes, extents, asymmetry) | LightGBM / logistic regression | 0.334 / 0.346 |

**Stacking** (`scripts/package.py`): Platt-scale each model's OOF logits, then non-negative weights on the scaled logits (L-BFGS-B, cross-fitted by fold). `2dt + 2dcor + feat` = 0.2585 OOF cross-fit, 0.2696 public.

**Size**: `src/quant.py` packs a truncated ResNet18 as int8 per-output-channel (2.8 MB per model, max logit drift 0.4). A full submission zip is 48 MB.

## Submissions

| # | stack | OOF cross-fit | public LB |
|---|---|---|---|
| 1 | 2dt + 2dcor + feat | 0.2585 | **0.2696** |
| 2 | 2dtc + 2dcorc + feat (retrained with the 3 % highest-OOF-loss training scans dropped) | 0.2502 | 0.2723 |
| 3 | 2dtc5 (5 % dropped) + 2dt6n (6-channel input) + 2dcorc + feat | 0.2424 | 0.2772 |

Shipped weights and Platt parameters: `results/blend_sub{1,2,3}.json`. Full log: `results/submissions.md`.

## What did not work

- **Dropping noisy-label training scans.** OOF improved monotonically (0 / 3 / 5 % dropped: 0.2585 / 0.2502 / 0.2424) while the public LB got monotonically worse (0.2696 / 0.2723 / 0.2772). The drop list came from OOF predictions that had already seen each validation fold's labels, so the gain was leakage.
- **Mixup**: +0.02 to +0.05 worse OOF.
- **3D CNN and sagittal view**: no weight in the stack.
- **6-channel input** (both channels x 3 slabs, no mixup): -0.022 OOF, only ever forward-tested inside submission 3.

## Layout

```
src/prep.py            preprocessing; `python src/prep.py` builds data/cache
src/quant.py           int8 state_dict pack / unpack
src/main_blend.py      inference entrypoint that shipped (copied to main.py in the zip)
submission_src/        submission package source: cnn2d.py, extra_models.py, cnn3d.py, Platt calibration json
scripts/               census, make_folds, features, train_*, blend, package
tests/                 int8 round-trip, smoke-set and garbage-input checks
plans/                 lane specs handed to the executor agents (Claude Code; the plans are as given)
results/               per-model OOF summaries, shipped blend weights, submission log
```

Not included: competition data (`data/`), trained weights (`submission_src/models/`), per-scan OOF predictions.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# put DrivenData's train_labels.csv and niftis/*.nii.gz under data/
python scripts/census.py         # data/meta_train.csv: voxel size and matrix per scan
python scripts/make_folds.py     # data/folds.csv (same recipe; 96.5 % of assignments match the campaign split)
python src/prep.py               # data/cache/crops_2mm.npy, 1362 x 2 x 48 x 64 x 64 float16
python scripts/features.py && python scripts/train_feat.py
python scripts/train_2dt.py && python scripts/train_2d_cor.py
python scripts/package.py s1 cnn_2dt cnn_cor feat_lgb feat_lr   # upload/submission_s1.zip
```

Container test: clone [drivendataorg/competition-sfmn-parkinsons-runtime](https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime), unzip the submission into its `submission_src/`, follow its README. Platform versions: Python 3.12, torch 2.12.1, torchvision 0.27.1, numpy 2.2.6, scipy 1.15.3, nibabel 5.4.2, pandas 3.0.3, scikit-learn 1.8.0, lightgbm 4.6.0.

## License

MIT
