# LANE 2 — handcrafted-feature model, then coronal 2.5D CNN, then blend script     HARD CAP: 21:40 UTC (check `date -u`)

You OWN: scripts/features.py, scripts/train_feat.py, scripts/train_2d_cor.py, scripts/blend.py, results/feats.csv,
results/oof_feat.csv, results/oof_lr.csv, results/oof_2dcor.csv, results/feat_summary.json, results/2dcor_summary.json,
runtime/submission_src/extra_models.py, runtime/submission_src/models/feat/*, runtime/submission_src/models/2dcor/*.
Do NOT touch main.py, cnn2d.py, prep.py, models/2d (lane 1). Do NOT create submission.zip.

## Step A — features (fixed list): scripts/features.py -> function `compute_features(crop: np.ndarray (2,48,64,64), sbr: float) -> dict`
Use channel 1 (c = crop[1], shape (Z,Y,X); midline x=32; striatal centre (24,32,32)). Restrict to the striatal box B = c[12:36, 16:48, 8:56].
peak = mean of the top 0.5% voxels of B. Bright mask M = B > 0.5*peak ; M70 = B > 0.7*peak.
Global: peak, sbr, vol50 = M.sum(), vol70 = M70.sum(), mean_B, p90_B, ring = mean of c outside B (proxy for background/scatter).
Per side s in {L: x<32, R: x>=32} (x in B coordinates: L = B[..., :24], R = B[..., 24:]):
  max_s, top1_s (mean of top 1% voxels), vol50_s, vol70_s, extent_y_s (max y - min y of M restricted to side, 0 if empty),
  extent_z_s, ycen_s (intensity-weighted y centroid of side bright voxels), pa_ratio_s = mean of B side voxels with y < ycen_s (posterior)
  over mean of voxels with y >= ycen_s (anterior), within the side's M; use 1.0 if undefined.
Asymmetry: asym_max = |max_L-max_R|/(max_L+max_R), asym_top1, asym_vol50, min_over_max_top1 = min(top1_L,top1_R)/max(...),
min_pa = min(pa_ratio_L, pa_ratio_R), min_extent_y = min(extent_y_L, extent_y_R). All NaN/inf -> 0.
Compute for all 1362 rows (multiprocessing fine) -> results/feats.csv (uid + features). Print feature AUCs (top 10 by |AUC-0.5|).

## Step B — models on features: scripts/train_feat.py (5 folds from data/folds.csv)
1) LightGBM: params = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=20, feature_fraction=0.8,
   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0), num_boost_round=400 (fixed, no early stopping).
   OOF logits (log(p/(1-p))) -> results/oof_feat.csv (uid,y,fold,family,logit,prob). Save the 5 boosters as text:
   runtime/submission_src/models/feat/lgb_fold{k}.txt (booster.save_model). Save the feature name order to models/feat/features.json.
2) Logistic regression: StandardScaler + LogisticRegression(C=0.3, max_iter=2000) per fold; OOF -> results/oof_lr.csv.
   Save per-fold mean/scale/coef/intercept as JSON in models/feat/lr.json (NO pickles).
Report for each: OOF log loss (raw), AUC, per-fold and per-family log loss -> results/feat_summary.json.

## Step C — coronal 2.5D CNN: scripts/train_2d_cor.py  (start only after Step B is reported; cap 40 min; skip if `date -u` > 21:15)
Same recipe as lane 1 but a different view and backbone — copy the recipe exactly from plans/LANE1_2D.md Step 2-3 with these substitutions:
input = channel 1 coronal slabs = mean over y-ranges [23:29), [29:35), [35:41) -> (3, 48(Z), 64(X)); scale x=slab/4-0.5; zero-pad Z to 64
(pad 8 top/bottom) -> (3,64,64); upsample to 128 in forward; backbone torchvision resnet34 IMAGENET1K_V1; L/R flip = flip last dim;
y-jitter in {-2..2} instead of z-jitter; seeds {0} only (5 models); 24 epochs. OOF -> results/oof_2dcor.csv; weights fp16 in
models/2dcor/f{k}_s0.pt; Platt -> models/2dcor/calib.json; summary -> results/2dcor_summary.json. Put the input function `make_input_cor`
and `NetCor` class in runtime/submission_src/extra_models.py.

## Step D — inference module: runtime/submission_src/extra_models.py must expose
`predict_extra(crops: list[np.ndarray|None], sbrs: list[float], device) -> dict[str, np.ndarray]` returning raw logits arrays
(np.nan where crop is None) for keys "feat_lgb", "feat_lr", and "cnn_cor" (only if Step C ran). Loads models from
Path(__file__).parent/"models". Uses only numpy/scipy/lightgbm/torch/torchvision. Test it on 20 crops from the cache and confirm the
logits match the OOF-time predictions for the same fold model (print max abs diff).

## Step E — blend: scripts/blend.py
Reads every results/oof_*.csv present (oof_2d.csv may appear later; run the script anyway on what exists and print).
For each model: Platt (a,b) on OOF logit. Blend = sigmoid(sum_i w_i*(a_i*logit_i+b_i)) with weights on a simplex grid (step 0.05),
choose weights minimising OOF log loss (clip [0.02,0.98]); ALSO report the equal-weight blend and every single model.
Print a table and write results/blend.json {weights, platt params, oof_logloss, auc, per-family}.

## Deliverable
status=complete with every printed metric, per-step timings, list of created files. status=blocked with discrepancies otherwise.
