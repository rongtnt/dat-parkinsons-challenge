# DaT Parkinson's Challenge — submissions (2026-09-16, UTC)
| # | id | submitted | models (stack) | OOF cross-fit | public LB | notes |
|---|---|---|---|---|---|---|
| smoke | 322916 | 20:24 | feat_lgb+feat_lr | 0.3245 | 0.2886 (smoke set) | container reproduces local run exactly |
| 1 | 322918 | 20:35 | 2dt+cor (trunc-r18 int8, 2 seeds) + feat_lgb+feat_lr | 0.2585 | 0.2696 (#85 at 21:20) | 41 min queue + 3 min run; gap +0.011 |
| 2 | 322932 | 21:21 | 2dtc(3%-drop)+corc(3%-drop)+feat | 0.2502 | 0.2723 (completed 22:20) | 50 min queue; WORSE than Sub 1 -> the drop-list OOF gain was leakage, did not transfer |
| 3 | 322942 | 22:21 | 2dtc5(5%-drop)+2dt6n(6ch)+corc+feat | 0.2424 | 0.2772 (completed 23:12; 46 min queue) | submitted 60 s after Sub 2 completed; presign is refused until the previous job is Completed (Scoring still counts as active) |

Lesson: the forward holdout (public LB) beat in-sample OOF: Sub 2's cleaned models looked 0.008 better OOF but scored 0.003 worse on LB. Unleaked levers (6-channel input, more seeds) are the ones to trust.

FINAL: best public 0.2696 (Sub 1, rank ~87 of ~1,006 joined). Drop-fraction vs public LB is monotone in the wrong direction: 0% 0.2696, 3% 0.2723, 5% 0.2772 while OOF went 0.2585 -> 0.2502 -> 0.2424. The un-nested drop list leaked validation labels; OOF gains were bookkeeping. The only clean positive lever (6-channel input without mixup, -0.022 OOF) never got its own forward test.
