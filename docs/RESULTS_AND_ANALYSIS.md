# Results and Analysis

## 1. Summary

The project was developed in three iterations. The best Kaggle
leaderboard score recorded by the project owner is **0.12427**, from the
V2 weighted-blend submission. Kaggle's competition metric is RMSE on
log-transformed sale prices; lower is better.

## 2. Recorded Kaggle submissions

  Version   Submission/model                 Kaggle score
  --------- ------------------------------ --------------
  V1        Initial submission                    0.13026
  V2        Best single-model submission          0.12600
  V2        Weighted blend                    **0.12427**
  V3        Weighted blend                        0.12444
  V3        Stacking                              0.12610

These are leaderboard scores supplied from the project's recorded Kaggle
submission history. They are not cross-validation scores.

## 3. Interpretation

-   V2 weighted blending is the strongest recorded submission at
    0.12427.
-   V3 weighted blending scored 0.12444, which is close but did not beat
    V2.
-   V3 stacking scored 0.12610, also better than the initial V1 result
    but not better than the V2 blend.
-   The results illustrate that adding model complexity or another
    ensemble technique does not automatically improve leaderboard
    performance.

## 4. Validation results are a different measure

Local cross-validation and OOF scores estimate performance using
held-out folds of the training data. Kaggle leaderboard scores are
calculated on the competition's hidden test labels for the submitted
test predictions. Do not combine these into one ranking table or
describe one as the other.

The project report records these V1 cross-validation values:

  V1 model              CV RMSE (log)
  ------------------- ---------------
  Gradient Boosting          0.129979
  Extra Trees                0.137422
  Random Forest              0.139437
  Lasso                      0.141692
  ElasticNet                 0.141872
  Ridge                      0.145126

The project report records these V3 base-model OOF values:

  V3 base model         OOF RMSE (log)
  ------------------- ----------------
  CatBoost                    0.125920
  XGBoost                     0.127790
  Gradient Boosting           0.134407

These local validation numbers should be interpreted in the context of
their respective version's split strategy and implementation. They are
not directly interchangeable with leaderboard scores.

## 5. V3 feature-importance snapshot

The project report records the following V3 model-derived importance
values:

  Feature                  Importance
  ---------------------- ------------
  Quality × TotalSF            28.876
  Quality × LivingArea         14.786
  TotalSF                       6.989
  OverallQual                   4.818
  TotalBsmtSF                   3.013
  TotalBathrooms                2.775

These are model-specific feature-importance scores, not causal effects
or percentages of price explained. See
`visualizations/feature_importance_v3.png`.

## 6. Charts

-   `visualizations/kaggle_submission_scores.png` compares the recorded
    Kaggle submissions. Lower is better.
-   `visualizations/feature_importance_v3.png` displays the V3
    importance snapshot recorded in the project report.

## 7. Recommended next experiments

1.  Preserve a fixed validation strategy and compare every change
    against it.
2.  Use nested cross-validation or a separate holdout set when tuning
    blend weights or a stacking meta-model.
3.  Tune boosting models with a controlled search budget and record
    parameters.
4.  Test log-target and feature-engineering changes independently before
    combining them.
5.  Track every experiment with a version, seed, validation score,
    Kaggle score, and submission filename.
6.  Avoid repeated leaderboard-driven tuning that overfits the public
    leaderboard.

## 8. Reproducibility note

The Kaggle scores above are the scores shown in the project's submission
history. Keep the original submission files and Kaggle submission
records as supporting evidence when presenting the results.
