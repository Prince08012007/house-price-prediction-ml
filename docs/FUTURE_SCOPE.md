# Future Scope

## 1. Stronger validation

-   Add a fixed holdout set or nested cross-validation for a less biased
    estimate of generalization.
-   Fit blending weights and stacking meta-models without evaluating
    them on the same OOF rows used for fitting.
-   Track random seeds and fold assignments for reproducibility.

## 2. More systematic model optimization

-   Run controlled hyperparameter searches for CatBoost, XGBoost,
    LightGBM, and regularized linear models.
-   Record best parameters, search ranges, runtime, and validation score
    for every experiment.
-   Compare tuned models against a stable baseline to verify that
    complexity provides a real benefit.

## 3. Better error analysis

-   Analyze residuals by sale-price range, house quality, neighborhood,
    and property type.
-   Identify systematic overprediction and underprediction.
-   Review influential observations carefully rather than deleting
    outliers automatically.

## 4. Explainability

-   Add permutation importance and SHAP analysis where appropriate.
-   Compare feature importance across model families.
-   Clearly distinguish predictive associations from causal
    explanations.

## 5. Reproducibility and engineering

-   Consolidate dependencies into a documented environment or separate
    baseline/advanced requirements with tested versions.
-   Add automated tests for feature engineering, preprocessing,
    prediction length, positive predictions, and submission schema.
-   Add a configuration file for paths, seeds, folds, and model
    parameters.
-   Add a lightweight CI workflow to run formatting and tests.
-   Save model metadata and experiment logs alongside results.

## 6. Productization

-   Build a small interface that accepts property attributes and returns
    an estimated price, with clear warnings that the model was trained
    on Ames, Iowa competition data and is not a professional appraisal.
-   Package preprocessing and the fitted model together to ensure
    inference uses the same transformations as training.
-   Monitor data drift and recalibrate only with suitable new labelled
    data.

## 7. Scope and limitations

The competition dataset represents residential sales in Ames, Iowa.
Applying the model to another city, country, or time period would
require suitable local data, retraining, and validation. Future work
should not present the current model as a universally accurate
real-estate valuation system.
