# Methodology --- House Price Prediction

## 1. Objective

Predict the `SalePrice` target for homes in the Ames, Iowa dataset used
by Kaggle's **House Prices - Advanced Regression Techniques**
competition. The competition evaluates the root mean squared error
between the logarithms of predicted and observed sale prices.

## 2. End-to-end workflow

1.  Download `train.csv` and `test.csv` from the competition's Data
    page.
2.  Keep `Id` as the prediction-row identifier and separate `SalePrice`
    as the training target.
3.  Apply feature engineering consistently to training and test data.
4.  Split features into numeric and categorical groups.
5.  Impute missing values and encode categorical values using the
    preprocessing pipeline in the selected training script.
6.  Train and validate candidate regression models.
7.  Compare validation results and generate test-set predictions.
8.  Where implemented, combine model predictions through weighted
    blending or stacking.
9.  Convert predictions from log-price space back to sale-price space.
10. Save a Kaggle submission with exactly `Id` and `SalePrice` columns.

## 3. Target transformation and metric

The competition metric is:

\[ RMSE\_{log} =
`\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\log(\hat y_i)-\log(y_i))^2}`{=tex} \]

The project scripts use a log-transformed target (`log1p(SalePrice)`) in
their regression workflow. Predictions are converted back to the
original currency scale using the inverse transformation (`expm1`)
before writing the submission file. A lower competition score is better.

## 4. Feature engineering

The project README and report describe engineered features intended to
summarize related property attributes. These include:

-   **TotalSF:** combined basement, first-floor, and second-floor area.
-   **TotalBathrooms:** full bathrooms plus half-weighted half
    bathrooms, including basement bathrooms.
-   **HouseAge / HouseAgeAtSale:** sale year minus construction year.
-   **RemodAge / RemodelAgeAtSale:** sale year minus remodel year.
-   **TotalPorchSF:** combined porch/deck area, according to the
    version's implementation.
-   **HasGarage, HasBasement, HasPool:** binary property-presence
    indicators.
-   **Quality × Living Area** and **Quality × TotalSF:** interactions
    between quality and area.
-   **TotalRooms / TotalRoomsPlusBedrooms:** aggregate room-count
    features, depending on version.

Feature definitions can differ slightly between V1, V2, and V3. The
corresponding source file is the authority for the exact formula used in
a particular run.

## 5. Missing values and categorical data

The documented preprocessing approach separates numeric and categorical
columns:

-   Numeric variables: median imputation; some pipelines also add
    missingness indicators and apply standard scaling.
-   Categorical variables: most-frequent imputation and one-hot
    encoding.
-   Unknown categories: ignored by the encoder so validation/test
    categories not seen during fitting do not cause a transform error.

Preprocessing should be fitted within each cross-validation training
fold. This reduces leakage from validation folds into preprocessing
statistics.

## 6. Model development by version

### V1 --- Baseline models

The repository describes Ridge, Lasso, ElasticNet, Random Forest, Extra
Trees, and Gradient Boosting regressors. The V1 workflow compares
candidate models using shuffled five-fold cross-validation and creates a
baseline blended prediction from selected models.

### V2 --- Advanced boosting and tuning

V2 adds or supports XGBoost, LightGBM, and CatBoost. The project
documentation describes an optional Optuna search for Gradient Boosting
hyperparameters such as estimator count, learning rate, maximum depth,
minimum samples per leaf, and subsampling ratio. Optuna tuning is
optional; it should not be described as tuning every model.

### V3 --- Feature refinements and ensembles

V3 explores additional engineered features and tree-based learners, and
generates outputs for model comparison, feature importance, outlier
review, weighted blending, and stacking. The documented V3 base learners
include CatBoost, XGBoost, and Gradient Boosting.

## 7. Cross-validation, blending, and stacking

Cross-validation estimates performance on held-out folds of the training
data. Out-of-fold (OOF) predictions are predictions for training rows
made by models that did not train on those rows.

-   **Weighted blending:** combines base-model predictions using
    weights. The V3 saved blend weights are stored in
    `results/blend_weights_v3.csv`.
-   **Stacking:** uses base-model predictions as inputs to a
    second-level model. The project report identifies Ridge as the V3
    stacking meta-model.

The project report notes an important validation limitation: if blend
weights or the stacking meta-model are fitted and evaluated on the same
OOF prediction matrix, that ensemble score is not a fully independent
nested estimate. A nested or untouched holdout evaluation is preferable
for a stricter estimate.

## 8. Reproducibility

Run commands from the repository root after placing the Kaggle files in
`data/`:

``` powershell
python src/train.py
python src/train_v2.py
python src/train_v3.py
```

Optional V2 tuning, as documented in the project README/report:

``` powershell
python src/train_v2.py --tune --trials 20
```

Check the actual script's command-line options and requirements files if
your local version differs. Do not commit the competition's raw CSV data
unless the competition's terms explicitly permit redistribution.
