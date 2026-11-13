# Project Report: House Price Prediction --- Advanced Regression & Ensemble Learning

**Competition:** [House Prices - Advanced Regression
Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)\
**Dataset:** Ames, Iowa residential housing data\
**Primary task:** Predict the `SalePrice` of each home\
**Best submitted Kaggle score in this project:** **0.12427 RMSE (log
scale)** --- V2 weighted blend

------------------------------------------------------------------------

## 1. Abstract

This project develops an end-to-end regression workflow for predicting
residential house sale prices using the Ames Housing dataset. The work
progressed through three iterations: a baseline regression pipeline
(V1), an advanced boosting and blending pipeline (V2), and a richer
feature-engineering and ensemble pipeline (V3).

The workflow includes data loading, feature construction, preprocessing
of numeric and categorical variables, cross-validation, model
comparison, optional hyperparameter optimization with Optuna, outlier
review, feature-importance analysis, weighted blending, stacking, and
Kaggle submission generation.

Across the submitted predictions, the best observed public leaderboard
result was **0.12427**, achieved by the V2 weighted-blend submission.
The V3 weighted blend scored 0.12444, while the V3 stacking submission
scored 0.12610. These leaderboard results are reported separately from
cross-validation and out-of-fold (OOF) validation results because they
are different evaluation procedures.

## 2. Problem Statement

House prices depend on many interacting characteristics, including
location-related attributes, living area, construction quality, age,
garage and basement features, and other property details. The objective
of this project is to learn relationships between these explanatory
variables and the final sale price.

Given the features for each property in the competition's test set, the
model must predict `SalePrice` and produce a CSV containing:

``` csv
Id,SalePrice
1461,169000.1
1462,187724.1233
```

The project uses the competition's supplied training and test files. The
training file contains the target column `SalePrice`; the test file does
not.

## 3. Dataset Overview

The competition is based on the Ames, Iowa housing dataset compiled by
Dean De Cock for data-science education. The competition describes 79
explanatory variables covering almost every aspect of residential homes.

-   **Training data:** `train.csv`
-   **Test data:** `test.csv`
-   **Target variable:** `SalePrice`
-   **Identifier:** `Id`
-   **Prediction output:** one predicted sale price for every test-set
    `Id`

The original data files are not included in this public repository.
Users should download them from the Kaggle competition Data page after
accepting the competition rules, then place `train.csv` and `test.csv`
in the local `data/` directory. Follow Kaggle's dataset and competition
terms when handling or redistributing the files.

## 4. Evaluation Metric

Kaggle evaluates submissions using Root Mean Squared Error (RMSE)
between the logarithms of the predicted and observed sale prices.

The training scripts therefore transform the target using:

``` python
y = np.log1p(train["SalePrice"])
```

The reported validation metric is RMSE in log-price space. A lower score
is better. Logarithmic evaluation reduces the dominance of high-priced
properties and makes relative prediction errors more comparable across
price ranges.

For final prediction files, the scripts transform log predictions back
to the original price scale using `expm1` and enforce a minimum positive
prediction.

## 5. Tools and Technologies

-   **Language:** Python
-   **Data handling:** pandas, NumPy
-   **Machine learning:** scikit-learn
-   **Boosting libraries:** XGBoost, LightGBM, CatBoost
-   **Hyperparameter optimization:** Optuna (optional V2 tuning
    workflow)
-   **Notebook environment:** Jupyter Notebook / VS Code
-   **Version control and publication:** Git and GitHub
-   **Competition platform:** Kaggle

## 6. Project Workflow

1.  Download the competition training and test datasets.
2.  Load the CSV files and separate identifiers, predictors, and target.
3.  Apply feature engineering consistently to training and test data.
4.  Identify numeric and categorical columns.
5.  Impute missing numeric values with the median and categorical values
    with the most frequent value.
6.  Scale numeric features where the pipeline uses scaling.
7.  One-hot encode categorical features while ignoring previously unseen
    categories.
8.  Evaluate models using five-fold cross-validation or out-of-fold
    predictions, depending on the project version.
9.  Compare models and generate test-set predictions.
10. Create submission files with the required `Id` and `SalePrice`
    columns.
11. Submit prediction files to Kaggle and record leaderboard scores.

The scikit-learn preprocessing steps are placed in pipelines/column
transformers so that preprocessing is fitted within each validation fold
rather than being fitted once on the full training data.

## 7. Feature Engineering

The project creates additional features intended to represent property
characteristics more directly.

### V1 and V2 features

-   **TotalSF:** sum of basement, first-floor, and second-floor square
    footage.
-   **TotalBathrooms:** full bathrooms plus half-weighted half
    bathrooms, including basement bathrooms.
-   **HouseAge:** sale year minus construction year.
-   **RemodAge:** sale year minus remodeling year.
-   **TotalPorchSF:** combined area of selected porch types.
-   **HasGarage:** indicator for a positive garage area.
-   **HasBasement:** indicator for a positive basement area.
-   **HasPool:** indicator for a positive pool area.
-   **OverallQuality_x_LivingArea (V2):** interaction between overall
    quality and above-ground living area.
-   **TotalRooms (V2):** total rooms above ground plus bedrooms above
    ground.

### Additional V3 features

V3 adds or refines engineered variables, including:

-   **TotalSF**
-   **TotalPorchSF**, including wood-deck area
-   **TotalBathrooms**
-   **TotalRoomsPlusBedrooms**
-   **HouseAgeAtSale**
-   **RemodelAgeAtSale**
-   **GarageAgeAtSale**
-   **Quality_x_LivingArea**
-   **Quality_x_TotalSF**
-   Property-presence indicators such as `HasBasement`

Feature engineering is applied to both training and test records so that
the model receives a consistent feature set.

## 8. Data Preprocessing

The training scripts use separate preprocessing paths for numeric and
categorical features.

### Numeric features

-   Median imputation
-   Missing-value indicators in the V1/V2 preprocessing pipeline
-   Standard scaling in the V1/V2 pipeline

### Categorical features

-   Most-frequent-value imputation
-   One-hot encoding
-   `handle_unknown="ignore"` so categories appearing only in validation
    or test data do not cause encoding errors

The V3 pipeline also separates numeric and categorical processing. The
exact transformations can vary by version; consult the corresponding
source script for implementation details.

## 9. Models Implemented

### V1: Baseline regression and tree models

-   Ridge Regression
-   Lasso Regression
-   ElasticNet
-   Random Forest Regressor
-   Extra Trees Regressor
-   Gradient Boosting Regressor

The V1 script compares these models using five-fold shuffled K-Fold
cross-validation with `random_state=42`. It creates a simple
equal-weight blend of the three best cross-validation models.

### V2: Advanced boosting

V2 preserves the baseline workflow and adds optional advanced boosting
models:

-   XGBoost
-   LightGBM
-   CatBoost

It also supports optional Optuna tuning for the Gradient Boosting model.
The tuning command is:

``` bash
python src/train_v2.py --tune --trials 20
```

The implemented Optuna search explores estimator count, learning rate,
maximum depth, minimum samples per leaf, and subsampling ratio. This
tuning workflow targets Gradient Boosting; it does not automatically
tune every boosting library.

V2 generates both a blended prediction file and a best-single-model
prediction file so that their Kaggle scores can be compared.

### V3: Feature engineering and ensemble methods

V3 uses:

-   CatBoost
-   XGBoost
-   Gradient Boosting
-   Non-negative weighted blending based on out-of-fold predictions
-   Ridge-based stacking
-   Tree-based feature-importance extraction
-   Outlier review for manual inspection

The outlier-review step flags unusual observations for review; it does
**not** automatically delete those rows from the training data.

## 10. Validation and Optimization

### V1/V2 validation

The baseline scripts use five-fold shuffled K-Fold cross-validation with
a fixed random seed. Model performance is measured as RMSE on the
log-transformed target.

### V3 validation

V3 generates out-of-fold predictions across five folds for its base
models. It uses those predictions to compare the base learners and fit
ensemble weights and the stacking meta-model.

**Validation limitation:** V3's blend weights and stacking meta-model
are learned from the same OOF prediction matrix used to report the
ensemble's development performance. The reported ensemble OOF score is
therefore not a fully independent, nested estimate of generalization. A
nested or separate holdout evaluation would be needed for a stricter
unbiased estimate.

V2 cross-validation results and V3 OOF results should not be treated as
perfectly interchangeable because the model settings and
prediction-generation procedures differ.

## 11. Experimental Results

### 11.1 Baseline V1 cross-validation

The saved V1 model-comparison output reports the following five-fold CV
RMSE values in log-price space:

  Model                 CV RMSE (log)
  ------------------- ---------------
  Gradient Boosting          0.129979
  Extra Trees                0.137422
  Random Forest              0.139437
  Lasso                      0.141692
  ElasticNet                 0.141872
  Ridge                      0.145126

### 11.2 V3 base-model OOF results

The saved V3 model-comparison output reports:

  Model                 OOF RMSE (log)
  ------------------- ----------------
  CatBoost                    0.125920
  XGBoost                     0.127790
  Gradient Boosting           0.134407

The saved V3 weighted-blend configuration assigns approximately 67.03%
to CatBoost, 32.97% to XGBoost, and effectively 0% to Gradient Boosting.
These weights are specific to the saved V3 run.

### 11.3 Kaggle leaderboard submissions

The following are the actual Kaggle scores recorded for the submitted
files in this project:

  Version   Submission            Kaggle score
  --------- ------------------- --------------
  V1        `submission.csv`           0.13026
  V2        Best single model          0.12600
  V2        Weighted blend         **0.12427**
  V3        Weighted blend             0.12444
  V3        Stacking                   0.12610

**Best observed submission:** V2 weighted blend --- **0.12427**.

The leaderboard score is the competition's evaluation result for the
submitted predictions. It is distinct from the local cross-validation or
OOF score. The V2 weighted blend is retained as the project's best
submission based on the recorded leaderboard results.

## 12. V3 Feature-Importance Analysis

The saved V3 feature-importance file ranks engineered quality-and-area
interactions among the most influential model features.

  Feature                  Importance
  ---------------------- ------------
  Quality × TotalSF            28.876
  Quality × LivingArea         14.786
  TotalSF                       6.989
  OverallQual                   4.818
  TotalBsmtSF                   3.013
  TotalBathrooms                2.775

These values are model-derived feature-importance scores, not causal
effects and not percentage contributions to sale price. They indicate
which features the fitted tree model relied on more heavily in that run.

## 13. Outlier Review

The V3 output flags two records for manual review based on unusually
large above-ground living areas relative to their sale prices. The
script writes these rows to `outlier_review_v3.csv`.

The flagged records are reviewed rather than automatically removed. This
avoids silently discarding potentially valid properties based on a
simple rule.

## 14. Reproducibility and Running the Project

### Requirements

Use Python 3.10 or newer as recommended by the project README. Create
and activate a virtual environment:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

Install baseline dependencies:

``` powershell
python -m pip install -r requirements.txt
```

Install advanced-model dependencies:

``` powershell
python -m pip install -r requirements_v2.txt
```

### Dataset placement

After downloading the competition data, the local project should
contain:

``` text
house-price-prediction-ml/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
├── outputs/
├── results/
├── src/
│   ├── train.py
│   ├── train_v2.py
│   └── train_v3.py
├── README.md
├── requirements.txt
└── requirements_v2.txt
```

The repository's `.gitignore` excludes CSV files placed directly in
`data/`. Keep the original competition data local and do not publish it
unless permitted by the applicable Kaggle terms.

### Run each version

From the repository root:

``` powershell
python src/train.py
python src/train_v2.py
python src/train_v3.py
```

For optional V2 Optuna tuning:

``` powershell
python src/train_v2.py --tune --trials 20
```

The scripts expect `data/train.csv` and `data/test.csv` and write
generated files under `outputs/`. The consolidated GitHub repository
currently stores selected experiment results under `results/`; generated
training outputs can remain local unless intentionally added to version
control.

## 15. Limitations

-   The dataset represents homes from Ames, Iowa; model performance may
    not transfer to other cities, regions, or time periods.
-   The available dataset is limited in size compared with many
    production real-estate datasets.
-   Public leaderboard performance is based on the competition's
    evaluation split and is not proof of performance on new real-world
    markets.
-   V3's ensemble development score is not based on a nested outer
    validation procedure.
-   Feature-importance scores describe model usage, not causal
    relationships.
-   The Optuna workflow in V2 tunes Gradient Boosting rather than all
    advanced models.
-   The project is a competition prediction workflow, not a deployed
    real-estate valuation service.

## 16. Conclusion

This project demonstrates a practical progression from baseline
regression models to advanced boosting and ensemble learning for a
structured tabular regression problem. Feature engineering,
preprocessing pipelines, cross-validation, model comparison, optional
hyperparameter tuning, weighted blending, and stacking were implemented
and evaluated.

The best recorded Kaggle result is **0.12427 RMSE**, achieved by the V2
weighted-blend submission. V3 explored richer features and additional
ensemble strategies, but its weighted blend (0.12444) and stacking
submission (0.12610) did not surpass that benchmark. The experiment
highlights an important machine-learning lesson: added complexity does
not guarantee better generalization, and models should be selected using
reliable validation and external evaluation.

## 17. References

1.  Kaggle, [House Prices - Advanced Regression
    Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
2.  De Cock, D. (2011), *Ames, Iowa: Alternative to the Boston Housing
    Data as an End of Semester Regression Project*, Journal of
    Statistics Education.
3.  scikit-learn documentation: https://scikit-learn.org/
4.  XGBoost documentation: https://xgboost.readthedocs.io/
5.  LightGBM documentation: https://lightgbm.readthedocs.io/
6.  CatBoost documentation: https://catboost.ai/docs/
7.  Optuna documentation: https://optuna.org/
