# 🏡 House Price Prediction – Advanced Regression & Ensemble Learning

An end-to-end Machine Learning project that predicts residential house prices using advanced regression algorithms, feature engineering, hyperparameter optimization, and ensemble learning.

This project is based on the **House Prices – Advanced Regression Techniques** competition hosted on Kaggle.

## 📌 Project Overview

The objective is to predict the final sale prices of residential houses in Ames, Iowa, using 79 explanatory variables describing different aspects of residential properties.

The project explores multiple regression algorithms and ensemble techniques to improve prediction performance.

**Best Kaggle Score: 0.12427 RMSE**

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA).
- Handle missing values and categorical variables.
- Apply feature engineering.
- Implement multiple regression algorithms.
- Compare model performance using cross-validation.
- Optimize hyperparameters using Optuna.
- Explore weighted blending and stacking.
- Generate predictions for Kaggle submission.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- Jupyter Notebook

## 🤖 Machine Learning Algorithms

### Traditional Regression Models
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor

### Advanced Boosting Algorithms
- XGBoost
- LightGBM
- CatBoost

### Ensemble Techniques
- Weighted Blending
- Stacking Regressor

## 📊 Kaggle Performance

The project was developed incrementally through three versions.

| Version | Model | Kaggle RMSE |
|---|---|---:|
| V1 | Gradient Boosting | 0.13026 |
| V2 | Weighted Blending | **0.12427** |
| V2 | CatBoost | 0.12600 |
| V3 | Weighted Blending | 0.12444 |
| V3 | Stacking | 0.12610 |

**Best Result: 0.12427 RMSE (V2 Weighted Blending)**

Lower RMSE indicates better prediction performance.

## 🔍 Feature Engineering

Several additional features were developed to capture meaningful relationships between house characteristics.

Examples include:

- Total House Area
- Total Bathrooms
- House Age
- Remodel Age
- Total Porch Area
- Garage Availability
- Basement Availability
- Pool Availability
- Quality × Living Area

## ⚙️ Hyperparameter Optimization

Optuna was incorporated to explore better Gradient Boosting hyperparameters.

The optimization process searches for suitable combinations of:

- Number of estimators
- Learning rate
- Maximum depth
- Minimum samples per leaf
- Subsample ratio

## 📁 Project Structure

```text
house-price-prediction-ml/
│
├── data/
│
├── notebooks/
│   └── house_prices_walkthrough.ipynb
│
├── src/
│   ├── train.py
│   ├── train_v2.py
│   └── train_v3.py
│
├── outputs/
│
├── results/
│   ├── best_submission_v2_blend.csv
│   ├── blend_weights_v3.csv
│   ├── feature_importance_v3.csv
│   ├── model_comparison_v2.csv
│   ├── model_comparison_v3.csv
│   └── outlier_review_v3.csv
│
├── docs/
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements_v2.txt
```

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/Prince08012007/house-price-prediction-ml.git
```

### 2. Navigate to the Project

```bash
cd house-price-prediction-ml
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

For the baseline project:

```bash
pip install -r requirements.txt
```

For advanced models:

```bash
pip install -r requirements_v2.txt
```

### 5. Download the Dataset

Download the following files from the Kaggle competition:

- train.csv
- test.csv

Place both files inside the `data/` directory.

Dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

### 6. Run the Models

Baseline:

```bash
python src/train.py
```

Advanced models:

```bash
python src/train_v2.py
```

Advanced feature engineering and ensemble learning:

```bash
python src/train_v3.py
```

## 📈 Evaluation Metric

The competition evaluates predictions using Root Mean Squared Error (RMSE) on logarithmic house prices.

A lower score indicates better prediction performance.

## 🏆 Key Learnings

- Practical implementation of regression algorithms.
- Handling missing values and categorical features.
- Importance of feature engineering.
- Understanding ensemble learning.
- Hyperparameter optimization using Optuna.
- Model validation using cross-validation.
- Comparing validation performance with Kaggle leaderboard results.

## 🔗 Kaggle Competition

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## 👨‍💻 Author

**Prince**

Machine Learning | Data Science | Python

---

⭐ If you find this project interesting, consider giving the repository a star!