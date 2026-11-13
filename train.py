from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple, interpretable features available in both train and test."""
    df = df.copy()
    # Fill component areas with zero only for sums; original columns are still imputed later.
    area_cols = ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea"]
    for col in area_cols:
        if col not in df:
            df[col] = np.nan
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0).sum(axis=1)
    df["TotalBathrooms"] = (
        df.get("FullBath", 0).fillna(0)
        + 0.5 * df.get("HalfBath", 0).fillna(0)
        + df.get("BsmtFullBath", 0).fillna(0)
        + 0.5 * df.get("BsmtHalfBath", 0).fillna(0)
    )
    df["HouseAge"] = df.get("YrSold", 0) - df.get("YearBuilt", 0)
    df["RemodAge"] = df.get("YrSold", 0) - df.get("YearRemodAdd", 0)
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0).fillna(0)
        + df.get("EnclosedPorch", 0).fillna(0)
        + df.get("3SsnPorch", 0).fillna(0)
        + df.get("ScreenPorch", 0).fillna(0)
    )
    df["HasGarage"] = (df.get("GarageArea", 0).fillna(0) > 0).astype(int)
    df["HasBasement"] = (df.get("TotalBsmtSF", 0).fillna(0) > 0).astype(int)
    df["HasPool"] = (df.get("PoolArea", 0).fillna(0) > 0).astype(int)
    return df

def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

def main():
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing data files. Download train.csv and test.csv from the Kaggle Data tab "
            f"and place them in: {DATA_DIR}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "SalePrice" not in train:
        raise ValueError("train.csv must contain the SalePrice target column.")

    y = np.log1p(train["SalePrice"])
    train_ids = train["Id"].copy()
    test_ids = test["Id"].copy()

    X = add_features(train.drop(columns=["SalePrice", "Id"]))
    X_test = add_features(test.drop(columns=["Id"]))
    # Align feature columns to protect against accidental train/test differences.
    X_test = X_test.reindex(columns=X.columns)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = "neg_root_mean_squared_error"

    models = {
        "Ridge": Ridge(alpha=15.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=20000),
        "ElasticNet": ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=20000),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_features=0.7, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=700, max_features=0.8, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=1800, learning_rate=0.03, max_depth=2,
            max_features=0.8, loss="huber", random_state=42
        ),
    }

    results, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])
        scores = -cross_val_score(pipe, X, y, cv=cv, scoring=rmse_scorer, n_jobs=1)
        results.append({"Model": name, "CV_RMSE_log": scores.mean(), "CV_Std": scores.std()})
        fitted[name] = pipe
        print(f"{name:18s} CV RMSE(log): {scores.mean():.5f} +/- {scores.std():.5f}")

    results_df = pd.DataFrame(results).sort_values("CV_RMSE_log")
    results_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:\n", results_df.to_string(index=False))

    # Fit the best three models on all training rows and blend their log predictions.
    top_names = results_df["Model"].head(3).tolist()
    blend_predictions = []
    for name in top_names:
        fitted[name].fit(X, y)
        blend_predictions.append(fitted[name].predict(X_test))

    # Equal-weight blend is a transparent starting point, not a guaranteed optimum.
    pred_log = np.mean(np.column_stack(blend_predictions), axis=1)
    predictions = np.expm1(pred_log)
    predictions = np.maximum(predictions, 1.0)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": predictions})
    submission.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'submission.csv'} using: {', '.join(top_names)}")
    print(submission.head())

if __name__ == "__main__":
    main()
