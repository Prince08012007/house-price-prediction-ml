from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUT_DIR = ROOT / "data", ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def add_features(df):
    df = df.copy()
    def col(name):
        return df[name] if name in df else pd.Series(0, index=df.index)
    df["TotalSF"] = col("TotalBsmtSF").fillna(0) + col("1stFlrSF").fillna(0) + col("2ndFlrSF").fillna(0)
    df["TotalBathrooms"] = (col("FullBath").fillna(0) + .5*col("HalfBath").fillna(0)
                            + col("BsmtFullBath").fillna(0) + .5*col("BsmtHalfBath").fillna(0))
    df["HouseAge"] = col("YrSold") - col("YearBuilt")
    df["RemodAge"] = col("YrSold") - col("YearRemodAdd")
    df["TotalPorchSF"] = sum(col(c).fillna(0) for c in ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"])
    df["HasGarage"] = (col("GarageArea").fillna(0) > 0).astype(int)
    df["HasBasement"] = (col("TotalBsmtSF").fillna(0) > 0).astype(int)
    df["HasPool"] = (col("PoolArea").fillna(0) > 0).astype(int)
    df["OverallQuality_x_LivingArea"] = col("OverallQual").fillna(0) * col("GrLivArea").fillna(0)
    df["TotalRooms"] = col("TotRmsAbvGrd").fillna(0) + col("BedroomAbvGr").fillna(0)
    return df

def make_preprocessor(X):
    nums = X.select_dtypes(include="number").columns.tolist()
    cats = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cats)
    ])

def get_models():
    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=1800, learning_rate=0.03, max_depth=2,
            max_features=0.8, loss="huber", random_state=42
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=700, max_features=0.8, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=600, max_features=0.7, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        )
    }
    optional = []
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=1800, learning_rate=0.025, max_depth=3,
            min_child_weight=2, subsample=0.75, colsample_bytree=0.8,
            reg_alpha=0.01, reg_lambda=1.5, objective="reg:squarederror",
            random_state=42, n_jobs=-1
        )
    except ImportError:
        optional.append("xgboost")
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=1800, learning_rate=0.02, num_leaves=15,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbosity=-1, random_state=42, n_jobs=-1
        )
    except ImportError:
        optional.append("lightgbm")
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=1800, learning_rate=0.03, depth=4,
            loss_function="RMSE", l2_leaf_reg=4,
            verbose=False, random_seed=42, allow_writing_files=False
        )
    except ImportError:
        optional.append("catboost")
    return models, optional

def tune_gradient_boosting(X, y, preprocessor, cv, trials=20):
    try:
        import optuna
    except ImportError:
        print("Optuna not installed; skipping tuning. Install with: pip install optuna")
        return None
    def objective(trial):
        model = GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 700, 2200, step=300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 1, 3),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            subsample=trial.suggest_float("subsample", 0.65, 1.0),
            loss="huber", random_state=42
        )
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        return -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1).mean()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    pd.DataFrame([{"best_cv_rmse_log": study.best_value, **study.best_params}]).to_csv(OUT_DIR/"optuna_best_params.csv", index=False)
    print("Optuna best:", study.best_value, study.best_params)
    return study.best_params

def main():
    parser = argparse.ArgumentParser(description="House Prices V2: advanced boosting and blending")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning for Gradient Boosting (takes longer)")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()

    train_path, test_path = DATA_DIR/"train.csv", DATA_DIR/"test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Put train.csv and test.csv in {DATA_DIR}")
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    y = np.log1p(train["SalePrice"].astype(float))
    test_ids = test["Id"].copy()
    X = add_features(train.drop(columns=["Id", "SalePrice"]))
    X_test = add_features(test.drop(columns=["Id"])).reindex(columns=X.columns)

    preprocessor = make_preprocessor(X)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models, missing = get_models()
    if missing:
        print("Optional models skipped (not installed):", ", ".join(missing))
        print("Install them with: pip install xgboost lightgbm catboost")

    if args.tune:
        best = tune_gradient_boosting(X, y, preprocessor, cv, args.trials)
        if best:
            models["GradientBoosting_Optuna"] = GradientBoostingRegressor(
                **best, loss="huber", random_state=42
            )

    rows, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", estimator)])
        scores = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
        rows.append({"Model": name, "CV_RMSE_log": scores.mean(), "CV_Std": scores.std()})
        fitted[name] = pipe
        print(f"{name:24s} CV RMSE(log): {scores.mean():.5f} +/- {scores.std():.5f}")

    comparison = pd.DataFrame(rows).sort_values("CV_RMSE_log")
    comparison.to_csv(OUT_DIR/"model_comparison_v2.csv", index=False)
    print("\nV2 comparison:\n", comparison.to_string(index=False))

    # Blend the best 3 models by CV score. This is a baseline blend, not guaranteed
    # to beat the best single model; compare both before choosing a Kaggle submission.
    selected = comparison["Model"].head(min(3, len(comparison))).tolist()
    test_predictions = []
    for name in selected:
        fitted[name].fit(X, y)
        test_predictions.append(fitted[name].predict(X_test))
    pred_log_blend = np.mean(np.column_stack(test_predictions), axis=1)
    pred_blend = np.maximum(np.expm1(pred_log_blend), 1.0)

    # Also save the single best model's predictions for direct comparison.
    best_name = comparison.iloc[0]["Model"]
    best_pred_log = fitted[best_name].predict(X_test)
    pred_best = np.maximum(np.expm1(best_pred_log), 1.0)

    pd.DataFrame({"Id": test_ids, "SalePrice": pred_blend}).to_csv(OUT_DIR/"submission_v2_blend.csv", index=False)
    pd.DataFrame({"Id": test_ids, "SalePrice": pred_best}).to_csv(OUT_DIR/"submission_v2_best_single.csv", index=False)
    print(f"\nSaved V2 submissions. Best CV model: {best_name}; blend: {selected}")
    print("Compare both Kaggle scores; retain whichever performs better.")

if __name__ == "__main__":
    main()
