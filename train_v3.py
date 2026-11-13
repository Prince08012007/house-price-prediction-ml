from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create consistent domain features for both training and test rows."""
    df = frame.copy()

    def s(name):
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index)

    # Area and room features
    df["TotalSF"] = s("TotalBsmtSF").fillna(0) + s("1stFlrSF").fillna(0) + s("2ndFlrSF").fillna(0)
    df["TotalPorchSF"] = sum(s(c).fillna(0) for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"])
    df["TotalBathrooms"] = (
        s("FullBath").fillna(0) + 0.5 * s("HalfBath").fillna(0)
        + s("BsmtFullBath").fillna(0) + 0.5 * s("BsmtHalfBath").fillna(0)
    )
    df["TotalRoomsPlusBedrooms"] = s("TotRmsAbvGrd").fillna(0) + s("BedroomAbvGr").fillna(0)

    # Age and renovation features; clip negative values caused by unusual dates.
    df["HouseAgeAtSale"] = (s("YrSold") - s("YearBuilt")).clip(lower=0)
    df["YearsSinceRemodel"] = (s("YrSold") - s("YearRemodAdd")).clip(lower=0)
    df["GarageAgeAtSale"] = (s("YrSold") - s("GarageYrBlt")).clip(lower=0)

    # Quality interactions and presence flags
    df["Quality_x_LivingArea"] = s("OverallQual").fillna(0) * s("GrLivArea").fillna(0)
    df["Quality_x_TotalSF"] = s("OverallQual").fillna(0) * df["TotalSF"]
    df["HasGarage"] = (s("GarageArea").fillna(0) > 0).astype(int)
    df["HasBasement"] = (s("TotalBsmtSF").fillna(0) > 0).astype(int)
    df["HasPool"] = (s("PoolArea").fillna(0) > 0).astype(int)
    df["HasFireplace"] = (s("Fireplaces").fillna(0) > 0).astype(int)
    df["Has2ndFloor"] = (s("2ndFlrSF").fillna(0) > 0).astype(int)

    # Ratios can contain missing values; pipeline imputes them later.
    df["LivingAreaPerRoom"] = s("GrLivArea") / s("TotRmsAbvGrd").replace(0, np.nan)
    df["GarageAreaPerCar"] = s("GarageArea") / s("GarageCars").replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


def make_preprocessor(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def build_models():
    # V3 focuses compute on the strongest V2 families rather than rerunning every baseline.
    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=2,
            max_features=0.8, loss="huber", random_state=RANDOM_STATE,
        ),
    }
    missing = []
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3,
            min_child_weight=2, subsample=0.75, colsample_bytree=0.8,
            reg_alpha=0.02, reg_lambda=1.5, objective="reg:squarederror",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:
        missing.append("xgboost")
    try:
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(
            iterations=600, learning_rate=0.04, depth=4,
            loss_function="RMSE", l2_leaf_reg=4,
            verbose=False, random_seed=RANDOM_STATE, allow_writing_files=False,
        )
    except ImportError:
        missing.append("catboost")
    return models, missing


def optimize_blend_weights(oof_matrix, y):
    """Find non-negative blend weights summing to one using training OOF predictions."""
    n_models = oof_matrix.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights):
        return mean_squared_error(y, oof_matrix @ weights) ** 0.5

    result = minimize(
        objective, initial, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        print("Blend optimizer warning:", result.message, "Using equal weights.")
        return initial
    return result.x


def create_outlier_review(train):
    # Flag only for human review. Do not remove rows automatically.
    flagged = train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].copy()
    columns = [c for c in ["Id", "GrLivArea", "OverallQual", "SalePrice"] if c in flagged]
    flagged[columns].to_csv(OUT_DIR / "outlier_review_v3.csv", index=False)
    return len(flagged)


def save_model_feature_importance(preprocessor, fitted_model):
    """Save native tree feature importance using transformed feature names."""
    try:
        names = preprocessor.get_feature_names_out()
        values = fitted_model.feature_importances_
    except Exception as exc:
        print("Could not extract native feature importance:", exc)
        return pd.DataFrame()
    importance = pd.DataFrame({
        "Feature": names,
        "Importance": values,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(OUT_DIR / "feature_importance_v3.csv", index=False)
    return importance


def main():
    train_path, test_path = DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Place train.csv and test.csv inside: {DATA_DIR}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = np.log1p(train["SalePrice"].astype(float)).to_numpy()
    test_ids = test["Id"].copy()
    X = add_features(train.drop(columns=["Id", "SalePrice"]))
    X_test = add_features(test.drop(columns=["Id"])).reindex(columns=X.columns)

    flagged_count = create_outlier_review(train)
    print(f"Rows flagged for manual outlier review (not removed): {flagged_count}")
    print(f"Training rows: {len(X)} | Features after engineering: {X.shape[1]}")

    models, missing = build_models()
    if missing:
        print("Optional models unavailable:", ", ".join(missing))
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof_predictions = {}
    fitted_models = {}
    fitted_preprocessors = {}
    result_rows = []
    splits = list(cv.split(X, y))
    for name, estimator in models.items():
        print(f"\nGenerating out-of-fold predictions for {name}...")
        oof = np.zeros(len(X), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            fold_preprocessor = make_preprocessor(X)
            X_fold_train = fold_preprocessor.fit_transform(X.iloc[train_idx])
            X_fold_valid = fold_preprocessor.transform(X.iloc[valid_idx])
            fold_model = clone(estimator)
            fold_model.fit(X_fold_train, y[train_idx])
            oof[valid_idx] = fold_model.predict(X_fold_valid)
            print(f"  fold {fold}/{len(splits)} complete")
        rmse = mean_squared_error(y, oof) ** 0.5
        oof_predictions[name] = oof
        result_rows.append({"Model": name, "OOF_RMSE_log": rmse})

        full_preprocessor = make_preprocessor(X)
        X_full = full_preprocessor.fit_transform(X)
        X_test_full = full_preprocessor.transform(X_test)
        full_model = clone(estimator)
        full_model.fit(X_full, y)
        fitted_preprocessors[name] = full_preprocessor
        fitted_models[name] = full_model
        print(f"{name:20s} OOF RMSE(log): {rmse:.5f}")

    comparison = pd.DataFrame(result_rows).sort_values("OOF_RMSE_log")
    comparison.to_csv(OUT_DIR / "model_comparison_v3.csv", index=False)
    print("\nV3 model comparison:\n", comparison.to_string(index=False))

    # Use the top three models by OOF RMSE for a data-driven non-negative blend.
    selected = comparison["Model"].head(min(3, len(comparison))).tolist()
    oof_matrix = np.column_stack([oof_predictions[name] for name in selected])
    weights = optimize_blend_weights(oof_matrix, y)
    weight_table = pd.DataFrame({"Model": selected, "Weight": weights})
    weight_table.to_csv(OUT_DIR / "blend_weights_v3.csv", index=False)
    blend_oof_rmse = mean_squared_error(y, oof_matrix @ weights) ** 0.5

    test_matrix = np.column_stack([
        fitted_models[name].predict(fitted_preprocessors[name].transform(X_test))
        for name in selected
    ])
    blend_test_log = test_matrix @ weights
    blend_test_price = np.maximum(np.expm1(blend_test_log), 1.0)
    pd.DataFrame({"Id": test_ids, "SalePrice": blend_test_price}).to_csv(
        OUT_DIR / "submission_v3_weighted_blend.csv", index=False
    )
    print(f"\nSelected blend: {dict(zip(selected, np.round(weights, 4)))}")
    print(f"Weighted blend OOF RMSE(log): {blend_oof_rmse:.5f}")

    # Stacking: train a Ridge meta-model on base-model OOF predictions.
    # Base models are already fitted on all training rows above for test inference.
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    meta_model.fit(oof_matrix, y)
    stacked_test_log = meta_model.predict(test_matrix)
    stacked_test_price = np.maximum(np.expm1(stacked_test_log), 1.0)
    pd.DataFrame({"Id": test_ids, "SalePrice": stacked_test_price}).to_csv(
        OUT_DIR / "submission_v3_stacking.csv", index=False
    )
    pd.DataFrame({"Model": selected, "Meta_Coefficient": meta_model.coef_}).to_csv(
        OUT_DIR / "stacking_meta_coefficients_v3.csv", index=False
    )
    print(f"Stacking meta-model alpha: {meta_model.alpha_}")

    # Explainability: permutation importance on a held-out validation split.
    best_name = comparison.iloc[0]["Model"]
    importance = save_model_feature_importance(
        fitted_preprocessors[best_name], fitted_models[best_name]
    )
    print(f"\nTop 15 model-based features for {best_name}:")
    print(importance.head(15).to_string(index=False))

    print("\nV3 files saved in:", OUT_DIR)
    print("- model_comparison_v3.csv")
    print("- blend_weights_v3.csv")
    print("- feature_importance_v3.csv")
    print("- outlier_review_v3.csv")
    print("- submission_v3_weighted_blend.csv")
    print("- submission_v3_stacking.csv")
    print("\nCompare both V3 Kaggle submissions with your V2 best score of 0.12427.")


if __name__ == "__main__":
    main()
