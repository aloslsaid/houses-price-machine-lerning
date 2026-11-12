from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "House Price Prediction Dataset.csv"
OUTPUT_DIR = ROOT / "artifacts"
TARGET = "Price"
SEED = 42


def make_pipeline(model, numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("model", model),
    ])


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {TARGET}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required.")
    X = df.drop(columns=[TARGET, "Id"], errors="ignore")
    y = df[TARGET]
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    if not numeric and not categorical:
        raise ValueError("No usable feature columns found.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED
    )
    candidates = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }
    results = []
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator, numeric, categorical)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        results.append((name, r2_score(y_val, pred), pipe))
    results.sort(key=lambda x: x[1], reverse=True)
    best_name = results[0][0]
    final_model = make_pipeline(candidates[best_name], numeric, categorical)
    final_model.fit(X_trainval, y_trainval)
    test_pred = final_model.predict(X_test)
    metrics = {
        "model": best_name,
        "validation_r2": float(results[0][1]),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "test_rows": int(len(y_test)),
        "features": list(X.columns),
        "target": TARGET,
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, OUTPUT_DIR / "house_price_model.joblib")
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"actual_price": y_test, "predicted_price": test_pred}).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )
    print(pd.DataFrame([{"model": n, "validation_r2": s} for n, s, _ in results]).to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
