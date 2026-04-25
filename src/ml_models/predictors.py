"""
Antrenare și comparare a mai multor algoritmi predictivi.

Algoritmi planificați pentru comparație:
    - LinearRegression (baseline)
    - RandomForestRegressor
    - XGBRegressor
    - LSTM (TensorFlow/Keras)

Metrici de evaluare: RMSE, MAE, R², MAPE.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass
class ModelResult:
    """Rezultatul evaluării unui model."""
    name: str
    rmse: float
    mae: float
    r2: float
    mape: float
    model: Any


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculează metricile standard pentru regresie."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    # MAPE robust la zerouri
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def train_linear(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42):
    """Import XGBoost lazy (e dependență grea)."""
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[ModelResult]:
    """
    Antrenează și compară mai mulți algoritmi pe același split.

    TODO:
        - Adăugare LSTM (necesită reshape pentru time-series)
        - Cross-validation pe ferestre de timp (TimeSeriesSplit)
        - Hyperparameter tuning (GridSearchCV / Optuna)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    results: list[ModelResult] = []
    trainers = {
        "LinearRegression": train_linear,
        "RandomForest": train_random_forest,
        "XGBoost": train_xgboost,
    }

    for name, trainer in trainers.items():
        try:
            model = trainer(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate(np.asarray(y_test), np.asarray(y_pred))
            results.append(ModelResult(name=name, model=model, **metrics))
        except ImportError as e:
            print(f"[skip] {name}: {e}")

    return results


if __name__ == "__main__":
    # Exemplu sintetic pentru a verifica că funcționează
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X.sum(axis=1) + rng.normal(scale=0.1, size=500))

    for r in compare_models(X, y):
        print(f"{r.name:20s}  RMSE={r.rmse:.4f}  R²={r.r2:.4f}")
