"""
Antrenare si comparare a algoritmilor predictivi pentru cele 3 seturi de date.

Algoritmi suportati:
    - LinearRegression (baseline)
    - RandomForestRegressor
    - XGBRegressor (gradient boosting)
    - LSTM (TensorFlow/Keras) - pentru serii temporale lungi
    - Prophet (Facebook) - pentru serii cu sezonalitate puternica (optional, USA)

Metrici de evaluare: RMSE, MAE, R^2, MAPE.

Validare: TimeSeriesSplit (cronologic) - fereastra de train se mareste progresiv,
fereastra de test ramane in viitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# ===========================================================================
# 1. METRICI
# ===========================================================================
@dataclass
class ModelResult:
    """Rezultatul evaluarii unui model."""
    name: str
    rmse: float
    mae: float
    r2: float
    mape: float
    model: Any = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.name,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "mape": self.mape,
        }


def evaluate(y_true, y_pred) -> dict[str, float]:
    """
    Calculeaza metricile standard pentru regresie.

    RMSE: Root Mean Squared Error (in unitati originale, penalizeaza erorile mari).
    MAE: Mean Absolute Error (mediana erorilor, robust la outliers).
    R^2: coeficient de determinare (1 = perfect, 0 = ca media, negativ = mai rau).
    MAPE: Mean Absolute Percentage Error (procentaj, util pentru comparatii cross-domain).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ===========================================================================
# 2. ANTRENARE INDIVIDUALA
# ===========================================================================
def train_linear(X_train, y_train, **kwargs) -> LinearRegression:
    model = LinearRegression(**kwargs)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42, **kwargs) -> RandomForestRegressor:
    defaults = dict(n_estimators=200, max_depth=None, n_jobs=-1)
    defaults.update(kwargs)
    model = RandomForestRegressor(random_state=random_state, **defaults)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42, **kwargs):
    """Import lazy - xgboost este greu, evitam la import-ul modulului."""
    from xgboost import XGBRegressor
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1)
    defaults.update(kwargs)
    model = XGBRegressor(random_state=random_state, **defaults)
    model.fit(X_train, y_train)
    return model


def train_lstm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    sequence_length: int = 24,
    units: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    dropout: float = 0.2,
    patience: int = 5,
    random_state: int = 42,
    verbose: int = 0,
):
    """
    Antreneaza un LSTM cu un singur strat recurent + dense output.

    Pentru ca LSTM cere input 3D (samples, timesteps, features), seria
    de features 2D este transformata in ferestre de `sequence_length` pasi.

    Args:
        sequence_length: cati pasi anteriori vede LSTM ca sa prezica pasul curent.
        units: numar de neuroni in stratul LSTM.
        epochs: numar maxim de epoci de antrenare.
        patience: early stopping pe val_loss daca e dat (X_val, y_val).

    Returns:
        dict cu chei: model (Keras), scaler_y (StandardScaler), seq_len, n_features.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Standardizam target-ul (LSTM converge mai usor pe valori in [-1, 1])
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).flatten()
    y_val_s = (
        scaler_y.transform(np.asarray(y_val).reshape(-1, 1)).flatten()
        if y_val is not None and len(y_val) > 0 else None
    )

    X_arr = np.asarray(X_train, dtype=np.float32)
    n_features = X_arr.shape[1]

    def _make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            ys.append(y[i])
        return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    X_seq, y_seq = _make_sequences(X_arr, y_train_s, sequence_length)

    val_data = None
    if X_val is not None and y_val_s is not None and len(X_val) > sequence_length:
        Xv_arr = np.asarray(X_val, dtype=np.float32)
        Xv_seq, yv_seq = _make_sequences(Xv_arr, y_val_s, sequence_length)
        val_data = (Xv_seq, yv_seq)

    model = Sequential([
        LSTM(units, input_shape=(sequence_length, n_features), return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    callbacks = []
    if val_data is not None:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))

    model.fit(
        X_seq, y_seq,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )

    return {
        "model": model,
        "scaler_y": scaler_y,
        "seq_len": sequence_length,
        "n_features": n_features,
    }


def predict_lstm(lstm_bundle: dict, X) -> np.ndarray:
    """
    Inferenta cu un bundle LSTM (model + scaler_y + seq_len).
    Primele `seq_len` predictii sunt NaN (nu avem istoric suficient).
    """
    model = lstm_bundle["model"]
    scaler_y = lstm_bundle["scaler_y"]
    seq_len = lstm_bundle["seq_len"]

    X_arr = np.asarray(X, dtype=np.float32)
    n = len(X_arr)
    Xs = np.array([X_arr[i - seq_len : i] for i in range(seq_len, n)], dtype=np.float32)
    y_pred_scaled = model.predict(Xs, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    out = np.full(n, np.nan)
    out[seq_len:] = y_pred
    return out


def train_prophet(df_train: pd.DataFrame, target: str, freq: str = "h", **kwargs):
    """
    Antreneaza un model Prophet pe o serie temporala.

    Prophet asteapta DataFrame cu coloane 'ds' (timestamp) si 'y' (target).
    Capteaza automat: trend, sezonalitate zilnica, saptamanala, anuala.

    Args:
        df_train: DataFrame cu DatetimeIndex.
        target: numele coloanei target.
        freq: frecventa pentru predictii viitoare ("h" pentru orar).

    Returns:
        modelul Prophet antrenat.
    """
    from prophet import Prophet

    prophet_df = pd.DataFrame({
        "ds": df_train.index,
        "y": df_train[target].values,
    })

    defaults = dict(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
    )
    defaults.update(kwargs)

    model = Prophet(**defaults)
    model.fit(prophet_df)
    return model


def predict_prophet(model, index: pd.DatetimeIndex) -> np.ndarray:
    """Inferenta Prophet pe un index dat."""
    future = pd.DataFrame({"ds": index})
    forecast = model.predict(future)
    return forecast["yhat"].values


# ===========================================================================
# 3. CROSS-VALIDATION CRONOLOGICA
# ===========================================================================
def time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    train_func: Callable,
    n_splits: int = 5,
    predict_func: Callable | None = None,
    show_progress: bool = False,
    label: str = "CV",
) -> list[dict]:
    """
    Cross-validation cu TimeSeriesSplit (ferestre cronologice extinse).

    Pentru fiecare fold:
        - Antrenam pe primele K randuri (K creste de la fold la fold)
        - Testam pe urmatoarele M randuri (M e fix per fold).

    Args:
        train_func: functie cu signature (X_train, y_train) -> model.
        predict_func: functie cu signature (model, X) -> y_pred. Daca None,
            se foloseste model.predict(X).
        show_progress: daca True, afiseaza fold-ul curent + timp scurs.
        label: prefix pentru mesajele de progres.

    Returns:
        lista de dict-uri cu metricile per fold.
    """
    import time as _time
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    t_start = _time.time()
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if show_progress:
            print(f"  [{label}] fold {fold+1}/{n_splits} - antrenare pe {len(X_train)} randuri...", flush=True)
            t_fold = _time.time()

        model = train_func(X_train, y_train)
        if predict_func is None:
            y_pred = model.predict(X_test)
        else:
            y_pred = predict_func(model, X_test)

        # Pentru LSTM, primele seq_len predictii sunt NaN
        mask = ~pd.isna(y_pred)
        if mask.sum() == 0:
            continue
        metrics = evaluate(np.asarray(y_test)[mask], np.asarray(y_pred)[mask])
        metrics["fold"] = fold
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = int(mask.sum())
        results.append(metrics)

        if show_progress:
            elapsed = _time.time() - t_fold
            total = _time.time() - t_start
            print(f"  [{label}] fold {fold+1} terminat in {elapsed:.1f}s (total scurs: {total:.1f}s) - RMSE={metrics['rmse']:.2f}", flush=True)
    return results


# ===========================================================================
# 4. HYPERPARAMETER TUNING
# ===========================================================================
def tune_with_gridsearch(
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    n_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 0,
):
    """
    GridSearchCV cu TimeSeriesSplit pentru tuning cronologic corect.

    Returns:
        GridSearchCV fitted (best_estimator_, best_params_, cv_results_).
    """
    from sklearn.model_selection import GridSearchCV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    gs.fit(X_train, y_train)
    return gs


# ===========================================================================
# 5. SALVARE / INCARCARE MODELE
# ===========================================================================
def tune_with_optuna(
    estimator_class,
    param_space: dict,
    X_train,
    y_train,
    n_trials: int = 50,
    n_splits: int = 3,
    direction: str = "minimize",
    random_state: int = 42,
    show_progress: bool = False,
    use_pruner: bool = True,
):
    """
    Bayesian hyperparameter optimization cu Optuna (Tree Parzen Estimator).

    Spre deosebire de GridSearchCV care testeaza toate combinatiile, Optuna
    foloseste rezultatele anterioare pentru a alege inteligent urmatoarea
    combinatie - de 5-10x mai eficient pe spatii mari de parametri.

    Args:
        estimator_class: clasa modelului (de exemplu, XGBRegressor).
        param_space: dict cu numele parametrilor si specificatii Optuna.
            Exemplu: {
                "max_depth": ("int", 3, 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.6, 1.0),
                "n_estimators": ("int", 100, 500),
            }
        n_trials: numar de combinatii de testat.
        n_splits: numar de folduri TimeSeriesSplit.
        direction: "minimize" pentru RMSE / MAE, "maximize" pentru R^2.
        use_pruner: daca True, foloseste MedianPruner pentru a opri trial-urile slabe.

    Returns:
        dict cu chei: best_params, best_value, study (obiectul Optuna).
    """
    import optuna
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING if not show_progress else optuna.logging.INFO)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # Construiesc parametrii din specificatii
        params = {}
        for name, spec in param_space.items():
            kind = spec[0]
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == "float_log":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
            else:
                raise ValueError(f"Tip parametru necunoscut: {kind}")

        # Cross-validation cronologica
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_vl = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_vl = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            model = estimator_class(**params, random_state=random_state) if 'random_state' in estimator_class.__init__.__code__.co_varnames else estimator_class(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            rmse = float(np.sqrt(mean_squared_error(y_vl, y_pred)))
            scores.append(rmse)

            # Reportez scor intermediar pentru pruner
            trial.report(rmse, fold_idx)
            if use_pruner and trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1) if use_pruner else None
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=pruner,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "n_trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "n_trials_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
    }


def compute_shap_values(model, X, max_samples: int = 1000, model_type: str = "auto"):
    """
    Calculeaza SHAP values pentru un model antrenat.

    Foloseste TreeExplainer pentru tree-based models (RF, XGBoost) - calcul exact si rapid.
    Pentru alte modele, fallback la KernelExplainer (lent dar universal).

    Args:
        model: model antrenat.
        X: features (DataFrame sau ndarray) pentru care se calculeaza SHAP.
        max_samples: limita numarului de observatii pentru calcul (pentru viteza).
        model_type: "tree" (RF, XGBoost), "linear", "kernel" (fallback), sau "auto".

    Returns:
        dict cu chei: shap_values (ndarray), expected_value (float), explainer.
    """
    import shap
    import warnings as _w

    if isinstance(X, pd.DataFrame) and len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    elif not isinstance(X, pd.DataFrame) and len(X) > max_samples:
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # Auto-detect model type
    if model_type == "auto":
        cls_name = type(model).__name__
        if any(k in cls_name for k in ("XGB", "Forest", "Tree", "GBM", "Boost", "Cat")):
            model_type = "tree"
        elif any(k in cls_name for k in ("Linear", "Ridge", "Lasso", "Elastic", "Logistic")):
            model_type = "linear"
        else:
            model_type = "kernel"

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        explainer = None
        shap_values = None
        expected_value = 0.0

        # Pentru XGBoost folosim API-ul built-in pred_contribs - cel mai robust si rapid
        if "XGB" in type(model).__name__:
            booster = model.get_booster()
            import xgboost as xgb_lib
            if isinstance(X_sample, pd.DataFrame):
                dmat = xgb_lib.DMatrix(X_sample.values, feature_names=list(X_sample.columns))
            else:
                dmat = xgb_lib.DMatrix(X_sample)
            contribs = booster.predict(dmat, pred_contribs=True)
            # Ultima coloana e bias (expected_value), restul sunt SHAP values
            shap_values = contribs[:, :-1]
            expected_value = float(contribs[0, -1])
            explainer = booster
            model_type = "tree_xgboost_native"

        elif model_type == "tree":
            # Pentru RandomForest si alte tree models - TreeExplainer SHAP
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                else:
                    expected_value = float(expected_value)
            except Exception:
                # Fallback la kernel
                background = shap.sample(X_sample, min(50, len(X_sample)), random_state=42)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
                expected_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])
        elif model_type == "linear":
            try:
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.0
                else:
                    expected_value = float(expected_value)
            except Exception:
                # Fallback: SHAP pentru modele liniare se poate calcula direct din coeficienti
                if hasattr(model, 'coef_'):
                    X_arr = X_sample.values if hasattr(X_sample, 'values') else X_sample
                    X_mean = X_arr.mean(axis=0)
                    shap_values = (X_arr - X_mean) * model.coef_
                    expected_value = float(model.predict(X_mean.reshape(1, -1))[0])
                    explainer = "linear_manual"
                else:
                    raise
        else:
            # Kernel (lent)
            background = shap.sample(X_sample, min(50, len(X_sample)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample)
            expected_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])

    return {
        "shap_values": shap_values,
        "expected_value": expected_value,
        "explainer": explainer,
        "X_sample": X_sample,
        "model_type": model_type,
    }


def save_model(model, path: str | Path, kind: str = "auto") -> None:
    """
    Salveaza modelul.

    Args:
        kind: 'sklearn' (joblib), 'xgboost' (json), 'lstm' (h5), 'prophet' (json),
              'auto' (detecteaza dupa tip).
    """
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "auto":
        cls = type(model).__name__
        if "XGB" in cls:
            kind = "xgboost"
        elif "Sequential" in cls or "Functional" in cls:
            kind = "lstm"
        elif "Prophet" in cls:
            kind = "prophet"
        else:
            kind = "sklearn"

    if kind == "sklearn":
        joblib.dump(model, path)
    elif kind == "xgboost":
        model.save_model(str(path))
    elif kind == "lstm":
        model.save(str(path))
    elif kind == "prophet":
        from prophet.serialize import model_to_json
        with open(path, "w") as f:
            f.write(model_to_json(model))
    else:
        raise ValueError(f"kind necunoscut: {kind!r}")


def load_model(path: str | Path, kind: str):
    """Incarca un model salvat anterior. `kind` e obligatoriu (nu putem detecta din path)."""
    import joblib
    path = Path(path)
    if kind == "sklearn":
        return joblib.load(path)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor()
        m.load_model(str(path))
        return m
    if kind == "lstm":
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(str(path))
    if kind == "prophet":
        from prophet.serialize import model_from_json
        with open(path) as f:
            return model_from_json(f.read())
    raise ValueError(f"kind necunoscut: {kind!r}")


# ===========================================================================
# 6. FEATURE IMPORTANCE
# ===========================================================================
def get_feature_importance(model, feature_names: Iterable[str]) -> pd.DataFrame:
    """
    Extrage feature_importances_ pentru tree-based models (RF, XGBoost) sau
    coef_ pentru modele liniare.

    Returns:
        DataFrame cu coloane 'feature' si 'importance', sortat descrescator.
    """
    feature_names = list(feature_names)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_).flatten())
    else:
        raise AttributeError(f"Modelul {type(model).__name__} nu are feature_importances_ / coef_.")
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ===========================================================================
# 7. COMPARATIE COMPLETA
# ===========================================================================
def compare_models(
    X_train, y_train, X_test, y_test,
    algorithms: Iterable[str] = ("LinearRegression", "RandomForest", "XGBoost"),
    random_state: int = 42,
) -> list[ModelResult]:
    """
    Antreneaza si compara mai multi algoritmi pe acelasi split.

    Args:
        algorithms: subset din {LinearRegression, RandomForest, XGBoost}.
            (LSTM si Prophet se trateaza separat din cauza API-ului diferit.)

    Returns:
        lista de ModelResult sortata descrescator dupa R^2.
    """
    trainers = {
        "LinearRegression": train_linear,
        "RandomForest": lambda Xt, yt: train_random_forest(Xt, yt, random_state=random_state),
        "XGBoost": lambda Xt, yt: train_xgboost(Xt, yt, random_state=random_state),
    }
    results: list[ModelResult] = []
    for name in algorithms:
        if name not in trainers:
            print(f"[skip] {name} nu este suportat in compare_models (foloseste train_lstm/train_prophet separat).")
            continue
        try:
            model = trainers[name](X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            results.append(ModelResult(name=name, model=model, **metrics))
        except Exception as e:
            print(f"[error] {name}: {e}")
    results.sort(key=lambda r: r.r2, reverse=True)
    return results


# ===========================================================================
# Demo
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(X.sum(axis=1) + rng.normal(scale=0.1, size=n))

    split = int(0.8 * n)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    print("=== compare_models ===")
    for r in compare_models(X_tr, y_tr, X_te, y_te):
        print(f"  {r.name:<18s}  RMSE={r.rmse:.4f}  R^2={r.r2:.4f}")

    print("=== TimeSeriesSplit CV ===")
    cv_results = time_series_cv(X, y, train_linear, n_splits=3)
    for r in cv_results:
        print(f"  fold {r['fold']}: train={r['train_size']}, RMSE={r['rmse']:.4f}")
