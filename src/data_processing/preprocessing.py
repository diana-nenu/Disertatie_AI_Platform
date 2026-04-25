"""
Preprocesare si feature engineering pentru cele trei seturi de date.

Modulul ofera:
    - functii atomice reutilizabile (curatare NaN, features temporale, lag-uri,
      rolling means, split cronologic, eliminare coloane goale);
    - pipeline-uri complete per set de date care combina pasii intr-un singur apel;
    - parametri implici incarcati din config.yaml (sectiunea "preprocessing").

Conventii:
    - Toate functiile primesc DataFrame cu DatetimeIndex (sortat crescator).
    - Nu se shuffle-uieste niciodata datele: split cronologic strict.
    - Lag-urile si rolling-urile se aplica DOAR pe target (sau coloane explicite),
      pentru a evita leakage la train.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config


# ===========================================================================
# 1. FUNCTII ATOMICE
# ===========================================================================
def drop_empty_columns(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Elimina coloanele cu prea multe valori NaN.

    Args:
        df: DataFrame de intrare.
        threshold: procent maxim acceptat de NaN (intre 0 si 1). Coloanele cu
            procent NaN strict mai mare decat acest prag sunt eliminate.

    Returns:
        DataFrame fara coloanele "goale".
    """
    nan_ratio = df.isna().mean()
    keep_cols = nan_ratio[nan_ratio <= threshold].index
    return df[keep_cols].copy()


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "interpolate",
    max_nan_run: int = 24,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Trateaza valorile lipsa.

    Args:
        df: DataFrame de intrare (DatetimeIndex sortat).
        strategy: "interpolate" (linear), "ffill" (forward fill) sau "drop".
        max_nan_run: numar maxim de NaN consecutivi peste care randurile sunt
            eliminate in loc de interpolate (evita interpolari pe goluri lungi).
            Folosit doar daca strategy = "interpolate".
        columns: subset de coloane pe care se aplica strategia. None = toate
            coloanele numerice.

    Returns:
        DataFrame curatat.
    """
    out = df.copy()
    cols = list(columns) if columns else list(out.select_dtypes(include="number").columns)

    if strategy == "drop":
        out = out.dropna(subset=cols)
        return out

    if strategy == "ffill":
        out[cols] = out[cols].ffill().bfill()
        return out

    if strategy == "interpolate":
        # Marcam runurile lungi de NaN inainte de interpolare
        for c in cols:
            mask = out[c].isna()
            if not mask.any():
                continue
            # Identificam grupuri consecutive de NaN
            groups = (mask != mask.shift()).cumsum()
            run_sizes = mask.groupby(groups).transform("sum")
            long_run = mask & (run_sizes > max_nan_run)
            # Interpolam doar runurile scurte
            out[c] = out[c].interpolate(method="linear", limit=max_nan_run, limit_direction="both")
            # Pastram NaN pe runurile lungi (vor fi eliminate ulterior daca user vrea)
            out.loc[long_run & out[c].isna(), c] = np.nan

        # Drop randuri care inca au NaN (din runuri lungi)
        out = out.dropna(subset=cols)
        return out

    raise ValueError(f"Strategie necunoscuta: {strategy!r}. Optiuni: interpolate, ffill, drop.")


def add_temporal_features(
    df: pd.DataFrame,
    cyclic_encoding: bool = True,
    holidays_country: str | None = None,
) -> pd.DataFrame:
    """
    Adauga features temporale derivate din DatetimeIndex.

    Features create:
        - hour, dayofweek, day, month, quarter, year, dayofyear, weekofyear
        - is_weekend (bool -> int)
        - is_holiday (bool -> int) daca se da holidays_country (cod ISO 2 caractere)
        - hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
          (encoding ciclic, doar daca cyclic_encoding=True)

    Args:
        df: DataFrame cu DatetimeIndex.
        cyclic_encoding: daca True, adauga sin/cos pentru hour, dow, month.
        holidays_country: cod tara pentru pachetul `holidays` (ex. "US", "ES",
            "IN"). Daca None, nu se adauga is_holiday.

    Returns:
        DataFrame cu coloanele suplimentare.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame-ul trebuie sa aiba DatetimeIndex.")

    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["day"] = idx.day
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["year"] = idx.year
    out["dayofyear"] = idx.dayofyear
    out["weekofyear"] = idx.isocalendar().week.astype(int).values
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    if cyclic_encoding:
        out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        out["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
        out["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

    if holidays_country:
        try:
            import holidays as hol_pkg  # import lazy
            years = list(range(int(idx.year.min()), int(idx.year.max()) + 1))
            country_holidays = hol_pkg.country_holidays(holidays_country, years=years)
            # Convertim setul de sarbatori la timestamps pandas pentru a evita FutureWarning
            holiday_dates = pd.DatetimeIndex(list(country_holidays.keys()))
            out["is_holiday"] = idx.normalize().isin(holiday_dates).astype(int)
        except ImportError:
            print("[warn] pachetul `holidays` nu este instalat; sar peste is_holiday.")

    return out


def add_lags(
    df: pd.DataFrame,
    target: str,
    lags: Iterable[int],
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Adauga coloane lag pentru target: target_lag_<n>.

    Args:
        df: DataFrame cu DatetimeIndex sortat.
        target: numele coloanei target.
        lags: lista de pasi (ex. [1, 24, 168]).
        drop_na: daca True, elimina randurile cu NaN dupa shift (de obicei
            preferi sa il faci dupa toate transformarile).

    Returns:
        DataFrame cu coloanele lag adaugate.
    """
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    if drop_na:
        out = out.dropna()
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: Iterable[int],
    stats: Iterable[str] = ("mean", "std"),
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Adauga features rolling pentru target: target_roll_<stat>_<w>.

    Important: rolling-ul se aplica pe seria SHIFTATA cu 1 pas (target.shift(1))
    pentru a evita data leakage (la momentul t nu cunoastem y_t).

    Args:
        df: DataFrame cu DatetimeIndex sortat.
        target: numele coloanei target.
        windows: ferestre (ex. [3, 24, 168]).
        stats: statistici (mean, std, min, max, median).
        drop_na: daca True, elimina randurile cu NaN.

    Returns:
        DataFrame cu features rolling.
    """
    out = df.copy()
    base = out[target].shift(1)  # evita leakage
    for w in windows:
        roll = base.rolling(window=w, min_periods=max(2, w // 2))
        for s in stats:
            out[f"{target}_roll_{s}_{w}"] = getattr(roll, s)()
    if drop_na:
        out = out.dropna()
    return out


def chronological_split(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    validation_size: float = 0.0,
    feature_cols: Iterable[str] | None = None,
) -> dict:
    """
    Imparte cronologic: train (cele mai vechi), validation (mijloc), test (cele mai noi).

    Args:
        df: DataFrame deja sortat dupa index temporal.
        target: numele coloanei target (y).
        test_size: fractiune din total pentru test.
        validation_size: fractiune din total pentru validation (poate fi 0).
        feature_cols: lista de coloane folosite ca features. None = toate
            in afara de target.

    Returns:
        dict cu chei: X_train, y_train, X_val, y_val, X_test, y_test.
        Daca validation_size = 0, X_val si y_val sunt None.
    """
    if not 0 <= test_size < 1:
        raise ValueError("test_size trebuie in [0, 1).")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size trebuie in [0, 1).")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size trebuie < 1.")

    n = len(df)
    n_test = int(np.floor(n * test_size))
    n_val = int(np.floor(n * validation_size))
    n_train = n - n_test - n_val

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target]
    feature_cols = list(feature_cols)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val] if n_val > 0 else None
    test = df.iloc[n_train + n_val :]

    return {
        "X_train": train[feature_cols],
        "y_train": train[target],
        "X_val": val[feature_cols] if val is not None else None,
        "y_val": val[target] if val is not None else None,
        "X_test": test[feature_cols],
        "y_test": test[target],
        "feature_cols": feature_cols,
    }


# ===========================================================================
# 2. PIPELINE-URI PER SET DE DATE
# ===========================================================================
def build_features_consum_usa(
    df: pd.DataFrame | None = None,
    target: str = "PJME_MW",
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Pipeline preprocesare pentru consum USA (PJME, orar).

    Pasi: missing values -> features temporale -> lags -> rolling -> dropna final.
    """
    from src.data_processing.loader import load_consum_usa

    cfg = (config or load_config())["preprocessing"]
    if df is None:
        df = load_consum_usa()

    df = handle_missing_values(df, strategy=cfg["missing_strategy"], max_nan_run=cfg["max_nan_run"])
    df = add_temporal_features(df, holidays_country=cfg["holidays_country"]["consum_usa"])
    df = add_lags(df, target=target, lags=cfg["lags"])
    df = add_rolling_features(df, target=target, windows=cfg["rolling_windows"])
    df = df.dropna()
    return df


def build_features_pret_spania(
    df: pd.DataFrame | None = None,
    target: str = "price actual",
    config: dict | None = None,
    city: str = "Madrid",
) -> pd.DataFrame:
    """
    Pipeline preprocesare pentru pret + cerere Spania (orar).

    Pasi: drop coloane goale -> missing values -> features temporale -> lags
    -> rolling -> dropna. Pastreaza un subset de coloane numerice utile.
    """
    from src.data_processing.loader import merge_spania

    cfg = (config or load_config())["preprocessing"]
    if df is None:
        df = merge_spania(city=city)

    # Eliminam coloane non-numerice (ex. city_name, weather_main, etc.)
    df_num = df.select_dtypes(include="number").copy()
    df_num = drop_empty_columns(df_num, threshold=cfg["drop_column_nan_threshold"])

    if target not in df_num.columns:
        raise KeyError(f"Coloana target {target!r} nu exista dupa filtrare numerica.")

    df_num = handle_missing_values(
        df_num, strategy=cfg["missing_strategy"], max_nan_run=cfg["max_nan_run"]
    )
    df_num = add_temporal_features(df_num, holidays_country=cfg["holidays_country"]["pret_spania"])
    df_num = add_lags(df_num, target=target, lags=cfg["lags"])
    df_num = add_rolling_features(df_num, target=target, windows=cfg["rolling_windows"])
    df_num = df_num.dropna()
    return df_num


def build_features_solar_india(
    df: pd.DataFrame | None = None,
    target: str = "AC_POWER",
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Pipeline preprocesare pentru solar India (Plant_1).

    Pasi: (resample optional la orar) -> missing values -> features temporale
    -> lags -> rolling -> dropna.

    Nota: lag-urile / rolling-urile sunt redimensionate daca solar_resample_to_hourly
    e True (deja sunt in pasi orari) sau False (un pas = 15 min, deci lag 1 = 15 min).
    """
    from src.data_processing.loader import merge_solar

    cfg_full = config or load_config()
    cfg = cfg_full["preprocessing"]
    if df is None:
        df = merge_solar()

    if cfg.get("solar_resample_to_hourly", True):
        df = df.resample("1h").mean()

    df = handle_missing_values(df, strategy=cfg["missing_strategy"], max_nan_run=cfg["max_nan_run"])
    df = add_temporal_features(df, holidays_country=cfg["holidays_country"]["solar_india"])
    df = add_lags(df, target=target, lags=cfg["lags"])
    df = add_rolling_features(df, target=target, windows=cfg["rolling_windows"])
    df = df.dropna()
    return df


# ===========================================================================
# 3. DEMO
# ===========================================================================
if __name__ == "__main__":
    cfg = load_config()
    p_cfg = cfg["preprocessing"]

    print("=== Pipeline consum USA ===")
    df_usa = build_features_consum_usa(config=cfg)
    print(f"  Shape: {df_usa.shape}, Range: {df_usa.index.min()} -> {df_usa.index.max()}")
    print(f"  NaN total: {df_usa.isna().sum().sum()}")

    split_usa = chronological_split(
        df_usa, target="PJME_MW",
        test_size=p_cfg["test_size"], validation_size=p_cfg["validation_size"]
    )
    print(f"  Train: {split_usa['X_train'].shape}, Test: {split_usa['X_test'].shape}")

    print("=== Pipeline pret Spania ===")
    df_es = build_features_pret_spania(config=cfg)
    print(f"  Shape: {df_es.shape}, Range: {df_es.index.min()} -> {df_es.index.max()}")
    print(f"  NaN total: {df_es.isna().sum().sum()}")

    print("=== Pipeline solar India ===")
    df_in = build_features_solar_india(config=cfg)
    print(f"  Shape: {df_in.shape}, Range: {df_in.index.min()} -> {df_in.index.max()}")
    print(f"  NaN total: {df_in.isna().sum().sum()}")
