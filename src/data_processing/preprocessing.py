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


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adauga coloana `season` (1=iarna, 2=primavara, 3=vara, 4=toamna)
    bazata pe luna calendaristica (emisfera nordica).

    Mapare: dec/ian/feb=iarna, mar/apr/mai=primavara, iun/iul/aug=vara,
    sep/oct/nov=toamna.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame-ul trebuie sa aiba DatetimeIndex.")
    out = df.copy()
    month = out.index.month
    season = np.where(month.isin([12, 1, 2]), 1,
              np.where(month.isin([3, 4, 5]), 2,
              np.where(month.isin([6, 7, 8]), 3, 4)))
    out["season"] = season.astype(int)
    return out


def add_virtual_year_index(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """
    Adauga coloana `virtual_hour_of_year` (0-8759) - pozitia in cadrul unui
    an "virtual" de 365 zile cu rezolutie orara. Permite suprapunerea seriilor
    care provin din ani diferiti pe acelasi grafic comparativ.

    Pentru frecventa orara: virtual_hour_of_year = (dayofyear-1) * 24 + hour.
    Anii bisecti (366 zile) sunt trimati la 365 (29 februarie e tratat ca 28).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame-ul trebuie sa aiba DatetimeIndex.")

    out = df.copy()
    idx = out.index

    # Day of year cu corectie pentru an bisect (29 feb -> tratam ca 28 feb)
    is_leap = idx.is_leap_year
    is_after_feb29 = (idx.month > 2) | ((idx.month == 2) & (idx.day == 29))
    doy = idx.dayofyear - (is_leap & is_after_feb29).astype(int)
    doy = np.clip(doy, 1, 365)

    out["virtual_hour_of_year"] = (doy - 1) * 24 + idx.hour
    out["virtual_day_of_year"] = doy
    return out


def add_solar_efficiency_features(
    df: pd.DataFrame,
    power_col: str = "AC_POWER",
    irradiation_col: str = "IRRADIATION",
    dc_power_col: str = "DC_POWER",
    module_temp_col: str = "MODULE_TEMPERATURE",
    irradiation_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Adauga features de eficienta specifice pentru centrale solare:
        - performance_ratio = AC_POWER / IRRADIATION (Yield)
        - dc_ac_ratio = AC_POWER / DC_POWER (eficienta invertor)
        - temp_excess = MODULE_TEMPERATURE - 25 (peste temperatura de referinta STC)
        - eff_temp_corrected = performance_ratio * (1 + 0.004 * temp_excess)
          (corectie liniara cu coeficient temperatura tipic -0.4%/grad)

    Args:
        df: DataFrame cu coloanele relevante.
        irradiation_threshold: prag sub care nu calculam ratio (noaptea, evita
            division by zero / valori absurde).
    """
    out = df.copy()

    # Performance ratio (Yield-ul cerut explicit in plan)
    if power_col in out.columns and irradiation_col in out.columns:
        mask_day = out[irradiation_col] >= irradiation_threshold
        out["performance_ratio"] = np.where(
            mask_day, out[power_col] / out[irradiation_col], 0.0
        )

    # Raport invertor (pierderi DC -> AC)
    if dc_power_col in out.columns and power_col in out.columns:
        mask_active = out[dc_power_col] > 0
        out["dc_ac_ratio"] = np.where(
            mask_active, out[power_col] / out[dc_power_col], 0.0
        )

    # Excedent temperatura modul fata de Standard Test Conditions (STC = 25 grade)
    if module_temp_col in out.columns:
        out["temp_excess"] = out[module_temp_col] - 25.0
        if "performance_ratio" in out.columns:
            out["eff_temp_corrected"] = out["performance_ratio"] * (1 + 0.004 * out["temp_excess"])

    return out


def detect_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    method: str = "iqr",
    iqr_multiplier: float = 3.0,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Detecteaza outliers si returneaza DataFrame cu o coloana booleana
    `is_outlier_<colname>` per coloana analizata.

    Args:
        df: DataFrame cu date numerice.
        columns: coloane pe care se face detectia. None = toate numericele.
        method: "iqr" (Interquartile Range) sau "zscore".
        iqr_multiplier: pentru metoda IQR, multiplicator (1.5 = standard, 3 = conservator).
        z_threshold: prag z-score absolut pentru metoda zscore.

    Returns:
        DataFrame original + coloane booleene `is_outlier_<col>`.
    """
    out = df.copy()
    cols = list(columns) if columns else list(out.select_dtypes(include="number").columns)

    for c in cols:
        if c not in out.columns:
            continue
        x = out[c]
        if method == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            out[f"is_outlier_{c}"] = ((x < lower) | (x > upper)).astype(int)
        elif method == "zscore":
            mu, sigma = x.mean(), x.std()
            if sigma > 0:
                z = (x - mu) / sigma
                out[f"is_outlier_{c}"] = (z.abs() > z_threshold).astype(int)
            else:
                out[f"is_outlier_{c}"] = 0
        else:
            raise ValueError(f"Metoda necunoscuta: {method!r}. Folosesti iqr sau zscore.")
    return out


def validate_solar_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity checks fizice pentru date solar:
        - AC_POWER nu poate fi mai mare ca DC_POWER (pierderi invertor sunt pozitive)
        - Productia noaptea (IRRADIATION ~ 0) ar trebui sa fie ~0
        - Toate puterile sunt non-negative

    Returns:
        DataFrame cu coloane booleene `flag_*` care marcheaza inconsistente.
    """
    out = df.copy()
    if "DC_POWER" in out.columns and "AC_POWER" in out.columns:
        out["flag_ac_gt_dc"] = (out["AC_POWER"] > out["DC_POWER"] * 1.01).astype(int)
        out["flag_negative_power"] = ((out["DC_POWER"] < 0) | (out["AC_POWER"] < 0)).astype(int)
    if "IRRADIATION" in out.columns and "AC_POWER" in out.columns:
        out["flag_night_production"] = ((out["IRRADIATION"] < 0.01) & (out["AC_POWER"] > 10)).astype(int)
    return out


def encode_categorical_top_n(
    df: pd.DataFrame,
    columns: Iterable[str],
    top_n: int = 5,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    One-hot encoding pentru variabile categorice, pastrand doar top N categorii
    cele mai frecvente. Restul sunt grupate intr-o categorie "other".

    Args:
        df: DataFrame cu coloane categorice.
        columns: coloane de encodat.
        top_n: numar de categorii cele mai frecvente pastrate.
        drop_original: daca True, coloana originala se elimina dupa encoding.

    Returns:
        DataFrame cu coloane noi `<col>_<categorie>`.
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            continue
        top_cats = out[c].value_counts().head(top_n).index.tolist()
        out[c + "_clean"] = out[c].where(out[c].isin(top_cats), other="other")
        dummies = pd.get_dummies(out[c + "_clean"], prefix=c, dtype=int)
        out = pd.concat([out, dummies], axis=1)
        out = out.drop(columns=[c + "_clean"])
        if drop_original:
            out = out.drop(columns=[c])
    return out


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    X_test: pd.DataFrame | None = None,
    method: str = "standard",
    columns: Iterable[str] | None = None,
):
    """
    Scaleaza features. Foarte important: scaler-ul se FITEAZA DOAR pe X_train,
    apoi se TRANSFORMA pe val si test (anti data leakage).

    Args:
        X_train, X_val, X_test: seturile de features.
        method: "standard" (medie 0, std 1), "minmax" (0-1), sau "robust" (mediana, IQR).
        columns: subset de coloane de scalat. None = toate numericele.

    Returns:
        tuplu (X_train_scaled, X_val_scaled, X_test_scaled, scaler).
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Metoda necunoscuta: {method!r}. Optiuni: {list(scalers)}")

    if columns is None:
        columns = list(X_train.select_dtypes(include="number").columns)
    columns = list(columns)

    scaler = scalers[method]()
    Xt = X_train.copy()
    Xt[columns] = scaler.fit_transform(X_train[columns])

    Xv = None
    if X_val is not None and len(X_val) > 0:
        Xv = X_val.copy()
        Xv[columns] = scaler.transform(X_val[columns])

    Xs = None
    if X_test is not None and len(X_test) > 0:
        Xs = X_test.copy()
        Xs[columns] = scaler.transform(X_test[columns])

    return Xt, Xv, Xs, scaler


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
    df = add_season(df)
    df = add_virtual_year_index(df)
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

    # Encoding categoric meteo INAINTE de filtrare numerica (top 5 categorii)
    cat_cols = [c for c in ["weather_main", "weather_description"] if c in df.columns]
    if cat_cols:
        df = encode_categorical_top_n(df, columns=cat_cols, top_n=5, drop_original=True)

    # Eliminam coloane non-numerice ramase (city_name, weather_icon)
    df_num = df.select_dtypes(include="number").copy()
    df_num = drop_empty_columns(df_num, threshold=cfg["drop_column_nan_threshold"])

    if target not in df_num.columns:
        raise KeyError(f"Coloana target {target!r} nu exista dupa filtrare numerica.")

    df_num = handle_missing_values(
        df_num, strategy=cfg["missing_strategy"], max_nan_run=cfg["max_nan_run"]
    )
    df_num = add_temporal_features(df_num, holidays_country=cfg["holidays_country"]["pret_spania"])
    df_num = add_season(df_num)
    df_num = add_virtual_year_index(df_num)
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

    # Features specifice solar (yield, dc_ac_ratio, temperature derating)
    df = add_solar_efficiency_features(df)

    df = handle_missing_values(df, strategy=cfg["missing_strategy"], max_nan_run=cfg["max_nan_run"])
    df = add_temporal_features(df, holidays_country=cfg["holidays_country"]["solar_india"])
    df = add_season(df)
    df = add_virtual_year_index(df)
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
