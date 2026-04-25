"""
Functii pentru incarcarea celor trei seturi de date ale disertatiei.

Fiecare functie returneaza un DataFrame pandas pregatit, cu DatetimeIndex
unde este aplicabil. Formatele de timp sunt diferite intre seturi:
    - PJME (USA): "%Y-%m-%d %H:%M:%S"
    - Solar India - generare: "%d-%m-%Y %H:%M"  (atentie!)
    - Solar India - meteo: "%Y-%m-%d %H:%M:%S"
    - Spania: ISO 8601 cu timezone (+01:00)
"""

from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, load_config


# ---------------------------------------------------------------------------
# 1. CONSUM ENERGETIC USA (PJME)
# ---------------------------------------------------------------------------
def load_consum_usa() -> pd.DataFrame:
    """
    Incarca datele orare de consum din reteaua PJM (USA).

    Returns:
        DataFrame cu DatetimeIndex si coloana PJME_MW (megawati).
    """
    cfg = load_config()
    file_path = PROJECT_ROOT / cfg["datasets"]["consum_usa"]["file"]

    df = pd.read_csv(file_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()

    # Eliminam duplicatele de timestamp daca exista (DST schimbari)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# 2. SOLAR INDIA (Plant_1)
# ---------------------------------------------------------------------------
def load_solar_generation() -> pd.DataFrame:
    """Incarca datele de generare solar (Plant_1)."""
    cfg = load_config()
    path = PROJECT_ROOT / cfg["datasets"]["solar_india"]["generation"]

    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    return df


def load_solar_weather() -> pd.DataFrame:
    """Incarca datele meteo de la senzorii centralei solare (Plant_1)."""
    cfg = load_config()
    path = PROJECT_ROOT / cfg["datasets"]["solar_india"]["weather"]

    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df


def load_solar_india() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Incarca ambele tabele (generare + meteo) pentru Plant_1."""
    return load_solar_generation(), load_solar_weather()


def merge_solar(
    gen: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combina datele de generare cu datele meteo, agregand productia
    tuturor invertoarelor pe acelasi timestamp.

    Returns:
        DataFrame cu DatetimeIndex (15 min) si coloanele:
            DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD,
            AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION
    """
    if gen is None or weather is None:
        gen, weather = load_solar_india()

    # Agregam pe timestamp (sumam puterea pe toate invertoarele)
    gen_agg = (
        gen.groupby("DATE_TIME")
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "mean"),
            TOTAL_YIELD=("TOTAL_YIELD", "sum"),
        )
    )

    weather_agg = (
        weather.groupby("DATE_TIME")
        .agg(
            AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
            MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
            IRRADIATION=("IRRADIATION", "mean"),
        )
    )

    merged = gen_agg.join(weather_agg, how="inner").sort_index()
    return merged


# ---------------------------------------------------------------------------
# 3. PRET + CERERE SPANIA
# ---------------------------------------------------------------------------
def load_pret_spania() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Incarca datele de generare/cerere/preturi si meteo din Spania.

    Returns:
        (energy_df, weather_df) cu DatetimeIndex naive (timezone eliminat).
    """
    cfg = load_config()
    energy_path = PROJECT_ROOT / cfg["datasets"]["pret_spania"]["energy"]
    weather_path = PROJECT_ROOT / cfg["datasets"]["pret_spania"]["weather"]

    energy_df = pd.read_csv(energy_path)
    energy_df["time"] = pd.to_datetime(energy_df["time"], utc=True).dt.tz_convert(None)
    energy_df = energy_df.set_index("time").sort_index()

    weather_df = pd.read_csv(weather_path)
    weather_df["dt_iso"] = pd.to_datetime(weather_df["dt_iso"], utc=True).dt.tz_convert(None)
    weather_df = weather_df.set_index("dt_iso").sort_index()

    return energy_df, weather_df


def merge_spania(
    energy: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
    city: str = "Madrid",
) -> pd.DataFrame:
    """
    Combina datele energetice cu meteo dintr-un singur oras.

    Args:
        city: orasul ales (Madrid, Barcelona, Valencia, Seville, Bilbao).

    Returns:
        DataFrame cu DatetimeIndex orar.
    """
    if energy is None or weather is None:
        energy, weather = load_pret_spania()

    # Pastram doar orasul ales
    weather_city = weather[weather["city_name"] == city].copy()

    # Eliminam duplicatele pe timp (apar uneori in dataset)
    weather_city = weather_city[~weather_city.index.duplicated(keep="first")]

    merged = energy.join(weather_city, how="inner")
    return merged


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Consum USA ===")
    c = load_consum_usa()
    print(f"  Shape: {c.shape}, Range: {c.index.min()} -> {c.index.max()}")

    print("=== Solar India (merged) ===")
    s = merge_solar()
    print(f"  Shape: {s.shape}, Range: {s.index.min()} -> {s.index.max()}")
    print(f"  Coloane: {list(s.columns)}")

    print("=== Spania (merged Madrid) ===")
    e = merge_spania(city="Madrid")
    print(f"  Shape: {e.shape}, Range: {e.index.min()} -> {e.index.max()}")
