"""
Funcții pentru încărcarea celor trei seturi de date ale disertației.

Fiecare funcție returnează un DataFrame pandas pregătit (cu datetime index unde e cazul).
"""

from pathlib import Path

import pandas as pd

from src.utils.config_loader import PROJECT_ROOT, load_config


def load_solar_india() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Încarcă datele de generare și senzori meteo de la centrala solară din India.

    Returns:
        (generation_df, weather_df) – ambele cu DATE_TIME convertit la datetime.
    """
    cfg = load_config()
    gen_path = PROJECT_ROOT / cfg["datasets"]["solar_india"]["generation"]
    weather_path = PROJECT_ROOT / cfg["datasets"]["solar_india"]["weather"]

    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    # TODO: parsare DATE_TIME (formate diferite în cele două fișiere)
    # gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], format="%d-%m-%Y %H:%M")
    # weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"])

    return gen_df, weather_df


def load_consum_usa() -> pd.DataFrame:
    """
    Încarcă datele de consum energetic orar din USA (PJM Interconnection).

    Returns:
        DataFrame cu coloanele Datetime (index), PJME_MW.
    """
    cfg = load_config()
    file_path = PROJECT_ROOT / cfg["datasets"]["consum_usa"]["file"]

    df = pd.read_csv(file_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    return df


def load_pret_spania() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Încarcă datele de cerere/generare/preț și meteo din Spania.

    Returns:
        (energy_df, weather_df) – cu time index pe orele orare.
    """
    cfg = load_config()
    energy_path = PROJECT_ROOT / cfg["datasets"]["pret_spania"]["energy"]
    weather_path = PROJECT_ROOT / cfg["datasets"]["pret_spania"]["weather"]

    energy_df = pd.read_csv(energy_path)
    weather_df = pd.read_csv(weather_path)

    # TODO: parsare time -> datetime, set_index, gestionare timezone
    return energy_df, weather_df


if __name__ == "__main__":
    print("=== Solar India ===")
    g, w = load_solar_india()
    print(f"  Generation: {g.shape}, Weather: {w.shape}")

    print("=== Consum USA ===")
    c = load_consum_usa()
    print(f"  Shape: {c.shape}, Range: {c.index.min()} → {c.index.max()}")

    print("=== Pret Spania ===")
    e, w = load_pret_spania()
    print(f"  Energy: {e.shape}, Weather: {w.shape}")
