"""
Utilitar pentru încărcarea configurării din config.yaml.

Exemplu de utilizare:
    from src.utils.config_loader import load_config
    cfg = load_config()
    print(cfg["datasets"]["solar_india"]["generation"])
"""

from pathlib import Path
from typing import Any

import yaml

# Rădăcina proiectului = două niveluri deasupra acestui fișier (src/utils/ -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Încarcă fișierul YAML de configurare și returnează un dicționar."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Fișierul de configurare nu a fost găsit: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_data_path(relative_path: str) -> Path:
    """Returnează calea absolută către un fișier din folderul data/."""
    return PROJECT_ROOT / relative_path


if __name__ == "__main__":
    cfg = load_config()
    print(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")
    print(f"Datasets disponibile: {list(cfg['datasets'].keys())}")
