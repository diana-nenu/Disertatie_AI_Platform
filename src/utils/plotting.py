"""
Functii helper pentru vizualizari consistente in toate notebook-urile EDA.

Stilul este unic: paleta de culori sobra, grila subtila, text fara diacritice.
"""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Paleta proprie pentru disertatie (sobru, prietenos cu print B&W)
PALETA = {
    "primary": "#2E5C8A",      # albastru profund
    "secondary": "#C96442",    # caramiziu (accent)
    "tertiary": "#4A7C59",     # verde olive
    "neutral": "#6B7280",      # gri
    "light": "#E5E7EB",        # gri deschis
}


def setup_style() -> None:
    """Aplica stilul global pentru figurile matplotlib/seaborn."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.figsize": (12, 4),
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def plot_timeseries(
    series: pd.Series,
    title: str = "",
    ylabel: str = "",
    color: str = PALETA["primary"],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Grafic simplu de serie de timp."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(series.index, series.values, color=color, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    return ax


def plot_timeseries_resampled(
    series: pd.Series,
    rule: str = "D",
    title: str = "",
    ylabel: str = "",
) -> plt.Axes:
    """Grafic cu agregare temporala (zilnic, lunar, etc.)."""
    setup_style()
    fig, ax = plt.subplots()
    daily = series.resample(rule).mean()
    ax.plot(daily.index, daily.values, color=PALETA["primary"], linewidth=1.0)
    ax.fill_between(daily.index, daily.values, alpha=0.15, color=PALETA["primary"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
# Distributii
# ---------------------------------------------------------------------------
def plot_distribution(
    series: pd.Series,
    title: str = "",
    xlabel: str = "",
    bins: int = 50,
) -> plt.Axes:
    """Histograma + KDE pentru o distributie."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(series.dropna(), bins=bins, kde=True, color=PALETA["primary"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecventa")
    return ax


# ---------------------------------------------------------------------------
# Patterns sezoniere
# ---------------------------------------------------------------------------
def plot_seasonal_patterns(
    series: pd.Series,
    title: str = "Patterns sezoniere",
) -> tuple[plt.Figure, np.ndarray]:
    """
    Patru subplots: media valorilor pe ora, zi a saptamanii, luna, an.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    # Pe ora (0..23)
    by_hour = series.groupby(series.index.hour).mean()
    axes[0, 0].plot(by_hour.index, by_hour.values, color=PALETA["primary"], marker="o")
    axes[0, 0].set_title("Media pe ora a zilei")
    axes[0, 0].set_xlabel("Ora")

    # Pe zi a saptamanii (0=Luni)
    by_dow = series.groupby(series.index.dayofweek).mean()
    nume_zile = ["Lu", "Ma", "Mi", "Jo", "Vi", "Sa", "Du"]
    axes[0, 1].bar(nume_zile, by_dow.values, color=PALETA["secondary"])
    axes[0, 1].set_title("Media pe zi a saptamanii")

    # Pe luna
    by_month = series.groupby(series.index.month).mean()
    nume_luni = ["I", "F", "M", "A", "M", "I", "I", "A", "S", "O", "N", "D"]
    axes[1, 0].bar(nume_luni, by_month.values, color=PALETA["tertiary"])
    axes[1, 0].set_title("Media pe luna")

    # Pe an
    by_year = series.groupby(series.index.year).mean()
    axes[1, 1].plot(by_year.index, by_year.values, color=PALETA["neutral"], marker="s")
    axes[1, 1].set_title("Media anuala")
    axes[1, 1].set_xlabel("An")

    fig.suptitle(title, fontsize=14, fontweight="600", y=1.02)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Corelatii
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Matrice corelatii",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Axes:
    """Heatmap cu coeficientii de corelatie Pearson."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Valori lipsa
# ---------------------------------------------------------------------------
def plot_missing_values(df: pd.DataFrame, title: str = "Valori lipsa pe coloana") -> plt.Axes:
    """Bar chart cu procent de valori lipsa per coloana."""
    setup_style()
    pct = df.isna().mean().sort_values(ascending=True) * 100
    pct = pct[pct > 0]
    if pct.empty:
        print("Nu exista valori lipsa.")
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(pct))))
    ax.barh(pct.index, pct.values, color=PALETA["secondary"])
    ax.set_xlabel("Procent valori lipsa (%)")
    ax.set_title(title)
    return ax
