"""
Aplicatie Streamlit - platforma AI integrata pentru suport decizional energetic.

Reuneste cele patru componente ale lucrarii:
  - Analiza datelor (EDA) pe cele 3 seturi energetice
  - Predictii ML (modelele antrenate in Etapa II)
  - Optimizare neliniara prescriptiva (Etapa III)
  - Explicatii in limbaj natural (LLM, Etapa IV)

Design modern (tema 2026): paleta indigo-violet-cyan, font Inter, carduri, grafice stilizate.

Rulare:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config_loader import load_config

# --- Paleta de culori (tema 2026) ---
INDIGO = "#6366F1"
VIOLET = "#8B5CF6"
CYAN = "#22D3EE"
AMBER = "#F59E0B"
PINK = "#EC4899"
GREEN = "#10B981"
INK = "#0F172A"
MUTED = "#64748B"
COLORWAY = [INDIGO, CYAN, AMBER, PINK, GREEN, VIOLET]

DATASETS = {
    "Productie solara (India)": {
        "key": "solar_india", "parquet": "solar_india_features.parquet",
        "target": "AC_POWER", "unit": "W", "icon": "&#9728;",
        "comparison": "ml_comparison_india.csv",
        "pred_fig": "fig_7_1_pred_vs_real_india.png", "shap_fig": "fig_7_3_shap_india.png",
    },
    "Consum energetic (USA - PJM)": {
        "key": "consum_usa", "parquet": "consum_usa_features.parquet",
        "target": "PJME_MW", "unit": "MW", "icon": "&#128268;",
        "comparison": "ml_comparison_usa_databricks.csv",
        "pred_fig": "fig_5_4_predictii_vs_real.png", "shap_fig": "fig_5_2_feature_importance.png",
    },
    "Pret energie (Spania)": {
        "key": "pret_spania", "parquet": "pret_spania_features.parquet",
        "target": "price actual", "unit": "EUR/MWh", "icon": "&#128176;",
        "comparison": "ml_comparison_spania_databricks.csv",
        "pred_fig": "fig_6_1_pred_vs_real_spania.png", "shap_fig": "fig_6_3_shap_summary_spania.png",
    },
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
GITHUB_URL = "https://github.com/diana-nenu/Disertatie_AI_Platform"
PAGES = ["Acasa", "Analiza date (EDA)", "Predictii ML", "Optimizare prescriptiva", "Insight-uri LLM"]


# ===========================================================================
# Stilizare (CSS modern)
# ===========================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%); }
        .block-container { padding-top: 2rem; max-width: 1200px; }

        /* Hero */
        .hero {
            background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 45%, #22D3EE 100%);
            border-radius: 20px; padding: 30px 36px; color: white;
            box-shadow: 0 12px 30px rgba(99,102,241,0.35); margin-bottom: 24px;
        }
        .hero h1 { color: white; font-size: 2.0rem; font-weight: 800; margin: 0 0 6px 0; }
        .hero p { color: rgba(255,255,255,0.92); font-size: 1.02rem; margin: 0; }
        .pill { display:inline-block; background: rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.35);
            color:#fff; padding:4px 12px; border-radius:999px; font-size:0.8rem; margin:8px 8px 0 0; }

        /* Metric cards */
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #6366F1;
            border-radius: 14px; padding: 16px 18px; box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        }
        [data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }
        [data-testid="stMetricValue"] { color: #0F172A; font-weight: 800; }

        /* Section heading */
        h2, h3 { color: #0F172A; font-weight: 700; }
        .sec { font-size: 1.25rem; font-weight: 700; color:#0F172A; margin: 18px 0 6px 0;
            padding-left: 12px; border-left: 4px solid #22D3EE; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        }
        section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
        section[data-testid="stSidebar"] a { color: #A5B4FC !important; }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white; border: none;
            border-radius: 10px; padding: 8px 18px; font-weight: 600;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3); transition: transform .08s ease;
        }
        .stButton > button:hover { transform: translateY(-1px); color: white; }

        /* Hide default chrome */
        #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pills_html = "".join(f"<span class='pill'>{p}</span>" for p in (pills or []))
    st.markdown(
        f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p>{pills_html}</div>",
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", colorway=COLORWAY,
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_dataset(parquet_name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / parquet_name)


@st.cache_data(show_spinner=False)
def load_comparison(csv_name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / csv_name
    return pd.read_csv(path) if path.exists() else None


# ===========================================================================
# Pagina 1: Acasa
# ===========================================================================
def page_home(cfg: dict) -> None:
    hero(
        "Platforma AI pentru suport decizional energetic",
        "Patru componente AI integrate: predictie (ML), optimizare neliniara, "
        "explicatii in limbaj natural (LLM) si o interfata interactiva.",
        pills=["Machine Learning", "Optimizare SLSQP", "LLM flan-t5", "3 seturi de date"],
    )

    section("Performanta modelelor predictive (R-patrat pe test)")
    cols = st.columns(3)
    for col, (label, meta) in zip(cols, DATASETS.items()):
        dfc = load_comparison(meta["comparison"])
        if dfc is not None and "r2" in dfc.columns:
            best = dfc.sort_values("r2", ascending=False).iloc[0]
            col.metric(label, f"{best['r2']:.3f}", f"{best['model']}")
        else:
            col.metric(label, "n/a")

    section("Arhitectura platformei")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Etapa I - Date**  \nTrei seturi energetice, feature engineering temporal "
                        "(lag-uri, rolling, encoding ciclic).")
        with st.container(border=True):
            st.markdown("**Etapa II - Predictie**  \nLinearRegression / RandomForest / XGBoost / LSTM, "
                        "validare cronologica, explicabilitate SHAP.")
        with st.container(border=True):
            st.markdown("**Etapa III - Optimizare**  \nDispatch baterie, load shifting, orientare panouri (SciPy / SLSQP).")
    with c2:
        with st.container(border=True):
            st.markdown("**Etapa IV - LLM**  \nExplicatii in limbaj natural ale rezultatelor (HuggingFace flan-t5).")
        with st.container(border=True):
            st.markdown("**Etapa V - Aplicatia de fata**  \nInterfata interactiva care reuneste totul.")
        with st.container(border=True):
            st.markdown(f"**Cod sursa**  \n[{GITHUB_URL}]({GITHUB_URL})")


# ===========================================================================
# Pagina 2: Analiza date (EDA)
# ===========================================================================
def page_eda() -> None:
    hero("Analiza exploratorie a datelor",
         "Exploreaza interactiv cele trei seturi energetice: evolutie temporala, distributii, statistici.")
    label = st.selectbox("Set de date:", list(DATASETS.keys()))
    meta = DATASETS[label]
    try:
        df = load_dataset(meta["parquet"])
    except Exception as e:
        st.error(f"Nu am putut incarca datele: {e}")
        return
    target = meta["target"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Inregistrari", f"{len(df):,}")
    c2.metric("Features", df.shape[1] - 1)
    c3.metric("Perioada", f"{df.index.min().date()}")

    section(f"Evolutia temporala - {target} ({meta['unit']})")
    n = min(len(df), 24 * 14)
    sub = df[target].iloc[-n:]
    fig = px.area(x=sub.index, y=sub.values, labels={"x": "Timp", "y": f"{target}"})
    fig.update_traces(line_color=INDIGO, fillcolor="rgba(99,102,241,0.12)")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        section("Distributia tintei")
        hist = px.histogram(df, x=target, nbins=50)
        hist.update_traces(marker_color=CYAN)
        st.plotly_chart(style_fig(hist, height=300), use_container_width=True)
    with col_b:
        section("Statistici")
        st.dataframe(df[target].describe().round(2), use_container_width=True)


# ===========================================================================
# Pagina 3: Predictii ML
# ===========================================================================
def page_predictions() -> None:
    hero("Predictii - comparatia modelelor",
         "Rezultatele modelelor antrenate (Etapa II): metrici, model castigator, predictii vs real.")
    label = st.selectbox("Set de date:", list(DATASETS.keys()))
    meta = DATASETS[label]

    dfc = load_comparison(meta["comparison"])
    if dfc is not None:
        dfc = dfc.sort_values("r2", ascending=False).reset_index(drop=True)
        best = dfc.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model castigator", best["model"])
        m2.metric("R-patrat", f"{best['r2']:.4f}")
        m3.metric("RMSE", f"{best['rmse']:.2f}")
        m4.metric("MAPE", f"{best['mape']:.2f}%")

        section("Tabel comparativ")
        cols = [c for c in ["model", "rmse", "mae", "r2", "mape"] if c in dfc.columns]
        st.dataframe(dfc[cols].round(4), use_container_width=True)
    else:
        st.warning("Tabel comparativ indisponibil.")

    fp = FIG_DIR / meta["pred_fig"]
    if fp.exists():
        section("Predictii vs valori reale")
        st.image(str(fp), use_container_width=True)
    sp = FIG_DIR / meta["shap_fig"]
    if sp.exists():
        section("Importanta features-urilor / SHAP")
        st.image(str(sp), use_container_width=True)


# ===========================================================================
# Pagina 4: Optimizare prescriptiva
# ===========================================================================
def page_optimization() -> None:
    from src.optimization.optimizer import BatteryConfig, solve_battery_dispatch
    from src.llm_integration.insights import summarize_optimization

    hero("Optimizare prescriptiva - dispatch baterie",
         "Optimizare neliniara (SLSQP) pe preturile Spania: planul orar de incarcare/descarcare pentru profit maxim.")

    c1, c2, c3 = st.columns(3)
    capacity = c1.slider("Capacitate baterie (MWh)", 2.0, 40.0, 10.0, 1.0)
    p_max = c2.slider("Putere maxima (MW)", 0.5, 10.0, 2.5, 0.5)
    window = c3.slider("Fereastra (ore)", 24, 168, 72, 24)

    try:
        df = load_dataset("pret_spania_features.parquet")
        prices = df["price actual"].iloc[-window:].values.astype(float)
        idx = list(range(window))
        cfg = BatteryConfig(capacity=capacity, p_max=p_max, soc_init=0.5, lambda_deg=0.02, cyclic=True)
        out = solve_battery_dispatch(prices, cfg)
        x, soc, pr = out["x"], out["soc"], out["profit"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Profit net", f"{pr['net_profit']:.0f} EUR")
        m2.metric("Venit brut", f"{pr['revenue']:.0f} EUR")
        m3.metric("Convergenta", "DA" if out["result"].success else "NU")

        section("Pret energie si decizia de dispatch")
        fig = go.Figure()
        cols = [PINK if v > 0 else CYAN for v in x]
        fig.add_bar(x=idx, y=x, marker_color=cols, name="Dispatch (roz=descarcare, cyan=incarcare)")
        fig.add_trace(go.Scatter(x=idx, y=soc, name="SOC (MWh)", yaxis="y2",
                                 line=dict(color=INK, width=2.5)))
        fig.add_trace(go.Scatter(x=idx, y=prices, name="Pret (EUR/MWh)", yaxis="y3",
                                 line=dict(color=AMBER, width=1.5, dash="dot")))
        fig.update_layout(
            yaxis=dict(title="Putere (MW)"),
            yaxis2=dict(title="SOC", overlaying="y", side="right", range=[0, capacity], showgrid=False),
            yaxis3=dict(overlaying="y", side="right", position=0.97, showgrid=False, showticklabels=False),
            xaxis=dict(title="Ora"),
        )
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

        rec = summarize_optimization(kind="battery", prices=prices, x=x, profit=pr["net_profit"])
        section("Recomandare (limbaj natural)")
        st.info(rec["template"])
    except Exception as e:
        st.error(f"Eroare la optimizare: {e}")


# ===========================================================================
# Pagina 5: Insight-uri LLM
# ===========================================================================
def page_llm() -> None:
    from src.llm_integration.insights import explain_prediction

    hero("Insight-uri in limbaj natural",
         "Componenta LLM traduce rezultatele numerice in explicatii inteligibile. "
         "Varianta determinista ruleaza instant; flan-t5 poate rafina formularea.")

    label = st.selectbox("Pentru ce rezultat generam explicatia?", list(DATASETS.keys()))
    meta = DATASETS[label]
    dfc = load_comparison(meta["comparison"])
    if dfc is None:
        st.warning("Rezultate indisponibile.")
        return
    best = dfc.sort_values("r2", ascending=False).iloc[0]
    metrics = {k: float(best[k]) for k in ["rmse", "mae", "r2", "mape"] if k in best}

    res = explain_prediction(dataset=label, target=meta["target"], metrics=metrics, unit=meta["unit"])
    section("Explicatie generata (determinist)")
    st.success(res["template"])

    with st.expander("Optional: rafineaza cu flan-t5 (descarca modelul la prima rulare)"):
        if st.button("Ruleaza flan-t5"):
            try:
                from src.llm_integration.insights import get_pipeline
                with st.spinner("Incarc flan-t5..."):
                    pipe = get_pipeline()
                    res2 = explain_prediction(dataset=label, target=meta["target"], metrics=metrics,
                                              unit=meta["unit"], pipe=pipe, use_llm=True)
                st.info(res2.get("llm", "(fara output)"))
            except Exception as e:
                st.warning(f"flan-t5 indisponibil ({type(e).__name__}). Instaleaza transformers + torch.")


# ===========================================================================
# Main
# ===========================================================================
def _default_page_index() -> int:
    try:
        val = st.query_params.get("page")
    except Exception:
        try:
            val = st.experimental_get_query_params().get("page", [None])[0]
        except Exception:
            val = None
    if val:
        for i, name in enumerate(PAGES):
            if val.lower() in name.lower():
                return i
    return 0


def main() -> None:
    cfg = load_config()
    st.set_page_config(
        page_title=cfg.get("streamlit", {}).get("page_title", "Energy AI"),
        page_icon="⚡", layout="wide",
    )
    inject_css()

    st.sidebar.markdown("## ⚡ Energy AI")
    st.sidebar.caption("Suport decizional prescriptiv")
    page = st.sidebar.radio("Navigare", PAGES, index=_default_page_index())
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"[Cod sursa GitHub]({GITHUB_URL})")
    st.sidebar.caption(f"{cfg.get('project', {}).get('name', '')} "
                       f"v{cfg.get('project', {}).get('version', '')}")

    if page == "Acasa":
        page_home(cfg)
    elif page == "Analiza date (EDA)":
        page_eda()
    elif page == "Predictii ML":
        page_predictions()
    elif page == "Optimizare prescriptiva":
        page_optimization()
    elif page == "Insight-uri LLM":
        page_llm()


if __name__ == "__main__":
    main()
