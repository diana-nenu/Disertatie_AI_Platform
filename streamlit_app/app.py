"""
Aplicatie Streamlit - platforma AI integrata pentru suport decizional energetic.

Reuneste cele patru componente ale lucrarii:
  - Analiza datelor (EDA) pe cele 3 seturi energetice
  - Predictii ML (modelele antrenate in Etapa II)
  - Optimizare neliniara prescriptiva (Etapa III)
  - Explicatii in limbaj natural (LLM, Etapa IV)

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

# --- Metadate seturi de date (target, fisier, etichete) ---
DATASETS = {
    "Productie solara (India)": {
        "key": "solar_india", "parquet": "solar_india_features.parquet",
        "target": "AC_POWER", "unit": "W",
        "comparison": "ml_comparison_india.csv",
        "pred_fig": "fig_7_1_pred_vs_real_india.png",
        "shap_fig": "fig_7_3_shap_india.png",
    },
    "Consum energetic (USA - PJM)": {
        "key": "consum_usa", "parquet": "consum_usa_features.parquet",
        "target": "PJME_MW", "unit": "MW",
        "comparison": "ml_comparison_usa_databricks.csv",
        "pred_fig": "fig_5_4_predictii_vs_real.png",
        "shap_fig": "fig_5_2_feature_importance.png",
    },
    "Pret energie (Spania)": {
        "key": "pret_spania", "parquet": "pret_spania_features.parquet",
        "target": "price actual", "unit": "EUR/MWh",
        "comparison": "ml_comparison_spania_databricks.csv",
        "pred_fig": "fig_6_1_pred_vs_real_spania.png",
        "shap_fig": "fig_6_3_shap_summary_spania.png",
    },
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
GITHUB_URL = "https://github.com/diana-nenu/Disertatie_AI_Platform"


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
    st.title("Platforma AI pentru suport decizional energetic")
    st.markdown(
        "Platforma integreaza patru componente de inteligenta artificiala pentru "
        "**suport decizional prescriptiv** in domeniul energetic: machine learning predictiv, "
        "optimizare neliniara, modele de limbaj (LLM) si o interfata interactiva."
    )

    st.subheader("Performanta modelelor predictive (R-patrat pe set de test)")
    cols = st.columns(3)
    for col, (label, meta) in zip(cols, DATASETS.items()):
        dfc = load_comparison(meta["comparison"])
        if dfc is not None and "r2" in dfc.columns:
            best = dfc.sort_values("r2", ascending=False).iloc[0]
            col.metric(label, f"R2 = {best['r2']:.3f}", f"model: {best['model']}")
        else:
            col.metric(label, "n/a")

    st.subheader("Arhitectura platformei")
    st.markdown(
        "- **Etapa I - Date**: trei seturi energetice (solar India, consum USA, pret Spania), "
        "preprocesate cu feature engineering temporal (lag-uri, rolling, encoding ciclic).\n"
        "- **Etapa II - Predictie**: comparatie LinearRegression / RandomForest / XGBoost / LSTM, "
        "validare cronologica, explicabilitate SHAP.\n"
        "- **Etapa III - Optimizare**: dispatch baterie, load shifting, orientare panouri (SciPy / SLSQP).\n"
        "- **Etapa IV - LLM**: explicatii in limbaj natural ale rezultatelor (HuggingFace flan-t5).\n"
        "- **Etapa V - Aplicatia de fata**: interfata interactiva care reuneste totul."
    )
    st.markdown(f"Cod sursa: [{GITHUB_URL}]({GITHUB_URL})")
    st.caption(f"{cfg.get('project', {}).get('name', 'Disertatie')} "
               f"v{cfg.get('project', {}).get('version', '')} - {cfg.get('project', {}).get('author', '')}")


# ===========================================================================
# Pagina 2: Analiza date (EDA)
# ===========================================================================
def page_eda() -> None:
    st.title("Analiza exploratorie a datelor")
    label = st.selectbox("Selecteaza setul de date:", list(DATASETS.keys()))
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
    c3.metric("Perioada", f"{df.index.min().date()} - {df.index.max().date()}")

    st.subheader(f"Evolutia temporala - {target} ({meta['unit']})")
    n = min(len(df), 24 * 14)  # max 2 saptamani pentru lizibilitate
    sub = df[target].iloc[-n:]
    fig = px.line(x=sub.index, y=sub.values, labels={"x": "Timp", "y": f"{target} ({meta['unit']})"})
    fig.update_traces(line_color="#2563EB")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distributia tintei")
        hist = px.histogram(df, x=target, nbins=50)
        hist.update_traces(marker_color="#0891B2")
        st.plotly_chart(hist, use_container_width=True)
    with col_b:
        st.subheader("Statistici descriptive")
        st.dataframe(df[target].describe().round(2))


# ===========================================================================
# Pagina 3: Predictii ML
# ===========================================================================
def page_predictions() -> None:
    st.title("Predictii - comparatia modelelor")
    label = st.selectbox("Selecteaza setul de date:", list(DATASETS.keys()))
    meta = DATASETS[label]

    dfc = load_comparison(meta["comparison"])
    if dfc is not None:
        dfc = dfc.sort_values("r2", ascending=False).reset_index(drop=True)
        best = dfc.iloc[0]
        st.success(f"Model castigator: **{best['model']}** "
                   f"(R2 = {best['r2']:.4f}, MAPE = {best['mape']:.2f}%)")
        st.subheader("Tabel comparativ")
        show_cols = [c for c in ["model", "rmse", "mae", "r2", "mape"] if c in dfc.columns]
        st.dataframe(dfc[show_cols].round(4), use_container_width=True)

        st.subheader("Metrici model castigator")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R-patrat", f"{best['r2']:.4f}")
        m2.metric("RMSE", f"{best['rmse']:.2f}")
        m3.metric("MAE", f"{best['mae']:.2f}")
        m4.metric("MAPE", f"{best['mape']:.2f}%")
    else:
        st.warning("Tabelul comparativ nu este disponibil pentru acest set.")

    fig_path = FIG_DIR / meta["pred_fig"]
    if fig_path.exists():
        st.subheader("Predictii vs valori reale")
        st.image(str(fig_path), use_container_width=True)
    shap_path = FIG_DIR / meta["shap_fig"]
    if shap_path.exists():
        st.subheader("Importanta features-urilor / SHAP")
        st.image(str(shap_path), use_container_width=True)


# ===========================================================================
# Pagina 4: Optimizare prescriptiva (dispatch baterie)
# ===========================================================================
def page_optimization() -> None:
    from src.optimization.optimizer import BatteryConfig, solve_battery_dispatch

    st.title("Optimizare prescriptiva - dispatch baterie")
    st.markdown(
        "Pe baza preturilor energiei (Spania), optimizarea neliniara (SLSQP) calculeaza "
        "planul orar de incarcare/descarcare a unei baterii pentru profit maxim."
    )

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

        st.subheader("Pret energie")
        fp = px.line(x=idx, y=prices, labels={"x": "Ora", "y": "Pret (EUR/MWh)"})
        fp.update_traces(line_color="#2563EB")
        st.plotly_chart(fp, use_container_width=True)

        st.subheader("Decizia de dispatch si starea de incarcare (SOC)")
        fig = go.Figure()
        colors = ["#E11D48" if v > 0 else "#0891B2" for v in x]
        fig.add_bar(x=idx, y=x, marker_color=colors, name="Dispatch (rosu=descarcare, albastru=incarcare)")
        fig.add_trace(go.Scatter(x=idx, y=soc, name="SOC (MWh)", yaxis="y2",
                                 line=dict(color="black", width=2)))
        fig.update_layout(
            yaxis=dict(title="Putere (MW)"),
            yaxis2=dict(title="SOC (MWh)", overlaying="y", side="right", range=[0, capacity]),
            xaxis=dict(title="Ora"), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recomandare in limbaj natural
        from src.llm_integration.insights import summarize_optimization
        rec = summarize_optimization(kind="battery", prices=prices, x=x, profit=pr["net_profit"])
        st.subheader("Recomandare (limbaj natural)")
        st.info(rec["template"])
        st.session_state["last_opt_profit"] = pr["net_profit"]
    except Exception as e:
        st.error(f"Eroare la optimizare: {e}")


# ===========================================================================
# Pagina 5: Insight-uri LLM
# ===========================================================================
def page_llm() -> None:
    from src.llm_integration.insights import explain_prediction, summarize_optimization

    st.title("Insight-uri in limbaj natural")
    st.markdown(
        "Componenta de limbaj natural traduce rezultatele numerice in explicatii inteligibile. "
        "Varianta determinista ruleaza instant; flan-t5 (HuggingFace) poate rafina formularea."
    )

    label = st.selectbox("Pentru ce rezultat generam explicatia?", list(DATASETS.keys()))
    meta = DATASETS[label]
    dfc = load_comparison(meta["comparison"])
    if dfc is None:
        st.warning("Rezultate indisponibile.")
        return
    best = dfc.sort_values("r2", ascending=False).iloc[0]
    metrics = {k: float(best[k]) for k in ["rmse", "mae", "r2", "mape"] if k in best}

    res = explain_prediction(
        dataset=label, target=meta["target"], metrics=metrics, unit=meta["unit"],
    )
    st.subheader("Explicatie generata (determinist)")
    st.info(res["template"])

    with st.expander("Optional: rafineaza cu flan-t5 (descarca modelul la prima rulare)"):
        if st.button("Ruleaza flan-t5"):
            try:
                from src.llm_integration.insights import get_pipeline
                with st.spinner("Incarc flan-t5..."):
                    pipe = get_pipeline()
                    res = explain_prediction(
                        dataset=label, target=meta["target"], metrics=metrics,
                        unit=meta["unit"], pipe=pipe, use_llm=True,
                    )
                st.success(res.get("llm", "(fara output)"))
            except Exception as e:
                st.warning(f"flan-t5 indisponibil ({type(e).__name__}). "
                           f"Instaleaza transformers + torch. Varianta determinista ramane disponibila mai sus.")


# ===========================================================================
# Main
# ===========================================================================
PAGES = ["Acasa", "Analiza date (EDA)", "Predictii ML", "Optimizare prescriptiva", "Insight-uri LLM"]


def _default_page_index() -> int:
    """Permite selectarea paginii din URL (?page=...), util pentru navigare directa."""
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
        layout=cfg.get("streamlit", {}).get("layout", "wide"),
    )
    st.sidebar.title("Energy AI")
    st.sidebar.caption("Suport decizional prescriptiv")
    page = st.sidebar.radio("Navigare:", PAGES, index=_default_page_index())
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"[Cod sursa GitHub]({GITHUB_URL})")

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
