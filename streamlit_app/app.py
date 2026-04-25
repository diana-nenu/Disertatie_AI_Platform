"""
Aplicație Streamlit – platformă AI integrată pentru suport decizional energetic.

Rulare:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Adaugă root-ul proiectului în PYTHONPATH ca să putem importa src/...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.utils.config_loader import load_config


def main() -> None:
    cfg = load_config()

    st.set_page_config(
        page_title=cfg["streamlit"]["page_title"],
        layout=cfg["streamlit"]["layout"],
    )

    st.title("⚡ Energy AI – Suport Decizional Prescriptiv")
    st.markdown(
        """
        Această aplicație integrează **ML predictiv**, **optimizare neliniară**
        și **LLM-uri** pentru a oferi recomandări inteligente în domeniul energetic.

        **Seturi de date analizate:**
        - 🌞 Producția panourilor solare (India)
        - 🔌 Consumul energetic orar (USA – rețeaua PJM)
        - 💶 Cererea, generarea și prețurile (Spania)
        """
    )

    st.sidebar.header("Navigare")
    page = st.sidebar.radio(
        "Selectează modulul:",
        [
            "Acasă",
            "Analiză date (EDA)",
            "Predicții ML",
            "Optimizare prescriptivă",
            "Insight-uri LLM",
        ],
    )

    if page == "Acasă":
        show_home(cfg)
    elif page == "Analiză date (EDA)":
        st.info("🚧 În dezvoltare – aici vor apărea grafice EDA pe cele 3 seturi de date.")
    elif page == "Predicții ML":
        st.info("🚧 În dezvoltare – aici vor apărea predicțiile celor 4 algoritmi comparați.")
    elif page == "Optimizare prescriptivă":
        st.info("🚧 În dezvoltare – aici va rula scipy.optimize pentru recomandări.")
    elif page == "Insight-uri LLM":
        st.info("🚧 În dezvoltare – aici LLM-ul va explica rezultatele în limbaj natural.")


def show_home(cfg: dict) -> None:
    st.subheader("Despre proiect")
    st.write(cfg["project"]["description"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Algoritmi ML comparați", len(cfg["ml"]["algorithms"]))
    with col2:
        st.metric("Metrici evaluare", len(cfg["ml"]["metrics"]))
    with col3:
        st.metric("Seturi de date", len(cfg["datasets"]))


if __name__ == "__main__":
    main()
