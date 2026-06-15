"""
Aplicatie Streamlit - platforma AI integrata pentru suport decizional energetic.

Reuneste cele patru componente ale lucrarii:
  - Analiza datelor (EDA) pe cele 3 seturi energetice
  - Predictii ML (modelele antrenate in Etapa II)
  - Optimizare neliniara prescriptiva (Etapa III)
  - Explicatii in limbaj natural (LLM, Etapa IV)

Design modern (tema 2026): paleta indigo-violet-cyan, font Inter mare, carduri, grafice stilizate,
explicatii bogate si interactivitate.

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
from streamlit_option_menu import option_menu

from src.utils.config_loader import load_config

# --- Paleta de culori (tema 2026) ---
INDIGO, VIOLET, CYAN, AMBER, PINK, GREEN = "#6366F1", "#8B5CF6", "#22D3EE", "#F59E0B", "#EC4899", "#10B981"
INK, MUTED = "#0F172A", "#475569"
COLORWAY = [INDIGO, CYAN, AMBER, PINK, GREEN, VIOLET]

DATASETS = {
    "Productie solara (India)": {
        "key": "solar_india", "parquet": "solar_india_features.parquet",
        "target": "AC_POWER", "unit": "W",
        "comparison": "ml_comparison_india.csv",
        "pred_fig": "fig_7_1_pred_vs_real_india.png", "shap_fig": "fig_7_3_shap_india.png",
        "desc": (
            "Date de la o <b>centrala solara din India</b> (sursa Kaggle), colectate la fiecare ora "
            "timp de 27 de zile. Tinta prezisa este <b>AC_POWER</b> - puterea livrata efectiv in retea. "
            "Predictorii principali sunt <b>fizici</b>: iradierea solara (cat de puternic bate soarele), "
            "temperatura panourilor si a aerului. Provocarea acestui set: este <b>mic</b> (doar 648 de ore) "
            "si productia este <b>zero noaptea</b>, ceea ce cere atentie la interpretarea erorilor."
        ),
        "challenge": "Set mic + relatie fizica directa (soare -> productie).",
    },
    "Consum energetic (USA - PJM)": {
        "key": "consum_usa", "parquet": "consum_usa_features.parquet",
        "target": "PJME_MW", "unit": "MW",
        "comparison": "ml_comparison_usa_databricks.csv",
        "pred_fig": "fig_5_4_predictii_vs_real.png", "shap_fig": "fig_5_2_feature_importance.png",
        "desc": (
            "Consumul orar de energie al retelei <b>PJM</b> din estul Statelor Unite (sursa Kaggle), "
            "pe <b>16 ani</b> de istoric. Tinta este <b>PJME_MW</b> - puterea ceruta de milioane de consumatori. "
            "Seria este puternic <b>ciclica</b>, cu tipare clare de zi, saptamana si anotimp. Fiind cel mai "
            "mare set de date, este ideal pentru modele care invata din istoric (inclusiv retele neuronale)."
        ),
        "challenge": "Serie lunga, foarte ciclica - domina istoricul recent (lag-uri).",
    },
    "Pret energie (Spania)": {
        "key": "pret_spania", "parquet": "pret_spania_features.parquet",
        "target": "price actual", "unit": "EUR/MWh",
        "comparison": "ml_comparison_spania_databricks.csv",
        "pred_fig": "fig_6_1_pred_vs_real_spania.png", "shap_fig": "fig_6_3_shap_summary_spania.png",
        "desc": (
            "Piata de energie din <b>Spania</b> (sursa ENTSOE), 2015-2018, la rezolutie orara. Tinta este "
            "<b>pretul orar</b> al energiei (EUR/MWh). Setul este cel mai <b>bogat</b>: peste 80 de variabile - "
            "28 de surse de generare (eoliana, solara, hidro, fosile etc.), meteo din cinci orase, cererea "
            "totala si pretul anuntat in piata day-ahead. Pretul este <b>volatil</b>, deci cel mai greu de prezis."
        ),
        "challenge": "Multe variabile + volatilitate ridicata a pretului.",
    },
}

SECTION_GUIDE = [
    ("Acasa", "&#127968;", "Prezentarea platformei: scopul, performanta modelelor si ghidul sectiunilor."),
    ("Analiza date (EDA)", "&#128202;", "Explorezi datele brute - cum arata in timp, cum sunt distribuite, ce factori conteaza."),
    ("Predictii ML", "&#129518;", "Vezi cat de precis prezic modelele si care algoritm castiga pe fiecare set de date."),
    ("Optimizare prescriptiva", "&#9889;", "Transformi predictiile in decizii concrete (cand sa incarci bateria) - reglezi parametrii cu sliderele."),
    ("Insight-uri LLM", "&#128172;", "Primesti explicatii in limbaj natural ale rezultatelor, pe intelesul oricui."),
]

# Explicatii despre algoritmii ML (afisate ca expander pe pagina Predictii)
ALGO_INFO = """
- **LinearRegression** - modelul de baza (baseline): traseaza o relatie liniara intre variabile si tinta. Simplu si rapid, dar nu prinde interactiuni complexe.
- **Ridge / Lasso** - regresie liniara cu regularizare (L2 / L1), utile cand sunt multe variabile. **Lasso** elimina automat variabilele inutile (le aduce coeficientul la zero).
- **RandomForest** - o "padure" de sute de arbori de decizie antrenati pe portiuni diferite din date; rezultatul e media lor. Robust si prinde neliniaritati.
- **XGBoost** - *gradient boosting*: construieste arbori succesiv, fiecare corectand greselile celui anterior. De obicei cel mai performant pe date tabelare.
- **LSTM** - retea neuronala recurenta cu "memorie", potrivita pentru serii temporale lungi; are nevoie de mult istoric ca sa invete bine.
- **Optuna / GridSearchCV** - cauta cei mai buni hiperparametri (reglajele fine ale modelului). Optuna foloseste optimizare bayesiana, mai eficienta pe spatii mari.
- **TimeSeriesSplit** - validare care respecta ordinea in timp (antrenezi pe trecut, testezi pe viitor), ca sa nu "trisezi" folosind viitorul.
"""

# Interpretarea rezultatelor per set de date (afisata sub tabelul comparativ)
RESULTS_INSIGHT = {
    "consum_usa": (
        "<b>Castigator: XGBoost</b> (R-patrat 0.998, eroare sub 1%). Descoperirea cheie: valoarea consumului "
        "de acum o ora prezice aproape perfect consumul curent (lag-ul de o ora face peste 90% din decizie) - "
        "consumul agregat al unei retele mari are o inertie naturala. LSTM se descurca bine, dar e mai lent; "
        "modelul Prophet esueaza pe acest tip de serie energetica."
    ),
    "pret_spania": (
        "<b>Castigator: XGBoost optimizat cu Optuna</b> (R-patrat 0.970). Aici reglajul hiperparametrilor aduce "
        "un castig real, iar dominanta lag-ului scade (~48%) - conteaza si sursele de generare si pretul "
        "day-ahead. <b>Lasso</b> a pastrat doar 19 din 78 de variabile, semn de redundanta in date."
    ),
    "solar_india": (
        "Pe R-patrat castiga <b>LinearRegression</b> (0.997), iar pe eroarea procentuala <b>RandomForest</b> "
        "(MAPE 4.9%) - productia solara este aproape liniara in iradiere. <b>Iradierea domina absolut</b> "
        "(confirmare fizica prin SHAP). LSTM esueaza pe setul mic. In prealabil am eliminat variabile cu "
        "'scurgere de informatie' (data leakage), precum DC_POWER, care ar fi falsificat rezultatele."
    ),
}

# Interpretarea graficelor EDA (distributie + corelatie) per set de date
DIST_INSIGHT = {
    "solar_india": "Vezi doua aglomerari: foarte multe valori la <b>zero</b> (noaptea, cand nu se produce energie) "
                   "si un domeniu de valori pozitive (ziua). E o distributie <b>bimodala</b>, tipica productiei solare.",
    "consum_usa": "Valorile se grupeaza in jurul mediei (~32.000 MW), intr-o forma apropiata de un <b>clopot</b> - "
                  "consumul rar coboara foarte jos sau urca foarte sus, semn de stabilitate.",
    "pret_spania": "Preturile sunt concentrate intr-un interval relativ ingust, dar cu cateva <b>valori extreme</b> "
                   "(varfuri rare de pret). Aceasta 'coada' face pretul mai greu de prezis decat consumul.",
}
CORR_INSIGHT = {
    "solar_india": "Atentie: <b>DC_POWER</b> apare cu corelatie 1.0 - este practic aceeasi marime ca tinta, de aceea "
                   "a fost <b>eliminata la modelare</b> (scurgere de informatie / data leakage). Dincolo de ea, conteaza "
                   "predictorii fizici (iradiere, temperatura) si valorile recente.",
    "consum_usa": "Domina <b>lag-ul de o ora</b> (~0.97) - consumul de acum o ora prezice aproape perfect consumul "
                  "curent. De aici si precizia foarte mare a modelelor pe acest set.",
    "pret_spania": "Lag-ul de o ora e cel mai legat de pret, dar <b>mai putin dominant</b> - conteaza si alti factori "
                   "(surse de generare, pretul day-ahead), de aceea problema e mai complexa.",
}

# Detalii tehnice (dropdown pe Acasa): cum s-a ajuns la rezultat, probleme per algoritm, de ce castigatorul
PERF_DETAIL = {
    "consum_usa": """
**Cum am ajuns la rezultat**
145.194 inregistrari orare, 30 features (lag-uri, rolling, encoding ciclic). Split cronologic 70/10/20.
Antrenare *full* pe Databricks (cluster UTM), cu tracking MLflow.

**Probleme intampinate, pe algoritmi, si cum le-am rezolvat**
- **LSTM / Prophet in mod demo**: cu parametri minimi dadeau rezultate foarte slabe (R-patrat negativ). *Rezolvat* prin rulare full pe Databricks (50.000 randuri, 30 epoci) - LSTM a ajuns la R-patrat 0.996.
- **Rularea full blocata local**: in PyCharm kernel-ul murea dupa ~1.5h pe celula GridSearchCV. *Rezolvat* mutand calculul pe Databricks (ne-blocant, multi-core).
- **Prophet a esuat** (R-patrat negativ): modelul aditiv nu poate captura interactiunile neliniare ale unei serii energetice agregate. *Decizie*: exclus pentru acest tip de date.
- **Tuning marginal**: GridSearchCV a adus sub 1% peste XGBoost default - semn ca features-urile sunt deja foarte informative.

**De ce XGBoost_tuned e modelul ales**
R-patrat 0.9975 si MAPE 0.75% (eroare medie ~236 MW pe un consum mediu de ~32.000 MW). E rapid, interpretabil (feature importance / SHAP) si compact. LSTM e valid (R-patrat 0.996) dar mai lent, fara castig. Insight cheie: lag-ul de o ora face ~90% din predictie.
""",
    "pret_spania": """
**Cum am ajuns la rezultat**
34.896 randuri, 78 features (28 surse de generare + meteo + lag-uri). Split cronologic. Rulare full pe Databricks + MLflow.

**Probleme intampinate, pe algoritmi, si cum le-am rezolvat**
- **Kernel crash pe Databricks**: `%pip install` fara versiuni fixate a adus numpy 2.x si protobuf 6.x, incompatibile cu runtime-ul -> kernel-ul nu mai pornea la `restartPython()`. *Rezolvat* fixand versiunile (numpy<2, protobuf<5, tensorflow==2.16.1).
- **Datele lipsa pe cloud**: fisierul parquet e gitignored, deci nu venea prin git. *Rezolvat* urcandu-l manual in Databricks.
- **Multicoliniaritate** (multe din cele 78 features corelate): am adaugat **Ridge** si **Lasso**. Lasso a pastrat doar 19/78 features fara pierdere mare - confirma redundanta.
- **GridSearch prea lent** pe 78 features: inlocuit cu **Optuna** (optimizare bayesiana + pruning), mult mai eficient.
- **LSTM slab** (R-patrat 0.836): pretul e prea volatil si avem mai putine date decat la USA.

**De ce XGBoost_tuned (Optuna) e modelul ales**
R-patrat 0.970 si MAPE 2.48% - cel mai bun pe toate metricile. Aici tuning-ul a adus un **castig real** (0.962 -> 0.970), spre deosebire de USA, pentru ca surprinde interactiunile complexe dintre generare, cerere si meteo.
""",
    "solar_india": """
**Cum am ajuns la rezultat**
648 randuri (27 zile), predictori fizici (iradiere, temperaturi) + lag-uri. Split cronologic. Rulare locala (set mic, ruleaza in minute).

**Probleme intampinate, pe algoritmi, si cum le-am rezolvat**
- **Data leakage (cea mai importanta problema)**: setul continea `DC_POWER` (corelatie 1.0000 cu tinta) si alte variabile derivate din tinta. Cu ele, orice model dadea R-patrat = 1.0000 *artificial* - nu prezicea, ci "citea raspunsul". *Rezolvat* detectand prin corelatie si eliminand 6 coloane cu scurgere de informatie. Abia apoi rezultatele au devenit reale si interpretabile.
- **Set mic -> LSTM esueaza** (R-patrat ~0.80, MAPE peste 100%, instabil intre rulari). Mentionat ca limitare; pe date putine, arborii sunt alegerea corecta.
- **MAPE nedefinit la zero** (productie zero noaptea): calculat doar pe valorile nenule.
- **Dezacord intre metrici**: LinearRegression e cel mai bun pe R-patrat/RMSE, RandomForest pe MAE/MAPE.

**De ce alegem acest rezultat**
Dupa eliminarea leakage-ului, productia solara este aproape **liniara in iradiere**, deci LinearRegression castiga pe R-patrat (0.997). Pentru acuratete procentuala (MAPE), **RandomForest** e mai bun (4.9%) si e recomandat practic. SHAP confirma fizica (iradierea domina). Transparenta asupra leakage-ului este, in sine, o validare metodologica importanta.
""",
}

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
GITHUB_URL = "https://github.com/diana-nenu/Disertatie_AI_Platform"
GH_SVG = ("<svg width='18' height='18' viewBox='0 0 16 16' aria-hidden='true'><path fill-rule='evenodd' "
          "d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
          "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 "
          "1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59"
          ".82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
          "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
          "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42"
          "-3.58-8-8-8z'></path></svg>")
PAGES = [s[0] for s in SECTION_GUIDE]


# ===========================================================================
# Stilizare (CSS modern, text marit)
# ===========================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
        html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; font-size: 17px; }
        .stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%); }
        .block-container { padding-top: 2rem; max-width: 1200px; }

        /* Text general mai mare */
        .stMarkdown p, .stMarkdown li { font-size: 1.06rem; line-height: 1.7; color: #1E293B; }
        [data-testid="stCaptionContainer"], .stCaption, small { font-size: 0.95rem !important; }

        /* Hero */
        .hero { background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 45%, #22D3EE 100%);
            border-radius: 20px; padding: 32px 38px; color: white;
            box-shadow: 0 12px 30px rgba(99,102,241,0.35); margin-bottom: 22px; }
        .hero h1 { color: white; font-size: 2.1rem; font-weight: 800; margin: 0 0 8px 0; }
        .hero p { color: rgba(255,255,255,0.95); font-size: 1.12rem; line-height: 1.6; margin: 0; }
        .pill { display:inline-block; background: rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.35);
            color:#fff; padding:5px 14px; border-radius:999px; font-size:0.92rem; margin:10px 8px 0 0; }

        /* Card de informare / explicatie */
        .infocard { background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid #22D3EE;
            border-radius:14px; padding:16px 20px; margin:10px 0 16px 0; box-shadow:0 4px 14px rgba(15,23,42,0.05); }
        .infocard b { color:#0F172A; }
        /* Carduri ghid - late, orizontale, cu efect la hover */
        .guidecard2 { display:flex; align-items:center; gap:20px; background:#FFFFFF; border:1px solid #E2E8F0;
            border-radius:16px; padding:18px 26px; margin:14px 0; box-shadow:0 4px 14px rgba(15,23,42,0.05);
            transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
        .guidecard2:hover { transform: scale(1.02); box-shadow:0 12px 28px rgba(99,102,241,0.20);
            border-color:#C7D2FE; }
        .guidecard2 .ic { font-size:2.1rem; flex:0 0 56px; text-align:center; }
        .guidecard2 .txt h4 { margin:0 0 4px 0; color:#4F46E5; font-size:1.2rem; }
        .guidecard2 .txt p { margin:0; color:#475569; font-size:1.05rem; line-height:1.5; }

        /* Flux de etape (secvential, cu sageata) */
        .flow-card { background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid #6366F1;
            border-radius:14px; padding:16px 26px; box-shadow:0 4px 14px rgba(15,23,42,0.05);
            max-width:840px; margin:0 auto; transition: transform .15s ease, box-shadow .15s ease; }
        .flow-card:hover { transform: scale(1.015); box-shadow:0 12px 26px rgba(99,102,241,0.18); }
        .flow-badge { display:inline-block; background:linear-gradient(135deg,#6366F1,#8B5CF6); color:#fff;
            font-weight:700; font-size:0.92rem; letter-spacing:.3px; padding:4px 14px; border-radius:999px;
            margin-right:12px; vertical-align:middle; }
        .flow-h { font-weight:700; color:#0F172A; font-size:1.22rem; vertical-align:middle; }
        .flow-card p { margin:10px 0 0 0; color:#475569; font-size:1.05rem; line-height:1.55; }
        .flow-arrow { text-align:center; color:#8B5CF6; font-size:1.8rem; line-height:1; margin:6px 0; }

        /* Metric cards */
        [data-testid="stMetric"] { background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #6366F1;
            border-radius:14px; padding:16px 18px; box-shadow:0 4px 14px rgba(15,23,42,0.05); }
        [data-testid="stMetricLabel"] { color:#475569; font-weight:600; font-size:1.0rem; }
        [data-testid="stMetricLabel"] p { font-size:1.0rem !important; }
        [data-testid="stMetricValue"] { color:#0F172A; font-weight:800; font-size:1.9rem; }

        /* Section heading */
        h2, h3 { color:#0F172A; font-weight:700; }
        .sec { font-size:1.4rem; font-weight:700; color:#0F172A; margin:22px 0 8px 0;
            padding-left:13px; border-left:5px solid #22D3EE; }

        /* Widget labels mai mari */
        .stSelectbox label, .stSlider label, .stRadio label { font-size:1.05rem !important; font-weight:600; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background:linear-gradient(180deg,#0F172A 0%,#1E293B 100%); }
        section[data-testid="stSidebar"] * { color:#E2E8F0 !important; }
        section[data-testid="stSidebar"] a { color:#A5B4FC !important; }
        section[data-testid="stSidebar"] [role="radiogroup"] label { font-size:1.05rem !important;
            padding:8px 12px; border-radius:10px; margin:2px 0; transition: background .15s ease; }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,0.08); }
        /* Brand + buton GitHub in sidebar */
        .sb-brand { display:flex; align-items:center; gap:12px; padding:4px 2px 12px 2px; }
        .sb-logo { width:46px; height:46px; border-radius:13px; background:linear-gradient(135deg,#6366F1,#22D3EE);
            display:flex; align-items:center; justify-content:center; font-size:1.55rem; flex:0 0 auto;
            box-shadow:0 6px 16px rgba(99,102,241,0.45); }
        .sb-name { font-size:1.45rem; font-weight:800; color:#FFFFFF !important; line-height:1.05; }
        .sb-sub { font-size:0.84rem; color:#94A3B8 !important; line-height:1.3; margin-top:3px; }
        .gh-btn { display:inline-flex; align-items:center; gap:9px; background:rgba(255,255,255,0.10);
            border:1px solid rgba(255,255,255,0.22); color:#E2E8F0 !important; padding:10px 16px; border-radius:11px;
            text-decoration:none !important; font-weight:600; transition: all .15s ease; }
        .gh-btn:hover { background:rgba(255,255,255,0.18); transform:translateY(-1px);
            box-shadow:0 6px 16px rgba(0,0,0,0.3); }
        .gh-btn svg { fill:#E2E8F0; }

        /* Navigare moderna (butoane in loc de radio) */
        .nav-label { color:#64748B !important; font-size:0.78rem; font-weight:700; letter-spacing:1.2px;
            text-transform:uppercase; margin:6px 2px 6px 2px; }
        /* baza: butoane de nav vizibile, cu font modern si text mare */
        section[data-testid="stSidebar"] .stButton > button {
            display:flex !important; justify-content:flex-start !important; align-items:center; width:100%;
            text-align:left; border-radius:13px; padding:13px 18px; font-weight:600; font-size:1.22rem;
            margin:5px 0; transition: all .15s ease;
            font-family:'Space Grotesk','Inter',sans-serif !important;
            background:rgba(255,255,255,0.06) !important; color:#E2E8F0 !important;
            border:1px solid rgba(255,255,255,0.14) !important; box-shadow:0 2px 8px rgba(0,0,0,0.15) !important; }
        section[data-testid="stSidebar"] .stButton > button p {
            font-size:1.22rem !important; font-family:'Space Grotesk','Inter',sans-serif !important; font-weight:600; }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background:rgba(255,255,255,0.13) !important; color:#FFFFFF !important;
            border-color:rgba(165,180,252,0.5) !important; transform:translateX(3px);
            box-shadow:0 6px 16px rgba(0,0,0,0.25) !important; }
        /* elementul activ (primary) - evidentiat cu gradient, mai multe selectoare pt compatibilitate */
        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"],
        section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
            background:linear-gradient(135deg,#6366F1,#8B5CF6) !important; color:#FFFFFF !important;
            border:none !important; box-shadow:0 6px 18px rgba(99,102,241,0.45) !important; transform:none; }

        /* Buttons */
        .stButton > button { background:linear-gradient(135deg,#6366F1,#8B5CF6); color:white; border:none;
            border-radius:10px; padding:9px 20px; font-weight:600; font-size:1.0rem;
            box-shadow:0 4px 12px rgba(99,102,241,0.3); transition:transform .08s ease; }
        .stButton > button:hover { transform:translateY(-1px); color:white; }

        /* Info/success boxes text */
        [data-testid="stAlert"] p { font-size:1.05rem; line-height:1.6; }

        #MainMenu, footer, header [data-testid="stToolbar"] { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, pills: list[str] | None = None) -> None:
    pills_html = "".join(f"<span class='pill'>{p}</span>" for p in (pills or []))
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p>{pills_html}</div>", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)


def infocard(html: str) -> None:
    st.markdown(f"<div class='infocard'>{html}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", colorway=COLORWAY,
        font=dict(family="Inter, sans-serif", color=INK, size=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=44, b=10), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=13)),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zeroline=False, title_font_size=14)
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False, title_font_size=14)
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
        "De la date, la predictie, la decizie optima, la explicatie pe intelesul omului - "
        "patru componente de inteligenta artificiala intr-o singura interfata.",
    )

    st.markdown("**Explica un concept cheie** (click pentru pagina dedicata, cu exemple din lucrare):")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Machine Learning", use_container_width=True):
        st.session_state["concept"] = "ml"; st.rerun()
    if b2.button("Optimizare SLSQP", use_container_width=True):
        st.session_state["concept"] = "opt"; st.rerun()
    if b3.button("LLM flan-t5", use_container_width=True):
        st.session_state["concept"] = "llm"; st.rerun()
    if b4.button("3 seturi de date", use_container_width=True):
        st.session_state["concept"] = "data"; st.rerun()

    infocard(
        "<b>Care este scopul aplicatiei?</b><br>"
        "Aceasta platforma demonstreaza un <b>flux decizional energetic complet</b>. Intai <b>invata</b> din "
        "date istorice sa prezica marimi cheie (consum, pret, productie solara). Apoi foloseste aceste "
        "predictii pentru a <b>recomanda decizii optime</b> (de exemplu, cand sa incarci o baterie ca sa "
        "castigi maxim). In final, <b>explica</b> totul in limbaj natural, astfel incat un operator uman sa "
        "inteleaga si sa aiba incredere in recomandari. Parcurge sectiunile din meniul din stanga pentru a "
        "vedea fiecare pas."
    )

    section("Ce face fiecare sectiune")
    for name, icon, desc in SECTION_GUIDE:
        st.markdown(
            f"<div class='guidecard2'><div class='ic'>{icon}</div>"
            f"<div class='txt'><h4>{name}</h4><p>{desc}</p></div></div>",
            unsafe_allow_html=True,
        )

    section("Performanta modelelor predictive (R-patrat pe test)")
    st.caption("R-patrat masoara cat din variatia realitatii explica modelul: 1.00 = perfect, peste 0.95 = foarte bun.")
    cols = st.columns(3)
    for col, (label, meta) in zip(cols, DATASETS.items()):
        dfc = load_comparison(meta["comparison"])
        if dfc is not None and "r2" in dfc.columns:
            best = dfc.sort_values("r2", ascending=False).iloc[0]
            col.metric(label, f"{best['r2']:.3f}", f"model: {best['model']}")
        else:
            col.metric(label, "n/a")

    st.markdown("**Cum am ajuns la aceste rezultate** - click pe fiecare set pentru detalii tehnice "
                "(metodologie, probleme pe algoritmi, solutii, de ce e cel mai bun):")
    for label, meta in DATASETS.items():
        with st.expander(label):
            st.markdown(PERF_DETAIL.get(meta["key"], "Detalii in capitolele lucrarii."))

    section("Cum este construita platforma")
    st.caption("Cele cinci etape formeaza un flux: fiecare se bazeaza pe rezultatul celei anterioare.")
    stages = [
        ("Etapa I", "Date", "Trei seturi energetice, curatate si imbogatite cu features temporale "
         "(valori din trecut, medii mobile, codificarea orei)."),
        ("Etapa II", "Predictie", "Comparam patru algoritmi (LinearRegression, RandomForest, XGBoost, LSTM) "
         "si alegem cel mai bun pentru fiecare set."),
        ("Etapa III", "Optimizare", "Folosim predictiile pentru decizii optime: dispatch baterie, "
         "load shifting, orientare panouri (SciPy / SLSQP)."),
        ("Etapa IV", "LLM", "Un model de limbaj (flan-t5) explica rezultatele in limbaj natural."),
        ("Etapa V", "Aceasta aplicatie", "Interfata interactiva care reuneste toate componentele."),
    ]
    parts = []
    for i, (badge, title, desc) in enumerate(stages):
        parts.append(f"<div class='flow-card'><span class='flow-badge'>{badge}</span>"
                     f"<span class='flow-h'>{title}</span><p>{desc}</p></div>")
        if i < len(stages) - 1:
            parts.append("<div class='flow-arrow'>&#8595;</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.markdown(
        f"<div class='infocard' style='max-width:840px;margin:18px auto 0 auto;'><b>Cod sursa</b> - "
        f"proiectul complet pe GitHub: <a href='{GITHUB_URL}' target='_blank'>{GITHUB_URL}</a></div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# Pagina 2: Analiza date (EDA)
# ===========================================================================
def page_eda() -> None:
    hero("Analiza exploratorie a datelor",
         "Inainte de a prezice, intelegem datele: cum evolueaza in timp, cum sunt distribuite si ce "
         "factori sunt legati de tinta. Alege un set de date si exploreaza-l interactiv.")
    infocard("<b>Ce faci aici?</b> Selectezi unul dintre cele trei seturi energetice si il explorezi: vezi "
             "evolutia in timp a marimii prezise, distributia valorilor si ce variabile sunt cel mai puternic "
             "corelate cu tinta. Foloseste controalele pentru a ajusta ce si cat afisezi.")

    with st.expander("Ce metode folosim in analiza datelor? (click)"):
        st.markdown(
            "- **Feature engineering temporal**: pe langa datele brute, construim variabile noi care ajuta modelele:\n"
            "  - **lag-uri** (valoarea de acum 1, 24 sau 168 de ore) - surprind dependenta de trecut;\n"
            "  - **rolling features** (medii/deviatii pe ferestre mobile) - surprind tendinta recenta;\n"
            "  - **encoding ciclic** al orei/zilei (sin/cos) - invata ca ora 23 e langa ora 0.\n"
            "- **Corelatia**: masoara cat de legata liniar e o variabila de tinta (de la 0 la 1 in valoare absoluta). "
            "Variabilele puternic corelate sunt candidate bune de predictori.\n"
            "- **Tratarea valorilor lipsa** si **split cronologic** (antrenare pe trecut, test pe viitor) - vezi capitolele 3-4 ale lucrarii."
        )

    label = st.selectbox("Set de date:", list(DATASETS.keys()))
    meta = DATASETS[label]
    try:
        df = load_dataset(meta["parquet"])
    except Exception as e:
        st.error(f"Nu am putut incarca datele: {e}")
        return
    target = meta["target"]

    st.markdown(f"<div class='infocard'>{meta['desc']}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Inregistrari (ore)", f"{len(df):,}")
    c2.metric("Variabile (features)", df.shape[1] - 1)
    c3.metric("Provocare", "vezi mai jos")
    st.caption(f"Provocarea specifica: {meta['challenge']}")

    section(f"Evolutia in timp - {target} ({meta['unit']})")
    max_days = max(2, min(30, len(df) // 24))
    days = st.slider("Cate zile afisam (de la final):", 2, max_days, min(7, max_days))
    sub = df[target].iloc[-days * 24:]
    fig = px.area(x=sub.index, y=sub.values, labels={"x": "Timp", "y": f"{target} ({meta['unit']})"})
    fig.update_traces(line_color=INDIGO, fillcolor="rgba(99,102,241,0.12)")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Observa tiparele care se repeta (zi/noapte, zile lucratoare vs weekend) - exact ce invata modelele.")

    col_a, col_b = st.columns(2)
    with col_a:
        section("Distributia valorilor")
        hist = px.histogram(df, x=target, nbins=50)
        hist.update_traces(marker_color=CYAN)
        st.plotly_chart(style_fig(hist, height=320), use_container_width=True)
    with col_b:
        section("Ce factori sunt legati de tinta?")
        num = df.select_dtypes("number")
        corr = num.corr()[target].drop(target).abs().sort_values(ascending=False).head(10)
        cfig = px.bar(x=corr.values[::-1], y=corr.index[::-1], orientation="h",
                      labels={"x": "Corelatie absoluta cu tinta", "y": ""})
        cfig.update_traces(marker_color=VIOLET)
        st.plotly_chart(style_fig(cfig, height=320), use_container_width=True)

    section("Cum citim aceste doua grafice")
    e1, e2 = st.columns(2)
    with e1:
        infocard(
            "<b>Distributia valorilor (histograma)</b><br>"
            "Imparte intervalul valorilor tintei in 'cosulete' si arata <b>cat de des</b> apare fiecare interval. "
            "Pe axa orizontala sunt valorile posibile, pe cea verticala de cate ori apar. Ne spune in jurul caror "
            "valori se concentreaza datele, daca sunt simetrice sau au valori extreme.<br><br>"
            f"<i>Pe acest set:</i> {DIST_INSIGHT.get(meta['key'], '')}"
        )
    with e2:
        infocard(
            "<b>Factorii legati de tinta (corelatia)</b><br>"
            "Corelatia masoara cat de strans e legata <b>liniar</b> o variabila de marimea prezisa, pe o scala de la "
            "0 (fara legatura) la 1 (legatura perfecta). Barele mai lungi = variabile mai puternic legate de tinta, "
            "deci candidati buni de predictori. <b>Atentie</b>: corelatie nu inseamna mereu cauzalitate.<br><br>"
            f"<i>Pe acest set:</i> {CORR_INSIGHT.get(meta['key'], '')}"
        )


# ===========================================================================
# Pagina 3: Predictii ML
# ===========================================================================
def page_predictions() -> None:
    hero("Predictii - comparatia modelelor",
         "Pentru fiecare set de date am antrenat si comparat mai multi algoritmi. Aici vezi care a castigat "
         "si cat de precis prezice.")
    infocard("<b>Ce faci aici?</b> Alegi un set de date si vezi tabelul cu toti algoritmii comparati, modelul "
             "castigator, si graficul predictii vs valori reale. Mai jos, explicatia metricilor te ajuta sa "
             "interpretezi cifrele.")

    with st.expander("Ce inseamna metricile? (click pentru explicatii)"):
        st.markdown(
            "- **R-patrat (R2)**: cat din variatia realitatii explica modelul. 1.00 = perfect; peste 0.95 = foarte bun.\n"
            "- **RMSE**: eroarea medie (in unitatile tintei), penalizeaza puternic greselile mari.\n"
            "- **MAE**: eroarea medie absoluta - cat greseste in medie, in unitatile tintei.\n"
            "- **MAPE**: eroarea medie procentuala - util pentru a compara seturi diferite (in %)."
        )

    section("Detalii tehnice despre metode")
    ta, tv, tt, tm, ts = st.tabs(["Algoritmi", "Validare", "Tuning", "Metrici", "Explicabilitate (SHAP)"])
    with ta:
        st.markdown(
            "- **LinearRegression (OLS)**: minimizeaza suma patratelor reziduurilor; solutie inchisa prin ecuatiile "
            "normale. Baseline rapid, dar presupune relatie liniara si e sensibil la multicoliniaritate.\n"
            "- **Ridge / Lasso**: adauga regularizare la functia de cost. Ridge (L2): `MSE + lambda*sum(w^2)` - "
            "micsoreaza coeficientii. Lasso (L1): `MSE + lambda*sum(|w|)` - ii aduce pe unii exact la zero "
            "(selectie de variabile). Controleaza compromisul **bias-varianta**.\n"
            "- **RandomForest**: *bagging* - sute de arbori pe esantioane bootstrap + subset aleator de features la "
            "fiecare split; media lor reduce varianta. Robust, prinde neliniaritati, putin sensibil la scalare.\n"
            "- **XGBoost**: *gradient boosting* - arbori aditivi construiti secvential pe gradientul pierderii, cu "
            "**regularizare** pe structura arborilor, **shrinkage** (learning_rate) si subsampling. De obicei "
            "state-of-the-art pe date tabelare.\n"
            "- **LSTM**: retea recurenta cu porti (input/forget/output) care combat *vanishing gradient* si permit "
            "memorie pe termen lung; opereaza pe **secvente** (ferestre de N pasi). Necesita mult istoric si calcul."
        )
    with tv:
        st.markdown(
            "- **Split cronologic**: antrenare pe trecut, test pe viitor. Amestecarea aleatoare (*shuffle*) ar "
            "produce **data leakage** temporal (modelul ar 'vedea' viitorul).\n"
            "- **TimeSeriesSplit**: cross-validation cu ferestre extinse - la fiecare fold antrenezi pe primele K "
            "segmente si testezi pe urmatorul (varianta de **walk-forward**), niciodata invers.\n"
            "- Raportam media +/- deviatia standard a metricilor intre folduri, ca masura a **stabilitatii** modelului."
        )
    with tt:
        st.markdown(
            "- **GridSearchCV**: cauta exhaustiv intr-o grila fixa - simplu, dar explodeaza combinatoric.\n"
            "- **Optuna**: optimizare **bayesiana** cu **Tree-structured Parzen Estimator (TPE)** - modeleaza "
            "distributia hiperparametrilor 'buni' vs 'rai' si esantioneaza inteligent. **MedianPruner** opreste "
            "devreme incercarile slabe, economisind 30-50% din timp.\n"
            "- Tuning-ul se face DOAR pe train/validare (cu TimeSeriesSplit), niciodata pe setul de test."
        )
    with tm:
        st.markdown(
            "- **RMSE** = `sqrt(mean((y-yhat)^2))` - in unitatile tintei; penalizeaza puternic erorile mari.\n"
            "- **MAE** = `mean(|y-yhat|)` - eroarea medie absoluta, robusta la outlieri.\n"
            "- **R-patrat** = `1 - SS_res/SS_tot` - fractiunea de varianta explicata (1=perfect, 0=cat media, <0=mai rau).\n"
            "- **MAPE** = `mean(|y-yhat|/|y|)*100` - eroare procentuala; nedefinita la y=0 (de aceea o calculam doar "
            "pe valorile nenule - relevant la productia solara, zero noaptea)."
        )
    with ts:
        st.markdown(
            "- **Valori Shapley** (teoria jocurilor cooperative, Nobel 2012): impart 'echitabil' contributia fiecarui "
            "feature la o predictie, satisfacand axiome (eficienta, simetrie, jucator nul, aditivitate).\n"
            "- **TreeSHAP**: algoritm exact si eficient (polinomial) pentru modele cu arbori (RandomForest, XGBoost).\n"
            "- **Proprietatea de aditivitate**: suma valorilor SHAP + valoarea de baza = predictia exacta - utila "
            "pentru audit. SHAP ofera atat importanta globala, cat si explicatii **locale** (per predictie)."
        )

    label = st.selectbox("Set de date:", list(DATASETS.keys()))
    meta = DATASETS[label]
    st.markdown(f"<div class='infocard'>{meta['desc']}</div>", unsafe_allow_html=True)

    dfc = load_comparison(meta["comparison"])
    if dfc is not None:
        dfc = dfc.sort_values("r2", ascending=False).reset_index(drop=True)
        best = dfc.iloc[0]
        st.markdown(
            f"Model castigator: <b style='color:#4F46E5; font-size:1.25rem'>{best['model']}</b>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R-patrat", f"{best['r2']:.4f}")
        m2.metric("RMSE", f"{best['rmse']:.2f}")
        m3.metric("MAE", f"{best['mae']:.2f}")
        m4.metric("MAPE", f"{best['mape']:.2f}%")

        section("Tabel comparativ - toti algoritmii")
        cols = [c for c in ["model", "rmse", "mae", "r2", "mape"] if c in dfc.columns]
        st.dataframe(dfc[cols].round(4), use_container_width=True)

        section("Interpretarea rezultatelor")
        infocard(RESULTS_INSIGHT.get(meta["key"], "Rezultate disponibile in tabelul de mai sus."))
    else:
        st.warning("Tabel comparativ indisponibil.")

    fp = FIG_DIR / meta["pred_fig"]
    if fp.exists():
        section("Predictii vs valori reale")
        st.image(str(fp), use_container_width=True)
        st.caption("Cu cat liniile modelelor urmaresc mai fidel linia neagra (realitatea), cu atat predictia e mai buna.")
    sp = FIG_DIR / meta["shap_fig"]
    if sp.exists():
        section("Ce factori conteaza cel mai mult (SHAP / importanta)")
        st.image(str(sp), use_container_width=True)
        st.caption("SHAP arata contributia fiecarui factor la predictie - utila pentru a avea incredere in model.")


# ===========================================================================
# Pagina 4: Optimizare prescriptiva
# ===========================================================================
def page_optimization() -> None:
    from src.optimization.optimizer import BatteryConfig, solve_battery_dispatch
    from src.llm_integration.insights import summarize_optimization

    hero("Optimizare prescriptiva - dispatch baterie",
         "Aici trecem de la 'ce se va intampla' la 'ce sa faci'. Pe baza preturilor energiei, optimizarea "
         "calculeaza planul orar de incarcare/descarcare a unei baterii pentru profit maxim.")
    infocard("<b>Cum folosesti?</b> Regleaza din slidere parametrii bateriei (capacitate, putere, fereastra de "
             "timp). Aplicatia rezolva instant problema de optimizare si iti arata planul de actiune, profitul "
             "estimat si o recomandare in limbaj natural. <b>Ideea</b>: incarci cand pretul e mic, descarci cand e mare.")

    section("Detalii tehnice despre optimizare")
    tf, ts, tl = st.tabs(["Formularea problemei", "Metoda SLSQP", "Liniar vs neliniar"])
    with tf:
        st.markdown(
            "Pentru o fereastra de T ore:\n"
            "- **Variabile de decizie**: `x_t` = puterea la ora t (x>0 = descarcare/vinzi, x<0 = incarcare/cumperi).\n"
            "- **Functia obiectiv** (de maximizat): `profit = sum(pret_t * x_t) - lambda * sum(x_t^2)`. Primul termen "
            "e venitul net; al doilea, un cost de **degradare** patratic (uzura + pierderi round-trip).\n"
            "- **Constrangeri**:\n"
            "  - *bounds*: `-p_max <= x_t <= p_max` (putere limitata);\n"
            "  - *inegalitate*: `0 <= SOC_t <= capacitate` la fiecare t, unde `SOC_t = SOC_0 - cumsum(x)`;\n"
            "  - *egalitate*: `sum(x_t) = 0` (ciclic - bateria revine la nivelul initial).\n"
            "- Pentru `scipy.minimize` rezolvam echivalent **minimizarea** lui `-profit`."
        )
    with ts:
        st.markdown(
            "- **SLSQP** = *Sequential Least SQuares Programming*, o metoda de tip **SQP** (programare patratica "
            "secventiala).\n"
            "- La fiecare iteratie aproximeaza problema printr-un **subproblem patratic** (obiectiv patratic + "
            "constrangeri liniarizate) si rezolva acel subproblem pentru directia de pas.\n"
            "- Foloseste **multiplicatorii Lagrange** si conditiile de optimalitate **KKT** (Karush-Kuhn-Tucker) "
            "pentru a trata constrangerile de egalitate si inegalitate.\n"
            "- Converge cand pasul si incalcarea constrangerilor scad sub tolerante - returneaza si un flag de succes."
        )
    with tl:
        st.markdown(
            "- O problema **liniara** are obiectiv si constrangeri liniare (se rezolva cu programare liniara). "
            "Termenul nostru de degradare `lambda*sum(x^2)` este **patratic**, deci problema e **neliniara**.\n"
            "- Concret, e o **problema patratica convexa** (QP convex): obiectiv convex + constrangeri liniare. "
            "Pentru probleme convexe, **optimul local gasit este si global** - deci solutia raportata e cea mai buna.\n"
            "- Convexitatea explica si convergenta rapida (cateva iteratii) observata pe acest set."
        )

    c1, c2, c3 = st.columns(3)
    capacity = c1.slider("Capacitate baterie (MWh)", 2.0, 40.0, 10.0, 1.0)
    p_max = c2.slider("Putere maxima (MW)", 0.5, 10.0, 2.5, 0.5)
    window = c3.slider("Fereastra analizata (ore)", 24, 168, 72, 24)

    try:
        df = load_dataset("pret_spania_features.parquet")
        prices = df["price actual"].iloc[-window:].values.astype(float)
        idx = list(range(window))
        cfg = BatteryConfig(capacity=capacity, p_max=p_max, soc_init=0.5, lambda_deg=0.02, cyclic=True)
        out = solve_battery_dispatch(prices, cfg)
        x, soc, pr = out["x"], out["soc"], out["profit"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Profit net estimat", f"{pr['net_profit']:.0f} EUR")
        m2.metric("Venit brut", f"{pr['revenue']:.0f} EUR")
        m3.metric("Solutie valida", "DA" if out["result"].success else "NU")

        section("Planul de actiune: pret, dispatch si stare de incarcare")
        fig = go.Figure()
        cols = [PINK if v > 0 else CYAN for v in x]
        fig.add_bar(x=idx, y=x, marker_color=cols, name="Dispatch (roz=descarcare/vinzi, cyan=incarcare/cumperi)")
        fig.add_trace(go.Scatter(x=idx, y=soc, name="SOC - cat e plina bateria (MWh)", yaxis="y2",
                                 line=dict(color=INK, width=2.5)))
        fig.add_trace(go.Scatter(x=idx, y=prices, name="Pret (EUR/MWh)", yaxis="y3",
                                 line=dict(color=AMBER, width=1.6, dash="dot")))
        fig.update_layout(
            yaxis=dict(title="Putere (MW)"),
            yaxis2=dict(title="SOC", overlaying="y", side="right", range=[0, capacity], showgrid=False),
            yaxis3=dict(overlaying="y", side="right", position=0.97, showgrid=False, showticklabels=False),
            xaxis=dict(title="Ora"),
        )
        st.plotly_chart(style_fig(fig, height=440), use_container_width=True)
        st.caption("Barele cyan (sub zero) = orele cand bateria se incarca; barele roz (peste zero) = cand se descarca. "
                   "Linia neagra arata cat e de plina, linia portocalie e pretul.")

        section("Recomandarea de operare")
        rec = summarize_optimization(kind="battery", prices=prices, x=x, profit=pr["net_profit"])
        st.success(rec["template"])

        # Rezumat schematic
        n_charge = int((x < -1e-3).sum()); n_discharge = int((x > 1e-3).sum())
        energy = float(np.abs(x[x > 0]).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Ore de incarcare", n_charge)
        k2.metric("Ore de descarcare", n_discharge)
        k3.metric("Energie tranzactionata", f"{energy:.1f} MWh")

        # Tabel orar interactiv (filtrabil)
        actions = np.where(x < -1e-3, "Incarcare", np.where(x > 1e-3, "Descarcare", "Inactiv"))
        plan = pd.DataFrame({
            "Ora": [f"Ziua {i // 24 + 1}, {i % 24:02d}:00" for i in range(len(x))],
            "Pret (EUR/MWh)": np.round(prices, 1),
            "Actiune": actions,
            "Putere (MW)": np.round(x, 2),
        })
        filtru = st.radio("Afiseaza:", ["Toate orele", "Doar incarcare", "Doar descarcare"],
                          horizontal=True)
        if filtru == "Doar incarcare":
            plan = plan[plan["Actiune"] == "Incarcare"]
        elif filtru == "Doar descarcare":
            plan = plan[plan["Actiune"] == "Descarcare"]
        st.dataframe(plan, use_container_width=True, height=340, hide_index=True)
        st.caption("Tabel interactiv: poti sorta dupa orice coloana (click pe antet) sau filtra cu butoanele de mai sus.")
    except Exception as e:
        st.error(f"Eroare la optimizare: {e}")


# ===========================================================================
# Pagina 5: Insight-uri LLM
# ===========================================================================
def page_llm() -> None:
    from src.llm_integration.insights import explain_prediction, llm_generate

    hero("Insight-uri in limbaj natural (NLG)",
         "Ultima componenta traduce rezultatele numerice in limbaj natural, folosind un model de limbaj "
         "de tip transformer (flan-t5). Mai jos: mecanica interna, parametrii de generare si o demonstratie reala.")
    infocard("<b>Ce faci aici?</b> Vezi cum functioneaza tehnic generarea de limbaj natural (NLG), apoi rulezi "
             "pe rezultatele reale ale platformei. Sectiunea e gandita sa fie transparenta inclusiv pentru un "
             "evaluator tehnic - arata arhitectura, tokenizarea, decodarea si prompt-ul efectiv trimis modelului.")

    section("Mecanica modelului de limbaj")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Arhitectura (T5)", "Tokenizare & embeddings", "Decodarea", "Prompt engineering", "Template vs LLM"])
    with t1:
        st.markdown(
            "**flan-t5** este un transformer de tip **encoder-decoder** (seq2seq). Encoder-ul citeste intregul "
            "prompt si construieste o reprezentare contextuala prin **self-attention** (fiecare token isi calculeaza "
            "relevanta fata de toti ceilalti, ponderat). Decoder-ul genereaza raspunsul **autoregresiv** (token cu "
            "token), folosind **cross-attention** catre iesirea encoder-ului plus **masked self-attention** pe ce a "
            "generat deja.\n\n"
            "- **T5** trateaza orice sarcina ca **text-to-text** (intrare text -> iesire text), ceea ce unifica "
            "traducere, sumarizare, QA etc.\n"
            "- **FLAN** = *Fine-tuned LAnguage Net*: T5 antrenat suplimentar pe sute de sarcini formulate ca "
            "instructiuni (**instruction tuning**), de unde capacitatea de a urma cereri in limbaj natural.\n"
            "- Complexitatea atentiei este **O(n^2)** in lungimea secventei - de aici limitele de contextul maxim."
        )
    with t2:
        st.markdown(
            "Modelul nu vede caractere, ci **tokeni** produsi de un tokenizer **SentencePiece** (subword): cuvintele "
            "rare sunt sparte in bucati (ex. `optimizare` -> `optim` + `izare`), ceea ce reduce vocabularul si "
            "permite tratarea cuvintelor necunoscute.\n\n"
            "- Fiecare token primeste un **embedding** - un vector dens (sute de dimensiuni) invatat, in care "
            "proximitatea geometrica codifica similaritatea semantica.\n"
            "- T5 foloseste **relative position bias** in loc de positional encoding absolut, ceea ce ajuta la "
            "generalizarea pe lungimi diferite.\n"
            "- Vocabularul flan-t5 e centrat pe engleza; romana e acoperita slab la nivel de tokenizare (multi tokeni "
            "per cuvant), o cauza directa a calitatii reduse in romana."
        )
    with t3:
        st.markdown(
            "Generarea e **autoregresiva**: la fiecare pas modelul produce o distributie de probabilitate peste "
            "vocabular (prin **softmax** pe logits) si alege urmatorul token. Strategia de selectie conteaza:\n\n"
            "- **Greedy** (temperature=0): alege mereu tokenul cel mai probabil - determinist, dar poate fi repetitiv.\n"
            "- **Sampling cu temperature**: imparte logits la `T`; `T<1` ascute distributia (mai sigur), `T>1` o "
            "aplatizeaza (mai creativ/riscant).\n"
            "- **top-k / top-p (nucleus)**: restrang esantionarea la cele mai probabile k tokeni, respectiv la masa "
            "de probabilitate p - controleaza diversitatea fara a permite tokeni absurzi.\n"
            "- **beam search**: exploreaza mai multe ipoteze in paralel, util pentru sarcini deterministe.\n"
            "- **max_new_tokens**: limiteaza lungimea raspunsului (si costul computational)."
        )
    with t4:
        st.markdown(
            "Calitatea iesirii depinde de **prompt** (cererea formulata). Construim un prompt cu rol, context si "
            "datele relevante, apoi il dam modelului. Mai jos vezi exact prompt-ul generat pentru rezultatul selectat."
        )
        st.caption("Prompt-ul efectiv apare in sectiunea de demonstratie de mai jos (expander 'Vezi prompt-ul').")
    with t5:
        st.markdown(
            "Platforma combina doua niveluri, cu motivatie tehnica clara:\n\n"
            "- **Generator determinist pe sabloane**: ia direct valorile numerice si completeaza o fraza. Este "
            "**grounded** (nu poate halucina cifre), **determinist** si **reproductibil** - garantii importante "
            "intr-un sistem de suport decizional.\n"
            "- **LLM (flan-t5)**: ofera formulare mai naturala si variata, dar poate **halucina** (inventa) sau "
            "parafraza imprecis. De aceea LLM-ul **rafineaza**, dar adevarul numeric vine din sablon.\n\n"
            "Acest design reflecta o practica reala de **AI responsabil**: folosesti puterea generativa acolo unde "
            "adauga valoare, dar pastrezi un strat verificabil pentru corectitudine si audit."
        )

    section("Demonstratie pe rezultate reale")
    label = st.selectbox("Pentru ce rezultat generam explicatia?", list(DATASETS.keys()))
    meta = DATASETS[label]
    dfc = load_comparison(meta["comparison"])
    if dfc is None:
        st.warning("Rezultate indisponibile.")
        return
    best = dfc.sort_values("r2", ascending=False).iloc[0]
    metrics = {k: float(best[k]) for k in ["rmse", "mae", "r2", "mape"] if k in best}
    top_features = ["valoarea din ora precedenta", "media mobila recenta", "pretul day-ahead"]

    res = explain_prediction(dataset=label, target=meta["target"], metrics=metrics, unit=meta["unit"],
                             top_features=top_features)
    st.markdown("**1. Varianta determinista (grounded, reproductibila):**")
    st.success(res["template"])

    # Reconstruim prompt-ul pentru transparenta tehnica
    metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    prompt = (
        "You are an energy analyst. Explain, for a non-technical reader, this machine learning result. "
        f"Context: predicting {meta['target']} for {label}. Metrics: {metrics_str}. "
        f"Most important features: {', '.join(top_features)}. Give a short, clear interpretation of the model quality."
    )
    with st.expander("Vezi prompt-ul trimis modelului (prompt engineering)"):
        st.code(prompt, language="text")

    st.markdown("**2. Varianta generata de flan-t5** (model de limbaj real; descarca modelul la prima rulare):")
    cpar1, cpar2 = st.columns(2)
    temperature = cpar1.slider("temperature (0 = determinist, >1 = creativ)", 0.0, 1.5, 0.0, 0.1)
    max_tok = cpar2.slider("max_new_tokens (lungimea raspunsului)", 40, 256, 120, 20)
    if st.button("Genereaza cu flan-t5"):
        try:
            from src.llm_integration.insights import get_pipeline
            with st.spinner("Incarc flan-t5 si generez..."):
                pipe = get_pipeline()
                out = llm_generate(prompt, pipe, max_new_tokens=max_tok, temperature=temperature)
            st.info(out or "(fara output)")
            st.caption(f"Parametri: temperature={temperature}, max_new_tokens={max_tok}. "
                       "Observa diferenta fata de varianta determinista - LLM-ul reformuleaza, dar cifrele de "
                       "incredere raman cele din sablon.")
        except Exception as e:
            st.warning(f"flan-t5 indisponibil in acest mediu ({type(e).__name__}). "
                       f"Ruleaza `pip install transformers torch`; prima rulare descarca modelul (~1 GB).")


# ===========================================================================
# Main
# ===========================================================================
# ===========================================================================
# Pagini dedicate de concept (accesate din butoanele de pe Acasa)
# ===========================================================================
def _back_button() -> None:
    if st.button("← Inapoi la Acasa"):
        st.session_state["concept"] = None
        st.rerun()


def concept_ml() -> None:
    _back_button()
    hero("Machine Learning - componenta predictiva",
         "Algoritmi care invata tipare din date istorice pentru a prezice marimi viitoare (consum, pret, productie).")
    section("Ce este")
    st.markdown(
        "Machine Learning inseamna a invata o functie din exemple, fara a o programa explicit. Aici rezolvam o "
        "problema de **regresie supervizata**: date fiind features-uri (valori din trecut, meteo, ora etc.), "
        "modelul invata sa estimeze o valoare numerica (tinta). Am comparat mai multe familii de algoritmi:")
    st.markdown(ALGO_INFO)
    section("Cum am folosit-o in lucrare")
    st.markdown(
        "- Am antrenat si comparat algoritmii pe **trei seturi de date** energetice (Etapa II).\n"
        "- Validare **cronologica** (TimeSeriesSplit) - antrenez pe trecut, testez pe viitor, fara data leakage.\n"
        "- Pentru fiecare set am ales **modelul castigator** dupa metrici (R-patrat, RMSE, MAE, MAPE) si l-am salvat.\n"
        "- Am folosit **SHAP** pentru a intelege ce factori conteaza in predictii (explicabilitate).")
    section("Exemple concrete din rezultate")
    c1, c2, c3 = st.columns(3)
    c1.metric("Consum USA", "R2 = 0.998", "XGBoost, MAPE 0.75%")
    c2.metric("Pret Spania", "R2 = 0.970", "XGBoost+Optuna")
    c3.metric("Solar India", "R2 = 0.997", "LinearReg / RandomForest")
    st.markdown("Exemplu de cod folosit pentru antrenarea modelului castigator (XGBoost):")
    st.code("from src.ml_models.predictors import train_xgboost, evaluate\n"
            "model = train_xgboost(X_train, y_train, n_estimators=300, max_depth=6)\n"
            "metrici = evaluate(y_test, model.predict(X_test))  # rmse, mae, r2, mape", language="python")
    infocard("Detaliile complete sunt in <b>Capitolele 5, 6 si 7</b> ale lucrarii si in notebook-urile 05-07.")


def concept_opt() -> None:
    _back_button()
    hero("Optimizare neliniara (SLSQP) - componenta prescriptiva",
         "Trece de la 'ce se va intampla' la 'ce sa faci': calculeaza decizia optima sub constrangeri fizice.")
    section("Ce este")
    st.markdown(
        "Optimizarea cauta valorile **variabilelor de decizie** care maximizeaza/minimizeaza o **functie obiectiv**, "
        "respectand **constrangeri**. Cand obiectivul sau constrangerile sunt neliniare, folosim un optimizator "
        "neliniar - aici **SLSQP** (Sequential Least Squares Programming, din SciPy), care rezolva iterativ "
        "subprobleme patratice folosind multiplicatorii Lagrange si conditiile KKT.")
    section("Cum am folosit-o in lucrare")
    st.markdown(
        "Am formulat **trei probleme** de decizie energetica (Etapa III), toate rezolvate cu acelasi motor SLSQP:\n"
        "- **Dispatch baterie (Spania)**: cand sa incarci/descarci pentru profit maxim, pe baza pretului prognozat.\n"
        "- **Load shifting (USA)**: muta consumul flexibil in orele ieftine, sub tarif time-of-use.\n"
        "- **Orientare panouri (India)**: unghiul de inclinare care maximizeaza energia captata.")
    section("Exemplu concret - dispatch baterie")
    st.markdown(
        "Pe o fereastra de 72 de ore, cu o baterie de 10 MWh, optimizarea a gasit un plan cu **profit ~561 EUR**, "
        "respectand toate limitele. Formularea matematica:")
    st.code("# maximizam profitul (minimizam negativul)\n"
            "profit = sum(pret_t * x_t) - lambda * sum(x_t**2)   # termen patratic = degradare (neliniar)\n"
            "# constrangeri: 0 <= SOC_t <= capacitate ; -p_max <= x_t <= p_max ; sum(x_t) = 0 (ciclic)\n"
            "from src.optimization.optimizer import solve_battery_dispatch\n"
            "rezultat = solve_battery_dispatch(preturi, config_baterie)", language="python")
    infocard("Termenul patratic de degradare face problema <b>neliniara</b> (QP convex), justificand SLSQP. "
             "Detalii in <b>Capitolul 8</b> si notebook-urile 08-10.")


def concept_llm() -> None:
    _back_button()
    hero("Modele de limbaj (LLM flan-t5) - explicatii in limbaj natural",
         "Traduce rezultatele numerice in text inteligibil, folosind un transformer text-to-text open-source.")
    section("Ce este")
    st.markdown(
        "Un **model de limbaj** invata sa genereze text. **flan-t5** (Google) este un **transformer** de tip "
        "encoder-decoder, *text-to-text* si *instruction-tuned* - raspunde la sarcini formulate clar. Functioneaza "
        "pe baza **atentiei** (cantareste relevanta cuvintelor in context), opereaza pe **tokeni** (subword) "
        "transformati in **embeddings**, si genereaza raspunsul **autoregresiv** (token cu token).")
    section("Cum am folosit-o in lucrare")
    st.markdown(
        "In Etapa IV, componenta LLM transforma metricile si recomandarile in explicatii pe intelesul unui operator. "
        "Am folosit o **arhitectura duala**: un generator determinist pe sabloane (corect, reproductibil, *grounded* "
        "- nu poate inventa cifre) plus flan-t5 care **rafineaza** formularea. Asa imbinam corectitudinea cu "
        "naturaletea limbajului - o practica de **AI responsabil**.")
    section("Exemplu concret - explicatie generata")
    st.success("Modelul pentru pretul energiei in Spania prezice pretul orar cu o precizie foarte buna. "
               "Coeficientul R-patrat este 0.970 (explica 97% din variatie), eroarea procentuala medie 2.5%. "
               "Cei mai influenti factori: pretul din ora precedenta, pretul day-ahead si media mobila recenta.")
    st.code("from src.llm_integration.insights import explain_prediction\n"
            "rez = explain_prediction(dataset='Spania', target='pret', metrics=metrici, use_llm=True)\n"
            "print(rez['template'])  # varianta determinista (grounded)\n"
            "print(rez['llm'])       # varianta generata de flan-t5", language="python")
    infocard("Detalii (transformer, tokenizare, decodare, prompt engineering) in <b>Capitolul 9</b> si notebook-ul 11.")


def concept_data() -> None:
    _back_button()
    hero("Cele trei seturi de date energetice",
         "Trei probleme cu naturi diferite, alese pentru a testa aceeasi metodologie cross-domain.")
    section("De ce trei seturi diferite")
    st.markdown(
        "Ipoteza centrala a lucrarii este ca aceeasi metodologie de pipeline AI functioneaza pe probleme energetice "
        "diferite. De aceea am ales trei seturi complementare - o serie lunga ciclica, una bogata si volatila, si "
        "una mica dar fizica:")
    for label, meta in DATASETS.items():
        with st.container(border=True):
            st.markdown(f"#### {label}")
            st.markdown(meta["desc"], unsafe_allow_html=True)
            st.caption(f"Provocarea specifica: {meta['challenge']}")
    infocard("Toate sunt preprocesate identic (feature engineering temporal: lag-uri, rolling, encoding ciclic) - "
             "vezi <b>Capitolele 3 si 4</b> si notebook-urile 01-04.")


CONCEPTS = {"ml": concept_ml, "opt": concept_opt, "llm": concept_llm, "data": concept_data}


def main() -> None:
    cfg = load_config()
    st.set_page_config(page_title=cfg.get("streamlit", {}).get("page_title", "Energy AI"),
                       page_icon="⚡", layout="wide")
    inject_css()

    if "route" not in st.session_state:
        st.session_state["route"] = "Acasa"

    st.sidebar.markdown(
        "<div class='sb-brand'><div class='sb-logo'>&#9889;</div>"
        "<div><div class='sb-name'>EnergyAI</div>"
        "<div class='sb-sub'>Platforma integrata pentru suport decizional prescriptiv in energie</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    concept = st.session_state.get("concept")
    NAV = ["Acasa", "Analiza date (EDA)", "Predictii ML", "Optimizare prescriptiva", "Insight-uri LLM"]
    ICONS = ["house-door", "bar-chart-line", "cpu", "lightning-charge", "chat-square-text"]
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=NAV,
            icons=ICONS,
            default_index=NAV.index(st.session_state["route"]) if st.session_state["route"] in NAV else 0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#A5B4FC", "font-size": "19px"},
                "nav-link": {
                    "font-family": "'Space Grotesk', 'Inter', sans-serif", "font-size": "1.06rem",
                    "font-weight": "600", "color": "#CBD5E1", "padding": "13px 16px",
                    "border-radius": "12px", "margin": "5px 0", "--hover-color": "rgba(255,255,255,0.08)",
                },
                "nav-link-selected": {
                    "background-color": "#6366F1", "color": "#FFFFFF", "font-weight": "700",
                    "box-shadow": "0 6px 18px rgba(99,102,241,0.45)",
                },
            },
            key="nav_menu",
        )
    if selected != st.session_state["route"]:
        st.session_state["route"] = selected
        st.session_state["concept"] = None
    page = st.session_state["route"]
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<a href='{GITHUB_URL}' target='_blank' class='gh-btn'>{GH_SVG}<span>Cod sursa</span></a>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Lucrare de disertatie · Stiinta Datelor si Inteligenta Artificiala")
    st.sidebar.markdown(
        "<div style='color:#E2E8F0; font-size:0.9rem; margin-top:2px;'>"
        "Autor: <b>Nenu Diana Andreea</b></div>",
        unsafe_allow_html=True,
    )

    # Pagina de concept (deschisa din butoanele de pe Acasa) are prioritate
    concept = st.session_state.get("concept")
    if concept and concept in CONCEPTS:
        CONCEPTS[concept]()
        return

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
