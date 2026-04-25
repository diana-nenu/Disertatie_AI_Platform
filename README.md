# Disertatie_AI_Platform

**Dezvoltarea unei platforme integrate de inteligență artificială, bazată pe ML predictiv, LLM și optimizare neliniară, pentru suport decizional prescriptiv în cloud**

Lucrare de disertație – Master Data Science and Artificial Intelligence
Autor: Diana Nenu

---

## Descriere

Această platformă integrează patru componente majore pentru suport decizional prescriptiv în domeniul energetic:

1. **ML Predictiv** – Comparație între mai mulți algoritmi (regresie liniară, Random Forest, XGBoost, LSTM) pentru a alege modelul optim pentru fiecare set de date.
2. **Optimizare Neliniară** – Folosind SciPy, recomandări inteligente despre ce ar trebui ajustat pentru a obține rezultatul optim (ex: dispatch baterie, alocare resurse).
3. **Integrare LLM** – HuggingFace Transformers pentru generarea automată de insight-uri și interpretări în limbaj natural ale rezultatelor obținute.
4. **Aplicație Streamlit** – Interfață interactivă cu vizualizări grafice și posibil deployment în cloud.

## Seturi de date utilizate

| Set de date | Țară | Scop |
|-------------|------|------|
| Solar Power Generation Data (Plant 1) | India | Analiza randamentului panourilor solare (producție și senzori) |
| PJME Hourly Energy Consumption | USA | Analiza consumului energetic orar |
| Hourly Energy Demand & Generation | Spania | Analiza prețului pentru luarea deciziilor economice |

## Structura proiectului

```
Disertatie_AI_Platform/
├── data/
│   ├── raw/              # Date brute (CSV originale)
│   ├── processed/        # Date curățate și pregătite pentru modele
│   └── external/         # Date suplimentare (dacă e cazul)
├── notebooks/            # Notebook-uri Jupyter pentru EDA și prototipare
│   ├── 01_eda_consum_usa.ipynb
│   ├── 02_eda_pret_spania.ipynb
│   └── 03_eda_solar_india.ipynb
├── src/                  # Cod sursă modular
│   ├── data_processing/  # ETL, curățare, feature engineering
│   ├── ml_models/        # Antrenare și comparare algoritmi predictivi
│   ├── optimization/     # Optimizare neliniară cu SciPy
│   ├── llm_integration/  # Integrare HuggingFace pentru insight-uri
│   └── utils/            # Funcții utilitare comune
├── streamlit_app/        # Aplicația interactivă Streamlit
│   ├── app.py            # Punct de intrare
│   └── pages/            # Pagini multi-page
├── models/               # Modele antrenate salvate (.pkl, .h5)
├── reports/              # Rapoarte și figuri pentru lucrarea scrisă
│   └── figures/
├── tests/                # Teste unitare
├── docs/                 # Documentație suplimentară
├── requirements.txt
├── config.yaml
├── .gitignore
└── README.md
```

## Pași principali ai lucrării

1. **Analiză exploratorie a datelor** – Înțelegere distribuții, corelații, sezonalitate
2. **Preprocesare** – Tratare valori lipsă, scalare, feature engineering
3. **Modele ML predictive** – Antrenare, validare, comparație metrici (RMSE, MAE, R²)
4. **Optimizare neliniară** – Recomandări prescriptive prin scipy.optimize
5. **Integrare LLM** – Generare automată de insight-uri textuale
6. **Aplicație Streamlit** – Vizualizări interactive și deploy
7. **Lucrare scrisă** – Documentație completă în Microsoft Word

## Instalare

```bash
# Clonează repo
git clone <github-url>
cd Disertatie_AI_Platform

# Creează virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Instalează dependențele
pip install -r requirements.txt
```

## Rulare

```bash
# Aplicația Streamlit
streamlit run streamlit_app/app.py

# Notebook-uri Jupyter
jupyter notebook notebooks/
```

## Status

Proiect în dezvoltare – aprilie 2026.
