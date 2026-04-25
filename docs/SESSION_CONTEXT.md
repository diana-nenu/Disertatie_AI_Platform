# Context sesiune — Disertatie_AI_Platform

Acest fișier conține toate **datele non-secrete** necesare pentru a relua lucrul în
sesiunile viitoare cu Claude. Fără secrete (parole, tokenuri) — acelea trăiesc în macOS Keychain.

---

## Identitate proiect

| Cheie | Valoare |
|-------|---------|
| Nume proiect | `Disertatie_AI_Platform` |
| Path local | `/Users/diana/PycharmProjects/Disertatie_AI_Platform` |
| Path proiect anterior (sursa CSV-uri) | `/Users/diana/PycharmProjects/MasterAN2/Disertatie` |
| Limbă cod & docs | Română (commenturi & docstring-uri) |
| Versiune Python țintă | 3.11+ |
| Virtual env | `.venv` (în root, gitignored) |

## GitHub

| Cheie | Valoare |
|-------|---------|
| Username | `diana-nenu` |
| Repo URL | https://github.com/diana-nenu/Disertatie_AI_Platform |
| Vizibilitate | **privat** |
| Branch principal | `main` |
| Remote name | `origin` |
| Email commit | `diana_nenu@yahoo.com` |
| Nume commit | `Diana Nenu` |

### Autentificare GitHub (unde sunt secretele)

- Token GitHub clasic (PAT) este salvat în **macOS Keychain** sub:
  - host: `github.com`
  - username: `diana-nenu`
  - protocol: `https`
- Scope-uri token: `repo, workflow`
- Verificare keychain (în Terminal):
  ```bash
  printf "protocol=https\nhost=github.com\nusername=diana-nenu\n\n" \
    | git credential-osxkeychain get
  ```
- **Pushurile și pull-urile NU mai cer token** — git îl ia automat din keychain.

### Re-autentificare (dacă tokenul expiră sau e revocat)

1. Generezi PAT nou: https://github.com/settings/tokens/new?description=Disertatie_AI_Platform&scopes=repo,workflow
2. Îl salvezi în keychain:
   ```bash
   printf "protocol=https\nhost=github.com\nusername=diana-nenu\npassword=<TOKEN_NOU>\n\n" \
     | git credential-osxkeychain store
   ```

---

## Tema disertației

> **Dezvoltarea unei platforme integrate de inteligență artificială, bazată pe ML predictiv,
> LLM și optimizare neliniară, pentru suport decizional prescriptiv în cloud**

Master: Data Science and Artificial Intelligence
Autor: Diana Nenu
Termen lucrare scrisă: Microsoft Word (paralel cu codul)

## Stack tehnologic

- **Date & ML**: pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, pmdarima
- **Deep learning**: tensorflow / keras (pentru LSTM)
- **Optimizare neliniară**: scipy.optimize (SLSQP / trust-constr)
- **LLM**: HuggingFace transformers (model de start: `google/flan-t5-base`)
- **Vizualizare**: matplotlib, seaborn, plotly
- **App**: Streamlit (multi-pagină)
- **Cloud (planificat)**: posibil deploy ulterior

## Seturi de date

| Set | Țară | Folder | Fișiere |
|-----|------|--------|---------|
| Solar Power Generation | India | `data/raw/` | `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv` |
| Hourly Energy Consumption | USA | `data/raw/` | `PJME_hourly.csv` |
| Hourly Energy Demand & Generation | Spania | `data/raw/` | `energy_dataset.csv`, `weather_features.csv` |

CSV-urile NU sunt urcate pe GitHub (`.gitignore` exclude `data/raw/*.csv`).
Sunt copiate din proiectul anterior `/Users/diana/PycharmProjects/MasterAN2/Disertatie/CSVs/`.

## Structura proiectului

```
Disertatie_AI_Platform/
├── data/{raw,processed,external}/
├── notebooks/                 # 01_eda_consum_usa, 02_eda_pret_spania, 03_eda_solar_india
├── src/
│   ├── data_processing/       # loader.py
│   ├── ml_models/             # predictors.py
│   ├── optimization/          # optimizer.py
│   ├── llm_integration/       # insights.py
│   └── utils/                 # config_loader.py
├── streamlit_app/app.py
├── tests/, docs/, models/, reports/figures/
├── README.md, requirements.txt, config.yaml, .gitignore, LICENSE, .env.example
```

---

## Pași planificați (roadmap)

1. ✅ Setup proiect (foldere, configs, git, GitHub remote)
2. ⏳ EDA aprofundat pe fiecare set (refactor notebook-urile existente)
3. ⏳ Preprocesare & feature engineering (datetime features, lags, rolling stats)
4. ⏳ Modele ML predictive (LinearReg / RF / XGBoost / LSTM) — comparație + tuning
5. ⏳ Optimizare neliniară SciPy (battery dispatch, load shifting, cost minimization)
6. ⏳ Integrare LLM HuggingFace (explicații în limbaj natural)
7. ⏳ Aplicație Streamlit completă (5 pagini: Acasă, EDA, Predicții, Optimizare, LLM)
8. ⏳ Deploy cloud (opțional)
9. ⏳ Lucrarea scrisă în Word (capitole)

---

## Comenzi rapide pentru sesiunile viitoare

```bash
# Activare virtual env
cd /Users/diana/PycharmProjects/Disertatie_AI_Platform
source .venv/bin/activate

# Update repo (pull înainte să lucrezi)
git pull

# Lansare aplicație Streamlit
streamlit run streamlit_app/app.py

# Jupyter
jupyter notebook notebooks/

# Push modificări
git add -A
git commit -m "mesaj"
git push
```

---

*Ultima actualizare: 26 aprilie 2026 (sesiune setup inițial proiect).*
