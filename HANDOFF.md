# HANDOFF - Stadiu lucrare disertatie

> **Pentru cine:** Claude (sau alt model AI) care preia sesiunea pe acest proiect.
> **Scris de:** Claude Sonnet 4.7, sesiune 26 aprilie 2026.
> **Limba:** romana fara diacritice (conventia proiectului).

---

## 1. Cine este utilizatoarea

**Diana Nenu** - studenta in anul 2 la masterul **Data Science and Artificial Intelligence**.
Email: `diana_nenu@yahoo.com`. Comunica in romana.

**Stilul ei de colaborare:**
- Vrea ca AI-ul sa faca efectiv treaba (sa creeze fisiere, sa faca push pe GitHub, sa rezolve autentificari) - nu doar sa-i spuna ce trebuie facut.
- Mi-a dat tokenul ei GitHub sa-l folosesc direct.
- Apreciaza intrebarile clarificatoare inainte de pasi mari (folosirea AskUserQuestion cu 2-4 variante).
- Pentru lucruri tehnice mai noi (LLM, optimizare neliniara, deploy cloud) prefera explicatii cu pas-cu-pas.

**Reguli de stil cerute explicit:**
- Comentariile, docstring-urile si celulele markdown din notebook-uri: **romana fara diacritice** (a/i/s/t simple, NU a-/a^/i^/s,/t,).
- Explicatii **scurte**, nu paragrafe lungi.
- Numele de variabile / functii / clase pot fi in engleza (conventie tehnica).
- **Diana NU cunoaste in detaliu conceptele tehnice** (lag-uri, rolling, encoding ciclic, MAPE, TimeSeriesSplit, leakage, optimizare neliniara, embeddings LLM etc.) - vrea sa inteleaga complet codul cand citeste. Vezi sectiunea 14 pentru regulile de livrare.

---

## 2. Tema disertatiei

> "Dezvoltarea unei platforme integrate de inteligenta artificiala, bazata pe ML predictiv,
> LLM si optimizare neliniara, pentru suport decizional prescriptiv in cloud."

**Patru componente AI integrate intr-o platforma:**
1. **ML predictiv** - compara mai multi algoritmi (LinearReg, RandomForest, XGBoost, LSTM) si alege cel mai bun pentru fiecare set de date.
2. **Optimizare neliniara** (SciPy) - genereaza recomandari prescriptive: "ce trebuie schimbat pentru rezultat optim".
3. **LLM** (HuggingFace) - genereaza insight-uri si interpretari in limbaj natural ale rezultatelor.
4. **Aplicatie Streamlit** - interfata interactiva cu vizualizari grafice; posibil deploy in cloud.

**In paralel:** lucrare scrisa in Microsoft Word (capitole de la Introducere la Concluzii).

**Domeniul de aplicare:** energetic - cele trei seturi de date acopera generare solara, consum si pret.

---

## 3. Setarile proiectului

### Locatii fizice

| Cheie | Path |
|---|---|
| Proiect curent (PyCharm) | `/Users/diana/PycharmProjects/Disertatie_AI_Platform` |
| Repo GitHub (privat) | https://github.com/diana-nenu/Disertatie_AI_Platform |
| Branch principal | `main` |
| Username GitHub | `diana-nenu` |
| Proiect anterior (sursa CSV-uri) | `/Users/diana/PycharmProjects/MasterAN2/Disertatie` |

### Autentificare GitHub

Tokenul Personal Access Token este salvat in **macOS Keychain** (NU intr-un fisier!) sub:
- protocol: `https`
- host: `github.com`
- username: `diana-nenu`
- scope-uri: `repo, workflow`

**Verificare token:**
```bash
printf "protocol=https\nhost=github.com\nusername=diana-nenu\n\n" | git credential-osxkeychain get
```

**Re-autentificare daca expira / e revocat:**
1. Genereaza nou la https://github.com/settings/tokens/new?description=Disertatie_AI_Platform&scopes=repo,workflow
2. Salveaza in keychain:
```bash
printf "protocol=https\nhost=github.com\nusername=diana-nenu\npassword=<TOKEN_NOU>\n\n" | git credential-osxkeychain store
```

**`git push` simplu functioneaza fara intrebari** - tokenul este luat automat din keychain.

### Stack tehnologic (vezi `requirements.txt`)

- **Date & ML clasic:** pandas, numpy, scikit-learn, xgboost, lightgbm
- **Time series:** statsmodels, pmdarima
- **Deep learning:** tensorflow / keras (pentru LSTM)
- **Optimizare neliniara:** scipy.optimize (SLSQP / trust-constr)
- **LLM:** transformers (HuggingFace), default `google/flan-t5-base`
- **Vizualizare:** matplotlib, seaborn, plotly
- **App:** streamlit
- **Dev:** jupyter, pytest, black, isort

Python target: 3.11+. Virtual env: `.venv` in root (gitignored).

---

## 4. Cele 3 seturi de date

Toate sunt in `data/raw/` (NU sunt urcate pe GitHub - exclude prin `.gitignore`).

| Set | Sursa | Fisiere | Coloane cheie | Periada |
|---|---|---|---|---|
| **Solar India** (Plant 1) | Kaggle | `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv` | DC_POWER, AC_POWER, IRRADIATION, MODULE_TEMPERATURE | 34 zile, 15 min |
| **Consum USA (PJME)** | Kaggle | `PJME_hourly.csv` | PJME_MW (cerere) | 2002-2018, orar |
| **Pret + cerere Spania** | ENTSOE | `energy_dataset.csv`, `weather_features.csv` | price actual, total load actual, 28 surse generare, meteo 5 orase | 2015-2018, orar |

Formate de timestamp **diferite** intre seturi - vezi `loader.py`:
- PJME: `%Y-%m-%d %H:%M:%S`
- Solar India generare: `%d-%m-%Y %H:%M` (dd-mm-yyyy!)
- Solar India meteo: ISO standard
- Spania: ISO 8601 cu timezone (+01:00)

---

## 5. Structura proiectului

```
Disertatie_AI_Platform/
├── data/
│   ├── raw/              # CSV-uri originale (gitignore)
│   ├── processed/        # date procesate (gol)
│   └── external/         # date suplimentare (gol)
├── notebooks/
│   ├── 01_eda_consum_usa.ipynb       # complet (23 celule)
│   ├── 02_eda_pret_spania.ipynb      # complet (28 celule)
│   └── 03_eda_solar_india.ipynb      # complet (23 celule)
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── loader.py                 # FUNCTIONAL: load_consum_usa, load_solar_india, merge_solar, load_pret_spania, merge_spania
│   ├── ml_models/
│   │   ├── __init__.py
│   │   └── predictors.py             # SCHELET: compare_models() cu LinearReg, RF, XGBoost
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── optimizer.py              # SCHELET: optimize_nonlinear() wrapper SciPy + battery_dispatch_problem (TODO)
│   ├── llm_integration/
│   │   ├── __init__.py
│   │   └── insights.py               # SCHELET: explain_prediction(), summarize_optimization()
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py          # FUNCTIONAL: load_config(), PROJECT_ROOT
│       └── plotting.py               # FUNCTIONAL: PALETA, setup_style, plot_timeseries, plot_distribution, plot_seasonal_patterns, plot_correlation_heatmap, plot_missing_values
├── streamlit_app/
│   ├── pages/                        # gol
│   └── app.py                        # SCHELET: 5 pagini (Acasa, EDA, Predictii, Optimizare, LLM)
├── tests/                            # gol (cu __init__.py)
├── docs/
│   ├── SESSION_CONTEXT.md            # detalii setup proiect (path, GitHub, comenzi rapide)
│   └── HANDOFF.md                    # ACEST FISIER
├── reports/figures/
│   └── architecture.svg              # schema vizuala completa a arhitecturii
├── models/                           # gol (gitignore .pkl, .h5, etc.)
├── requirements.txt
├── config.yaml                       # setari centralizate (paths, ML, optim, LLM, Streamlit)
├── .gitignore                        # complet
├── .env.example
├── LICENSE                           # MIT
└── README.md
```

---

## 6. Ce s-a facut in sesiunea anterioara

### Pas 1: Setup proiect (rezolvat)
- Creata structura completa de foldere.
- Copiate cele 5 CSV-uri din proiectul vechi `MasterAN2/Disertatie/CSVs/` in `data/raw/`.
- Copiate cele 3 notebook-uri originale (in versiune draft).
- Creat `README.md`, `requirements.txt`, `.gitignore`, `config.yaml`, `LICENSE`, `.env.example`.

### Pas 2: GitHub (rezolvat)
- Token PAT generat de Diana cu scope `repo, workflow`.
- Repo `diana-nenu/Disertatie_AI_Platform` creat **privat** prin GitHub API.
- Token salvat in macOS Keychain (URL remote curat, fara secrete).
- Branch `main` urcat pe GitHub.

### Pas 3: Schema arhitectura (rezolvat)
- `reports/figures/architecture.svg` - vector, ~940x920px, contine:
  - 3 surse de date
  - Procesare
  - Comparatie 4 algoritmi ML
  - Optimizare neliniara
  - LLM
  - Aplicatie Streamlit (5 pagini)
  - Sidebar cu lucrarea Word + stack tehnologic
- Util pentru insertia in lucrarea Word la capitolul de metodologie.

### Pas 4: EDA notebook-uri rescrise (rezolvat)
- Toate cele 3 notebook-uri rescrise cu structura uniforma:
  1. Titlu + scop
  2. Setup (PYTHONPATH + import-uri)
  3. Incarcare date + interpretare
  4. Statistici descriptive + interpretare
  5. Valori lipsa
  6. Vizualizari (time series, distributii, sezonalitate)
  7. Corelatii
  8. Concluzii + pasi urmatori
- Toate explicatiile in **romana fara diacritice**.
- Markdown si interpretari dupa fiecare grafic.

### Pas 5: Module functionale `src/`
- **`loader.py`** - functional pentru toate cele 3 seturi (parsare timestamp corecta, agregare invertoare solar, selectie oras meteo Spania).
- **`plotting.py`** - paleta unitara (`PALETA`), `setup_style()`, helpers consistenti pentru toate notebook-urile.
- **`config_loader.py`** - incarcare YAML, expune `PROJECT_ROOT`.
- **`predictors.py`, `optimizer.py`, `insights.py`** - **schelete cu TODO-uri**, urmeaza sa fie implementate.

### Pas 6: Memorie persistenta
Salvat in `~/Library/Application Support/Claude/.../memory/`:
- `MEMORY.md` - index
- `user_diana.md` - profil utilizator
- `disertatie_project.md` - detalii proiect
- `github_setup.md` - cum se autentifica pe GitHub

---

## 7. Stadiu commituri

```
33c4145 feat: EDA complet pentru cele 3 seturi de date
5470dd6 docs: schema arhitecturii proiectului (SVG) in reports/figures
bf63354 docs: adauga SESSION_CONTEXT.md cu detalii proiect si setup GitHub
aea11be Initial commit: structura proiectului Disertatie_AI_Platform
```

Toate sunt urcate pe `origin/main`. Repo e curat (`git status` = clean).

---

## 8. Restrictii tehnice descoperite (IMPORTANT pentru sesiunea noua)

### a) Bash sandbox NU poate accesa github.com
`mcp__workspace__bash` are un proxy care blocheaza `api.github.com` cu HTTP 403.
**Solutie:** foloseste `mcp__Control_your_Mac__osascript` cu `do shell script "..."` pentru orice apel curl catre GitHub.

### b) Bash sandbox are restrictii pe operatii git pe foldere mount-uite
`git init`, `git add`, `git commit` din bash sandbox dau "Operation not permitted" pe fisiere temporare in `.git/`.
**Solutie:** ruleaza git prin `osascript` direct pe host.

### c) Stergerea de fisiere in foldere mount-uite necesita aprobare
Daca apare "Operation not permitted" la `rm`, foloseste `mcp__cowork__allow_cowork_file_delete` pentru a cere permisiune.
**Status curent:** permisiunea este deja activa pentru folderul `PycharmProjects`.

### d) Folderele accesibile (mounts)
- `/Users/diana/PycharmProjects` - parinte (toate proiectele Python)
- `/Users/diana/PycharmProjects/MasterAN2` - proiect vechi (sursa CSV-uri)
- `/Users/diana/PycharmProjects/MasterAN2/Disertatie` - subfolder vechi
- `/Users/diana/Documents/Claude/Projects/Disertatie` - folder gol Cowork (NU este folosit)

---

## 9. Ce urmeaza (roadmap propus)

### Faza 1 - Preprocesare si features (urmator pas natural)
- `src/data_processing/preprocessing.py` - functii pentru:
  - Eliminare coloane goale (Spania).
  - Interpolare valori lipsa.
  - Feature engineering temporal: `hour`, `dayofweek`, `month`, `is_weekend`, `is_holiday`.
  - Lag-uri: t-1, t-24, t-168 (saptamana).
  - Rolling means: 3h, 24h, 7zile.
  - Train/test split cronologic (NU shuffle).

### Faza 2 - Modele ML predictive
Pentru fiecare set de date, antreneaza si compara:
- Linear Regression (baseline)
- Random Forest
- XGBoost
- LSTM (cu tensorflow/keras)

Metrici: RMSE, MAE, R^2, MAPE. Evaluare cu `TimeSeriesSplit` pentru cross-validation.

Notebook-uri propuse:
- `notebooks/04_ml_consum_usa.ipynb`
- `notebooks/05_ml_pret_spania.ipynb`
- `notebooks/06_ml_solar_india.ipynb`

Salveaza modelele cele mai bune in `models/` (cu joblib sau h5).

### Faza 3 - Optimizare neliniara
Implementare in `src/optimization/optimizer.py`:
- **Battery dispatch** (Spania): max profit prin charge/discharge in functie de pretul prognozat.
  - Variabile: x_t (charge/discharge la fiecare ora)
  - Obiectiv: max sum(price_t * x_t)
  - Constrangeri: SOC limits, eta charge/discharge, energy balance
- **Load shifting** (USA): minimizare cost prin mutarea sarcinilor flexibile in ore mai ieftine.
- **Eficienta solar** (India): orientare/inclinare optime din panouri (parametri continui cu bounds fizice).

Notebook: `notebooks/07_optimization.ipynb`.

### Faza 4 - Integrare LLM
Implementare in `src/llm_integration/insights.py`:
- `explain_prediction()` - generator de explicatii pentru rezultatele ML.
- `summarize_optimization()` - generator de recomandari prescriptive.
- Model implicit: `google/flan-t5-base` (mic, rapid, cpu).
- Pipeline: input = metrici / rezultate optim → output = paragraf in romana.

Notebook: `notebooks/08_llm_insights.ipynb`.

### Faza 5 - Aplicatia Streamlit
Extinde `streamlit_app/app.py` (acum este schelet) cu cele 5 pagini:
- **Acasa** - overview proiect, link GitHub, KPI-uri.
- **EDA** - selectie set de date + vizualizari interactive (plotly).
- **Predictii** - rulare model salvat pe noi date, comparatie metrici.
- **Optimizare** - input parametri (capacitate baterie, eta), output recomandari + grafic dispatch.
- **Insight LLM** - generare explicatii pentru rezultatele afisate.

Posibil deploy: Streamlit Cloud (gratuit, integrat cu GitHub).

### Faza 6 - Lucrarea scrisa in Word
Capitole sugerate (vezi schema SVG, sidebar dreapta):
1. Introducere
2. Stadiul actual al cunoasterii (literatura)
3. Date si metodologie
4. Modele ML
5. Optimizare neliniara
6. Integrare LLM si insight-uri
7. Aplicatie integrata
8. Concluzii si dezvoltari ulterioare

Schema SVG poate fi inserata direct in capitolul de metodologie.

---

## 10. Comenzi rapide

```bash
# Sesiune noua, deschide proiectul:
cd /Users/diana/PycharmProjects/Disertatie_AI_Platform

# Activare venv (creat de PyCharm)
source .venv/bin/activate

# Pull modificari recente
git pull

# Test loader
python -m src.data_processing.loader

# Test plotting
python -c "from src.utils.plotting import setup_style; setup_style(); print('OK')"

# Lansare Streamlit
streamlit run streamlit_app/app.py

# Jupyter
jupyter notebook notebooks/

# Commit + push
git add -A
git commit -m "..."
git push
```

---

## 11. Ce sa NU faci in sesiunea noua

- **NU** pune secrete in fisiere (`.env`, `config.yaml`, plain text). Tokenul GitHub este in keychain.
- **NU** scrie cu diacritice in cod / notebook-uri. Conventia este romana fara diacritice.
- **NU** rula git din bash sandbox - va da "Operation not permitted". Foloseste osascript.
- **NU** folosi `mcp__workspace__bash` pentru curl catre github.com - foloseste osascript.
- **NU** crea fisiere noi de documentatie / README dincolo de cele cerute explicit.
- **NU** suprascrie notebook-urile fara sa citesti `Read` mai intai (regula tool Write).

---

## 12. Tot ce am discutat (rezumat conversational)

Sesiunea s-a desfasurat in 7 schimburi mari:

1. **"Ai inteles ce trebuie sa facem?"** - i-am facut un rezumat al disertatiei si i-am cerut detalii despre date, GitHub, structura.

2. **"Creeaza un nou proiect in PyCharm si pornesti de acolo"** - am explorat folderul vechi, am creat folderul nou cu structura modulara, am copiat datele si notebook-urile, am inceput sa cer access la `/Users/diana/PycharmProjects` (era nevoie de mount). Am creat README, requirements, gitignore, config.yaml, schelete Python.

3. **"Poti sa-mi accesezi Safari?"** - i-am explicat ce pot si ce nu pot face cu osascript pe Safari, am subliniat ca nu pot face login in numele ei.

4. **"Ia tokenul si fa singur permisiunile"** - i-am explicat ca GitHub cere reconfirmare parola la generare token (security boundary), am propus alternativa cu PAT. I-am deschis pagina de generare token cu parametri pre-completati.

5. Diana mi-a dat un prim token care era invalid (`Bad credentials`), am cerut altul. Al doilea token a functionat (`ghp_OY03u9...`). Am creat repo-ul privat prin GitHub API, am facut push, am salvat tokenul in keychain.

6. **"Salveaza intr-un fisier tot ce ai nevoie pentru conectare"** - am creat `docs/SESSION_CONTEXT.md` (date non-secrete) si memorii persistente (`user_diana.md`, `disertatie_project.md`, `github_setup.md`). Tokenul a ramas DOAR in keychain.

7. **"Vreau sa imi faci o schema a intregului proiect"** - am creat schema vizuala interactiva cu instrumentul show_widget, apoi am salvat-o ca `reports/figures/architecture.svg`.

8. **"Vreau sa lucrezi in proiectul din PyCharm si sa incepi implementarea codului"** - am refacut `loader.py` cu parsare corecta (formate de timestamp diferite intre seturi), am creat `plotting.py` cu paleta unitara, am rescris cele 3 notebook-uri EDA cu structura completa si interpretari in romana fara diacritice. Toate validate sintactic, commitate, urcate.

9. **"Salveaza intr-un fisier tot ce ai facut"** - acest fisier (HANDOFF.md).

---

## 13. Punctul de start pentru sesiunea noua

Cand Diana incepe sesiunea cu Opus, prima ta actiune ar trebui sa fie:

1. **Citeste `docs/SESSION_CONTEXT.md` si acest `HANDOFF.md`.**
2. **Verifica `git log --oneline`** pentru a vedea daca au fost adaugate commituri intre timp.
3. **Intreaba-o ce vrea sa atace** din roadmap (Faza 1, 2, 3 etc.).
4. **Continua in stilul stabilit** - romana fara diacritice, lucruri concrete (nu doar sfaturi).
5. **Citeste sectiunea 14** pentru regulile obligatorii de livrare (.py + notebook + explicatii didactice).

---

## 14. Reguli OBLIGATORII de livrare (cerute explicit de Diana)

### 14.1. Forma livrarii: fisier .py + notebook .ipynb

**TOT proiectul** trebuie facut sub aceasta forma duala:

- **Modul Python (`.py`)** in `src/` - codul "de productie": functii reutilizabile, docstring-uri, logic curat. Acesta este codul pe care il importa Streamlit-ul si pe care il vor citi profesoara/comisia ca structura modulara.
- **Notebook (`.ipynb`)** in `notebooks/` - codul "didactic": demonstreaza modulul `.py` pas cu pas, cu **explicatii intercalate**, vizualizari si interpretari. Acesta este ce va prezenta Diana in fata profesoarei si comisiei.

Niciodata doar unul din cele doua. Daca scrii o functie noua in `.py`, fa si o demonstratie in notebook. Daca explorezi ceva intr-un notebook, refactorizeaza in `.py` cand ajunge stabil.

### 14.2. Structura obligatorie a notebook-urilor

Fiecare notebook trebuie sa aiba:

1. **Header cu titlu si scop** (markdown) - ce face notebook-ul, de ce e important pentru disertatie.
2. **Setup** (markdown + cod) - import-uri, sys.path, incarcare config.
3. **Sectiuni numerotate** cu pattern: markdown explicativ -> celula cod -> markdown interpretare.
4. **Concluzii** (markdown) - ce s-a obtinut, pasul urmator.

### 14.3. Continutul celulelor markdown - reguli pentru explicatii

**Diana NU are background tehnic puternic pe ML/data science.** Cand ii livrezi cod, ea vrea sa-l inteleaga complet citind notebook-ul. Asadar:

**Pentru fiecare concept tehnic folosit, scrie o celula markdown care explica:**
- **Ce este** conceptul (definitie scurta in romana fara diacritice).
- **De ce** e folosit aici (motivatie pentru contextul disertatiei).
- **Cum** functioneaza (intuitie matematica simpla, nu demonstratii formale).
- **Exemple** concrete cand e relevant (numere, scenarii).

**Lista (neexhaustiva) de concepte care TREBUIE explicate cand sunt folosite:**
- features informative, feature engineering
- lag-uri (de ce 1, 24, 168 - de unde apar numerele astea)
- rolling features (medie/deviatie pe fereastra mobila)
- dependentele temporale, sezonalitate (zilnica, saptamanala, anuala)
- encoding ciclic (sin/cos pe ora, zi, luna - de ce e nevoie)
- data leakage (ce e, cum se evita)
- split cronologic vs split aleator (de ce shuffle e gresit pentru time series)
- TimeSeriesSplit (cross-validation pe ferestre de timp)
- metrici regresie: RMSE, MAE, R², MAPE (cum se interpreteaza, ce inseamna valori bune)
- algoritmi ML folositi: ce face fiecare, cand functioneaza bine, hyperparametri principali
  - LinearRegression, RandomForest, XGBoost (gradient boosting), LSTM (retele recurente)
- normalizare / standardizare (StandardScaler, MinMaxScaler)
- one-hot encoding pentru variabile categorice
- optimizare neliniara: ce inseamna, diferenta vs liniara, SLSQP / trust-constr / COBYLA
- functie obiectiv, constrangeri (egalitate, inegalitate, bounds)
- variabile de decizie, lagrangean (intuitie, nu formal)
- pentru LLM: tokeni, embeddings, prompt engineering, transformer (intuitie)
- modele HuggingFace: flan-t5, ce inseamna instruction-tuned
- temperatura, max_new_tokens (efectul lor pe iesire)
- pipeline HuggingFace, generare text-to-text
- pentru Streamlit: session_state, cache, layout columns

**Reguli pentru textul explicatiilor:**
- **Limba: romana fara diacritice** (a/i/s/t simple).
- **Lungime moderata**: nu paragrafe academice, dar nici prea scurt incat sa fie criptic. Scop: Diana sa inteleaga complet la prima citire.
- **Foloseste analogii** cand ajuta (ex. "rolling mean = ca o medie mobila pe ultimele N ore").
- **Adauga dupa fiecare grafic** o celula markdown cu interpretare (ce vede, ce semnifica).
- **NU presupune cunostinte avansate** - explica si lucruri "evidente" pentru un data scientist (ex. de ce facem dropna dupa lag-uri).

### 14.4. Validare: ruleaza notebook-ul end-to-end inainte de a-l livra

Inainte sa zici "gata", **executa notebook-ul** cu `nbclient` / `jupyter nbconvert --execute` ca sa fii sigur ca:
- Toate celulele trec fara eroare.
- Graficele se genereaza corect.
- Outputs-urile sunt salvate in fisier (Diana il deschide si vede rezultate fara sa-l ruleze).

### 14.4.bis. Progress bars la celulele lungi - OBLIGATORIU

**Diana ruleaza notebook-urile pe masina ei in PyCharm**, unde celulele cu LSTM, GridSearchCV, TimeSeriesSplit CV sau orice antrenare pe seturi mari pot dura zeci de minute. Daca celula nu afiseaza nimic in tot acest timp, Diana nu stie daca proiectul ruleaza, daca s-a blocat, sau cat mai are.

**Reguli obligatorii pentru orice celula care depaseste ~30 secunde estimate:**

1. **Print initial cu estimare durata** ("Estimare: ~10-30 min, depinde de CPU/GPU").
2. **Progress per pas** - epoch (Keras `verbose=1`), fold (CV manual cu `print` + timp), antrenare (sklearn / XGBoost `verbose=2`).
3. **Print final cu durata efectiva** ("LSTM antrenat in 12.4 minute").
4. **In modul demo**, progress-urile pot fi tacute (ruleaza rapid oricum); in modul **full**, OBLIGATORIU active.

**Implementare standard (pattern):**

```python
import time as _time
print(f"Antrenare X pe N randuri... estimare ~Y minute\\n")
t = _time.time()
# ... call training cu verbose=1 sau show_progress=True ...
print(f"\\nTerminat in {(_time.time()-t)/60:.1f} minute.")
```

**Pentru CV manual cu fold-uri:** `time_series_cv` din `predictors.py` accepta `show_progress=True` si `label="<nume>"` care printeaza fiecare fold cu timp scurs.

**Pentru Keras LSTM:** `verbose=1` afiseaza bara de progres standard pe fiecare epoch.
**Pentru sklearn GridSearchCV:** `verbose=2` afiseaza fiecare combinatie de parametri si timpul de antrenare.

### 14.5. Salvare iesiri intermediare in `data/processed/`

Notebook-urile care produc DataFrame-uri folosibile mai tarziu (post-preprocesare, predictii, optimizari rezolvate) trebuie sa salveze rezultatul ca **parquet** in `data/processed/` astfel incat notebook-urile urmatoare sa nu fie nevoite sa rezerve pipeline-ul de la zero. Format `parquet` (compact, pastreaza tipuri).

### 14.6. Mapare faze -> fisiere

| Faza | Modul `.py` | Notebook(s) `.ipynb` |
|---|---|---|
| EDA | (folosesc loader, plotting) | `01_eda_consum_usa.ipynb`, `02_eda_pret_spania.ipynb`, `03_eda_solar_india.ipynb` |
| Preprocesare | `src/data_processing/preprocessing.py` | `04_preprocessing.ipynb` |
| ML predictiv | `src/ml_models/predictors.py` (extins) | `05_ml_consum_usa.ipynb`, `06_ml_pret_spania.ipynb`, `07_ml_solar_india.ipynb` |
| Optimizare | `src/optimization/optimizer.py` | `08_optimization_battery.ipynb`, `09_optimization_load_shifting.ipynb`, `10_optimization_solar.ipynb` |
| LLM | `src/llm_integration/insights.py` | `11_llm_insights.ipynb` |
| Streamlit | `streamlit_app/app.py` + pages | (nu are notebook - app live) |

Cifrele se pot ajusta dupa cum cere fluxul; cheia e pattern-ul .py + notebook.

---

## 15. Reguli pentru prezentarea planului fiecarei etape

**Inainte de a incepe orice etapa noua a lucrarii**, AI-ul trebuie sa prezinte Dianei un plan **structurat si detaliat** in stilul folosit de ea (vezi exemplul din sectiunea 16). Diana NU vrea sa intri direct in implementare - vrea sa vada planul, sa-l aprobe sau sa-l modifice, si abia apoi sa atacam.

**Forma obligatorie a planului de etapa:**

1. **Titlu** in formatul `Etapa N: <nume etapa> (<categorie pe scurt, ex. Machine Learning>)` + o propozitie scurta cu scopul global ("Construirea creierului care anticipeaza...", "Integrarea inteligentei lingvistice care explica...", etc.).
2. **Sub-puncte numerotate** - cate unul pentru fiecare model / componenta / livrabil major. Fiecare sub-punct contine:
   - **Bold pe titlul** sub-punctului cu specificarea concreta (set de date, algoritm, target).
   - Una-doua propozitii descriptive cu **specificul** componentei (de ce e o provocare, de ce e potrivit pentru context).
   - Lista marcata cu detalii: algoritmi comparati / framework folosit, asteptare ce va functiona cel mai bine si **de ce**, **concepte cheie ce trebuie explicate didactic** in notebook-uri (Diana citeste si trebuie sa inteleaga complet).
3. **Validare / Metrici / Criterii de succes** - sub-punct dedicat cu criterii cuantificabile (RMSE, MAE, R², MAPE, accuracy, BLEU, latenta, costuri etc.) si protocoale (TimeSeriesSplit, k-fold, holdout etc.).
4. **Livrabile concrete** - lista cu fisierele exacte care vor fi create (.py si .ipynb conform sectiunii 14), tabele de raport, modele salvate.
5. **Estimare efort si optionale** - cate sesiuni estimezi, ce e obligatoriu vs nice-to-have. Mentioneaza limitarile cunoscute (ex.: "LSTM pe set scurt nu va functiona bine").

**Apoi pune in HANDOFF.md** planul detaliat ca sectiune noua (16, 17, 18...). Diana va putea reciti planul oricand si va sti exact ce ai propus.

**Doar dupa aprobarea ei**, ataci implementarea.

---

## 16. Plan Etapa II: Dezvoltarea Modelelor Predictive (Machine Learning)

> Status: **in lucru**. Sesiunea 1 (USA) **finalizata** - rezultate la finalul sectiunii.
> Urmeaza Sesiunea 2 (Spania) si Sesiunea 3 (India).

Construirea "creierului" care anticipeaza variabilele de intrare. Pentru fiecare din cele 3 seturi se compara **mai multi algoritmi** si se alege cel mai potrivit (asa cum cere scopul disertatiei). Validarile sunt cronologice - fara shuffle.

**1. Model Productie Solara (India): cea mai mare provocare** - target `AC_POWER`. Set mic (648 ore), dar bogat in predictori fizici: iradiere, temperatura modul, performance_ratio, dc_ac_ratio, lag-uri.
- Algoritmi comparati: **LinearRegression** (baseline), **RandomForestRegressor**, **XGBoost** (gradient boosting), **LSTM** (retele recurente).
- Asteptare: **RandomForest si XGBoost vor castiga** pentru ca surprind interactiuni neliniare cu putine date. LSTM-ul probabil nu va converge bine cu doar 27 zile de antrenare - mentionat explicit ca limitare in lucrare.
- Concepte cheie de explicat: ce face fiecare algoritm, **feature importance** (cum aflam care predictori conteaza cel mai mult - esential pentru capitolul de interpretare).

**2. Model Consum (USA): aici LSTM are loc sa straluceasca** - target `PJME_MW`. Serie orara lunga 16 ani (145.194 randuri), foarte ciclica.
- Algoritmi comparati: **LinearRegression** (baseline), **RandomForest**, **XGBoost**, **LSTM**.
- Asteptare: **LSTM va prinde** sezonalitatile multi-scalare (zi / saptamana / an). XGBoost e candidat puternic pentru robustete cu lag-uri si rolling.
- Concepte cheie de explicat: LSTM (Long Short-Term Memory) - cum functioneaza celulele de memorie, de ce sunt potrivite pentru serii temporale; **secventiere** (cum impartim seria in ferestre de N pasi); **early stopping** (cand oprim antrenarea).

**3. Model Pret (Spania): cel mai bogat in features (80)** - target `price actual`. Predictori inclusi: 28 surse de generare, meteo Madrid (dupa one-hot), cerere totala, lag-uri pret.
- Algoritmi comparati: **LinearRegression**, **RandomForest**, **XGBoost**, **LSTM**.
- Asteptare: **XGBoost va fi greu de batut** datorita features-urilor numeroase si interactiunilor complexe. Optional: **Prophet** ca alternativa specializata pe sezonalitate, dar nu in scope-ul initial.
- Concepte cheie de explicat: **regularizare** (L1/L2 in LinearRegression) - de ce previne overfitting-ul cand avem 80 features si ~24K randuri de train.

**4. Validare cronologica si selectie model castigator.** Pentru fiecare set de date:
- **Metrici raportate**: RMSE (eroare absoluta in unitati originale), MAE (mediana erorilor), R² (cat din varianta explica modelul), MAPE (eroare procentuala - util pentru comparatii cross-domain).
- **TimeSeriesSplit cu 5 folduri** (in loc de KFold clasic) - fereastra de train se mareste progresiv, fereastra de test ramane in viitor. Simuleaza scenariul real de productie.
- **Tabel comparativ** algoritm x metrici pentru fiecare set de date, plus grafic **predictii vs valori reale** pe ultima saptamana din test.
- **Salvare model castigator** in `models/` (joblib pentru sklearn/XGBoost, .h5 pentru LSTM/Keras) - reutilizat in Streamlit si Etapa IV (LLM).

**5. Livrare conform regulilor (sectiunea 14).**
- **Modul .py**: extind `src/ml_models/predictors.py` existent cu functii pentru LSTM, TimeSeriesSplit cv, feature importance, salvare/incarcare modele. Posibil split in submodule daca devine prea mare.
- **3 notebook-uri didactice**: `05_ml_consum_usa.ipynb`, `06_ml_pret_spania.ipynb`, `07_ml_solar_india.ipynb`. Structura uniforma per notebook: incarcare parquet -> split cronologic -> scaler -> antrenare 4 modele -> CV -> comparatie metrici -> grafic predictii -> salvare model.
- **Concepte explicate didactic in fiecare notebook** (romana fara diacritice): ce face fiecare algoritm, hyperparametrii principali, **bias-variance tradeoff**, cum se interpreteaza fiecare metrica, ce inseamna underfitting vs overfitting, **feature importance** pentru tree-based models.
- **Validare**: fiecare notebook executat end-to-end cu nbclient inainte de livrare.

**6. Livrabile concrete la final Etapa II.**
- 3 modele salvate in `models/` (cate unul per set de date, cel mai bun din comparatie).
- Tabel master cu toate rezultatele (`reports/ml_comparison.parquet` sau csv).
- 3 notebook-uri rulate complete cu graficele inline.
- Update HANDOFF.md cu lista de modele castigatoare si metrici pentru fiecare.

**Estimare efort si abordare.**
- **Un set de date per sesiune** (decis explicit de Diana). Ritmul: 3 sesiuni de implementare ML, fiecare focusata pe un singur notebook.
- **Optionale incluse direct** in plan (NU mai sunt optionale - vor fi facute la sesiunea respectiva).
- **Ordinea propusa** (de la simplu la complex, didactic):
  1. **Sesiunea 1: USA** - introduce framework-ul ML clasic (4 algoritmi + Prophet + GridSearchCV).
  2. **Sesiunea 2: Spania** - introduce **Optuna** (tuning mai avansat) + **SHAP values** pentru explicarea predictiilor (esential pentru Etapa IV LLM).
  3. **Sesiunea 3: India** - inchide cu un set fizic mic, focus pe **feature importance** specifice (iradiere, temp), tuning lightweight + SHAP pe eficienta panourilor.

**Distributie optionale pe notebook-uri:**

| Notebook | Algoritmi obligatorii | Hyperparameter tuning | Explicabilitate | Extra |
|---|---|---|---|---|
| 05 USA (Sesiunea 1) | LinearReg, RF, XGBoost, LSTM | **GridSearchCV** (intro la concept) | feature_importances_ | **Prophet** (alternativa pentru long time series) |
| 06 Spania (Sesiunea 2) | LinearReg, RF, XGBoost, LSTM | **Optuna** (mai eficient pe 80 features) | **SHAP values** (full demo) | - |
| 07 India (Sesiunea 3) | LinearReg, RF, XGBoost, LSTM (mentionat ca limitat) | **Optuna lightweight** | **SHAP** focus pe eficienta solar | Validari fizice (din 04b) |

**Concepte noi introduse pe parcurs (in plus fata de cele de baza):**
- Sesiunea 1 (USA): GridSearchCV (cum se cauta cei mai buni hyperparametri), TimeSeriesSplit (cv pe ferestre), Prophet (ce face Facebook Prophet, cand functioneaza), early stopping pentru LSTM.
- Sesiunea 2 (Spania): Optuna (Bayesian optimization vs grid search), SHAP values (cum se interpreteaza, force_plot, summary_plot), regularizare L1/L2.
- Sesiunea 3 (India): walk-forward validation (alternativa la TimeSeriesSplit pentru seturi foarte mici), SHAP cu focus pe context fizic (iradiere -> productie).

---

### 16.A. Rezultate Sesiunea 1 - Consum USA (PJME) - FINALA

**Status:** FINALIZATA cu rulare full pe Databricks (UTM Medium-1 cluster, ~50 min total).

**Rulare pe Databricks (28 aprilie 2026, mod full)**:

| Loc | Model | RMSE (MW) | MAE (MW) | R² | MAPE % |
|---|---|---|---|---|---|
| 1 | **XGBoost_tuned (CASTIGATOR)** | **324.89** | **235.98** | **0.9975** | **0.75** |
| 2 | XGBoost (default) | 328.05 | 237.85 | 0.9974 | 0.76 |
| 3 | RandomForest | 358.61 | 253.32 | 0.9969 | 0.81 |
| 4 | LSTM (50K rows, 30 epochs) | 407.22 | 300.34 | 0.9958 | 0.98 |
| 5 | LinearRegression | 514.83 | 382.55 | 0.9937 | 1.22 |
| 6 | Prophet (45K ore) | 7869.67 | 6639.33 | -0.47 | 21.11 |

**Best params XGBoost (din GridSearchCV)**: `learning_rate=0.1, max_depth=6, n_estimators=300`. Best CV RMSE: 391.24.

**Feature importance top 10 (XGBoost_tuned)**:
1. PJME_MW_lag_1: **0.901** (dominant!)
2. hour_cos: 0.051
3. PJME_MW_lag_2: 0.009
4. hour: 0.007
5. month_cos: 0.005
6. hour_sin: 0.005
7. PJME_MW_lag_3: 0.004
8. PJME_MW_roll_mean_24: 0.004
9. dayofweek: 0.003
10. PJME_MW_roll_std_3: 0.003

**Insight major**: lag 1 face 90% din predictie. Restul features sunt aproape neimportante. Pentru consum electric pe orizont scurt, valoarea de acum 1 ora prezice aproape perfect valoarea curenta.

**LSTM training history** (30 epochs, fara overfitting - val_loss < loss):
- Epoch 1: loss 0.111 / val_loss 0.038
- Epoch 10: loss 0.013 / val_loss 0.008
- Epoch 30: loss 0.0094 / val_loss 0.0047

**Concluzii pentru lucrare:**
- **XGBoost este modelul recomandat** pentru consum electric. Tuning-ul aduce <1% imbunatatire fata de default.
- **LSTM e o alternativa valida** (R² 0.9958), dar overhead-ul nu se justifica fata de XGBoost.
- **Prophet este EXCLUS** pentru acest tip de date - modelul aditiv simplu nu poate captura complexitatea unei serii energetice agregate cu 16 ani.
- **Lag 1 ora domina** - implicatie puternica pentru forecast pe termen scurt.

**Output-uri salvate (commit din Databricks)**:
- `models/usa_winner_xgboost_databricks.json` (1.97 MB)
- `reports/ml_comparison_usa_databricks.csv`
- `reports/training_log_usa_databricks.md` (cu best params, feature importance, LSTM history)
- `notebooks/05_databricks_ml_consum_usa.ipynb` (executat pe cluster)

**Setari folosite**:
- LSTM: 50.000 train + 5.000 val + 10.000 test, 64 units, 30 epochs, batch 128
- Prophet: 45.000 ore istoric
- GridSearchCV: 30.000 randuri, 3 folduri, 18 combinatii
- TimeSeriesCV: 50.000 randuri, 5 folduri
- RandomForest: 200 arbori, depth nelimitat
- XGBoost: 300 arbori, depth 6

**Lectii invatate**:
1. Workspace Files in `/Workspace/Users/` NU sunt accesibile prin Python `open()` direct - trebuie download via REST API in `/tmp/`.
2. `dbutils.fs.cp` esueaza cu permission error pe cluster shared - folosim `shutil.copy` ca fallback.
3. Notebook-ul Databricks salveaza fisierele in `/tmp/` initial, trebuie copiate inapoi in workspace pentru git commit.

---

### 16.A.OLD. Rezultate Sesiunea 1 - mod DEMO (lasata pentru istoric)

**Status:** finalizata. `notebooks/05_ml_consum_usa.ipynb` - 33 celule, executat end-to-end.

**Modele evaluate** (test set 29.038 ore din 2015-2018):

| Model | RMSE (MW) | MAE (MW) | R² | MAPE % |
|---|---|---|---|---|
| **XGBoost (castigator)** | **345.2** | **252.6** | **0.9972** | ~0.78 |
| RandomForest | 494.8 | 364.5 | 0.9942 | ~1.1 |
| LinearRegression | 514.9 | 382.7 | 0.9937 | ~1.2 |
| XGBoost_tuned (GridSearchCV) | 877.7 | 684.8 | 0.9817 | ~2.1 |
| LSTM (DEMO param. minimi) | 11.874 | - | -13.96 | - |
| Prophet (DEMO param. minimi) | enorm | - | -349.899 | - |

**Observatii:**
- **Castigator: XGBoost** cu defaults (n=200, max_depth=6) - R² = 0.997, eroare medie ~250 MW pe target ~32.000 MW (**< 1% MAPE**).
- LSTM si Prophet au valori extrem de proaste pentru ca au fost rulate cu **parametri minimi** (LSTM: 1000 randuri, 3 epochs; Prophet: 2000 ore istorice). Pentru lucrarea finala, **maresti valorile** la N_TRAIN_LSTM=50.000 si PROPHET_TRAIN_HOURS=45.000 si rezultatele se vor schimba dramatic (in PyCharm pe masina locala, nu in sandbox-ul cu timeout 45s).
- XGBoost tuned a iesit mai slab decat default - grila prea mica (4 combinatii). Pentru lucrare, foloseste Optuna (Sesiunea 2 va introduce conceptul) cu cel putin 50 trials.

**TimeSeriesSplit CV (pe subset 10K randuri, 3 folduri):**
- LinearRegression: RMSE = 670.7 ± 33.1, R² = 0.983
- RandomForest: RMSE = 717.8 ± 167.5, R² = 0.977
- XGBoost: RMSE = 740.3 ± 155.1, R² = 0.976

**Feature importance top 5 (XGBoost tunat):** lag-urile (`PJME_MW_lag_1`, `PJME_MW_lag_24`, `PJME_MW_lag_168`) si rolling means de 24h domina.

**Output-uri salvate:**
- `models/usa_winner_xgboost.json` - modelul XGBoost castigator (1.4 MB).
- `reports/ml_comparison_usa.csv` - tabel comparativ cu toate metricile.

**Concepte explicate didactic in notebook (in plus fata de baseline):** baseline LinearReg, RandomForest (bagging), XGBoost (boosting), LSTM (forget/input/output gates, sequencing), Prophet (trend + seasonality + holidays + noise), TimeSeriesSplit (expanding window), GridSearchCV (cost computational), feature_importances_, RMSE/MAE/R²/MAPE cu interpretare.

---

### 16.B. Plan Sesiunea 2 - Pret energie Spania

> Status: **in lucru** (MODE=demo local). Ulterior se va rula full pe Databricks.

Construirea modelelor predictive pentru pretul orar al energiei in piata spaniola, cu inovatii metodologice fata de Sesiunea 1: **Optuna** in loc de GridSearchCV (Bayesian optimization mai eficient pentru spatii mari de hyperparametri) si **SHAP values** pentru explicabilitate (esential pentru un set cu 80 features).

**1. Particularitatile setului Spania**: 80 features (28 surse generare + meteo 5 orase + ingineresti), volatilitate ridicata (0-100+ EUR/MWh, ocazional valori negative), perioada 2015-2018 (~35.000 randuri).

**2. Modele**: LinearReg (baseline), Ridge, Lasso (regularizare critica cu 80 features), RandomForest, XGBoost (default + Optuna-tuned), LSTM. **Prophet EXCLUS** (am demonstrat in USA ca esueaza pe acest tip de date).

**3. Optuna** cu Tree Parzen Estimator (TPE) si MedianPruner. Search space cu 5-6 hyperparametri liber alesi, 50 trials in mod full (5 trials in mod demo).

**4. SHAP values**: TreeExplainer pentru XGBoost, summary_plot, force_plot, dependence_plot pentru top 5 features.

**5. Validare**: TimeSeriesSplit 5 folduri, metrici RMSE/MAE/R²/MAPE.

**6. Livrabile**:
- `notebooks/06_ml_pret_spania.ipynb` (local) si `06_databricks_ml_pret_spania.ipynb` (cloud).
- `models/spania_winner_*.json`, `reports/ml_comparison_spania*.csv`, `reports/training_log_spania*.md`.
- 4-5 grafice noi pentru capitolul 6 din lucrare (incluzand SHAP summary_plot).
- Capitol nou in `Disertatie.docx` cu interpretare detaliata.

**7. Asteptari rezultate**: XGBoost va castiga, dar diferenta fata de RF/Lasso va fi mai mica decat la USA. Lag 1 NU va mai domina la fel - generarea solar/eoliana si pretul precedent vor imparti importanta. LSTM mai competitiv. Lasso va elimina automat redundante.

**8. Pasi de executie**:
- Pas A (in curs): notebook locala MODE=demo, validat end-to-end.
- Pas B (urmator): rulare full pe Databricks, salvare rezultate.
- Pas C: capitol 6 in `Disertatie.docx` + grafice integrate.

---

### 16.B.REZULTATE. Sesiunea 2 Spania - FINALA (rulare full Databricks, 15 iunie 2026)

**Status:** FINALIZATA. Rulat full pe Databricks (MODE=full), MLflow tracking. Notebook: `notebooks/06_databricks_ml_pret_spania.ipynb`.

| Loc | Model | RMSE (EUR/MWh) | MAE | R^2 | MAPE % |
|---|---|---|---|---|---|
| 1 | **XGBoost_tuned_Optuna (CASTIGATOR)** | **2.00** | 1.49 | **0.9696** | 2.48 |
| 2 | RandomForest | 2.13 | 1.58 | 0.9655 | 2.61 |
| 3 | XGBoost (default) | 2.24 | 1.71 | 0.9619 | 2.83 |
| 4 | Lasso | 2.40 | 1.70 | 0.9564 | 2.93 |
| 5 | Ridge | 2.63 | 2.05 | 0.9476 | 3.46 |
| 6 | LinearRegression | 2.64 | 2.07 | 0.9470 | 3.49 |
| 7 | LSTM | 4.72 | 3.88 | 0.8361 | 7.07 |

- **Optuna best**: n_estimators=495, max_depth=6, lr=0.041, subsample=0.936, colsample=0.867 (best CV RMSE 2.71; 28 trials completate, 22 pruned din 50).
- **Lasso**: doar 19/78 features non-zero (confirma redundanta).
- **SHAP/feature importance**: price actual_lag_1 domina (48% importanta interna, 9.04 EUR SHAP), dar mai putin coplesitor ca la USA; apar si pretul day-ahead si surse de generare.
- **Insight**: pret mai greu de prezis decat consumul (MAPE 2.48% vs 0.75% USA); tuning-ul aduce castig real (vs marginal la USA); LSTM ramane slab.
- Output: `models/spania_winner_xgboost_databricks.json`, `reports/ml_comparison_spania_databricks.csv`, `reports/training_log_spania_databricks.md`, `reports/figures/fig_6_*.png`.
- **Capitolul 6** scris in `Disertatie.docx`.

### 16.C.REZULTATE. Sesiunea 3 India - FINALA (rulare locala PyCharm, 15 iunie 2026)

**Status:** FINALIZATA. Notebook LOCAL (set mic, ruleaza in minute - NU necesita Databricks): `notebooks/07_ml_solar_india.ipynb` (executat, cu outputs inline).

**ATENTIE - data leakage eliminat:** setul brut continea coloane derivate din tinta - `DC_POWER` (corr 1.0000 cu AC_POWER, AC=DC*0.0977), `dc_ac_ratio`, `performance_ratio`, `eff_temp_corrected`, `DAILY_YIELD`, `TOTAL_YIELD`. Cu ele, LinearRegression dadea R^2=1.0000 ARTIFICIAL. Au fost eliminate (raman 33 features legitime: iradiere, temperaturi, temporale, lag-uri/rolling AC_POWER). Vezi sectiunea 1.bis din notebook + subcapitolul 7.1 din lucrare.

Target AC_POWER, 648 randuri (27 zile), train 455/val 64/test 129.

| Loc | Model | RMSE | MAE | R^2 | MAPE % |
|---|---|---|---|---|---|
| 1 | LinearRegression (best R^2/RMSE) | 416 | 315 | 0.9972 | 15.0 |
| 2 | **RandomForest (best MAE/MAPE - recomandat practic)** | 448 | 232 | 0.9967 | 4.90 |
| 3 | XGBoost_tuned_Optuna | 528 | 280 | 0.9955 | 6.27 |
| 4 | XGBoost (default) | 588 | 321 | 0.9944 | 6.40 |
| 5 | LSTM | ~3300 | ~2500 | ~0.80 | >100 |

- **Castigatorul depinde de metrica**: Linear pe R^2/RMSE (productia ~ liniara in iradiere), RandomForest pe MAE/MAPE (mai bun in regimul de tranzitie zori/amurg). LSTM stochastic, variaza intre rulari.
- **Feature importance / SHAP**: IRRADIATION domina absolut (~98% importanta interna, SHAP ~6817 - de 100x al doilea). Validare fizica: temperatura modulelor are contributie corectiva (panouri supraincalzite = eficienta scazuta).
- **LSTM esueaza** pe 27 zile - exact ipoteza din plan.
- Output: `models/india_winner.pkl` (Linear), `reports/ml_comparison_india.csv`, `reports/figures/fig_7_*.png`.
- **Capitolul 7** scris in `Disertatie.docx`.

**=> ETAPA II (modelare ML) COMPLETA pe toate 3 seturile (cod + capitole 5, 6, 7).** Pas urmator: Etapa III - optimizare neliniara.

---

## 17. Databricks - rularea modelelor lungi in cloud

> Status: configurat. Vezi `docs/DATABRICKS_SETUP.md` pentru ghid complet.

Diana are cont **Databricks** prin facultate. Folosim Databricks pentru:
- **LSTM in mod full** (50k × 30 epochs) - 30+ min CPU local devine 2-5 min pe GPU.
- **GridSearchCV** cu grila mare - multi-core mai eficient decat local.
- **Optuna** (Sesiunea 2) - paralelizare mai buna.
- **MLflow tracking** - frumos pentru capitolul de Implementare in lucrare.

**Conventie de notebook-uri duala:**

Pentru fiecare set de date din Sesiunile 1-3, exista **doua notebook-uri** in `notebooks/`:
- `05_ml_consum_usa.ipynb` - **versiunea locala** (ruleaza in PyCharm cu MODE demo/full).
- `05_databricks_ml_consum_usa.ipynb` - **versiunea Databricks** (cu MLflow + DBFS paths + GPU detection).

Aceeasi structura didactica, aceleasi explicatii. Diferenta: paths, MLflow logging, %pip install.

**Pentru sesiunile urmatoare** (Spania, India), AI-ul trebuie sa creeze **ambele variante**.

**Conventia paths Databricks:**
- Cod: `/Workspace/Repos/diana_nenu@yahoo.com/Disertatie_AI_Platform/` (din Repos integration cu GitHub).
- Date: `/dbfs/FileStore/disertatie/data/processed/<dataset>_features.parquet` (urcate manual prin UI sau CLI).
- Output modele: `/dbfs/FileStore/disertatie/models/`
- Output rapoarte: `/dbfs/FileStore/disertatie/reports/`

**Pattern MLflow obligatoriu pentru notebook-urile Databricks:**

```python
import mlflow
EXPERIMENT_NAME = "/Users/<email>/disertatie_<dataset>"
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="<algoritm>"):
    mlflow.log_params({...})           # hyperparametri
    # ... antrenare ...
    mlflow.log_metrics(metrics)        # rmse, mae, r2, mape
    mlflow.<framework>.log_model(...)  # model serializat
```

`<framework>` poate fi `sklearn`, `xgboost`, `tensorflow`, sau pentru Prophet `mlflow.log_artifact(json_path)`.

---

## 18. Stadiu actualizat (sesiunea 26 aprilie 2026, parte 2)

Aceasta sectiune captureaza stadiul real al proiectului dupa o sesiune lunga de lucru continuat. Descrie ce s-a finalizat, ce a ramas blocat si materialele de prezentare generate pentru profesoara.

### 18.1. Sesiunea 1 USA - status real

**Notebook 05 ruleaza in modul demo curat** (pipeline complet, 35 celule, ~30s total). Modelele clasice ies cu:
- LinearRegression: R² 0.994
- RandomForest: R² 0.994
- XGBoost: R² 0.997 (CASTIGATOR)

**Rularea full a fost incercata local in PyCharm** dar a ramas blocata pe celula `gs.best_estimator_.predict(Xs)` dupa GridSearchCV (probabil kernel mort dupa ~1.5h). Diana a renuntat la rularea full local si nu am rezultate finale pentru LSTM/Prophet/XGBoost_tuned cu setarile complete.

**Pentru rularea finala** in viitor: foloseste varianta Databricks (ne-blocanta, cu MLflow tracking) sau lasa peste noapte in PyCharm cu MODE=full.

### 18.2. Imbunatatiri majore aduse notebook-ului 05

1. **Variabila `MODE`** (demo / full) la celula 4 - schimbi un singur cuvant si toate parametrii pentru LSTM, Prophet, GridSearchCV, TimeSeriesCV se ajusteaza automat.
2. **Progress bars** la celulele lungi:
   - `verbose=1` pentru Keras LSTM (bara per epoch).
   - `show_progress=True` pentru `time_series_cv` (afiseaza fiecare fold cu timp scurs).
   - `verbose=2` pentru GridSearchCV (afiseaza fiecare combinatie cu timp).
   - Print-uri pre/post cu durata estimata si efectiva.
3. **Regula 14.4.bis** in HANDOFF: progress bars la orice celula >30s e obligatoriu.

### 18.3. Integrare Databricks

**Workspace Diana:** UTM (Universitatea Titu Maiorescu) - `utm-dbx-platform`. Are 7 clustere shared (Small/Medium/Large × 2 + un autoscale), toate cu runtime **16.4 LTS general (NU ML)**, cu Unity Catalog activat. Niciun cluster cu GPU.

**Cluster recomandat: `UTM Shared Cluster Medium-1`** - 32 GB / 4 cores driver, 2-10 workers.

**Notebook adaptat:** `notebooks/05_databricks_ml_consum_usa.ipynb`
- `%pip install` include explicit tensorflow, mlflow, sklearn (nu doar xgboost/prophet/holidays - cele preinstalate doar pe ML runtime).
- `find_parquet()` incearca 3 locatii pentru date: workspace files in Repo, DBFS legacy, Unity Catalog Volumes.
- `find_writable_output_dir()` analog pentru output (preferinta workspace files - se sincronizeaza cu git).
- MLflow tracking automat pentru toate modelele (`mlflow.log_params`, `log_metrics`, `<framework>.log_model`).
- GPU detection cu fallback CPU.

**Setup:** vezi `docs/DATABRICKS_SETUP.md` (ghid pas-cu-pas: cluster, Repo, upload date, run, MLflow UI, export rezultate). Sectiunea "Specific UTM" descrie cluster-ele si recomandarile.

**Limitari descoperite la rularea pe Databricks:**
- Diana a importat repo-ul ca Git folder, dar a avut probleme cu primul push/pull (modificari locale in conflict cu remote). S-au rezolvat prin Discard local + Pull.
- Datele parquet trebuie urcate in `data/processed/` din Repo (workspace files) sau in DBFS legacy. Notebook-ul le gaseste automat.
- Rularea efectiva pe cluster nu a fost finalizata in aceasta sesiune (cluster-ul Medium-1 era inca in starea "loading").

### 18.4. Materiale de prezentare generate

Pentru intalnirea cu profesoara coordonatoare am generat o serie de materiale didactice si academice in folderul `docs/`:

| Fisier | Status | Continut |
|---|---|---|
| `Disertatie.docx` | **PREZENT** (editat de Diana) | Document compilat de Diana cu capitolele 3 si 4 integrate. **NU se regenereaza automat.** |
| `Mind_Maps_Seturi_Date.pdf` | **PREZENT** | 3 mind maps (cate unul per set), layout tree orizontal, A4 landscape per pagina. |
| `Mind_Map_Solar_India.svg` | **PREZENT** | Mind map Solar India - vector, prezentabil in PPT/Word/browser. |
| `Mind_Map_Consum_USA.svg` | **PREZENT** | Mind map Consum USA - vector. |
| `Mind_Map_Pret_Spania.svg` | **PREZENT** | Mind map Pret Spania - vector. |
| `Concepte_ML_explicate.pdf` | regenerabil | Ghid didactic pentru: encoding ciclic, lag-uri, rolling features, data leakage. **Continutul integrat in Disertatie.docx**, dar PDF-ul standalone poate fi regenerat din scriptul `outputs/build_concepte_pdf.py`. |
| `Capitolul_3_Date_si_surse.docx` | regenerabil | Capitolul 3 al lucrarii - 3 seturi date + justificare cross-domain. **Continutul integrat in Disertatie.docx**, dar fisierul separat poate fi regenerat din scriptul `outputs/build_capitol_3.js`. |
| `Capitolul_4_Preprocesare.docx` | regenerabil | Capitolul 4 al lucrarii - 6 sub-capitole + cuprins. **Continutul integrat in Disertatie.docx**, dar fisierul separat poate fi regenerat din scriptul `outputs/build_capitol_word.js`. |

**Mind maps - structura uniforma per set:**
- Nod central: numele setului + dimensiuni (randuri × features).
- 6 ramuri (3 stanga + 3 dreapta) cu titluri: "Cum am incarcat datele", "EDA pe care am facut-o", "Preprocesarea aplicata", "Armonizarea avansata" (sau "Modelele ML antrenate" pentru USA), "Ce am salvat ca output" (sau "Rezultatele Sesiunii 1"), "Ce mai am de facut".
- Fiecare ramura are 5 sub-puncte concrete cu actiuni la prima persoana ("Am combinat...", "Am calculat...", "Astept ca...").

### 18.5. Comenzi git ramase pentru Diana

Push pentru toate modificarile din aceasta sesiune (commituri locale + fisiere noi de prezentare):

```bash
cd /Users/diana/PycharmProjects/Disertatie_AI_Platform
git add docs/Concepte_ML_explicate.pdf docs/Capitolul_3_Date_si_surse.docx \
        docs/Capitolul_4_Preprocesare.docx docs/Mind_Maps_Seturi_Date.pdf \
        docs/Mind_Map_*.svg docs/Disertatie.docx
git add notebooks/05_databricks_ml_consum_usa.ipynb docs/DATABRICKS_SETUP.md
git commit -m "docs: materiale prezentare profesoara + integrare Databricks"
git push
```

**Atentie:** in `git status` apar fisiere `~$*.docx` - acestea sunt fisiere temporare Word de la documentele deschise, **NU le commiteaza** (vor disparea cand inchizi Word). Adauga in `.gitignore`:
```
docs/~$*
```

### 18.6. Pas urmator natural - Sesiunea 2 (Pret Spania)

Conform planului din sectiunea 16:
- Notebook `06_ml_pret_spania.ipynb` cu MODE switch + progress bars (acelasi pattern ca 05).
- Algoritmi: LinearRegression, RandomForest, XGBoost, LSTM (4 algoritmi obligatorii pentru Spania).
- **Optuna** pentru tuning (in loc de GridSearchCV - mai eficient pe cele 80 features).
- **SHAP values** pentru explicabilitate (esential pentru capitolul de interpretare a rezultatelor).
- Salvare model castigator in `models/spania_winner_*.json`.
- Tabel comparativ in `reports/ml_comparison_spania.csv`.
- Update HANDOFF sectiunea 16.B cu rezultatele.

**De asemenea, varianta Databricks** `06_databricks_ml_pret_spania.ipynb` cu acelasi pattern (MLflow + paths flexibile + %pip install).

### 18.7. Cum sa incepi sesiunea urmatoare

1. **Citeste sectiunile 14, 15, 16, 17, 18, 19 din acest HANDOFF** ca sa cunosti regulile si stadiul.
2. **Verifica `git log --oneline -10`** sa vezi ultimele commituri.
3. **Verifica daca exista commituri ne-pushate** (`git status` cu "Your branch is ahead").
4. **Intreaba Diana** ce vrea sa atace in sesiune (Sesiunea 2 Spania, modificari capitole Word, cerere de explicatii noi etc.).
5. **NU PORNI direct in implementare** - intai prezinta planul detaliat conform sectiunii 15 (titlu etapa, sub-puncte numerotate cu specific + algoritmi + asteptari + concepte didactice + metrici + livrabile + estimare).

---

## 19. FILOSOFIA proiectului si stilul didactic OBLIGATORIU

> Status: regula esentiala adaugata in sesiunea 27 aprilie 2026 dupa cererea explicita a Dianei.

### 19.1. Contextul si scopul proiectului

**Diana NU este o studenta eminenta de Data Science.** A terminat facultatea de informatica (BSc) si acum e in masterul Data Science and AI, dar nu cunoaste in profunzime tehnicile ML, optimizarea neliniara, retelele neurale sau LLM-urile. **Citeste cod pentru a-l intelege**, nu il scrie ea.

**Cum a ales tema:** a cautat job-uri pe LinkedIn pentru pozitiile **Data Scientist** si **ML Engineer**, a extras cerintele cele mai frecvente din anunturi, si a compus titlul lucrarii ca un **proiect-portofoliu cuprinzator** care sa aiba toate elementele unei pozitii reale din aceste domenii.

**Implicatia pentru AI:** lucrarea NU e un exercitiu academic abstract. E un **portfolio piece** care va fi prezentat la angajatori. Trebuie sa contina **practici reale din industrie** (versionare git, cloud deploy, MLflow tracking, modele in productie, explicabilitate prin SHAP, RESTful endpoints daca e cazul, documentare profesionala etc.). NU e suficient sa rulam un model si sa raportam metrici - trebuie sa demonstram un **flux complet end-to-end**.

### 19.2. Stilul didactic OBLIGATORIU pentru orice explicatie

Diana imi va cere des explicatii. Pentru fiecare conceptu, metoda, librarie, decizie de design - explicatia trebuie sa contina:

**1. Scopul (de ce facem asta).** Punctul de pornire - care e problema concreta pe care o rezolvam? Niciodata sa nu introduc o tehnica fara sa explic intai problema pe care o adreseaza.

Exemplu corect:
> "Lag-urile rezolva o problema fundamentala in serii temporale: valoarea de la momentul t depinde adesea de valori din trecut. Daca nu ofer modelului aceste valori istorice ca features, el nu are de unde sa le invete."

Exemplu **gresit**:
> "Adaug lag-uri 1, 24, 168."  *(fara context, fara motivatie)*

**2. Definitia clara cu cuvinte simple.** Niciodata cu jargon nedefinit. Daca folosesc un termen tehnic, il explic imediat.

Exemplu corect:
> "Un lag este o coloana noua in DataFrame care contine, pentru fiecare moment, valoarea variabilei tinta dintr-un moment anterior. Lag de 24 = valoarea de acum 24 de ore."

**3. Metoda cu exemple concrete.** Cum se calculeaza? Cu numere reale, daca e cazul. Cu cod scurt, daca ajuta.

Exemplu corect:
> "Aplic `df['target'].shift(24)` - asta deplaseaza intreaga coloana cu 24 de pozitii in jos, deci valoarea de la randul X devine valoare la randul X+24. Practic, in dreptul orei 14 azi voi avea valoarea de la ora 14 ieri."

**4. Analogii din viata reala.** Ajuta enorm la fixare.

Exemplu corect:
> "E ca atunci cand iei in calcul vremea de saptamana trecuta cand iti planifici ce sa imbraci azi - daca a fost frig, probabil mai e."

**5. Capcanele tipice.** Ce se poate gresi? Ce trebuie evitat?

Exemplu corect:
> "Atentie: rolling fara shift introduce data leakage - modelul vede indirect raspunsul. Solutia: aplic shift(1) inainte de rolling."

**6. Rezultatul concret.** Ce iese? Cum se vede in date?

Exemplu corect:
> "Dupa aplicare: am 5 coloane noi (lag_1, lag_2, lag_3, lag_24, lag_168). Primele 168 randuri vor avea NaN si vor fi eliminate la finalul pipeline-ului."

### 19.3. Reguli de stil pentru explicatii lungi

**Cand explicatia e lunga (paragrafe sau pagini):**

- **Foloseste headers ierarhice** ca sa structurezi continutul (titlul mare al conceptului, sub-titluri pentru fiecare aspect, mini-concluzie la final).
- **Evita zidul de text** - paragrafele scurte de 2-4 propozitii sunt mai usor de citit.
- **Foloseste tabele** cand compari mai multe optiuni sau ai date numerice.
- **Foloseste exemple cu numere reale** din proiect, nu abstracte.
- **Citeaza fisierele si liniile** - "vezi `src/data_processing/preprocessing.py`, functia `add_lags` linia 95" - asa Diana stie unde sa caute in cod.
- **Mentioneaza alternativele** pe care nu le-am ales si motivul. "Am ales XGBoost in loc de LightGBM pentru ca XGBoost e mai matur si are documentatie mai buna pentru incepatori."
- **Termina cu un mini-rezumat** in 2-3 propozitii care reia esenta.

### 19.4. Exemplu standard de explicatie didactica completa

Cand Diana cere "explica-mi X", structura ideala a raspunsului:

```
## Ce este X
[1-2 propozitii cu definitie simpla]

## De ce avem nevoie de X (problema rezolvata)
[descrierea problemei pe care X o adreseaza, cu un exemplu concret]

## Cum functioneaza X
[mecanism intuitiv, daca e cazul cu o analogie]

## Formula / Algoritmul
[doar daca e relevant - cu explicatii pentru fiecare termen]

## Exemplu pas-cu-pas pe datele noastre
[un exemplu concret cu numere din proiectul Dianei]

## Capcane si decizii metodologice
[ce trebuie evitat, ce decizii am luat in proiect, de ce]

## Cum se aplica in proiect
[unde e implementat X, ce face concret, ce iese din rulare]

## Mini-rezumat
[2-3 propozitii care reiau ideea principala]
```

### 19.5. Reguli pentru limbajul de proiect

Dat fiind ca Diana **nu va prezinta lucrarea unei comisii oarecare**, ci va folosi proiectul si in interviurile pentru job-uri de Data Scientist / ML Engineer:

- **Toate alegerile tehnice trebuie justificabile fata de un evaluator profesionist** - nu doar fata de profesoara.
- **Practicile aplicate trebuie sa reflecte standarde de productie**: versionare semantica, code review, CI/CD-ready, dependency management, documentare API, test coverage etc.
- **Materialele didactice (capitole, mind maps, slide-uri)** sa para opera unei studente serioase si exigente, chiar daca tehnologia e noua pentru ea.
- **Stack-ul ales** sa fie cel din lumea reala: Python + pandas/scikit-learn/XGBoost/TensorFlow + Streamlit + Databricks + MLflow + GitHub. Toate apar in anunturile LinkedIn pentru aceste pozitii.

### 19.6. Lista de cunostinte pe care AI-ul TREBUIE sa le explice cand apar

In ordinea in care apar in proiect, am explicat sau voi explica:

| Concept | Stadiul explicatiei | Locul |
|---|---|---|
| Procesul EDA si ce contine | facut | notebook 01-03 |
| Pandas/DataFrame, parquet | facut | notebook 04, sectiune docs |
| Tratarea valorilor lipsa (interpolare, ffill, drop) | facut | notebook 04, capitol 4 |
| Encoding ciclic (sin/cos) | facut | notebook 04, PDF concepte |
| Lag-uri si rolling features | facut | notebook 04, PDF concepte |
| Data leakage | facut | notebook 04, PDF concepte |
| Split cronologic vs random | facut | notebook 04 |
| Performance ratio (Yield) solar | facut | notebook 04b, capitol 3 |
| Detectie outliers IQR | facut | notebook 04b |
| One-hot encoding | facut | notebook 04b, capitol 4 |
| Normalizare (StandardScaler, MinMax) | facut | notebook 04b, capitol 4 |
| LinearRegression - cum functioneaza | facut | notebook 05 |
| RandomForest si bagging | facut | notebook 05 |
| XGBoost si gradient boosting | facut | notebook 05 |
| LSTM si retele recurente | facut | notebook 05 |
| Prophet si decompozitia aditiva | facut | notebook 05 |
| TimeSeriesSplit | facut | notebook 05 |
| GridSearchCV | facut | notebook 05 |
| Feature importance | facut | notebook 05 |
| RMSE, MAE, R², MAPE | facut | notebook 05 |
| MLflow tracking | facut partial | notebook 05_databricks |
| **Optuna (tuning Bayesian)** | TBD | notebook 06 (Sesiunea 2) |
| **SHAP values** | TBD | notebook 06 (Sesiunea 2) |
| **Regularizare L1/L2** | TBD | notebook 06 (Sesiunea 2) |
| **Walk-forward validation** | TBD | notebook 07 (Sesiunea 3) |
| **Optimizare neliniara - SciPy** | TBD | notebook 08+ (Etapa III) |
| **Functie obiectiv, constrangeri, SLSQP** | TBD | notebook 08+ |
| **Battery dispatch (problema reala)** | TBD | notebook 08+ |
| **Hugging Face transformers** | TBD | notebook 11 (Etapa IV) |
| **Tokeni, embeddings, prompt engineering** | TBD | notebook 11 |
| **Streamlit components, session_state** | TBD | aplicatia |
| **Streamlit Cloud deploy** | TBD | aplicatia |

**Pentru fiecare item TBD, AI-ul va aplica obligatoriu structura din 19.4.**

---

## 20. Cerinte oficiale ghid facultate (UTM) - OBLIGATORIU de respectat

> Sursa: `docs/Ghid_pregatire_lucrare_2025-2026.pdf` (salvat in proiect, mereu la indemana in PyCharm).
> Adaugat in sesiunea 15 iunie 2026 dupa ce George a incarcat ghidul oficial.
> Lucrarea noastra este **lucrare de DISERTATIE** (master), nu licenta.

### 20.1. Structura obligatorie a lucrarii (in ordine)
a) **Pagina de titlu** - conform Anexa 3: "UNIVERSITATEA «TITU MAIORESCU» DIN BUCURESTI", "FACULTATEA DE INFORMATICA", "LUCRARE DE DISERTATIE", titlul, "COORDONATOR STIINTIFIC: (grad didactic, nume)", "ABSOLVENT: (nume)", "SESIUNEA IUNIE/IULIE 2026".
b) **Referatul de apreciere** (Anexa 2) - il completeaza coordonatorul, se indosariaza cu lucrarea (nu il scriem noi).
c) **Cuprins** - titluri + paginile de inceput ale capitolelor si subcapitolelor.
d) **Introducere** (2-3 pagini) - prezentarea generala a temei, motivatia alegerii, gradul de noutate, prezentarea pe scurt a structurii lucrarii, **sublinierea contributiei personale**. NU se numeroteaza ca si capitol.
e) **Capitole** - **NUMAR 3-5 capitole** numerotate crescator, cu subcapitole. Contin abordarea teoretica + prezentarea aplicatiei practice si a tehnicilor de programare. Se includ **capturi de ecran de la rularea aplicatiei, secvente de cod, figuri si tabele**. **Un capitol separat trebuie sa descrie aplicatia software** realizata.
f) **Concluzii** (minim 1 pagina) - concluzii proprii, directii noi, imbunatatiri. NU se numeroteaza ca si capitol.
g) **Bibliografie** - lista tuturor surselor.
h) **Anexe** (daca e cazul) - secvente de cod, studii de caz, rezultate experimentale.

### 20.2. Norme de redactare (verificate pe Disertatie.docx)
- **Microsoft Word**, font **Times New Roman 12pt**, **1.5 randuri**, aliniere **Justify**, fara greseli ortografice.
- **Disertatie: 40-70 pagini** (fara anexe).
- Status verificat: corpul textului e DEJA TNR 12 / 1.5 / Justify (conform). **Atentie: titlurile (Heading 1/2/3) sunt acum Arial** - ghidul cere TNR; de schimbat la final.

### 20.3. Originalitate
- Citarile in text se fac cu mentionarea exacta a sursei **dupa numarul din lista bibliografica**.
- **Declaratie pe proprie raspundere** privind originalitatea - se indosariaza cu lucrarea.
- Coordonatorul verifica antiplagiat.

### 20.4. Prezentarea (pentru sustinere)
- PowerPoint, ~10 slide-uri, 15-20 min. Repartitie sugerata: 1 titlu, 1 cuprins, 1 obiective/stadiu actual, 2 contributie+rezultate, **4 implementarea aplicatiei**, 1 concluzii. Accent pe aplicatia practica + contributia personala.

### 20.5. STADIUL CONFORMITATII (la 15 iunie 2026)

**Conform / prezent:**
- Pagina de titlu - EXISTA (necesita mici ajustari Anexa 3: nume universitate complet, camp COORDONATOR STIINTIFIC, formularea "SESIUNEA IUNIE/IULIE 2026").
- Cuprins - EXISTA (planifica Cap 1-9).
- Corp text: TNR 12 / 1.5 / Justify - CONFORM.
- Capitole scrise: 3 (Date), 4 (Preprocesare), 5 (ML USA), 6 (ML Spania).

**Lipseste / de facut:**
- **Introducere** (2-3 pg) - planificata in cuprins (Cap 1), NEscrisa.
- **Capitolul 2** (stadiul cunoasterii/fundamente) - NEscris.
- **Capitol dedicat aplicatiei Streamlit** cu capturi de ecran de la rulare - NEscris (planificat Cap 8).
- **Concluzii** (min 1 pg), **Bibliografie**, **Declaratie de originalitate** - lipsesc.
- Capitole LLM (7) si Optimizare neliniara - NEscrise.

**PROBLEMA MAJORA de structura:**
- Ghidul cere **3-5 capitole** numerotate (Introducerea si Concluziile sunt separate, NEnumerotate). Planul actual are **9 "capitole"** (Cap 1 Introducere ... Cap 9 Concluzii), ceea ce depaseste limita.
- **Recomandare**: de confirmat cu coordonatorul. Schema conforma propusa (Introducere + 5 capitole + Concluzii):
  - Introducere (NEnumerotata, 2-3 pg)
  - Cap 1: Stadiul actual al cunoasterii / fundamente teoretice
  - Cap 2: Date si metodologie de preprocesare (fuziune Cap 3+4 actuale)
  - Cap 3: Modele predictive ML (USA + Spania + India - fuziune Cap 5+6 + Sesiunea 3)
  - Cap 4: Optimizare neliniara si integrare LLM
  - Cap 5: Aplicatia integrata in cloud (Streamlit) - **capitolul obligatoriu despre aplicatie**, cu capturi de ecran
  - Concluzii (NEnumerotate)
  - Bibliografie
- Renumerotarea capitolelor existente se va face DUPA ce tot continutul exista, ca sa nu refacem munca de mai multe ori.

### 20.6. Reguli derivate pentru sesiunile urmatoare
- Orice capitol nou trebuie sa includa, unde e relevant, **capturi de ecran de la rularea aplicatiei**, **secvente de cod** si **figuri/tabele** (cerinta explicita din ghid).
- Cand adaug surse (articole, librarii), le notez ca sa construiesc **Bibliografia** cu citari numerotate in text.
- Tinta finala: 40-70 pagini, 3-5 capitole, TNR 12 peste tot (inclusiv titluri la final).

---

## 21. Etapa III - Optimizare neliniara (COMPLETA, 15 iunie 2026)

Componenta prescriptiva: transforma predictiile (Etapa II) in decizii. Tot codul in `src/optimization/optimizer.py` (extins peste wrapper-ul SLSQP generic). Toate notebook-urile ruleaza LOCAL in PyCharm (rapide, nu necesita Databricks) si sunt executate (cu outputs inline).

| Notebook | Problema | Set | Rezultat |
|---|---|---|---|
| `08_optimization_battery.ipynb` | Dispatch baterie (arbitraj pret) | Spania | profit 561 EUR/72h; bate strategia naiva; profit creste cu capacitatea |
| `09_optimization_load_shifting.ipynb` | Load shifting (tarif time-of-use) | USA | economie 4.74% (1.4%-7.4% cu flexibilitatea); energie conservata |
| `10_optimization_solar_tilt.ipynb` | Orientare panouri (model geometric simplificat) | India | unghi optim ~27 grade, +12% vs orizontal |

**Functii adaugate in optimizer.py:** `BatteryConfig`, `battery_dispatch_problem`, `battery_soc`, `battery_profit`, `solve_battery_dispatch`; `LoadShiftConfig`, `time_of_use_tariff`, `load_shifting_problem`, `solve_load_shifting`, `load_cost`; `solar_elevation`, `solar_captured_energy`, `solve_solar_tilt`.

**Formulari NELINIARE** (justifica optimizatorul neliniar): baterie - cost degradare patratic; load shifting - penalizare confort patratica; solar - obiectiv trigonometric. Toate cu SLSQP.

**Figuri:** `reports/figures/fig_8_1`, `fig_8_2` (baterie), `fig_9_1`, `fig_9_2` (load shifting), `fig_10_1`, `fig_10_2`, `fig_10_3` (solar).

**Capitolul 8** ("Optimizare neliniara pentru suport decizional prescriptiv") scris in `Disertatie.docx` - 3 figuri + tabel sinteza. Lucrarea: 48 pagini.

**Nota onesta (de stiut):** problema solar (10) foloseste un model geometric simplificat al pozitiei soarelui, NU datele masurate (care nu au geometria soarelui). E declarat explicit ca model didactic atat in notebook, cat si in capitolul 8.4. Bateria si load shifting-ul folosesc date reale.

**=> Etapa III COMPLETA (cod + capitol 8).** Pas urmator: Etapa IV - integrare LLM (HuggingFace) pentru explicarea in limbaj natural a predictiilor si recomandarilor (vezi `src/llm_integration/insights.py`, schelet existent).

---

## 22. Etapa IV - Integrare LLM (COMPLETA, 15 iunie 2026)

Componenta de limbaj natural: traduce rezultatele numerice (metrici ML, recomandari de optimizare) in explicatii in romana.

**Backend ales: flan-t5 local (HuggingFace)**, cu generator determinist pe sabloane ca fallback/baseline.

`src/llm_integration/insights.py` rescris complet:
- `get_pipeline`, `llm_generate` - apelul flan-t5 (text2text-generation).
- Generatoare deterministe: `explain_prediction_template`, `summarize_dispatch_template`, `summarize_load_shifting_template`.
- Functii nivel inalt: `explain_prediction(...)`, `summarize_optimization(kind='battery'|'load_shifting', ...)` - returneaza dict cu 'template' (mereu) si 'llm' (daca use_llm=True).

`notebooks/11_llm_insights.ipynb` (executat) - explica didactic: tokeni, embeddings, transformer, prompt engineering, flan-t5, temperature, max_new_tokens. Demonstreaza explicatii pe rezultate reale Spania + dispatch baterie + load shifting.

**IMPORTANT - arhitectura cu degradare eleganta:** notebook-ul ruleaza ORIUNDE (generatorul determinist nu necesita model). flan-t5 ruleaza in PyCharm dupa `pip install transformers torch` (prima rulare descarca modelul ~1GB). Celula flan-t5 e protejata cu try/except - daca modelul lipseste, foloseste fallback-ul determinist fara sa crape.

**Nota onesta:** flan-t5-base e mic si slab pe romana - de aceea generatorul determinist (romana corecta) e baseline-ul, iar flan-t5 e stratul optional de rafinare. Discutat transparent in notebook si in capitolul 9.

**Capitolul 9** ("Integrarea modelelor de limbaj natural...") scris in `Disertatie.docx` cu exemple de explicatii generate. Lucrarea: 50 pagini.

**=> Etapa IV COMPLETA (cod + capitol 9).** Pas urmator: Etapa V - aplicatia Streamlit (capitolul OBLIGATORIU despre aplicatie, cu capturi de ecran). Apoi sectiunile Word ramase: Introducere, Cap. stadiul cunoasterii, Concluzii, Bibliografie. Vezi sectiunea 20.5 pentru checklist conformitate ghid.

---

## 23. Etapa V - Aplicatia Streamlit (COMPLETA, 16 iunie 2026)

Aplicatia integrata care reuneste toate componentele - capitolul OBLIGATORIU despre aplicatie din ghid.

`streamlit_app/app.py` rescris complet (un singur fisier, 5 pagini + pagini de concept):
- **Acasa, Analiza date (EDA), Predictii ML, Optimizare prescriptiva, Insight-uri LLM**.
- Reutilizeaza direct modulele din `src/` (predictors, optimizer, insights, plotting) - fara duplicare de cod.
- Foloseste: `st.cache_data`, `st.session_state`, Plotly (grafice interactive), `streamlit-option-menu` (meniu lateral modern), CSS propriu.
- Pagini de concept dedicate (butoane pe Acasa): Machine Learning, Optimizare SLSQP, LLM flan-t5, seturile de date - fiecare cu explicatii + exemple.
- Detalii tehnice pe fiecare pagina (tab-uri): algoritmi, validare, tuning, metrici, SHAP / formulare optimizare, SLSQP / arhitectura transformer.
- Dropdown-uri pe Acasa cu detalii per set (probleme pe algoritmi, solutii, alegerea castigatorului).

**Design (multe iteratii cu Diana):** tema deschisa moderna, sidebar alb cu titlul lucrarii (succint) + autor "Nenu Diana Andreea", meniu cu iconite si pilula activa gradient, font Space Grotesk, text marit, carduri cu hover, flux de etape cu sageti. Sidebar fortat mereu vizibil (initial_sidebar_state + CSS).

**Dependinte noi:** `streamlit-option-menu` (adaugat in requirements.txt).

**Rulare:** `streamlit run streamlit_app/app.py`. Posibil deploy Streamlit Cloud (integrat GitHub).

**Capturi de ecran (cerinta ghid):** facute prin `screencapture -l<window_id>` pe fereastra Safari (captura curata, fara overlay-uri), salvate in `reports/figures/app_01_acasa.png ... app_05_llm.png` + `app_04b_optimizare_grafic.png`.

**Capitolul 10** ("Aplicatia integrata in cloud (Streamlit)") scris in `Disertatie.docx` cu cele 6 capturi inserate. Lucrarea: 55 pagini.

**Cum capturez aplicatia (pt sesiuni viitoare):** browserul Claude-in-Chrome e REMOTE (nu vede localhost-ul Mac-ului). Solutia: Diana deschide app in Safari/Chrome local, navigheaza, iar eu capturez fereastra prin `screencapture -l<id>` (id obtinut cu Quartz: `CGWindowListCopyWindowInfo`). NU folosi screencapture full-screen (prinde fereastra Claude peste app).

**=> Etapa V COMPLETA (aplicatie + capitol 10).**

### Stadiu GENERAL disertatie (16 iunie 2026)
- Etapa I (date+preprocesare): cod + Cap 3, 4 - GATA
- Etapa II (ML, 3 seturi): cod + Cap 5, 6, 7 - GATA
- Etapa III (optimizare): cod + Cap 8 - GATA
- Etapa IV (LLM): cod + Cap 9 - GATA
- Etapa V (Streamlit): cod + Cap 10 - GATA
- **Sectiuni scrise adaugate (16 iunie)**: Cap 1 Introducere (motivatie, scop/obiective, structura+contributie), Cap 2 Stadiul cunoasterii (ML serii temporale, optimizare, LLM), Concluzii, Bibliografie (18 referinte reale), Declaratie de originalitate (cu numele autoarei). **Lucrarea: 59 pagini.** Structura completa: Pagina titlu -> Cuprins -> Cap 1-10 -> Concluzii -> Bibliografie -> Declaratie.
- **FINISARI FACUTE (16 iunie, parte 2):**
  - **Restructurat in 5 capitole** (cu subcapitole, permis de ghid): Introducere (ne-numerotata) -> Cap 1 Stadiul cunoasterii -> Cap 2 Date si preprocesare -> Cap 3 Dezvoltarea modelelor predictive (3.1 USA, 3.2 Spania, 3.3 India) -> Cap 4 Optimizare + LLM (4.1, 4.2) -> Cap 5 Aplicatia -> Concluzii/Bibliografie/Declaratie. Backup vechi: `docs/Disertatie_backup_10cap.docx`.
  - **Font titluri** schimbat Arial -> Times New Roman (corpul era deja TNR 12).
  - **Cuprins** inlocuit cu **camp TOC automat Word** (`TOC \o "1-3"`). IMPORTANT: in Word, click dreapta pe cuprins -> Update Field -> Update entire table (LibreOffice nu il populeaza).
- **RAMAS (de finisat):**
  - **Numerotarea figurilor/tabelelor + referintele din text** inca folosesc vechea schema (ex. "Figura 5.2", "Tabelul 6.1") - dupa consolidare nu mai corespund capitolelor (Figura 5.2 e acum in Cap 3). De renumerotat consecvent (Figura 3.1, 3.2...) SAU de lasat asa daca coordonatoarea accepta. NU s-a facut automat (risc de erori).
  - Completare **nume coordonator** pe pagina de titlu (placeholder).
  - Confirmare structura 5 capitole cu coordonatoarea.
  - Referatul coordonatorului (Anexa 2) - il completeaza coordonatoarea.

---

Mult succes la disertatie, Diana!
