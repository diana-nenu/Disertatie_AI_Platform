# DATABRICKS_SETUP - Ghid pas-cu-pas

> Pentru rularea notebook-urilor ML din `Disertatie_AI_Platform` pe **Databricks** \
> (cont facultate). Rezolva: cluster, conectare GitHub Repo, upload date in DBFS, \
> rulare cu MLflow, export rezultate.

---

## 1. Verifica ce ai disponibil pe contul de facultate

In UI-ul Databricks (Safari deschis):

1. Click pe **Compute** (in panoul lateral stang).
2. Verifica daca poti **Create Cluster**. Daca nu vezi butonul, contul are restrictii - foloseste un cluster existent.
3. Click pe **Workspace** > **Repos**. Daca exista optiunea "Add Repo", e disponibila integrarea GitHub.

Daca **NU** ai voie sa creezi cluster, sari direct la sectiunea 4 (folosesti cluster existent).

### Specific UTM (Universitatea Titu Maiorescu)

Workspace-ul UTM are 7 clustere shared preconfigurate:
- `UTM Shared Cluster Small-1/2` (16.4 LTS) - mai putin RAM, pentru teste.
- `UTM Shared Cluster Medium-1/2` (16.4 LTS) - **recomandat** pentru notebook-urile noastre. Driver 32 GB / 4 cores, autoscale 2-10 workers.
- `UTM Shared Cluster Large-1/2` (16.4 LTS) - cel mai puternic, dar poate fi aglomerat.
- `UTM shared autoscale cluster B` (15.4 LTS) - runtime mai vechi, evita.

**Toate sunt CPU-only (fara GPU)**, **runtime general (NU "ML")**, cu **Unity Catalog activat**. Asta inseamna:
- LSTM va dura ~25-40 min in mod full (vs 30+ min local + nu poti face nimic altceva).
- Notebook-ul instaleaza automat `tensorflow`, `mlflow`, `xgboost`, `prophet`, `holidays` la inceput (~3-5 min prima rulare).
- Pentru date, **foloseste Workspace files in Repo** (sectiunea 4 optiunea A) - cel mai simplu cu Unity Catalog.

---

## 2. Creeaza un cluster (daca ai voie)

Click pe **Compute > Create Compute**, apoi:

| Setare | Valoare recomandata |
|---|---|
| **Cluster name** | `disertatie-ml` |
| **Cluster mode** | Single Node |
| **Databricks Runtime** | **15.4 LTS ML** (sau cea mai recenta cu sufix "ML") - asta include TF, sklearn, MLflow preinstalate |
| **Node type** | Pentru CPU: `Standard_DS3_v2` (4 core, 14 GB) sau echivalent. Pentru GPU: `Standard_NC4as_T4_v3` (T4, ~5x mai rapid LSTM) sau `Standard_NC6s_v3` (V100, ~10x mai rapid) |
| **Auto termination** | 30 min de inactivitate |

Click **Create Cluster**. Pornirea dureaza 3-5 min.

**IMPORTANT:** daca selectezi GPU, asigura-te ca runtime-ul este cu sufix "**ML**" (ex. `15.4 LTS ML (includes Apache Spark 3.5.0, GPU, Scala 2.12)`). Fara "ML" lipsesc librariile.

---

## 3. Conecteaza GitHub Repo

Cea mai eleganta cale e ca tot codul tau (incluzand notebook-ul Databricks) sa fie sincronizat din repo-ul tau privat de pe GitHub.

1. **Workspace** (panou lateral) > **Repos**.
2. Click pe folderul tau (ex. `Repos/diana_nenu@yahoo.com/`).
3. **Add Repo** (sau "+", in functie de versiune).
4. Completezi:
   - **URL repo**: `https://github.com/diana-nenu/Disertatie_AI_Platform.git`
   - **Provider**: GitHub
   - **Repo Name**: `Disertatie_AI_Platform` (auto-completat).
5. Daca repo-ul e privat, Databricks iti cere **autentificare GitHub**:
   - Click **Add Git credential**.
   - **Git provider**: GitHub
   - **Personal Access Token**: foloseste **acelasi token** care e in macOS Keychain (cel cu scopes `repo, workflow`). Daca nu il mai ai, regenereaza la <https://github.com/settings/tokens>.
6. Click **Create Repo**.

Acum tot codul tau e la `/Workspace/Repos/diana_nenu@yahoo.com/Disertatie_AI_Platform/`.

**Cand faci commituri noi local pe GitHub**, in Databricks Repo: click pe icon-ul de Git > **Pull**.

**Cand faci modificari in Databricks** si vrei sa salvezi in GitHub: click Git icon > **Commit & Push**.

---

## 4. Upload-eaza datele procesate

Notebook-ul detecteaza **automat** unde sunt datele - incearca 3 locatii in ordine:
1. **Workspace files** (in folderul Repos, alaturi de cod) - **recomandat pentru UTM**.
2. **DBFS legacy** (`/dbfs/FileStore/disertatie/data/processed/`).
3. **Unity Catalog Volumes** (`/Volumes/main/default/disertatie/...`).

### Optiunea A: Workspace files in Repo (CEL MAI SIMPLU pentru UTM)

Repo-ul tau e deja in `/Workspace/Repos/<email>/Disertatie_AI_Platform/`. Pune datele in folderul `data/processed/` din interiorul repo-ului:

1. **Workspace > Repos > Disertatie_AI_Platform**.
2. Click pe folderul `data` (daca nu exista, click drept > Create > Folder, dai numele `data`).
3. Intra in `data` > Create folder `processed`.
4. Click drept in `processed` > **Import**.
5. Selecteaza fisierul `consum_usa_features.parquet` din `/Users/diana/PycharmProjects/Disertatie_AI_Platform/data/processed/` (drag & drop sau browse).
6. Repeta pentru celelalte 2 fisiere.

**Atentie:** datele sunt in `.gitignore`, deci NU vor fi commit-ate inapoi pe GitHub. Asta e bine - le pastrezi local in workspace fara sa poluezi repo-ul.

**Limita marime fisiere workspace:** ~500 MB per fisier in Databricks 16.4. Datele tale (10.6 MB max) sunt sub limita.

### Optiunea B: DBFS legacy (clasic)

Daca workspace files nu functioneaza:

1. In panoul lateral stang, click pe **Catalog**.
2. **Browse DBFS** (sau cauta "DBFS" in submeniu).
3. Navigheaza la `/FileStore/`.
4. **Upload** > selectezi fisierele.

Salveaza-le in: `/FileStore/disertatie/data/processed/`.

### Optiunea C: Unity Catalog Volumes (modern, daca esti familiar)

1. Catalog > `main` (sau alt catalog cu drepturi de scriere) > schema `default`.
2. Click drept > **Create > Volume**.
3. Nume: `disertatie`, tip: Managed.
4. Upload fisierele in folderul nou creat.

### Optiunea D: Databricks CLI (daca preferi terminal)

```bash
pip install databricks-cli
databricks configure --token  # cere host + PAT databricks
databricks fs mkdirs dbfs:/FileStore/disertatie/data/processed
databricks fs cp /Users/diana/PycharmProjects/Disertatie_AI_Platform/data/processed/consum_usa_features.parquet \
                 dbfs:/FileStore/disertatie/data/processed/
```

---

## 5. Deschide notebook-ul si configureaza cluster-ul

1. **Workspace > Repos > Disertatie_AI_Platform > notebooks**.
2. Click pe **`05_databricks_ml_consum_usa.ipynb`**.
3. In bara de sus a notebook-ului, in dreapta sus: **Connect** > selecteaza cluster-ul `disertatie-ml`.
4. Asteapta 1-2 minute sa se ataseze.

---

## 6. Ruleaza notebook-ul

**Run All** (butonul cu sageata dubla in bara de sus).

**Ce vei vedea:**

1. **`%pip install`** instaleaza xgboost, prophet, holidays - dureaza ~30 sec.
2. **`dbutils.library.restartPython()`** restarteaza kernelul - normal.
3. **Detectie GPU** - `len(gpus) > 0` daca cluster-ul are GPU.
4. **MLflow** seteaza experimentul automat.
5. **Antrenarile** ruleaza una cate una, fiecare loga in MLflow:
   - LinearRegression: < 1 s
   - RandomForest (n=200): ~30 s pe CPU, mai rapid pe GPU/multi-core
   - XGBoost: ~30 s
   - LSTM (50k × 30 epochs):
     - **CPU**: 30-45 min (vezi bara progres Keras pe fiecare epoch)
     - **GPU T4**: 5-8 min
     - **GPU V100**: 2-4 min
   - Prophet: 5-10 min (Stan/CmdStanPy)
   - GridSearchCV (18 combos × 3 folduri = 54 antrenari): 5-15 min CPU multi-core
6. **Tabel comparativ + grafic** la final.
7. **Salvare in DBFS**: model castigator + CSV.

---

## 7. Vezi rezultatele in MLflow UI

In timpul rularii sau la final:

1. Click pe **Experiments** (icon flask in panoul lateral - sau cauta "Experiments" in submeniu).
2. Vei gasi experimentul `/Users/diana_nenu@yahoo.com/disertatie_consum_usa`.
3. Click - vei vedea o tabela cu 6 randuri (cate unul per model: LinearRegression, RandomForest, XGBoost, LSTM, Prophet, XGBoost_tuned).

**Ce poti face in MLflow UI:**

- Sortezi dupa orice metrica (click pe header coloana `rmse` sau `r2`).
- Selectezi 2+ runs > **Compare** - vezi side-by-side.
- Click pe un run > vezi:
  - Parametrii (hyperparametri).
  - Metricile.
  - **Artifacts** - modelul serializat (poti descarca + reincarca).
- Pentru lucrare: poti face **screenshot** al tabelei comparative MLflow ca dovada metodologica.

---

## 8. Aduce rezultatele inapoi in repo GitHub

**Optiunea A** (recomandat) - **Databricks Repos commit**:

1. In notebook, click pe **Git** icon (pe bara de sus a notebook-ului).
2. Vei vedea **Changes**: `notebooks/05_databricks_ml_consum_usa.ipynb` (executat cu output-uri).
3. Mesaj commit: `results: notebook 05 Databricks - rulare full pe GPU`.
4. Click **Commit & Push**.
5. Local in PyCharm: `git pull` - ai notebook-ul executat.

**Optiunea B** - manual (modele + CSV):

Modelele si CSV-urile sunt in `/dbfs/FileStore/disertatie/models/` si `/dbfs/FileStore/disertatie/reports/`. Pentru a le aduce local:

```bash
# In terminal local (din PyCharm)
databricks fs cp dbfs:/FileStore/disertatie/models/usa_winner_xgboost_databricks.json \
                 /Users/diana/PycharmProjects/Disertatie_AI_Platform/models/

databricks fs cp dbfs:/FileStore/disertatie/reports/ml_comparison_usa_databricks.csv \
                 /Users/diana/PycharmProjects/Disertatie_AI_Platform/reports/

cd /Users/diana/PycharmProjects/Disertatie_AI_Platform
git add models/ reports/
git commit -m "results: rezultate Databricks USA"
git push
```

---

## 9. Best practices

**Cluster auto-termination.** Seteaz-o la 30 min - daca uiti notebook-ul deschis, nu pierzi credite.

**MLflow naming.** Daca vrei sa rulezi de mai multe ori cu setari diferite (ex. comparare LSTM 32 vs 64 units), seteaza `run_name` diferit la fiecare apel.

**Git in Databricks.** Cand lucrezi pe Repo, fa commituri **frecvent** ca sa nu pierzi modificari daca cluster-ul moare.

**GPU vs CPU decizie.** Pentru:
- LSTM full (50k × 30 epochs): GPU obligatoriu (altfel iei 30+ min vs 3 min).
- RandomForest, XGBoost: indiferent - sunt mai eficiente pe CPU multi-core decat pe GPU.
- Prophet: CPU - foloseste Stan in C++, GPU-ul nu ajuta.

---

## 10. Troubleshooting

**"No GPU detected" desi cluster-ul e cu GPU.**
- Verifica ca runtime-ul are sufix "ML" (Standard runtime nu are TF GPU).
- `nvidia-smi` in cell-ul de cod ar trebui sa arate placa.

**"Import error: src.data_processing.preprocessing".**
- Verifica ca repo-ul e atasat (`/Workspace/Repos/.../Disertatie_AI_Platform/`).
- Restart Python si re-rul cell-ul de imports.

**"Quota exceeded" sau "Compute limit reached".**
- Contul facultatii are limite. Vezi ce ai disponibil in **Admin Console** sau intreaba responsabilul.
- Ca alternativa, foloseses **Databricks Community Edition** (gratuit, dar limitat la 1 cluster mic) - <https://community.cloud.databricks.com>.

**"Repo: failed to authenticate".**
- Token GitHub expirat. Regenereaza la <https://github.com/settings/tokens> cu scopes `repo, workflow`.
- In Databricks: **Settings > User Settings > Linked Accounts > Git** - inlocuieste tokenul.

**Notebook ramane "Running" infinit.**
- Verifica daca celula e in mod ascuns (output Pretty hidden). Click pe celula > vezi output-uri.
- Daca cluster-ul e adormit, da **Detach > Attach** din nou.
