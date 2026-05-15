# Notebook-uri didactice SCSS - Feature Engineering pentru time series energetice

**Autor:** Diana Nenu  ·  Master DSAI, Universitatea Titu Maiorescu  ·  SCSS 2026

Acest folder contine 3 notebook-uri Jupyter care explica concret, pe cod si numere, cele 3 tehnici de
feature engineering folosite in capitolele 4.2-4.5 ale disertatiei.

## Ordine recomandata

### fe_01_encoding_ciclic.ipynb
Problema reprezentarii liniare a orei (23 si 0 par departe). Solutia matematica cu sin/cos pe cerc.
Verificare numerica + impact pe model (Ridge cu vs fara encoding ciclic).

### fe_02_lag_si_rolling.ipynb
Lag features (memoria scurta) + rolling features (trend si volatilitate). Vizualizari pe date
sintetice tip pret energetic. Corelatii numerice cu pretul real.

### fe_03_anti_leakage_shift1.ipynb
**Notebook-ul cel mai important.** Demonstreaza concret cum implementarea naiva a rolling features
introduce data leakage. Antreneaza 2 XGBoost paralel - cu si fara shift(1) - si arata diferenta de R².
Discutie despre cum se identifica acest bug in audit.

## Rulare

Notebook-urile sunt self-contained (genereaza propriile date sintetice). Nu necesita modele antrenate
sau fisiere externe. Ruleaza in 5-10 secunde fiecare.

Dependente: numpy, pandas, matplotlib, scikit-learn, xgboost (toate in requirements.txt).

## Conexiunea cu disertatia

| Notebook | Capitol Disertatie |
|---|---|
| fe_01 | 4.2 - Encoding ciclic pentru variabilele temporale |
| fe_02 | 4.3 - Lag features + 4.4 - Rolling features |
| fe_03 | 4.5 - Prevenirea data leakage |

## Materiale conexe pentru SCSS
- `../SCSS_FeatureEngineering_Diana_Nenu.pptx` - prezentarea principala (13 slide-uri)
- `../SCSS_FeatureEngineering_Handout_Diana_Nenu.pdf` - handout 2 pagini
