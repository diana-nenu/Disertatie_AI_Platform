# Notebook-uri didactice SCSS 2026 - SHAP si teoria jocurilor cooperative

**Autor:** Diana Nenu
**Curs:** Master Data Science & AI - Universitatea Titu Maiorescu
**Context:** material de pregatire pentru SCSS 2026 si anexa metodologica la lucrarea de disertatie

## Ordinea recomandata

Notebook-urile sunt progresive: fiecare construieste pe intelegerea celui anterior. Recomand parcurgerea
in ordine:

### 01. Jocul cooperativ si valoarea Shapley
**`01_jocul_cooperativ_si_valoarea_shapley.ipynb`**

Fundamentele teoriei jocurilor cooperative. Implementare in Python a doua formule echivalente de
calcul al valorii Shapley (combinatoriala vs prin permutari) si verificare numerica a echivalentei.
Exemplu cu 3 features inspirat din modelul Spania.

### 02. Demonstrarea celor 4 axiome
**`02_demonstrarea_celor_4_axiome.ipynb`**

Pentru fiecare din cele 4 axiome (eficienta, simetrie, dummy, aditivitate) construiesc un joc
ad-hoc care evidentiaza comportamentul axiomei si o demonstrez numeric. Schita demonstratiei
de unicitate a teoremei lui Shapley.

### 03. Translatia de la Shapley la SHAP
**`03_de_la_shapley_la_shap_translatia.ipynb`**

Construiesc DUBLU valorile contributiilor pentru un model XGBoost cu 3 features sintetice:
- Valoarea Shapley clasica prin retraining pe fiecare subset
- Valoarea SHAP prin TreeSHAP pe modelul unic

Demonstrez cand cele doua coincid si cand difera, explicand diferentele conceptuale.

### 04. TreeSHAP pe modelul propriu Spania
**`04_treeshap_pe_modelul_meu_spania.ipynb`**

Aplicare directa pe modelul XGBoost antrenat real (`models/spania_winner_xgboost.json`).
Calculez SHAP pe 500 esantioane, validez aditivitatea, generez 3 waterfall-uri (pret ridicat /
mediu / scazut) si compar cu feature_importances_ clasic.

**Prerequisite:** rularea `notebooks/06_ml_pret_spania.ipynb` din directorul radacina pentru a
salva modelul si datele procesate.

### 05. Complexitate si eficienta algoritmilor
**`05_complexitate_si_eficienta_algoritmi.ipynb`**

Analiza empirica a complexitatii. Demonstrez exploziunea combinatoriala $O(2^n)$ a Shapley exact
si o compar cu complexitatea polinomiala $O(T \cdot L \cdot D^2)$ a TreeSHAP. Inteleg de ce
TreeSHAP a fost inovatia decisiva care a facut SHAP practic pentru ML.

## Dependente Python

```
numpy
pandas
matplotlib
scikit-learn
xgboost >= 3.0
shap >= 0.49 (opțional pentru notebook 3)
```

Toate sunt deja in `requirements.txt` al proiectului parinte.

## Rulare

Din PyCharm:
1. Deschide notebook-ul dorit
2. Apasa "Run All" sau ruleaza celula cu celula
3. Toate notebook-urile sunt self-contained (in afara de #4 care necesita modelul antrenat)

## Materiale conexe

- `../SCSS_SHAP_Shapley_Diana_Nenu.pptx` - prezentarea principala (13 slide-uri)
- `../SCSS_Handout_SHAP_Diana_Nenu.pdf` - handout 2 pagini pentru profesori
- `../../Disertatie.docx` subcapitolul 5.6 - tratamentul academic complet
