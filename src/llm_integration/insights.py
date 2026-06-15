"""
Integrare LLM (HuggingFace) pentru generarea automata de insight-uri si interpretari.

Componenta de limbaj natural a platformei: traduce rezultatele numerice (metrici ML,
importanta features, recomandari de optimizare) in explicatii pe intelesul unui operator.

Doua nivele, complementare:
  1. LLM (HuggingFace flan-t5) - model open-source text-to-text, ruleaza local fara API key.
     Foloseste prompt engineering pentru a genera/rafina explicatii.
  2. Generator determinist pe sabloane (in romana) - reproductibil, robust, folosit ca
     fallback cand modelul nu e disponibil si ca baseline de comparatie.

Modelul implicit: google/flan-t5-base (mic, instruction-tuned, ruleaza pe CPU).
"""

from __future__ import annotations
from typing import Any, Sequence

import numpy as np


# ===========================================================================
# 1. Backend LLM (HuggingFace flan-t5)
# ===========================================================================

def get_pipeline(model_name: str = "google/flan-t5-base", device: str = "cpu"):
    """Construieste un pipeline HuggingFace text2text-generation.

    La prima rulare descarca modelul (~1 GB pentru flan-t5-base; ~300 MB pentru -small).
    Lazy import: transformers e o dependenta grea, o incarcam doar la cerere.
    """
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model=model_name,
        device=-1 if device == "cpu" else 0,
    )


def llm_generate(prompt: str, pipe: Any, max_new_tokens: int = 160,
                 temperature: float = 0.0) -> str:
    """Genereaza text cu un pipeline flan-t5.

    temperature=0 -> greedy (determinist); temperature>0 -> sampling (mai variat).
    max_new_tokens controleaza lungimea maxima a raspunsului.
    """
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature))
    out = pipe(prompt, **kwargs)
    return out[0]["generated_text"].strip()


# ===========================================================================
# 2. Generator determinist pe sabloane (romana, reproductibil)
# ===========================================================================

def _quality_phrase(r2: float) -> str:
    if r2 >= 0.99:
        return "o precizie excelenta - predictiile sunt foarte apropiate de valorile reale"
    if r2 >= 0.95:
        return "o precizie foarte buna, utila in practica"
    if r2 >= 0.85:
        return "o precizie buna, dar cu loc de imbunatatire"
    return "o precizie limitata - modelul ar trebui revizuit"


def explain_prediction_template(dataset: str, target: str, metrics: dict[str, float],
                                unit: str = "", top_features: Sequence[str] | None = None) -> str:
    """Explicatie determinista, in romana, a rezultatului unui model ML."""
    r2 = metrics.get("r2", float("nan"))
    mae = metrics.get("mae", float("nan"))
    mape = metrics.get("mape", float("nan"))
    u = f" {unit}" if unit else ""
    parts = [
        f"Modelul pentru {dataset} prezice {target} cu {_quality_phrase(r2)}.",
        f"Coeficientul de determinare R2 este {r2:.3f}, adica modelul explica "
        f"{r2*100:.1f}% din variatia tintei.",
        f"Eroarea medie absoluta este de aproximativ {mae:.2f}{u}"
        + (f", iar eroarea procentuala medie (MAPE) de {mape:.1f}%." if mape == mape else "."),
    ]
    if top_features:
        feats = ", ".join(top_features[:3])
        parts.append(f"Cei mai influenti factori in predictie sunt: {feats}.")
    return " ".join(parts)


def _hour_ranges(hours: list[int]) -> str:
    """Comprima o lista de ore (0-23) in intervale lizibile, ex: '02:00-05:00, 11:00, 13:00-16:00'."""
    hrs = sorted(set(int(h) % 24 for h in hours))
    if not hrs:
        return "niciuna"
    out, start, prev = [], hrs[0], hrs[0]
    for h in hrs[1:]:
        if h == prev + 1:
            prev = h
        else:
            out.append((start, prev)); start = prev = h
    out.append((start, prev))
    return ", ".join(f"{a:02d}:00" if a == b else f"{a:02d}:00-{b:02d}:00" for a, b in out)


def summarize_dispatch_template(prices: Sequence[float], x: Sequence[float],
                                profit: float, currency: str = "EUR") -> str:
    """Recomandare determinista, scurta si schematica, pentru dispatch-ul bateriei.

    Orele sunt deduplicate (pe ora din zi) si comprimate in intervale, ca sa fie lizibile
    chiar si pe ferestre lungi (mai multe zile).
    """
    x = np.asarray(x, dtype=float)
    charge_h = [i for i, v in enumerate(x) if v < -1e-3]
    discharge_h = [i for i, v in enumerate(x) if v > 1e-3]
    return (
        f"Strategia: incarci (cumperi) in orele cu pret scazut ({_hour_ranges(charge_h)}) "
        f"si descarci (vinzi) in orele cu pret ridicat ({_hour_ranges(discharge_h)}). "
        f"Profit estimat: ~{profit:.0f} {currency}, respectand limitele bateriei."
    )


def summarize_load_shifting_template(savings_pct: float, peak_reduction_pct: float | None = None,
                                     currency: str = "EUR") -> str:
    """Recomandare determinista pentru load shifting."""
    txt = (
        f"Recomandare de gestionare a consumului: muta consumul flexibil din orele "
        f"de varf (seara) catre orele ieftine (noaptea). Aceasta reorganizare reduce "
        f"factura cu aproximativ {savings_pct:.1f}%, fara a scadea energia totala consumata."
    )
    if peak_reduction_pct is not None:
        txt += f" Consumul la varf scade cu circa {peak_reduction_pct:.0f}%, usurand presiunea pe retea."
    return txt


# ===========================================================================
# 3. Functii de nivel inalt: LLM cu fallback determinist
# ===========================================================================

def explain_prediction(dataset: str, target: str, metrics: dict[str, float],
                       unit: str = "", top_features: Sequence[str] | None = None,
                       pipe: Any | None = None, use_llm: bool = False,
                       max_new_tokens: int = 160) -> dict[str, str]:
    """Explica un rezultat ML. Returneaza dict cu 'template' (mereu) si 'llm' (daca use_llm).

    - Varianta determinista (template) e mereu produsa: romana corecta, reproductibila.
    - Daca use_llm=True si pipe e dat (sau se creeaza), flan-t5 genereaza o varianta proprie
      pornind de la aceleasi fapte, prin prompt engineering.
    """
    template = explain_prediction_template(dataset, target, metrics, unit, top_features)
    result = {"template": template}
    if use_llm:
        pipe = pipe or get_pipeline()
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        feats = ", ".join(top_features[:3]) if top_features else "n/a"
        prompt = (
            "You are an energy analyst. Explain, for a non-technical reader, this machine "
            f"learning result. Context: predicting {target} for {dataset}. "
            f"Metrics: {metrics_str}. Most important features: {feats}. "
            "Give a short, clear interpretation of the model quality."
        )
        result["llm"] = llm_generate(prompt, pipe, max_new_tokens=max_new_tokens)
    return result


def summarize_optimization(kind: str, pipe: Any | None = None, use_llm: bool = False,
                           max_new_tokens: int = 160, **data) -> dict[str, str]:
    """Genereaza recomandarea prescriptiva pentru o problema de optimizare.

    kind: 'battery' (necesita prices, x, profit) sau 'load_shifting' (necesita savings_pct).
    """
    if kind == "battery":
        template = summarize_dispatch_template(data["prices"], data["x"], data["profit"],
                                               data.get("currency", "EUR"))
    elif kind == "load_shifting":
        template = summarize_load_shifting_template(data["savings_pct"],
                                                    data.get("peak_reduction_pct"),
                                                    data.get("currency", "EUR"))
    else:
        raise ValueError(f"kind necunoscut: {kind!r}")

    result = {"template": template}
    if use_llm:
        pipe = pipe or get_pipeline()
        prompt = (
            "You are an energy operations assistant. Rephrase the following recommendation "
            f"clearly for an operator: {template}"
        )
        result["llm"] = llm_generate(prompt, pipe, max_new_tokens=max_new_tokens)
    return result


if __name__ == "__main__":
    # Demo determinist (nu necesita modelul descarcat)
    print(explain_prediction(
        dataset="pretul energiei in Spania", target="pretul orar (EUR/MWh)",
        metrics={"rmse": 2.00, "mae": 1.49, "r2": 0.9696, "mape": 2.48}, unit="EUR/MWh",
        top_features=["price actual_lag_1", "price day ahead", "price actual_roll_mean_3"],
    )["template"])
    print()
    print(summarize_optimization(
        kind="battery", prices=[50, 40, 80], x=[-2.0, -1.5, 2.5], profit=561.0,
    )["template"])
