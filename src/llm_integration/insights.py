"""
Integrare LLM (HuggingFace) pentru generarea automată de insight-uri și interpretări.

Folosește un model text-to-text (ex: flan-t5) pentru:
    - Explicarea în limbaj natural a rezultatelor de la modelele ML
    - Sumarizarea recomandărilor de la optimizarea neliniară
    - Generarea de descrieri pentru rapoartele Streamlit

Pentru rulare locală fără API:
    pipeline("text2text-generation", model="google/flan-t5-base")
"""

from typing import Any

# Lazy import - transformers e o dependență grea, o încărcăm doar când e nevoie.


def get_pipeline(model_name: str = "google/flan-t5-base", device: str = "cpu"):
    """Construiește un pipeline HuggingFace text2text-generation."""
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model=model_name,
        device=-1 if device == "cpu" else 0,
    )


def explain_prediction(
    metrics: dict[str, float],
    context: str,
    pipe: Any | None = None,
    max_new_tokens: int = 200,
) -> str:
    """
    Generează o explicație în limbaj natural pentru rezultatul unui model.

    Args:
        metrics: dicționar de metrici (rmse, mae, r2, ...)
        context: descrierea problemei (ex: "predicția consumului orar pe rețeaua PJM")
        pipe: pipeline HuggingFace (dacă None, se creează unul implicit)
        max_new_tokens: lungimea maximă a output-ului

    Returns:
        Text explicativ.
    """
    if pipe is None:
        pipe = get_pipeline()

    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    prompt = (
        f"Explain in Romanian the following machine learning result for a "
        f"non-technical reader. Context: {context}. Metrics: {metrics_str}. "
        f"Provide an interpretation and what the metrics imply about model quality."
    )
    output = pipe(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
    return output


def summarize_optimization(
    x_optim: list[float],
    f_optim: float,
    problem_description: str,
    pipe: Any | None = None,
) -> str:
    """Generează o recomandare prescriptivă din rezultatul optimizării."""
    if pipe is None:
        pipe = get_pipeline()

    prompt = (
        f"Generate a Romanian prescriptive recommendation based on the optimization result. "
        f"Problem: {problem_description}. Optimal solution x = {x_optim}. "
        f"Optimal objective value = {f_optim:.4f}. Explain what action should be taken."
    )
    return pipe(prompt, max_new_tokens=200)[0]["generated_text"]


if __name__ == "__main__":
    # Demo (necesită transformers instalat și descărcarea modelului)
    print("Testez pipeline-ul... (poate dura la primul import)")
    pipe = get_pipeline()
    print(explain_prediction(
        metrics={"rmse": 1234.5, "mae": 980.2, "r2": 0.87},
        context="predicția consumului energetic orar pe rețeaua PJM (USA)",
        pipe=pipe,
    ))
