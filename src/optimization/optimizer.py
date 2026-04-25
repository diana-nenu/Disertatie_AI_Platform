"""
Optimizare neliniară cu SciPy pentru recomandări prescriptive.

Exemple de probleme țintă pentru disertație:
    1. Solar India: maximizarea producției AC ajustând orientare/înclinare panouri
       (variabile continue, constrâns de limite fizice).
    2. Consum USA: minimizarea costului de operare prin ajustare load shifting.
    3. Spania: dispatch optim baterie (charge/discharge) în funcție de prețuri prognozate.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Rezultat al unei probleme de optimizare."""
    x_optim: np.ndarray
    f_optim: float
    success: bool
    message: str
    n_iter: int


def optimize_nonlinear(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    constraints: list[dict] | None = None,
    method: str = "SLSQP",
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> OptimizationResult:
    """
    Wrapper general peste scipy.optimize.minimize.

    Args:
        objective: funcția de minimizat f(x) -> float (pentru maximizare, returnează -f(x))
        x0: punct inițial
        bounds: limite pe variabile [(min, max), ...]
        constraints: lista de constrângeri SciPy (eq/ineq)
        method: algoritm (SLSQP, trust-constr, COBYLA)
        max_iter: număr maxim de iterații
        tol: toleranță

    Returns:
        OptimizationResult cu valoarea optimă și diagnostic.
    """
    result = minimize(
        objective,
        x0=x0,
        method=method,
        bounds=bounds,
        constraints=constraints or [],
        options={"maxiter": max_iter, "ftol": tol},
    )
    return OptimizationResult(
        x_optim=result.x,
        f_optim=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        n_iter=int(result.nit) if hasattr(result, "nit") else -1,
    )


# -------------------------------------------------------------------
# TODO: Probleme concrete pentru disertație
# -------------------------------------------------------------------

def battery_dispatch_problem(prices: np.ndarray, capacity: float, eta: float = 0.9):
    """
    Schemă: dispatch optim al unei baterii pentru maximizarea profitului.

    Variabile: x_t in [-P_max, P_max] (negativ = charge, pozitiv = discharge)
    Obiectiv: max sum(price_t * x_t)
    Constrângeri:
        - SOC_t in [SOC_min, SOC_max]
        - SOC_t = SOC_{t-1} - eta * x_t
        - SOC_T = SOC_0  (opțional)

    TODO: implementare completă.
    """
    raise NotImplementedError("De implementat în următoarea iterație")


if __name__ == "__main__":
    # Demo: minimizarea funcției Rosenbrock în 2D cu bounds
    def rosenbrock(x: np.ndarray) -> float:
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    result = optimize_nonlinear(
        rosenbrock,
        x0=np.array([0.0, 0.0]),
        bounds=[(-2, 2), (-2, 2)],
    )
    print(f"x* = {result.x_optim}")
    print(f"f* = {result.f_optim:.6e}")
    print(f"success: {result.success} | iterații: {result.n_iter}")
