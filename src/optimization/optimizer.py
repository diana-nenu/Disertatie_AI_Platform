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


# ===========================================================================
# Problema 1: Dispatch optim al unei baterii (arbitraj de pret) - Spania
# ===========================================================================

@dataclass
class BatteryConfig:
    """Parametrii fizici ai bateriei pentru problema de dispatch.

    capacity: capacitatea utila a bateriei (MWh).
    p_max: puterea maxima de charge/discharge (MW) pe interval de o ora.
    soc_init: starea de incarcare initiala, ca fractiune din capacitate [0, 1].
    lambda_deg: coeficient de penalizare a degradarii (cost ~ patratul puterii).
                Modeleaza pierderile round-trip si uzura bateriei; face obiectivul NELINIAR.
    cyclic: daca True, bateria revine la SOC initial la final (sum(x) = 0),
            ca sa nu obtinem profit "golind" bateria.
    """
    capacity: float = 10.0
    p_max: float = 2.5
    soc_init: float = 0.5
    lambda_deg: float = 0.01
    cyclic: bool = True


def battery_soc(x: np.ndarray, cfg: BatteryConfig) -> np.ndarray:
    """Traiectoria starii de incarcare (MWh) data fiind schema de dispatch x.

    Conventie: x_t > 0 = descarcare (vindem in retea), x_t < 0 = incarcare (cumparam).
    Descarcarea scade SOC: E_t = E_{t-1} - x_t (pas de o ora, deci MW -> MWh).
    """
    e0 = cfg.soc_init * cfg.capacity
    return e0 - np.cumsum(x)


def battery_profit(x: np.ndarray, prices: np.ndarray, cfg: BatteryConfig) -> dict:
    """Descompune rezultatul economic al unei scheme de dispatch.

    revenue: incasari nete din vanzare minus cost cumparare = sum(price_t * x_t).
    degradation: cost de uzura/pierderi = lambda * sum(x_t^2).
    net_profit: revenue - degradation (ce maximizeaza optimizarea).
    """
    revenue = float(np.sum(prices * x))
    degradation = float(cfg.lambda_deg * np.sum(x ** 2))
    return {
        "revenue": revenue,
        "degradation": degradation,
        "net_profit": revenue - degradation,
    }


def battery_dispatch_problem(prices: np.ndarray, cfg: BatteryConfig) -> dict:
    """Construieste componentele problemei de optimizare pentru dispatch-ul bateriei.

    Variabile de decizie: x_t (MW) pentru fiecare ora, x_t in [-p_max, p_max].
    Obiectiv (de MAXIMIZAT): profit = sum(price_t * x_t) - lambda * sum(x_t^2).
        Pentru scipy.minimize returnam negativul: f(x) = -profit.
        Termenul patratic (degradarea) face obiectivul NELINIAR (convex quadratic),
        justificand folosirea unui optimizator neliniar (SLSQP).
    Constrangeri:
        - bounds: -p_max <= x_t <= p_max (putere limitata).
        - inegalitate: 0 <= SOC_t <= capacity pentru fiecare t (bateria nu se supra/sub-incarca).
        - egalitate (daca cyclic): SOC_final = SOC_init  <=>  sum(x) = 0.

    Returneaza un dict cu cheile: objective, x0, bounds, constraints - gata
    pentru a fi pasate functiei optimize_nonlinear().
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    cap = cfg.capacity
    e0 = cfg.soc_init * cap

    def objective(x: np.ndarray) -> float:
        revenue = np.sum(prices * x)
        degradation = cfg.lambda_deg * np.sum(x ** 2)
        return -(revenue - degradation)  # minimizam negativul profitului

    bounds = [(-cfg.p_max, cfg.p_max)] * T

    constraints = []
    # SOC >= 0 pentru orice t:  e0 - cumsum(x)[t] >= 0
    constraints.append({
        "type": "ineq",
        "fun": lambda x: e0 - np.cumsum(x),  # vector >= 0
    })
    # SOC <= capacity pentru orice t:  cap - (e0 - cumsum(x)[t]) >= 0
    constraints.append({
        "type": "ineq",
        "fun": lambda x: cap - (e0 - np.cumsum(x)),  # vector >= 0
    })
    if cfg.cyclic:
        constraints.append({
            "type": "eq",
            "fun": lambda x: np.sum(x),  # sum(x) = 0  =>  SOC_final = SOC_init
        })

    x0 = np.zeros(T)  # pornim de la "bateria inactiva"
    return {"objective": objective, "x0": x0, "bounds": bounds, "constraints": constraints}


def solve_battery_dispatch(prices: np.ndarray, cfg: BatteryConfig | None = None,
                           max_iter: int = 1000) -> dict:
    """Rezolva problema de dispatch si returneaza solutia + diagnosticul economic.

    Returneaza dict cu: x (schema optima), soc (traiectorie), result (OptimizationResult),
    profit (descompunere economica), baseline_naive (profit fara optimizare = 0).
    """
    cfg = cfg or BatteryConfig()
    prob = battery_dispatch_problem(prices, cfg)
    result = optimize_nonlinear(
        prob["objective"], x0=prob["x0"], bounds=prob["bounds"],
        constraints=prob["constraints"], method="SLSQP", max_iter=max_iter,
    )
    x = result.x_optim
    return {
        "x": x,
        "soc": battery_soc(x, cfg),
        "result": result,
        "profit": battery_profit(x, np.asarray(prices, dtype=float), cfg),
        "cfg": cfg,
    }


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
