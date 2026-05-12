"""
BESS schedule evaluation script.
Called by the web app:  python3 scripts/bess.py <csv_path>
Outputs a single JSON object to stdout.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ADD YOUR IMPLEMENTATIONS IN THE SECTION BELOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import io
import json
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # matplotlib optional — only needed when plot=True


# ══════════════════════════════════════════════════════════════════════════════
#  USER-DEFINED COMPONENTS  —  fill in your implementations here
# ═════════════════════════════════════════════════════════════════════════════


from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, Any

import pyomo.environ as pyo


@dataclass
class BESSConfig:
    dt_h: float = 0.25

    # Energy capacity / SOC bounds
    e_max_mwh: float = 100.0
    soc_min_mwh: float = 0.0
    soc_max_mwh: float = 100.0

    # Power limits
    p_charge_max_mw: float = 50.0
    p_discharge_max_mw: float = 50.0

    # Efficiencies
    eta_charge: float = 1.0
    eta_discharge: float = 1.0

    # Initial / final SOC
    soc_init_mwh: float = 0.0
    soc_final_target_mwh: Optional[float] = 0.0

    # Throughput penalty
    throughput_cost_per_mwh: float = 0.0

    # Logic
    prevent_simultaneous: bool = True
    n_cycles: Optional[float] = 2.0

    # Solver
    solver_name: str = "highs"

# BESS configuration — replace with your actual config object
cfg = BESSConfig()

def solve_bess_pyomo(
    prices: Sequence[float] | np.ndarray,
    cfg: BESSConfig,
    *,
    timestamps: Optional[Sequence[Any]] = None,
    scenario_probabilities: Optional[Sequence[float]] = None,
    mode: str = "deterministic",
    cvar_beta: float = 0.95,
    risk_lambda: float = 0.25,
    tee: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame], pyo.ConcreteModel]:

    prices = np.asarray(prices, dtype=float)

    if prices.ndim == 1:
        prices = prices[None, :]
        deterministic_input = True
    elif prices.ndim == 2:
        deterministic_input = (prices.shape[0] == 1)
    else:
        raise ValueError("prices must have shape (T,) or (S,T)")

    S, T = prices.shape
    if S == 0 or T == 0:
        raise ValueError("prices cannot be empty")

    if mode not in {"deterministic", "expected_value", "mean_cvar", "maximin"}:
        raise ValueError("mode must be one of: deterministic, expected_value, mean_cvar, maximin")

    if mode == "deterministic" and S != 1:
        raise ValueError("mode='deterministic' requires one price path of shape (T,) or (1,T)")

    if timestamps is None:
        timestamps = pd.RangeIndex(start=0, stop=T, step=1, name="interval")
    else:
        if len(timestamps) != T:
            raise ValueError("timestamps length must match horizon T")

    if scenario_probabilities is None:
        probs = np.full(S, 1.0 / S, dtype=float)
    else:
        probs = np.asarray(scenario_probabilities, dtype=float)
        if len(probs) != S:
            raise ValueError("scenario_probabilities length must match number of scenarios")
        if np.any(probs < 0):
            raise ValueError("scenario_probabilities must be nonnegative")
        if probs.sum() <= 0:
            raise ValueError("scenario_probabilities must sum to > 0")
        probs = probs / probs.sum()

    m = pyo.ConcreteModel("bess_unified")

    m.S = pyo.RangeSet(0, S - 1)
    m.T = pyo.RangeSet(0, T - 1)
    m.T_SOC = pyo.RangeSet(0, T)

    price_dict = {(s, t): float(prices[s, t]) for s in range(S) for t in range(T)}
    prob_dict = {s: float(probs[s]) for s in range(S)}

    m.price = pyo.Param(m.S, m.T, initialize=price_dict)
    m.prob = pyo.Param(m.S, initialize=prob_dict)

    m.charge_mw = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.discharge_mw = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc_mwh = pyo.Var(
        m.T_SOC,
        domain=pyo.NonNegativeReals,
        bounds=(cfg.soc_min_mwh, cfg.soc_max_mwh),
    )

    if cfg.prevent_simultaneous:
        m.is_charge = pyo.Var(m.T, domain=pyo.Binary)
        m.is_discharge = pyo.Var(m.T, domain=pyo.Binary)

    m.initial_soc = pyo.Constraint(expr=m.soc_mwh[0] == cfg.soc_init_mwh)

    def soc_balance_rule(m, t):
        return m.soc_mwh[t + 1] == (
            m.soc_mwh[t]
            + cfg.eta_charge * m.charge_mw[t] * cfg.dt_h
            - (m.discharge_mw[t] * cfg.dt_h) / cfg.eta_discharge
        )
    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance_rule)

    if cfg.prevent_simultaneous:
        def charge_limit_rule(m, t):
            return m.charge_mw[t] <= cfg.p_charge_max_mw * m.is_charge[t]
        m.charge_limit = pyo.Constraint(m.T, rule=charge_limit_rule)

        def discharge_limit_rule(m, t):
            return m.discharge_mw[t] <= cfg.p_discharge_max_mw * m.is_discharge[t]
        m.discharge_limit = pyo.Constraint(m.T, rule=discharge_limit_rule)

        def no_simultaneous_rule(m, t):
            return m.is_charge[t] + m.is_discharge[t] <= 1
        m.no_simultaneous = pyo.Constraint(m.T, rule=no_simultaneous_rule)
    else:
        def charge_limit_rule(m, t):
            return m.charge_mw[t] <= cfg.p_charge_max_mw
        m.charge_limit = pyo.Constraint(m.T, rule=charge_limit_rule)

        def discharge_limit_rule(m, t):
            return m.discharge_mw[t] <= cfg.p_discharge_max_mw
        m.discharge_limit = pyo.Constraint(m.T, rule=discharge_limit_rule)


    if cfg.n_cycles is not None:
        volume_limit = cfg.e_max_mwh * cfg.n_cycles
        m.charge_cycle_limit = pyo.Constraint(
            expr=sum(m.charge_mw[t] * cfg.dt_h for t in m.T) <= volume_limit
        )
        m.discharge_cycle_limit = pyo.Constraint(
            expr=sum(m.discharge_mw[t] * cfg.dt_h for t in m.T) <= volume_limit
        )

    if cfg.soc_final_target_mwh is not None:
        m.final_soc = pyo.Constraint(expr=m.soc_mwh[T] == cfg.soc_final_target_mwh)

    def scenario_profit_rule(m, s):
        return sum(
            cfg.dt_h * (
                m.price[s, t] * m.discharge_mw[t]
                - m.price[s, t] * m.charge_mw[t]
                - cfg.throughput_cost_per_mwh * (m.charge_mw[t] + m.discharge_mw[t])
            )
            for t in m.T
        )
    m.scenario_profit = pyo.Expression(m.S, rule=scenario_profit_rule)

    m.expected_profit = pyo.Expression(
        expr=sum(m.prob[s] * m.scenario_profit[s] for s in m.S)
    )

    if mode == "deterministic":
        m.obj = pyo.Objective(expr=m.scenario_profit[0], sense=pyo.maximize)

    elif mode == "expected_value":
        m.obj = pyo.Objective(expr=m.expected_profit, sense=pyo.maximize)

    elif mode == "maximin":
        m.worst_case_profit = pyo.Var(domain=pyo.Reals)

        def worst_case_link_rule(m, s):
            return m.worst_case_profit <= m.scenario_profit[s]
        m.worst_case_link = pyo.Constraint(m.S, rule=worst_case_link_rule)

        m.obj = pyo.Objective(expr=m.worst_case_profit, sense=pyo.maximize)

    elif mode == "mean_cvar":
        m.eta = pyo.Var(domain=pyo.Reals)
        m.excess_loss = pyo.Var(m.S, domain=pyo.NonNegativeReals)

        def excess_loss_rule(m, s):
            return m.excess_loss[s] >= -m.scenario_profit[s] - m.eta
        m.excess_loss_def = pyo.Constraint(m.S, rule=excess_loss_rule)

        tail_prob = 1.0 - cvar_beta
        m.cvar_loss = pyo.Expression(
            expr=m.eta + (1.0 / tail_prob) * sum(m.prob[s] * m.excess_loss[s] for s in m.S)
        )

        m.obj = pyo.Objective(
            expr=m.expected_profit - risk_lambda * m.cvar_loss,
            sense=pyo.maximize,
        )


    solver = pyo.SolverFactory(cfg.solver_name)
    if solver is None:
        raise RuntimeError(f"Solver '{cfg.solver_name}' is not available.")

    results = solver.solve(m, tee=tee)
    term = results.solver.termination_condition
    status = results.solver.status

    if term not in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
        pyo.TerminationCondition.feasible,
    }:
        raise RuntimeError(
            f"Solver did not find a usable solution. status={status}, termination={term}"
        )

    charge = np.array([pyo.value(m.charge_mw[t]) for t in m.T], dtype=float)
    discharge = np.array([pyo.value(m.discharge_mw[t]) for t in m.T], dtype=float)
    soc = np.array([pyo.value(m.soc_mwh[t]) for t in m.T_SOC], dtype=float)

    weighted_mean_price = np.average(prices, axis=0, weights=probs)

    schedule = pd.DataFrame(
        {
            "price_reference": weighted_mean_price,
            "charge_mw": charge,
            "discharge_mw": discharge,
            "net_export_mw": discharge - charge,
            "soc_start_mwh": soc[:-1],
            "soc_end_mwh": soc[1:],
            "soc_mwh": soc[1:],
        },
        index=timestamps,
    )

    if cfg.prevent_simultaneous:
        schedule["is_charge"] = np.array([round(pyo.value(m.is_charge[t])) for t in m.T], dtype=int)
        schedule["is_discharge"] = np.array([round(pyo.value(m.is_discharge[t])) for t in m.T], dtype=int)

    schedule["revenue_discharge_ref"] = cfg.dt_h * schedule["price_reference"] * schedule["discharge_mw"]
    schedule["cost_charge_ref"] = cfg.dt_h * schedule["price_reference"] * schedule["charge_mw"]
    schedule["throughput_cost"] = cfg.dt_h * cfg.throughput_cost_per_mwh * (
        schedule["charge_mw"] + schedule["discharge_mw"]
    )
    schedule["interval_profit_ref"] = (
        schedule["revenue_discharge_ref"]
        - schedule["cost_charge_ref"]
        - schedule["throughput_cost"]
    )

    scenario_profit_values = np.array([pyo.value(m.scenario_profit[s]) for s in m.S], dtype=float)
    scenario_results = None
    if S > 1 or mode != "deterministic":
        scenario_results = pd.DataFrame(
            {
                "scenario": np.arange(S, dtype=int),
                "probability": probs,
                "profit": scenario_profit_values,
            }
        )

    summary = {
        "solver_status": str(status),
        "termination_condition": str(term),
        "mode": mode,
        "objective_value": float(pyo.value(m.obj)),
        "expected_profit": float(np.dot(probs, scenario_profit_values)),
        "worst_case_profit": float(np.min(scenario_profit_values)),
        "best_case_profit": float(np.max(scenario_profit_values)),
        "profit_std": float(np.std(scenario_profit_values)),
        "total_charge_mwh": float((schedule["charge_mw"] * cfg.dt_h).sum()),
        "total_discharge_mwh": float((schedule["discharge_mw"] * cfg.dt_h).sum()),
        "final_soc_mwh": float(soc[-1]),
        "n_charge_intervals": int((schedule["charge_mw"] > 1e-6).sum()),
        "n_discharge_intervals": int((schedule["discharge_mw"] > 1e-6).sum()),
    }

    if cfg.n_cycles is not None:
        summary["n_cycles_limit"] = float(cfg.n_cycles)

    if mode == "mean_cvar":
        summary["cvar_beta"] = float(cvar_beta)
        summary["risk_lambda"] = float(risk_lambda)
        summary["cvar_loss"] = float(pyo.value(m.cvar_loss))

    return schedule, summary, scenario_results, m


def realized_profit_from_schedule(
    schedule: pd.DataFrame,
    actual_prices: Sequence[float],
    cfg: BESSConfig,
) -> float:
    """
    Evaluate realized profit of a fixed schedule using actual prices.
    """
    actual_prices = np.asarray(actual_prices, dtype=float)

    if len(actual_prices) != len(schedule):
        raise ValueError("actual_prices length must match schedule length")

    charge = schedule["charge_mw"].to_numpy(dtype=float)
    discharge = schedule["discharge_mw"].to_numpy(dtype=float)

    return float(np.sum(
        cfg.dt_h * (
            actual_prices * discharge
            - actual_prices * charge
            - cfg.throughput_cost_per_mwh * (charge + discharge)
        )
    ))


def make_naive_schedule(cfg: BESSConfig, timestamps) -> pd.DataFrame:
    """
    Charge   04:00–03:45  and  12:00–13:45
    Discharge 06:00–07:45  and  19:00–20:45
    """
    idx = pd.DatetimeIndex(timestamps)
    schedule = pd.DataFrame(index=idx)
    schedule["charge_mw"] = 0.0
    schedule["discharge_mw"] = 0.0

    t_min = idx.hour * 60 + idx.minute  # minutes since midnight

    charge_mask = (
        ((t_min >= 2 * 60) & (t_min <= 3 * 60 + 45)) |
        ((t_min >= 13 * 60) & (t_min <= 14 * 60 + 45))
    )
    discharge_mask = (
        ((t_min >= 6 * 60) & (t_min <= 7 * 60 + 45)) |
        ((t_min >= 19 * 60) & (t_min <= 20 * 60 + 45))
    )

    schedule.loc[charge_mask,    "charge_mw"]    = cfg.p_charge_max_mw
    schedule.loc[discharge_mask, "discharge_mw"] = cfg.p_discharge_max_mw

    net = (cfg.eta_charge * schedule["charge_mw"]
           - schedule["discharge_mw"] / cfg.eta_discharge) * cfg.dt_h
    schedule["soc_mwh"] = (cfg.soc_init_mwh + net.cumsum()).clip(
        cfg.soc_min_mwh, cfg.soc_max_mwh
    )

    return schedule

# ══════════════════════════════════════════════════════════════════════════════
#  INTERVAL EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

try:
    from scipy.stats import norm as _sp_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _normal_ppf(p: float) -> float:
    """Approximate normal quantile without scipy (Beasley-Springer-Moro)."""
    if _HAS_SCIPY:
        return float(_sp_norm.ppf(p))
    # Rational approximation good to ~1e-4 for p in (0.5, 1)
    t = np.sqrt(-2.0 * np.log(1.0 - p)) if p > 0.5 else np.sqrt(-2.0 * np.log(p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + t * (c[1] + t * c[2])
    den = 1.0 + t * (d[0] + t * (d[1] + t * d[2]))
    result = t - num / den
    return result if p > 0.5 else -result


def build_daily_residual_library(
    y_true: pd.Series,
    y_pred: pd.Series,
    steps_per_day: int = 96,
) -> pd.DataFrame:
    """Return a (n_complete_days × steps_per_day) DataFrame of residuals."""
    residuals = (y_true - y_pred).to_numpy()
    n_days = len(residuals) // steps_per_day
    matrix = residuals[: n_days * steps_per_day].reshape(n_days, steps_per_day)
    return pd.DataFrame(matrix)


def generate_mc_price_scenarios_from_intervals(
    y_pred: np.ndarray,
    y_pis: np.ndarray,
    alpha: float,
    n_scenarios: int = 500,
    rho: float = 0.90,
    clip_to_interval: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """
    Draw (n_scenarios, T) price paths with AR(1)-correlated innovations.
    Marginal std is inferred from the PI width under a normal assumption.
    """
    rng = np.random.default_rng(random_state)
    T = len(y_pred)
    z_score = _normal_ppf(1.0 - alpha / 2.0)
    sigma = (y_pis[:, 1] - y_pis[:, 0]) / (2.0 * z_score)
    sigma = np.maximum(sigma, 1e-6)

    scenarios = np.empty((n_scenarios, T), dtype=float)
    z_prev = rng.standard_normal(n_scenarios)
    sqrt1m = np.sqrt(max(1.0 - rho ** 2, 0.0))

    for t in range(T):
        eps = rng.standard_normal(n_scenarios)
        z_t = rho * z_prev + sqrt1m * eps
        z_prev = z_t
        scenarios[:, t] = y_pred[t] + sigma[t] * z_t

    if clip_to_interval:
        scenarios = np.clip(scenarios, y_pis[:, 0], y_pis[:, 1])

    return scenarios


def bootstrap_scenarios_from_residual_days(
    y_pred: np.ndarray,
    residual_library: np.ndarray,
    n_scenarios: int = 500,
    random_state: int = 42,
    clip_low: Optional[np.ndarray] = None,
    clip_high: Optional[np.ndarray] = None,
    recenter: bool = True,
    scale_residuals: float = 1.0,
) -> np.ndarray:
    """Sample whole residual days, add to y_pred, optionally clip to PI."""
    rng = np.random.default_rng(random_state)
    n_days = residual_library.shape[0]
    idx = rng.integers(0, n_days, size=n_scenarios)
    scenarios = y_pred[None, :] + scale_residuals * residual_library[idx, :]
    if recenter:
        scenarios = scenarios - scenarios.mean(axis=0, keepdims=True) + y_pred[None, :]
    if clip_low is not None and clip_high is not None:
        scenarios = np.clip(scenarios, clip_low, clip_high)
    return scenarios


def compute_mapie_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pis: np.ndarray,
    alpha: float,
) -> dict:
    yt = np.asarray(y_true)
    covered = float(np.mean((yt >= y_pis[:, 0]) & (yt <= y_pis[:, 1])))
    avg_width = float(np.mean(y_pis[:, 1] - y_pis[:, 0]))
    return {
        "coverage": covered,
        "avg_width": avg_width,
        "target_coverage": 1.0 - alpha,
    }


_EMPTY_COLS = [
    "charge_pred", "discharge_pred",
    "charge_true", "discharge_true",
    "charge_naive", "discharge_naive",
    "soc_pred", "soc_true", "soc_naive",
    "day_profit_pred", "day_profit_true", "day_profit_naive",
]


def _det_day_task(args):
    """Module-level worker for eval_interval_det — one day, two solves + naive baseline."""
    ts, y_pred_price, y_true_price = args
    schedule, _, _, _ = solve_bess_pyomo(
        prices=y_pred_price, cfg=cfg, timestamps=ts, mode="deterministic"
    )
    schedule_true, _, _, _ = solve_bess_pyomo(
        prices=y_true_price, cfg=cfg, timestamps=ts, mode="deterministic"
    )
    naive = make_naive_schedule(cfg, ts)
    profit       = realized_profit_from_schedule(schedule,       y_true_price, cfg)
    profit_true  = realized_profit_from_schedule(schedule_true,  y_true_price, cfg)
    profit_naive = realized_profit_from_schedule(naive,          y_true_price, cfg)
    return pd.DataFrame(
        {
            "charge_pred":      schedule["charge_mw"].to_numpy(),
            "discharge_pred":   schedule["discharge_mw"].to_numpy(),
            "charge_true":      schedule_true["charge_mw"].to_numpy(),
            "discharge_true":   schedule_true["discharge_mw"].to_numpy(),
            "charge_naive":     naive["charge_mw"].to_numpy(),
            "discharge_naive":  naive["discharge_mw"].to_numpy(),
            "soc_pred":         schedule["soc_mwh"].to_numpy(),
            "soc_true":         schedule_true["soc_mwh"].to_numpy(),
            "soc_naive":        naive["soc_mwh"].to_numpy(),
            "day_profit_pred":  profit,
            "day_profit_true":  profit_true,
            "day_profit_naive": profit_naive,
        },
        index=ts,
    )


def _stoch_day_task(args):
    """Module-level worker for eval_interval_stoch — one day, stoch + oracle + naive."""
    ts, y_pred_day, y_true_day, y_pis_day, alpha, n_scenarios, mode, scenarios, resid_lib = args
    if scenarios == "mc":
        price_scenarios = generate_mc_price_scenarios_from_intervals(
            y_pred=y_pred_day, y_pis=y_pis_day, alpha=alpha,
            n_scenarios=n_scenarios, rho=0.90, clip_to_interval=True, random_state=42,
        )
    elif scenarios == "bootstrap":
        price_scenarios = bootstrap_scenarios_from_residual_days(
            y_pred=y_pred_day, residual_library=resid_lib,
            n_scenarios=n_scenarios, random_state=42,
            clip_low=y_pis_day[:, 0], clip_high=y_pis_day[:, 1],
            recenter=True, scale_residuals=1.0,
        )
    else:
        raise ValueError("scenarios must be one of: mc, bootstrap")
    schedule_stoch, _, _, _ = solve_bess_pyomo(
        prices=price_scenarios, cfg=cfg, timestamps=ts,
        mode=mode, cvar_beta=0.95, risk_lambda=0.05,
    )
    schedule_true, _, _, _ = solve_bess_pyomo(
        prices=y_true_day, cfg=cfg, timestamps=ts, mode="deterministic",
    )
    naive = make_naive_schedule(cfg, ts)
    profit       = realized_profit_from_schedule(schedule_stoch, y_true_day, cfg)
    profit_true  = realized_profit_from_schedule(schedule_true,  y_true_day, cfg)
    profit_naive = realized_profit_from_schedule(naive,          y_true_day, cfg)
    return pd.DataFrame(
        {
            "charge_pred":      schedule_stoch["charge_mw"].to_numpy(),
            "discharge_pred":   schedule_stoch["discharge_mw"].to_numpy(),
            "charge_true":      schedule_true["charge_mw"].to_numpy(),
            "discharge_true":   schedule_true["discharge_mw"].to_numpy(),
            "charge_naive":     naive["charge_mw"].to_numpy(),
            "discharge_naive":  naive["discharge_mw"].to_numpy(),
            "soc_pred":         schedule_stoch["soc_mwh"].to_numpy(),
            "soc_true":         schedule_true["soc_mwh"].to_numpy(),
            "soc_naive":        naive["soc_mwh"].to_numpy(),
            "day_profit_pred":  profit,
            "day_profit_true":  profit_true,
            "day_profit_naive": profit_naive,
        },
        index=ts,
    )

def eval_interval_naive(
    df_p: pd.DataFrame,
    rolling_widow : int,
) -> pd.DataFrame:

    start_day = df_p.index.min().normalize()
    end_day   = df_p.index.max().normalize()

    resid_lib_arr = build_daily_residual_library(
        y_true=df_p["y_true"], y_pred=df_p["y_pred"], steps_per_day=96,
    ).to_numpy() if scenarios == "bootstrap" else None

    tasks = []
    for day in pd.date_range(start_day, end_day, freq="D", tz="Europe/Prague"):
        mask_day = df_p.index.normalize() == day
        if mask_day.sum() != 96:
            continue
        ts         = df_p.loc[mask_day].index
        y_pred_day = df_p.loc[mask_day, "y_pred"].to_numpy()
        y_true_day = df_p.loc[mask_day, "y_true"].to_numpy()
        y_pis_day  = df_p.loc[mask_day][["pi_lo", "pi_hi"]].to_numpy()
        tasks.append((ts, y_pred_day, y_true_day, y_pis_day, alpha, n_scenarios, mode, scenarios, resid_lib_arr))

    if not tasks:
        return pd.DataFrame(columns=_EMPTY_COLS)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_stoch_day_task, tasks))

    return pd.concat(results)


def eval_interval_stoch(
    df_p: pd.DataFrame,
    n_scenarios: int = 1000,
    mode: str = "mean_cvar",
    scenarios: str = "mc",
    resid_lib_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.10,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Stochastic BESS evaluation using prediction-interval scenarios.
    Returns the same DataFrame structure as eval_interval_det so that
    main() can process both paths identically.
    """
    start_day = df_p.index.min().normalize()
    end_day   = df_p.index.max().normalize()

    resid_lib_arr = build_daily_residual_library(
        y_true=df_p["y_true"], y_pred=df_p["y_pred"], steps_per_day=96,
    ).to_numpy() if scenarios == "bootstrap" else None

    tasks = []
    for day in pd.date_range(start_day, end_day, freq="D", tz="Europe/Prague"):
        mask_day = df_p.index.normalize() == day
        if mask_day.sum() != 96:
            continue
        ts         = df_p.loc[mask_day].index
        y_pred_day = df_p.loc[mask_day, "y_pred"].to_numpy()
        y_true_day = df_p.loc[mask_day, "y_true"].to_numpy()
        y_pis_day  = df_p.loc[mask_day][["pi_lo", "pi_hi"]].to_numpy()
        tasks.append((ts, y_pred_day, y_true_day, y_pis_day, alpha, n_scenarios, mode, scenarios, resid_lib_arr))

    if not tasks:
        return pd.DataFrame(columns=_EMPTY_COLS)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_stoch_day_task, tasks))

    return pd.concat(results)


# ══════════════════════════════════════════════════════════════════════════════
#  PROVIDED EVALUATION FUNCTION  (do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def eval_interval_det(df_p: pd.DataFrame, plot: bool = False) -> pd.DataFrame:
    start_day = df_p.index.min().normalize()
    end_day   = df_p.index.max().normalize()

    tasks = []
    for day in pd.date_range(start_day, end_day, freq="D", tz="Europe/Prague"):
        mask_day = df_p.index.normalize() == day
        if mask_day.sum() != 96:
            continue
        day_idx      = df_p.loc[mask_day].index
        y_true_price = df_p.loc[mask_day, "y_true"].to_numpy()
        y_pred_price = df_p.loc[mask_day, "y_pred"].to_numpy()
        tasks.append((day_idx, y_pred_price, y_true_price))

    if not tasks:
        return pd.DataFrame(columns=_EMPTY_COLS)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_det_day_task, tasks))

    return pd.concat(results)



def main():
    import pickle
    import os

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: bess.py <csv_path> [--robust]"}))
        sys.exit(1)

    csv_path = sys.argv[1]
    robust = "--robust" in sys.argv[2:]

    # Infer alpha from filename (e.g. interval_preditions_80 → 80 % PI → alpha=0.20)
    alpha = 0.05
    if "_80" in csv_path:
        alpha = 0.20
    elif "_90" in csv_path:
        alpha = 0.10
    elif "_95" in csv_path:
        alpha = 0.05

    # Cache file: same dir as input CSV, stem + _solved_det/_solved_robust.pkl
    base = os.path.splitext(csv_path)[0]
    mode_tag = "robust" if robust else "det"
    cache_path = f"{base}_solved_{mode_tag}.pkl"

    # Load from cache if it exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                result_df = pickle.load(fh)
        except Exception as e:
            print(json.dumps({"error": f"Failed to load cache: {e}"}))
            sys.exit(1)
    else:
        # Load CSV
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(json.dumps({"error": f"Failed to load CSV: {e}"}))
            sys.exit(1)

        # Ensure UTC-aware index, then shift to Prague local time
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index = df.index.tz_convert("Europe/Prague")

        # Run evaluation — capture stdout (the function prints per-day progress)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                if robust:
                    result_df = eval_interval_stoch(df, alpha=alpha, plot=False)
                else:
                    result_df = eval_interval_det(df, plot=False)
        except NotImplementedError as e:
            print(json.dumps({"error": f"Not implemented: {e}"}))
            sys.exit(1)
        except Exception as e:
            print(json.dumps({"error": f"Evaluation failed: {e}"}))
            sys.exit(1)

        # Save result to cache
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(result_df, fh)
        except Exception as e:
            sys.stderr.write(f"Warning: could not write cache: {e}\n")

    # Build daily_profits from result_df (day_profit_* is repeated per interval, take first per day)
    daily_profit_map = {}
    for ts, row in result_df.iterrows():
        date_str = str(ts.date())
        if date_str not in daily_profit_map:
            daily_profit_map[date_str] = {
                "pred":  round(float(row["day_profit_pred"]),  4),
                "true":  round(float(row["day_profit_true"]),  4),
                "naive": round(float(row["day_profit_naive"]), 4),
            }

    daily_profits = [
        {"date": date, "predProfit": v["pred"], "trueProfit": v["true"], "naiveProfit": v["naive"]}
        for date, v in sorted(daily_profit_map.items())
    ]

    pred_total  = sum(d["predProfit"]  for d in daily_profits)
    true_total  = sum(d["trueProfit"]  for d in daily_profits)
    naive_total = sum(d["naiveProfit"] for d in daily_profits)
    n = len(daily_profits)
    summary = {
        "predProfit":    round(pred_total,  4),
        "trueProfit":    round(true_total,  4),
        "naiveProfit":   round(naive_total, 4),
        "lostProfit":    round(true_total - pred_total, 4),
        "lostProfitAvg": round((true_total - pred_total) / n, 4) if n else None,
        "daysEvaluated": n,
    }

    # Build schedule records
    records = []
    for ts, row in result_df.iterrows():
        records.append({
            "ts":             ts.strftime("%Y-%m-%d %H:%M"),
            "chargePred":     round(float(row["charge_pred"]),      6),
            "dischargePred":  round(float(row["discharge_pred"]),   6),
            "chargeTrue":     round(float(row["charge_true"]),      6),
            "dischargeTrue":  round(float(row["discharge_true"]),   6),
            "chargeNaive":    round(float(row["charge_naive"]),     6),
            "dischargeNaive": round(float(row["discharge_naive"]),  6),
            "socPred":        round(float(row["soc_pred"]),         4),
            "socTrue":        round(float(row["soc_true"]),         4),
            "socNaive":       round(float(row["soc_naive"]),        4),
            "dayProfitPred":  round(float(row["day_profit_pred"]),  4),
            "dayProfitTrue":  round(float(row["day_profit_true"]),  4),
            "dayProfitNaive": round(float(row["day_profit_naive"]), 4),
        })

    print(json.dumps({
        "schedule":     records,
        "dailyProfits": daily_profits,
        "summary":      summary,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
