"""
calibration.py
==============
Model calibration, sensitivity analysis, and uncertainty quantification.

- Least-squares calibration (scipy.optimize.minimize)
- Bayesian MCMC (Metropolis-Hastings)
- Sobol sensitivity indices (Saltelli sampling)
- Performance metrics: RMSE, NSE, MAE, R²
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Callable, Optional
import copy

from config import ModelConfig
from solver import run_simulation


# ================================================================
# PERFORMANCE METRICS
# ================================================================
def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((obs - sim) ** 2))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(obs - sim))


def nash_sutcliffe(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Nash–Sutcliffe Efficiency:
        NSE = 1 - Σ(obs - sim)² / Σ(obs - obs_mean)²

    NSE = 1   : perfect model
    NSE = 0   : model is as good as the mean
    NSE < 0   : model is worse than the mean
    """
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator < 1e-15:
        return -np.inf
    return 1.0 - numerator / denominator


def r_squared(obs: np.ndarray, sim: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_all_metrics(obs, sim):
    """Return dict of all performance metrics."""
    return {
        'RMSE': rmse(obs, sim),
        'MAE': mae(obs, sim),
        'NSE': nash_sutcliffe(obs, sim),
        'R2': r_squared(obs, sim),
    }


# ================================================================
# OBJECTIVE FUNCTION BUILDER
# ================================================================
def build_objective(cfg_base: ModelConfig,
                    param_names: List[str],
                    obs_data: Dict[float, np.ndarray],
                    obs_times: np.ndarray,
                    monitor_locs: List[float]) -> Callable:
    """
    Build an objective function for calibration.

    Parameters
    ----------
    cfg_base     : base ModelConfig (parameters not being calibrated)
    param_names  : list of parameter paths, e.g. 'kinetic.k_bio_max'
    obs_data     : {location_m: observed_concentration_array}
    obs_times    : times [s] at which observations are available
    monitor_locs : monitoring locations [m]

    Returns
    -------
    objective(theta) -> float  (sum of squared residuals)
    """

    def objective(theta):
        cfg = copy.deepcopy(cfg_base)

        # Set calibration parameters
        for name, val in zip(param_names, theta):
            parts = name.split('.')
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        # Run simulation
        try:
            result = run_simulation(cfg, monitor_locs=monitor_locs,
                                    save_interval=600.0, verbose=False)
        except Exception:
            return 1e12  # penalise failures

        # Compute residuals
        ssr = 0.0
        for loc in obs_data:
            if loc not in result.C_w_monitor:
                continue
            sim_full = result.C_w_monitor[loc]
            sim_times = result.times

            # Interpolate sim to obs times
            sim_at_obs = np.interp(obs_times, sim_times, sim_full)
            obs_arr = obs_data[loc]

            n = min(len(obs_arr), len(sim_at_obs))
            ssr += np.sum((obs_arr[:n] - sim_at_obs[:n]) ** 2)

        return ssr

    return objective


# ================================================================
# LEAST-SQUARES CALIBRATION
# ================================================================
def calibrate_least_squares(cfg: ModelConfig,
                            param_names: List[str],
                            bounds: List[Tuple[float, float]],
                            obs_data: Dict[float, np.ndarray],
                            obs_times: np.ndarray,
                            monitor_locs: List[float],
                            method: str = 'differential_evolution',
                            **kwargs) -> dict:
    """
    Calibrate model parameters by minimising sum-of-squares.

    Parameters
    ----------
    method : 'differential_evolution' (global) or 'L-BFGS-B' (local)

    Returns
    -------
    dict with 'best_params', 'objective_value', 'result'
    """
    obj = build_objective(cfg, param_names, obs_data,
                          obs_times, monitor_locs)

    if method == 'differential_evolution':
        res = differential_evolution(obj, bounds, seed=42,
                                     maxiter=kwargs.get('maxiter', 50),
                                     tol=1e-6, disp=True)
    else:
        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
        res = minimize(obj, x0, method=method, bounds=bounds)

    best = {name: val for name, val in zip(param_names, res.x)}

    return {
        'best_params': best,
        'objective_value': res.fun,
        'result': res,
    }


# ================================================================
# BAYESIAN MCMC  (Metropolis-Hastings)
# ================================================================
def mcmc_calibration(cfg: ModelConfig,
                     param_names: List[str],
                     bounds: List[Tuple[float, float]],
                     obs_data: Dict[float, np.ndarray],
                     obs_times: np.ndarray,
                     monitor_locs: List[float],
                     n_samples: int = 2000,
                     sigma_obs: float = 0.5,
                     proposal_scale: float = 0.05,
                     seed: int = 42) -> dict:
    """
    Bayesian parameter estimation via Metropolis-Hastings MCMC.

    Likelihood: L(θ) ∝ exp(-SSR / (2 σ²))
    Prior:      uniform within bounds

    Returns dict with 'chain', 'acceptance_rate', 'log_posterior'
    """
    np.random.seed(seed)
    n_params = len(param_names)

    obj = build_objective(cfg, param_names, obs_data,
                          obs_times, monitor_locs)

    # Initial point (midpoint of bounds)
    theta = np.array([(lo + hi) / 2 for lo, hi in bounds])
    bounds_arr = np.array(bounds)

    # Log-posterior (up to constant)
    def log_posterior(th):
        # Uniform prior: -inf if outside bounds
        if np.any(th < bounds_arr[:, 0]) or np.any(th > bounds_arr[:, 1]):
            return -np.inf
        ssr = obj(th)
        return -ssr / (2.0 * sigma_obs ** 2)

    lp_current = log_posterior(theta)
    chain = np.zeros((n_samples, n_params))
    lp_chain = np.zeros(n_samples)
    accepted = 0

    # Proposal standard deviations
    prop_std = proposal_scale * (bounds_arr[:, 1] - bounds_arr[:, 0])

    for i in range(n_samples):
        # Propose
        theta_prop = theta + prop_std * np.random.randn(n_params)

        lp_prop = log_posterior(theta_prop)

        # Accept / reject
        log_alpha = lp_prop - lp_current
        if np.log(np.random.rand()) < log_alpha:
            theta = theta_prop
            lp_current = lp_prop
            accepted += 1

        chain[i] = theta
        lp_chain[i] = lp_current

        if (i + 1) % 100 == 0:
            print(f"  MCMC step {i+1}/{n_samples}, "
                  f"accept rate = {accepted/(i+1):.2%}")

    return {
        'chain': chain,
        'param_names': param_names,
        'acceptance_rate': accepted / n_samples,
        'log_posterior': lp_chain,
    }


# ================================================================
# SOBOL SENSITIVITY ANALYSIS
# ================================================================
def sobol_sensitivity(cfg: ModelConfig,
                      param_names: List[str],
                      bounds: List[Tuple[float, float]],
                      obs_loc: float,
                      obs_time_idx: int = -1,
                      N: int = 256,
                      monitor_locs: Optional[List[float]] = None,
                      seed: int = 42) -> dict:
    """
    Variance-based Sobol sensitivity indices (first-order and total).

    Uses Saltelli's sampling scheme:
    - Generate two (N × k) quasi-random matrices A, B
    - For each parameter i, construct AB_i (A with column i from B)
    - Compute f(A), f(B), f(AB_i)
    - Estimate S_i and S_Ti

    Parameters
    ----------
    N : base sample size (total model runs ≈ N(2k+2))

    Returns
    -------
    dict with 'S1' (first-order), 'ST' (total), 'param_names'
    """
    np.random.seed(seed)
    k = len(param_names)
    bounds_arr = np.array(bounds)

    if monitor_locs is None:
        monitor_locs = [obs_loc]

    # --- Saltelli sampling ---
    A = np.random.rand(N, k)
    B = np.random.rand(N, k)

    # Scale to parameter bounds
    for j in range(k):
        A[:, j] = bounds_arr[j, 0] + A[:, j] * (bounds_arr[j, 1] - bounds_arr[j, 0])
        B[:, j] = bounds_arr[j, 0] + B[:, j] * (bounds_arr[j, 1] - bounds_arr[j, 0])

    def evaluate(params_matrix):
        """Run model for each row of the parameter matrix."""
        results = np.zeros(len(params_matrix))
        for i, theta in enumerate(params_matrix):
            cfg_i = copy.deepcopy(cfg)
            for name, val in zip(param_names, theta):
                parts = name.split('.')
                obj = cfg_i
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], val)
            try:
                res = run_simulation(cfg_i, monitor_locs=monitor_locs,
                                     save_interval=3600, verbose=False)
                if obs_loc in res.C_w_monitor:
                    arr = res.C_w_monitor[obs_loc]
                    results[i] = arr[obs_time_idx] if len(arr) > 0 else 0.0
            except Exception:
                results[i] = 0.0
        return results

    print(f"Sobol analysis: {N*(2*k+2)} model evaluations...")
    f_A = evaluate(A)
    f_B = evaluate(B)

    f_AB = np.zeros((k, N))
    for j in range(k):
        AB_j = A.copy()
        AB_j[:, j] = B[:, j]
        f_AB[j] = evaluate(AB_j)

    # --- Estimate indices ---
    f0_sq = np.mean(f_A) * np.mean(f_B)
    var_total = np.var(np.concatenate([f_A, f_B]))

    S1 = np.zeros(k)
    ST = np.zeros(k)
    for j in range(k):
        # First-order: S_i = V_i / V(Y)
        V_j = np.mean(f_B * (f_AB[j] - f_A))
        S1[j] = V_j / max(var_total, 1e-15)

        # Total: S_Ti = 1 - V_{~i} / V(Y)
        V_not_j = np.mean(f_A * (f_AB[j] - f_A))  # alternative estimator
        ST[j] = 0.5 * np.mean((f_A - f_AB[j]) ** 2) / max(var_total, 1e-15)

    return {
        'S1': S1,
        'ST': ST,
        'param_names': param_names,
        'var_total': var_total,
    }
