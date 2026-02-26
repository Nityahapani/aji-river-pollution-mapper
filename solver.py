"""
solver.py
=========
Core numerical engine.

- Crank–Nicolson scheme for advection–dispersion
- Strang operator splitting for reactions
- Thomas algorithm for tridiagonal solve
- Adaptive time-stepping
- Multi-compartment coupling loop
"""

import numpy as np
from scipy.linalg import solve_banded
from typing import Dict, List, Tuple, Optional

from config import ModelConfig, build_time_varying_params
from hydrodynamics import (bed_shear_stress, shear_velocity,
                           adaptive_timestep, compute_cfl)
from kinetics import total_reaction_rate, DAY
from sorption import (langmuir_equilibrium, freundlich_equilibrium,
                      sorption_kinetic_rate, particulate_fraction,
                      partition_coefficient_dynamic)
from sediment_exchange import (compute_all_fluxes, update_sediment,
                               update_hyporheic)


# ================================================================
# TRIDIAGONAL SOLVER  (banded format for scipy)
# ================================================================
def solve_tridiag(a, b, c, d):
    """
    Solve tridiagonal system using scipy banded solver.

    a : sub-diagonal  (length n, a[0] unused)
    b : main diagonal (length n)
    c : super-diagonal (length n, c[-1] unused)
    d : RHS vector (length n)

    Returns x such that  A x = d.
    """
    n = len(d)
    ab = np.zeros((3, n))
    ab[0, 1:] = c[:-1]   # super-diagonal
    ab[1, :] = b          # main diagonal
    ab[2, :-1] = a[1:]    # sub-diagonal
    return solve_banded((1, 1), ab, d)


# ================================================================
# CRANK–NICOLSON TRANSPORT STEP
# ================================================================
def crank_nicolson_step(C, u, D, dx, dt, C_left_bc, C_right_bc='neumann'):
    """
    Advance the advection–dispersion equation one step
    using the Crank–Nicolson scheme with upwind advection.

    ∂C/∂t = -u ∂C/∂x + D ∂²C/∂x²

    Parameters
    ----------
    C          : concentration array (nx,) [mg/L]
    u          : velocity [m/s] (scalar or array)
    D          : dispersion coefficient [m²/s]
    dx         : spatial step [m]
    dt         : time step [s]
    C_left_bc  : upstream Dirichlet BC value [mg/L]
    C_right_bc : 'neumann' (zero-gradient) or float (Dirichlet)

    Returns
    -------
    C_new : updated concentration array (nx,)

    Numerical scheme coefficients (upwind, u > 0):
        σ = u dt / (2 dx)
        r = D dt / (2 dx²)

    LHS: [-(σ+r)] C_{i-1}^{n+1} + [1+σ+2r] C_i^{n+1} + [-r] C_{i+1}^{n+1}
    RHS: [(σ+r)] C_{i-1}^n     + [1-σ-2r] C_i^n     + [r]  C_{i+1}^n
    """
    nx = len(C)
    sigma = u * dt / (2.0 * dx)
    r = D * dt / (2.0 * dx ** 2)

    # --- build coefficient arrays for interior nodes ---
    a = np.zeros(nx)  # sub-diagonal
    b = np.zeros(nx)  # main diagonal
    c = np.zeros(nx)  # super-diagonal
    d = np.zeros(nx)  # RHS

    # Interior nodes (i = 1 .. nx-2)
    for i in range(1, nx - 1):
        # LHS coefficients
        a[i] = -(sigma + r)
        b[i] = 1.0 + sigma + 2.0 * r
        c[i] = -r

        # RHS
        d[i] = ((sigma + r) * C[i - 1]
                + (1.0 - sigma - 2.0 * r) * C[i]
                + r * C[i + 1])

    # --- Boundary conditions ---
    # Left (upstream) Dirichlet: C[0] = C_left_bc
    b[0] = 1.0
    c[0] = 0.0
    d[0] = C_left_bc

    # Right (downstream) zero-gradient Neumann: ∂C/∂x = 0  →  C[nx-1] = C[nx-2]
    if C_right_bc == 'neumann':
        a[nx - 1] = -1.0
        b[nx - 1] = 1.0
        d[nx - 1] = 0.0
    else:
        b[nx - 1] = 1.0
        a[nx - 1] = 0.0
        d[nx - 1] = float(C_right_bc)

    # --- Solve tridiagonal system ---
    C_new = solve_tridiag(a, b, c, d)
    return np.maximum(C_new, 0.0)  # enforce non-negativity


# ================================================================
# REACTION HALF-STEP (for Strang splitting)
# ================================================================
def reaction_half_step(C_w, C_s, C_h, q, dt_half, params,
                       env, tau_b):
    """
    Apply reactions for dt_half seconds using implicit Euler.

    Updates C_w, C_s, C_h, and sorbed q in-place.

    Parameters
    ----------
    C_w     : water-column dissolved concentration [mg/L]
    C_s     : sediment concentration [mg/L equiv]
    C_h     : hyporheic concentration [mg/L]
    q       : sorbed phase concentration [mg/kg]
    dt_half : half time-step [s]
    params  : KineticParams
    env     : environment dict with current conditions
    tau_b   : bed shear stress [Pa]

    Returns
    -------
    C_w, C_s, C_h, q  (updated arrays)
    """
    dt_day = dt_half / DAY  # convert to days

    # --- Water column reactions ---
    R_w = total_reaction_rate(C_w, C_s, params, env, tau_b)
    # Implicit Euler for stability:  C_new = C_old / (1 - dt · k_eff)
    # For first-order-like terms, k_eff ≈ R/C (effective first-order rate)
    safe_C = np.where(C_w > 1e-15, C_w, 1e-15)
    k_eff = -R_w / safe_C  # positive when R is negative (decay)
    k_eff = np.maximum(k_eff, 0.0)
    C_w_new = C_w / (1.0 + k_eff * dt_day)
    C_w_new = np.maximum(C_w_new, 0.0)

    # --- Sorption kinetics ---
    dC_sorp, dq = sorption_kinetic_rate(
        C_w_new, q, params.k_ads, params.k_des,
        params.q_max, params.K_L, env['spm'])
    C_w_new = np.maximum(C_w_new + dC_sorp * dt_day, 0.0)
    q_new = np.maximum(q + dq * dt_day, 0.0)

    # --- Sediment reactions (simple first-order decay) ---
    k_sed = params.k_bio_max * 0.3  # reduced rate in sediment
    C_s_new = C_s / (1.0 + k_sed * dt_day)

    # --- Hyporheic reactions ---
    C_h_new = update_hyporheic(
        C_h, C_w_new, params_sed_alpha_hyp(env),
        env.get('k_bio_hyp', 0.05), params.theta_bio,
        env['T'], dt_day)

    return C_w_new, C_s_new, C_h_new, q_new


def params_sed_alpha_hyp(env):
    """Extract hyporheic exchange coefficient from environment."""
    return env.get('alpha_hyp', 5e-6)


# ================================================================
# SOURCE TERM (point source + pulse)
# ================================================================
def apply_sources(C, x, dx, t, source_cfg):
    """
    Add pollutant mass from point sources and pulse injections.

    Parameters
    ----------
    C          : concentration array (nx,) [mg/L]
    x          : spatial coordinate array (nx,) [m]
    dx         : spatial step [m]
    t          : current time [s]
    source_cfg : SourceConfig
    """
    sc = source_cfg
    C_new = C.copy()

    # --- Continuous point source ---
    if sc.inject_mass_rate > 0:
        idx = np.argmin(np.abs(x - sc.inject_loc))
        # mass_rate [mg/s] into a cell of width dx and depth H (≈ area)
        # Approximate: dC = mass_rate * dt / (dx * H * W)
        # Here we add per unit volume: handled in main loop

    # --- Pulse injection ---
    if sc.pulse_mass > 0:
        t_start = sc.pulse_time
        t_end = sc.pulse_time + sc.pulse_duration
        if t_start <= t < t_end:
            idx = np.argmin(np.abs(x - sc.pulse_loc))
            # Mass rate during pulse [mg/s]
            rate = sc.pulse_mass / sc.pulse_duration
            C_new[idx] += rate  # will be scaled by dt and volume in caller

    return C_new


# ================================================================
# MAIN SIMULATION LOOP
# ================================================================
class SimulationResult:
    """Container for simulation outputs."""
    def __init__(self):
        self.times = []
        self.C_w_series = []          # water column concentration snapshots
        self.C_s_series = []          # sediment concentration snapshots
        self.C_h_series = []          # hyporheic concentration snapshots
        self.C_w_monitor = {}         # time-series at monitoring points
        self.mass_balance = []        # total mass in system over time
        self.x = None
        self.dt_history = []


def run_simulation(cfg: ModelConfig,
                   monitor_locs: Optional[List[float]] = None,
                   save_interval: float = 600.0,
                   verbose: bool = True) -> SimulationResult:
    """
    Execute the full coupled simulation.

    Parameters
    ----------
    cfg            : ModelConfig with all parameters
    monitor_locs   : list of x-positions [m] for time-series extraction
    save_interval  : time between saved snapshots [s]
    verbose        : print progress

    Returns
    -------
    SimulationResult
    """
    dom = cfg.domain
    phy = cfg.physical
    chem = cfg.chemical
    bio = cfg.biological
    kin = cfg.kinetic
    sed = cfg.sediment
    src = cfg.source

    nx = dom.nx
    dx = dom.dx
    x = dom.x

    # --- Monitor points ---
    if monitor_locs is None:
        monitor_locs = [dom.L * f for f in [0.1, 0.25, 0.5, 0.75, 0.9]]
    monitor_idx = [np.argmin(np.abs(x - loc)) for loc in monitor_locs]

    # --- Initial conditions ---
    C_w = np.full(nx, src.C_upstream)     # water column [mg/L]
    C_s = np.zeros(nx)                     # sediment [mg/L equiv]
    C_h = np.zeros(nx)                     # hyporheic [mg/L]
    C_pw = np.zeros(nx)                    # pore water [mg/L]
    q = np.zeros(nx)                       # sorbed phase [mg/kg]

    # --- Build time-varying parameter arrays ---
    t_max = dom.T_total
    # Estimate number of steps for pre-building parameter arrays
    t_param = np.arange(0, t_max + 1, 60.0)  # 1-min resolution for params
    tv = build_time_varying_params(cfg, t_param)

    # --- Result container ---
    result = SimulationResult()
    result.x = x.copy()
    for mi, loc in zip(monitor_idx, monitor_locs):
        result.C_w_monitor[loc] = []

    # --- Time integration ---
    t = 0.0
    dt = dom.dt
    step = 0
    last_save = -save_interval

    g = 9.81  # gravitational acceleration [m/s²]

    if verbose:
        print(f"Starting simulation: L={dom.L}m, nx={nx}, "
              f"T={t_max/3600:.1f}h, dt0={dt:.1f}s")

    while t < t_max:
        # --- Interpolate time-varying parameters ---
        tidx = np.searchsorted(t_param, t, side='right') - 1
        tidx = max(0, min(tidx, len(t_param) - 1))

        u = tv['velocity'][tidx]
        T = tv['temperature'][tidx]
        I0 = tv['light'][tidx]
        H = tv['depth'][tidx]
        spm = tv['spm'][tidx]
        pH_val = tv['pH'][tidx]
        DO = tv['DO'][tidx]
        orp_val = tv['orp'][tidx]
        X = tv['biomass'][tidx]
        D = tv['dispersion'][tidx]

        # --- Adaptive time-step ---
        dt = adaptive_timestep(u, D, dx, dt, dom.dt_min,
                               dom.dt_max, dom.cfl_target)
        if t + dt > t_max:
            dt = t_max - t

        # Environment dictionary for kinetics
        env = {'T': T, 'pH': pH_val, 'DO': DO, 'orp': orp_val,
               'biomass': X, 'light': I0, 'depth': H, 'spm': spm,
               'alpha_hyp': sed.alpha_hyp, 'k_bio_hyp': sed.k_bio_hyp}

        # --- Bed shear stress ---
        tau_b = bed_shear_stress(phy.rho_water, g, H, phy.bed_slope)

        # --- Particulate fraction ---
        K_d_dyn = partition_coefficient_dynamic(
            kin.K_d, spm, pH_val, chem.ionic_strength)
        f_p = particulate_fraction(K_d_dyn, spm)

        # ========================================================
        # STRANG SPLITTING
        # ========================================================

        # STEP 1: Half-step reactions (dt/2)
        C_w, C_s, C_h, q = reaction_half_step(
            C_w, C_s, C_h, q, dt / 2.0, kin, env, tau_b)

        # STEP 2: Full-step transport (Crank–Nicolson)
        # --- Apply source before transport ---
        C_w_src = C_w.copy()
        if src.pulse_mass > 0:
            t_start = src.pulse_time
            t_end = src.pulse_time + src.pulse_duration
            if t_start <= t < t_end:
                idx_p = np.argmin(np.abs(x - src.pulse_loc))
                mass_rate = src.pulse_mass / src.pulse_duration  # mg/s
                # Convert to concentration increment:
                # dC = mass_rate · dt / (dx · H · 1) assuming unit width
                C_w_src[idx_p] += mass_rate * dt / (dx * H * 1000.0)

        C_w = crank_nicolson_step(C_w_src, u, D, dx, dt,
                                  src.C_upstream)

        # STEP 3: Half-step reactions (dt/2)
        C_w, C_s, C_h, q = reaction_half_step(
            C_w, C_s, C_h, q, dt / 2.0, kin, env, tau_b)

        # ========================================================
        # INTER-COMPARTMENT FLUXES
        # ========================================================
        fluxes = compute_all_fluxes(C_w, C_s, C_h, C_pw,
                                    kin, sed, tau_b, f_p)

        dt_day = dt / DAY

        # Update sediment
        J_net = fluxes['J_net_ws']
        R_sed = -kin.k_bio_max * 0.3 * kin.theta_bio ** (T - 20) * C_s
        C_s, C_pw = update_sediment(C_s, C_pw, J_net, R_sed,
                                    sed.porosity, sed.sed_depth, dt_day)

        # Flux effect on water column
        C_w -= (fluxes['J_settle'] - fluxes['J_resus']) / (H * 1000) * dt_day
        C_w -= fluxes['J_hyp'] / (H * 1000) * dt_day
        C_w = np.maximum(C_w, 0.0)

        # Update hyporheic
        C_h = update_hyporheic(C_h, C_w, sed.alpha_hyp,
                               sed.k_bio_hyp, kin.theta_bio, T, dt_day)

        # ========================================================
        # BOOKKEEPING
        # ========================================================
        t += dt
        step += 1
        result.dt_history.append(dt)

        # Save snapshots
        if t - last_save >= save_interval or t >= t_max:
            result.times.append(t)
            result.C_w_series.append(C_w.copy())
            result.C_s_series.append(C_s.copy())
            result.C_h_series.append(C_h.copy())

            # Monitor points
            for mi, loc in zip(monitor_idx, monitor_locs):
                result.C_w_monitor[loc].append(C_w[mi])

            # Total mass (water + sediment + hyporheic)
            mass_w = np.trapz(C_w, x) * H * 1.0  # assuming unit width
            mass_s = np.trapz(C_s, x) * sed.sed_depth * sed.porosity
            mass_h = np.trapz(C_h, x) * sed.hyp_depth
            result.mass_balance.append(mass_w + mass_s + mass_h)

            last_save = t

        if verbose and step % 500 == 0:
            Cr, Fo = compute_cfl(u, D, dx, dt)
            print(f"  t={t/3600:.2f}h  step={step}  dt={dt:.2f}s  "
                  f"Cr={Cr:.3f}  max(C_w)={C_w.max():.4f}")

    # Convert lists to arrays
    result.times = np.array(result.times)
    result.C_w_series = np.array(result.C_w_series)
    result.C_s_series = np.array(result.C_s_series)
    result.C_h_series = np.array(result.C_h_series)
    result.mass_balance = np.array(result.mass_balance)
    for loc in result.C_w_monitor:
        result.C_w_monitor[loc] = np.array(result.C_w_monitor[loc])

    if verbose:
        print(f"Simulation complete: {step} steps, "
              f"{len(result.times)} snapshots saved.")

    return result
