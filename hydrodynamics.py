"""
hydrodynamics.py
================
Compute flow-derived quantities: shear stress, friction velocity,
bed shear, dispersion from Fischer's formula, and dynamic flow updates.
"""

import numpy as np
from config import PhysicalParams


# ---------- Bed shear stress ----------
def bed_shear_stress(rho: float, g: float, H: float,
                     S0: float) -> float:
    """
    τ_b = ρ g H S0    [Pa]

    Parameters
    ----------
    rho : water density [kg/m³]
    g   : gravitational acceleration [m/s²]
    H   : water depth [m]
    S0  : bed slope [-]
    """
    return rho * g * H * S0


def shear_velocity(tau_b: float, rho: float) -> float:
    """
    u* = sqrt(τ_b / ρ)   [m/s]
    """
    return np.sqrt(np.abs(tau_b) / rho)


# ---------- Fischer longitudinal dispersion ----------
def fischer_dispersion(u: float, H: float, W: float,
                       u_star: float) -> float:
    """
    Fischer (1979) formula for longitudinal dispersion:
        D_L = 0.011 u² W² / (H u*)       [m²/s]

    Parameters
    ----------
    u     : cross-section average velocity [m/s]
    H     : depth [m]
    W     : width [m] (use H*10 if unknown)
    u_star: shear velocity [m/s]
    """
    if u_star < 1e-12:
        return 10.0  # fallback
    return 0.011 * u ** 2 * W ** 2 / (H * u_star)


# ---------- Dynamic velocity with seasonal / shock -----------
def velocity_at_time(t: float, u_base: float,
                     seasonal_amp: float = 0.0,
                     seasonal_period: float = 86400.0,
                     shock_time: float = -1.0,
                     shock_magnitude: float = 0.0,
                     shock_duration: float = 3600.0) -> float:
    """
    Return velocity at time t incorporating seasonal variation
    and optional shock (flood pulse).
    """
    u = u_base + seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
    if 0 <= (t - shock_time) <= shock_duration:
        u += shock_magnitude
    return max(u, 0.01)  # enforce positivity


# ---------- CFL / stability ----------
def compute_cfl(u: float, D: float, dx: float, dt: float):
    """
    Return Courant number and diffusion number.

    CFL (Courant):   Cr = u dt / dx
    Diffusion:       Fo = D dt / dx²

    For explicit schemes: Cr ≤ 1 and Fo ≤ 0.5.
    Crank–Nicolson is unconditionally stable, but accuracy
    degrades when Cr >> 1.
    """
    Cr = u * dt / dx
    Fo = D * dt / dx ** 2
    return Cr, Fo


def adaptive_timestep(u: float, D: float, dx: float,
                      dt_current: float, dt_min: float,
                      dt_max: float, cfl_target: float = 0.8) -> float:
    """
    Adjust dt to maintain target CFL ≤ cfl_target.

    Returns adjusted dt clamped to [dt_min, dt_max].
    """
    # CFL limit
    dt_cfl = cfl_target * dx / max(abs(u), 1e-12)
    # Diffusion limit (not strictly needed for CN, but keeps accuracy)
    dt_diff = 0.4 * dx ** 2 / max(D, 1e-12)
    dt_new = min(dt_cfl, dt_diff, dt_max)
    return max(dt_new, dt_min)
