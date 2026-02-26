"""
sediment_exchange.py
====================
Fluxes between water column, benthic sediment, and hyporheic zone.

Conventions
-----------
Positive flux = into the compartment receiving it.
All fluxes are in [mg/(m²·day)].
"""

import numpy as np

DAY = 86400.0


# ================================================================
# SETTLING FLUX  (water → sediment)
# ================================================================
def settling_flux(C_w, v_s, f_p):
    """
    J_settle = v_s · f_p · C_w     [mg/(m²·day)]

    Parameters
    ----------
    C_w : water-column concentration [mg/L]
    v_s : settling velocity [m/day]
    f_p : particulate fraction [-]
    """
    return v_s * f_p * np.maximum(C_w, 0)


# ================================================================
# RESUSPENSION FLUX  (sediment → water)
# ================================================================
def resuspension_flux(C_s, E0, tau_b, tau_c):
    """
    J_resus = E0 · max(τ_b/τ_c - 1, 0) · C_s   [mg/(m²·day)]

    Parameters
    ----------
    C_s   : sediment concentration [mg/kg] (or mg/L equiv)
    E0    : erosion rate constant [m/day]
    tau_b : bed shear stress [Pa]
    tau_c : critical shear stress for erosion [Pa]
    """
    excess = np.maximum(tau_b / max(tau_c, 1e-12) - 1.0, 0.0)
    return E0 * excess * np.maximum(C_s, 0)


# ================================================================
# DIFFUSIVE FLUX  (water ↔ pore-water across boundary layer)
# ================================================================
def diffusive_flux(C_w, C_pw, D_eff, delta_bl):
    """
    J_diff = D_eff / δ_bl · (C_w - C_pw)    [mg/(m²·day)]

    Positive = into sediment (when C_w > C_pw).

    Parameters
    ----------
    C_w      : water-column dissolved concentration [mg/L]
    C_pw     : pore-water concentration [mg/L]
    D_eff    : effective diffusion coefficient [m²/s]
    delta_bl : diffusive boundary-layer thickness [m]
    """
    return (D_eff / delta_bl) * (C_w - C_pw) * DAY  # convert m/s→m/day


# ================================================================
# HYPORHEIC EXCHANGE  (water ↔ hyporheic)
# ================================================================
def hyporheic_flux(C_w, C_h, alpha_hyp, H):
    """
    J_hyp = α_hyp · H · (C_w - C_h)    [mg/(m²·day)]

    α_hyp : first-order exchange coefficient [1/s]
    H     : water depth [m]
    """
    return alpha_hyp * H * (C_w - C_h) * DAY


# ================================================================
# NET FLUX AND COMPARTMENT UPDATES
# ================================================================
def compute_all_fluxes(C_w, C_s, C_h, C_pw, params, sed_params,
                       tau_b, f_p):
    """
    Compute all inter-compartment fluxes.

    Returns dict of arrays:
        'J_settle'  : water → sediment  [mg/(m²·day)]
        'J_resus'   : sediment → water  [mg/(m²·day)]
        'J_diff'    : water → sediment  [mg/(m²·day)]
        'J_hyp'     : water → hyporheic [mg/(m²·day)]
        'J_net_ws'  : net water→sediment [mg/(m²·day)]
    """
    kp = params
    sp = sed_params
    H = 2.5  # overridden by caller

    J_set = settling_flux(C_w, kp.v_settling, f_p)
    J_res = resuspension_flux(C_s, kp.E_resuspension * DAY,
                              tau_b, kp.tau_critical)
    J_dif = diffusive_flux(C_w, C_pw, sp.D_eff, sp.delta_bl)
    J_hyp = hyporheic_flux(C_w, C_h, sp.alpha_hyp, sp.hyp_depth)

    return {
        'J_settle': J_set,
        'J_resus': J_res,
        'J_diff': J_dif,
        'J_hyp': J_hyp,
        'J_net_ws': J_set - J_res + J_dif,
    }


def update_sediment(C_s, C_pw, J_net_ws, R_sed, porosity,
                    sed_depth, dt_day):
    """
    Update sediment concentration for one time-step.

    dC_s/dt = J_net_ws / (ε · H_s) + R_sed

    Parameters
    ----------
    C_s      : current sediment concentration [mg/L equiv]
    C_pw     : pore-water concentration [mg/L]
    J_net_ws : net flux into sediment [mg/(m²·day)]
    R_sed    : in-situ reaction rate in sediment [mg/L/day]
    porosity : ε [-]
    sed_depth: active layer depth [m]
    dt_day   : time-step [day]
    """
    dC = (J_net_ws / (porosity * sed_depth) + R_sed) * dt_day
    C_s_new = np.maximum(C_s + dC, 0.0)
    # Pore-water equilibrium (instantaneous partitioning)
    C_pw_new = C_s_new * porosity  # simplified
    return C_s_new, C_pw_new


def update_hyporheic(C_h, C_w, alpha_hyp, k_bio_hyp, theta,
                     T, dt_day):
    """
    Update hyporheic zone concentration.

    dC_h/dt = α_hyp (C_w - C_h) · 86400 - k_bio_hyp θ^(T-20) C_h
    """
    exchange = alpha_hyp * DAY * (C_w - C_h)
    decay = -k_bio_hyp * theta ** (T - 20.0) * C_h
    C_h_new = np.maximum(C_h + (exchange + decay) * dt_day, 0.0)
    return C_h_new
