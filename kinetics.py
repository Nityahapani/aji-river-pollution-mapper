"""
kinetics.py
===========
Reaction-rate computations for all transformation pathways.

Every function returns a rate R [mg/L/day] (negative = loss).
The solver converts to per-second internally.
"""

import numpy as np


DAY = 86400.0  # seconds in a day


# ================================================================
# 1. BIODEGRADATION — dual Monod with Arrhenius temperature scaling
# ================================================================
def biodegradation_rate(C, k_bio_max, K_s, K_X, X,
                        theta, T, T_ref=20.0):
    """
    R_bio = -k_bio_max · θ^(T-T_ref) · C/(K_s+C) · X/(K_X+X)

    Parameters
    ----------
    C         : dissolved concentration [mg/L]
    k_bio_max : maximum first-order rate [1/day]
    K_s       : substrate half-saturation [mg/L]
    K_X       : biomass half-saturation [mg/L]
    X         : microbial biomass [mg/L]
    theta     : Arrhenius temperature coefficient [-]
    T         : temperature [°C]
    T_ref     : reference temperature [°C]

    Returns
    -------
    R_bio [mg/L/day]  (negative)
    """
    temp_factor = theta ** (T - T_ref)
    monod_C = C / (K_s + np.maximum(C, 0))
    monod_X = X / (K_X + X)
    return -k_bio_max * temp_factor * monod_C * monod_X * C


# ================================================================
# 2. PHOTOLYSIS — depth-averaged light with Beer–Lambert
# ================================================================
def photolysis_rate(C, k_photo_ref, I0, k_ext, H):
    """
    Depth-averaged photolysis:
        I_avg = I0 / (k_ext · H) · (1 - exp(-k_ext · H))
        R_photo = -k_photo_ref · (I_avg / I_ref) · C

    with I_ref = 500 W/m² (reference intensity for k_photo_ref).
    """
    I_ref = 500.0
    if k_ext * H > 1e-6:
        I_avg = I0 / (k_ext * H) * (1 - np.exp(-k_ext * H))
    else:
        I_avg = I0
    return -k_photo_ref * (I_avg / I_ref) * C


# ================================================================
# 3. HYDROLYSIS — pH-dependent three-term rate law
# ================================================================
def hydrolysis_rate(C, k_acid, k_neutral, k_base, pH,
                    theta, T, T_ref=20.0):
    """
    R_hyd = -(k_a [H+] + k_n + k_b [OH-]) θ^(T-T_ref) C

    [H+] = 10^(-pH),   [OH-] = 10^(pH-14)     [mol/L]
    """
    H_plus = 10.0 ** (-pH)
    OH_minus = 10.0 ** (pH - 14.0)
    k_obs = k_acid * H_plus + k_neutral + k_base * OH_minus
    temp_factor = theta ** (T - T_ref)
    return -k_obs * temp_factor * C


# ================================================================
# 4. VOLATILISATION — two-film theory
# ================================================================
def volatilization_rate(C, k_vol, H):
    """
    R_vol = -(k_vol / H) · C

    k_vol : overall gas-transfer velocity [m/day]
    H     : water depth [m]
    """
    return -(k_vol / H) * C


# ================================================================
# 5. REDOX-CONTROLLED TRANSFORMATION — sigmoid switch
# ================================================================
def redox_rate(C, k_aer, k_anaer, orp, orp_threshold, beta):
    """
    Smooth sigmoid blending between aerobic and anaerobic rates:

        σ = 1 / (1 + exp(-β (E_h - E_h*)))
        k_eff = k_aer · σ + k_anaer · (1 - σ)
        R_redox = -k_eff · C
    """
    sigma = 1.0 / (1.0 + np.exp(-beta * (orp - orp_threshold)))
    k_eff = k_aer * sigma + k_anaer * (1.0 - sigma)
    return -k_eff * C


# ================================================================
# 6. SETTLING OF PARTICULATE FRACTION
# ================================================================
def settling_rate(C, v_s, K_d, spm, H):
    """
    Particulate fraction f_p = K_d · SPM / (1 + K_d · SPM)
    R_settle = -(v_s · f_p / H) · C

    v_s : settling velocity [m/day]
    K_d : distribution coefficient [L/kg]  (convert SPM mg/L → kg/L)
    spm : suspended particulate matter [mg/L]
    H   : depth [m]
    """
    spm_kg = spm * 1e-6  # mg/L → kg/L
    f_p = K_d * spm_kg / (1.0 + K_d * spm_kg)
    return -(v_s * f_p / H) * C


# ================================================================
# 7. RESUSPENSION
# ================================================================
def resuspension_rate(C_sed, E0, tau_b, tau_c, H):
    """
    R_resus = (E0 / H) · max(τ_b/τ_c - 1, 0) · C_sed

    E0    : erosion rate coefficient [m/day]
    tau_b : bed shear stress [Pa]
    tau_c : critical shear stress [Pa]
    C_sed : sediment-phase concentration [mg/L equiv]
    """
    excess = np.maximum(tau_b / tau_c - 1.0, 0.0)
    return (E0 / H) * excess * C_sed


# ================================================================
# AGGREGATE REACTION VECTOR
# ================================================================
def total_reaction_rate(C_w, C_sed, params, env, tau_b):
    """
    Compute R_total [mg/L/day] for the water column.

    Parameters
    ----------
    C_w    : dissolved water-column concentration [mg/L]
    C_sed  : sediment concentration [mg/L equivalent]
    params : KineticParams dataclass
    env    : dict with keys 'T', 'pH', 'DO', 'orp', 'biomass',
             'light', 'depth', 'spm'
    tau_b  : bed shear stress [Pa]

    Returns
    -------
    R_total : aggregate rate [mg/L/day]
    """
    p = params
    T = env['T']
    pH = env['pH']
    H = env['depth']
    I0 = env['light']

    R = np.zeros_like(C_w, dtype=float)

    R += biodegradation_rate(C_w, p.k_bio_max, p.K_s, p.K_X,
                             env['biomass'], p.theta_bio, T)
    R += photolysis_rate(C_w, p.k_photo_ref, I0, p.k_extinction, H)
    R += hydrolysis_rate(C_w, p.k_hyd_acid, p.k_hyd_neutral,
                         p.k_hyd_base, pH, p.theta_hyd, T)
    R += volatilization_rate(C_w, p.k_vol_mass_transfer, H)
    R += redox_rate(C_w, p.k_redox_aer, p.k_redox_anaer,
                    env['orp'], p.orp_threshold, p.orp_beta)
    R += settling_rate(C_w, p.v_settling, p.K_d, env['spm'], H)
    R += resuspension_rate(C_sed, p.E_resuspension * DAY,
                           tau_b, p.tau_critical, H)
    return R
