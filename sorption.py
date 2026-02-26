"""
sorption.py
===========
Sorption equilibria and kinetics.

Supports Langmuir and Freundlich isotherms, plus kinetic (non-equilibrium)
adsorption–desorption.
"""

import numpy as np


def langmuir_equilibrium(C, q_max, K_L):
    """
    Langmuir isotherm:   q_eq = q_max K_L C / (1 + K_L C)

    Parameters
    ----------
    C     : dissolved concentration [mg/L]
    q_max : maximum adsorption capacity [mg/kg]
    K_L   : Langmuir affinity constant [L/mg]

    Returns
    -------
    q_eq  : equilibrium sorbed concentration [mg/kg]
    """
    return q_max * K_L * C / (1.0 + K_L * np.maximum(C, 0))


def freundlich_equilibrium(C, K_F, n_F):
    """
    Freundlich isotherm:  q_eq = K_F · C^(1/n)

    Parameters
    ----------
    C   : dissolved concentration [mg/L]
    K_F : Freundlich coefficient [(mg/kg)/(mg/L)^(1/n)]
    n_F : Freundlich exponent [-]
    """
    return K_F * np.maximum(C, 0) ** (1.0 / n_F)


def sorption_kinetic_rate(C, q, k_ads, k_des, q_max, K_L,
                          spm, model='langmuir'):
    """
    Kinetic (non-equilibrium) sorption rate.

    dq/dt = k_ads · C · (q_max - q) - k_des · q      [mg/(kg·day)]
    dC/dt = -spm_kg · dq/dt                            [mg/(L·day)]

    Parameters
    ----------
    C     : dissolved concentration [mg/L]
    q     : current sorbed concentration [mg/kg]
    k_ads : adsorption rate constant [L/(mg·day)]
    k_des : desorption rate constant [1/day]
    q_max : Langmuir capacity [mg/kg]
    K_L   : Langmuir affinity (unused in kinetic, kept for interface)
    spm   : suspended particulate matter [mg/L]
    model : 'langmuir' or 'freundlich'

    Returns
    -------
    dC_dt : rate of change of dissolved concentration [mg/L/day]
    dq_dt : rate of change of sorbed concentration [mg/kg/day]
    """
    spm_kg = spm * 1e-6  # mg/L → kg/L

    dq_dt = k_ads * np.maximum(C, 0) * (q_max - q) - k_des * q
    dC_dt = -spm_kg * dq_dt  # mass balance: what sorbs leaves solution

    return dC_dt, dq_dt


def partition_coefficient_dynamic(K_d_ref, spm, pH, ionic_strength):
    """
    Compute an adjusted distribution coefficient accounting for
    pH and ionic-strength effects (empirical correction).

    K_d = K_d_ref · 10^(0.3·(pH-7)) · (1 + 2·I)^{-0.5}

    This represents:
    - Higher K_d at higher pH (more sorption for cationic species)
    - Lower K_d at higher ionic strength (competition for sites)

    Parameters
    ----------
    K_d_ref         : reference K_d at pH 7, I=0.01 [L/kg]
    spm             : SPM [mg/L] (unused but available for extension)
    pH              : [-]
    ionic_strength  : [mol/L]
    """
    pH_factor = 10.0 ** (0.3 * (pH - 7.0))
    IS_factor = (1.0 + 2.0 * ionic_strength) ** (-0.5)
    return K_d_ref * pH_factor * IS_factor


def particulate_fraction(K_d, spm):
    """
    f_p = K_d · SPM_kg / (1 + K_d · SPM_kg)

    Parameters
    ----------
    K_d : distribution coefficient [L/kg]
    spm : suspended particulate matter [mg/L]
    """
    spm_kg = spm * 1e-6
    return K_d * spm_kg / (1.0 + K_d * spm_kg)
