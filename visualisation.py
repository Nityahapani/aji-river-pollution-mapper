"""
visualization.py
================
Publication-quality plotting for simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, List


def setup_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# ================================================================
# 1. SPATIAL CONCENTRATION PROFILES AT MULTIPLE TIMES
# ================================================================
def plot_spatial_profiles(result, times_hours=None, compartment='water',
                          title=None, ax=None, savefig=None):
    """
    Plot concentration vs. distance at selected times.

    Parameters
    ----------
    result      : SimulationResult
    times_hours : list of times [hours] to plot (default: 5 equally spaced)
    compartment : 'water', 'sediment', or 'hyporheic'
    """
    setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    data_map = {
        'water': result.C_w_series,
        'sediment': result.C_s_series,
        'hyporheic': result.C_h_series,
    }
    data = data_map[compartment]
    x_km = result.x / 1000.0
    t_hours = result.times / 3600.0

    if times_hours is None:
        indices = np.linspace(0, len(t_hours) - 1, 6, dtype=int)[1:]
    else:
        indices = [np.argmin(np.abs(t_hours - th)) for th in times_hours]

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.85, len(indices)))

    for idx, color in zip(indices, colors):
        ax.plot(x_km, data[idx], color=color, linewidth=1.5,
                label=f't = {t_hours[idx]:.1f} h')

    ax.set_xlabel('Distance downstream [km]')
    ax.set_ylabel('Concentration [mg/L]')
    ax.set_title(title or f'{compartment.title()} Column — Spatial Profiles')
    ax.legend(loc='best')
    ax.set_xlim(x_km[0], x_km[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)
    return ax


# ================================================================
# 2. TIME-SERIES AT MONITORING POINTS
# ================================================================
def plot_time_series(result, obs_data=None, obs_times=None,
                     title=None, ax=None, savefig=None):
    """
    Plot concentration time-series at all monitoring locations.

    Parameters
    ----------
    obs_data  : dict {location_m: observed_array} (optional overlay)
    obs_times : observation times [s] (if obs_data provided)
    """
    setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    t_hours = result.times / 3600.0
    cmap = plt.cm.tab10

    for i, (loc, series) in enumerate(result.C_w_monitor.items()):
        color = cmap(i / max(len(result.C_w_monitor) - 1, 1))
        ax.plot(t_hours[:len(series)], series, color=color,
                linewidth=1.5, label=f'x = {loc/1000:.1f} km')

        if obs_data and loc in obs_data and obs_times is not None:
            ax.scatter(obs_times / 3600, obs_data[loc],
                       color=color, marker='o', s=20, zorder=5,
                       edgecolors='k', linewidths=0.5)

    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Concentration [mg/L]')
    ax.set_title(title or 'Concentration Time-Series at Monitoring Points')
    ax.legend(loc='best')
    ax.set_xlim(0, t_hours[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)
    return ax


# ================================================================
# 3. SEDIMENT ACCUMULATION CURVES
# ================================================================
def plot_sediment_accumulation(result, title=None, savefig=None):
    """Plot total sediment-phase mass over time."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_km = result.x / 1000
    t_hours = result.times / 3600

    # (a) Spatial sediment profiles at multiple times
    ax = axes[0]
    n_snap = min(5, len(t_hours))
    indices = np.linspace(0, len(t_hours) - 1, n_snap, dtype=int)
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, n_snap))
    for idx, c in zip(indices, colors):
        ax.plot(x_km, result.C_s_series[idx], color=c,
                label=f't={t_hours[idx]:.1f}h')
    ax.set_xlabel('Distance [km]')
    ax.set_ylabel('Sediment concentration [mg/L eq.]')
    ax.set_title('Sediment Spatial Profiles')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Total sediment mass over time
    ax = axes[1]
    sed_mass = np.trapz(result.C_s_series, result.x, axis=1)
    ax.plot(t_hours, sed_mass, 'r-', linewidth=2)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Total sediment mass [mg·m]')
    ax.set_title('Sediment Mass Accumulation')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title or 'Sediment Compartment Analysis', fontsize=14)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)


# ================================================================
# 4. SENSITIVITY RANKING CHART
# ================================================================
def plot_sensitivity(sobol_result, title=None, savefig=None):
    """Horizontal bar chart of Sobol S1 and ST indices."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [n.split('.')[-1] for n in sobol_result['param_names']]
    S1 = sobol_result['S1']
    ST = sobol_result['ST']

    y = np.arange(len(names))
    height = 0.35

    ax.barh(y - height / 2, S1, height, label='First-order (S₁)',
            color='steelblue', edgecolor='white')
    ax.barh(y + height / 2, ST, height, label='Total (Sₜ)',
            color='coral', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Sensitivity Index')
    ax.set_title(title or 'Sobol Sensitivity Indices')
    ax.legend()
    ax.set_xlim(0, max(np.max(ST), np.max(S1)) * 1.2)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)


# ================================================================
# 5. UNCERTAINTY ENVELOPES (from MCMC posterior)
# ================================================================
def plot_uncertainty_envelope(cfg, mcmc_result, obs_loc,
                              monitor_locs, obs_data=None,
                              obs_times=None,
                              n_posterior=50, savefig=None):
    """
    Draw uncertainty envelopes by sampling from the MCMC posterior.
    """
    import copy
    from solver import run_simulation

    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    chain = mcmc_result['chain']
    param_names = mcmc_result['param_names']
    n_total = chain.shape[0]
    burn_in = n_total // 2
    posterior = chain[burn_in:]

    # Sample from posterior
    idx_sample = np.random.choice(len(posterior), size=n_posterior,
                                  replace=False)
    all_series = []

    for idx in idx_sample:
        theta = posterior[idx]
        cfg_i = copy.deepcopy(cfg)
        for name, val in zip(param_names, theta):
            parts = name.split('.')
            obj = cfg_i
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)

        res = run_simulation(cfg_i, monitor_locs=monitor_locs,
                             save_interval=600, verbose=False)
        if obs_loc in res.C_w_monitor:
            all_series.append(res.C_w_monitor[obs_loc])

    # Align lengths
    min_len = min(len(s) for s in all_series)
    mat = np.array([s[:min_len] for s in all_series])
    t_hrs = np.linspace(0, cfg.domain.T_total / 3600, min_len)

    median = np.median(mat, axis=0)
    q05 = np.percentile(mat, 5, axis=0)
    q95 = np.percentile(mat, 95, axis=0)
    q25 = np.percentile(mat, 25, axis=0)
    q75 = np.percentile(mat, 75, axis=0)

    ax.fill_between(t_hrs, q05, q95, alpha=0.2, color='steelblue',
                    label='90% CI')
    ax.fill_between(t_hrs, q25, q75, alpha=0.4, color='steelblue',
                    label='50% CI')
    ax.plot(t_hrs, median, 'b-', linewidth=2, label='Median')

    if obs_data and obs_loc in obs_data and obs_times is not None:
        ax.scatter(obs_times / 3600, obs_data[obs_loc], color='red',
                   marker='o', s=30, zorder=5, label='Observations')

    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Concentration [mg/L]')
    ax.set_title(f'Uncertainty Envelope at x = {obs_loc/1000:.1f} km')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)


# ================================================================
# 6. MASS BALANCE PLOT
# ================================================================
def plot_mass_balance(result, title=None, savefig=None):
    """Plot total system mass over time."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    t_hours = result.times / 3600
    ax.plot(t_hours, result.mass_balance, 'k-', linewidth=2)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Total mass in system [mg·m]')
    ax.set_title(title or 'System Mass Balance')
    ax.grid(True, alpha=0.3)

    if savefig:
        plt.savefig(savefig)


# ================================================================
# 7. COMPREHENSIVE DASHBOARD
# ================================================================
def plot_dashboard(result, obs_data=None, obs_times=None,
                   savefig=None):
    """Create a 2×2 comprehensive results dashboard."""
    setup_style()
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (a) Spatial profiles
    ax1 = fig.add_subplot(gs[0, 0])
    plot_spatial_profiles(result, compartment='water', ax=ax1,
                          title='Water Column Profiles')

    # (b) Time series
    ax2 = fig.add_subplot(gs[0, 1])
    plot_time_series(result, obs_data=obs_data, obs_times=obs_times,
                     ax=ax2, title='Monitoring Time-Series')

    # (c) Sediment
    ax3 = fig.add_subplot(gs[1, 0])
    indices = np.linspace(0, len(result.times) - 1, 4, dtype=int)
    for idx in indices:
        ax3.plot(result.x / 1000, result.C_s_series[idx],
                 label=f't={result.times[idx]/3600:.1f}h')
    ax3.set_xlabel('Distance [km]')
    ax3.set_ylabel('Sediment conc. [mg/L eq.]')
    ax3.set_title('Sediment Profiles')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # (d) Mass balance
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(result.times / 3600, result.mass_balance, 'k-', lw=2)
    ax4.set_xlabel('Time [hours]')
    ax4.set_ylabel('Total mass [mg·m]')
    ax4.set_title('Mass Balance')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Pollutant Fate & Transport — Simulation Dashboard',
                 fontsize=15, fontweight='bold')

    if savefig:
        plt.savefig(savefig)
    plt.show()
