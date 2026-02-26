"""
main.py
=======
Master driver script demonstrating the full Pollutant Fate & Transport
Simulator workflow:

  1. Configure model
  2. Run baseline simulation
  3. Generate synthetic observations
  4. Calibrate parameters
  5. Sensitivity analysis
  6. Uncertainty quantification (MCMC)
  7. ML-based rate-constant prediction
  8. Visualise everything
"""

import numpy as np
import matplotlib.pyplot as plt

from config import (ModelConfig, DomainConfig, PhysicalParams, ChemicalParams,
                    BiologicalParams, KineticParams, SedimentParams,
                    SourceConfig)
from solver import run_simulation
from calibration import (calibrate_least_squares, sobol_sensitivity,
                         mcmc_calibration, compute_all_metrics)
from visualization import (plot_dashboard, plot_spatial_profiles,
                           plot_time_series, plot_sediment_accumulation,
                           plot_sensitivity, plot_uncertainty_envelope,
                           plot_mass_balance)
from ml_module import (generate_synthetic_training_data,
                       RateConstantPredictor, plot_ml_diagnostics)


def main():
    # ==============================================================
    # 1. CONFIGURE MODEL
    # ==============================================================
    print("=" * 60)
    print("POLLUTANT FATE & TRANSPORT SIMULATOR")
    print("=" * 60)

    cfg = ModelConfig(
        domain=DomainConfig(
            L=10_000.0,         # 10 km reach
            nx=201,
            T_total=2 * 86400,  # 2 days
            dt=30.0,
            dt_min=1.0,
            dt_max=120.0,
            cfl_target=0.8,
        ),
        physical=PhysicalParams(
            velocity=0.4,
            depth=2.5,
            temperature=18.0,
            spm=40.0,
            dispersion_coeff=15.0,
            bed_slope=0.0005,
        ),
        chemical=ChemicalParams(
            pH=7.5,
            dissolved_oxygen=8.0,
            orp=220.0,
        ),
        biological=BiologicalParams(
            microbial_biomass=8.0,
        ),
        kinetic=KineticParams(
            k_bio_max=0.4,
            K_s=5.0,
            k_photo_ref=0.02,
            k_vol_mass_transfer=0.3,
            v_settling=2.0,
            K_d=12.0,
            k_ads=2.0,
            k_des=0.2,
        ),
        sediment=SedimentParams(
            porosity=0.4,
            sed_depth=0.1,
            alpha_hyp=5e-6,
        ),
        source=SourceConfig(
            C_upstream=0.0,
            pulse_loc=500.0,
            pulse_mass=8000.0,        # 8 g total
            pulse_time=0.0,
            pulse_duration=300.0,     # 5 min pulse
            seasonal_amplitude=0.1,   # ±0.1 m/s velocity variation
            seasonal_period=86400.0,  # 1-day period
        ),
    )

    monitor_locs = [1000.0, 2500.0, 5000.0, 7500.0, 9000.0]

    # ==============================================================
    # 2. RUN BASELINE SIMULATION
    # ==============================================================
    print("\n--- Running baseline simulation ---")
    result = run_simulation(cfg, monitor_locs=monitor_locs,
                            save_interval=300.0, verbose=True)

    # ==============================================================
    # 3. GENERATE SYNTHETIC OBSERVATIONS
    # ==============================================================
    print("\n--- Generating synthetic observations ---")
    np.random.seed(123)

    # Sample from simulation result + noise
    obs_interval = 1800.0  # every 30 min
    obs_times = np.arange(obs_interval, cfg.domain.T_total,
                          obs_interval)
    obs_data = {}
    for loc in [2500.0, 5000.0, 7500.0]:
        if loc in result.C_w_monitor:
            sim_series = result.C_w_monitor[loc]
            sim_times = result.times
            sim_interp = np.interp(obs_times, sim_times, sim_series)
            noise = 0.05 * sim_interp * np.random.randn(len(obs_times))
            obs_data[loc] = np.maximum(sim_interp + noise, 0.0)

    print(f"  Generated {len(obs_times)} observation times "
          f"at {len(obs_data)} locations")

    # ==============================================================
    # 4. VISUALISE BASELINE RESULTS
    # ==============================================================
    print("\n--- Creating visualisations ---")

    plot_dashboard(result, obs_data=obs_data, obs_times=obs_times,
                   savefig='dashboard.png')

    plot_sediment_accumulation(result, savefig='sediment.png')
    plot_mass_balance(result, savefig='mass_balance.png')

    # ==============================================================
    # 5. PARAMETER SENSITIVITY ANALYSIS
    # ==============================================================
    print("\n--- Sobol sensitivity analysis ---")
    # (Reduced N for demo speed; increase N for publication)
    param_names_sa = [
        'kinetic.k_bio_max',
        'kinetic.k_photo_ref',
        'kinetic.k_vol_mass_transfer',
        'kinetic.v_settling',
        'kinetic.K_d',
    ]
    bounds_sa = [
        (0.1, 1.0),    # k_bio_max
        (0.001, 0.1),  # k_photo_ref
        (0.05, 1.0),   # k_vol
        (0.5, 5.0),    # v_settling
        (1.0, 50.0),   # K_d
    ]

    sobol_res = sobol_sensitivity(
        cfg, param_names_sa, bounds_sa,
        obs_loc=5000.0, N=32,    # N=32 for fast demo; use N≥512 for research
        monitor_locs=monitor_locs)

    plot_sensitivity(sobol_res, savefig='sensitivity.png')
    print("  First-order indices:", dict(zip(
        [n.split('.')[-1] for n in param_names_sa], sobol_res['S1'])))

    # ==============================================================
    # 6. CALIBRATION (Least-Squares)
    # ==============================================================
    print("\n--- Least-squares calibration ---")
    param_names_cal = ['kinetic.k_bio_max', 'kinetic.K_d']
    bounds_cal = [(0.1, 1.0), (1.0, 50.0)]

    cal_result = calibrate_least_squares(
        cfg, param_names_cal, bounds_cal,
        obs_data, obs_times, monitor_locs,
        method='differential_evolution', maxiter=15)

    print("  Best parameters:", cal_result['best_params'])
    print("  Objective (SSR):", f"{cal_result['objective_value']:.4f}")

    # ==============================================================
    # 7. MCMC UNCERTAINTY QUANTIFICATION
    # ==============================================================
    print("\n--- MCMC uncertainty estimation ---")
    mcmc_res = mcmc_calibration(
        cfg, param_names_cal, bounds_cal,
        obs_data, obs_times, monitor_locs,
        n_samples=300,  # increase for publication (≥5000)
        sigma_obs=0.2, proposal_scale=0.03)

    print(f"  Acceptance rate: {mcmc_res['acceptance_rate']:.2%}")

    # Posterior statistics
    burn_in = 150
    for i, name in enumerate(param_names_cal):
        post = mcmc_res['chain'][burn_in:, i]
        print(f"  {name}: mean={post.mean():.4f} "
              f"std={post.std():.4f} "
              f"95%CI=[{np.percentile(post,2.5):.4f}, "
              f"{np.percentile(post,97.5):.4f}]")

    # Uncertainty envelope plot
    plot_uncertainty_envelope(
        cfg, mcmc_res, obs_loc=5000.0,
        monitor_locs=monitor_locs,
        obs_data=obs_data, obs_times=obs_times,
        n_posterior=20, savefig='uncertainty.png')

    # ==============================================================
    # 8. MACHINE LEARNING MODULE
    # ==============================================================
    print("\n--- ML rate-constant prediction ---")

    # Generate synthetic training data
    X_train, y_train = generate_synthetic_training_data(n_samples=1000)

    # Train models
    for model_type in ['random_forest', 'neural_network']:
        print(f"\n  Training {model_type}...")
        predictor = RateConstantPredictor(model_type=model_type)
        metrics = predictor.fit(X_train, y_train, verbose=True)

        if model_type == 'random_forest':
            plot_ml_diagnostics(X_train, y_train, predictor,
                                savefig=f'ml_{model_type}.png')

            # Demonstrate integration: predict k for current conditions
            k_pred = predictor.predict(
                T=cfg.physical.temperature,
                pH=cfg.chemical.pH,
                DO=cfg.chemical.dissolved_oxygen,
                orp=cfg.chemical.orp,
                spm=cfg.physical.spm,
                biomass=cfg.biological.microbial_biomass)
            print(f"\n  ML-predicted k_bio for current conditions: "
                  f"{k_pred[0]:.4f} 1/day")
            print(f"  Model default k_bio_max: "
                  f"{cfg.kinetic.k_bio_max} 1/day")

    # ==============================================================
    # 9. SHOCK-LOADING EVENT
    # ==============================================================
    print("\n--- Shock loading scenario ---")
    cfg_shock = ModelConfig(
        domain=DomainConfig(L=10000, nx=201, T_total=86400, dt=20),
        physical=PhysicalParams(velocity=0.6, depth=2.0),
        source=SourceConfig(
            pulse_loc=500, pulse_mass=20000, pulse_time=0,
            pulse_duration=120),  # 2-min high-mass pulse
        kinetic=KineticParams(k_bio_max=0.5),
    )
    result_shock = run_simulation(cfg_shock, monitor_locs=monitor_locs,
                                  save_interval=300, verbose=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_spatial_profiles(result_shock, compartment='water', ax=ax,
                          title='Shock Loading — Water Column Response')
    plt.savefig('shock_loading.png')
    plt.show()

    # ==============================================================
    # 10. PRINT METRICS SUMMARY
    # ==============================================================
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    for loc in obs_data:
        sim_interp = np.interp(obs_times, result.times,
                               result.C_w_monitor.get(loc, [0]))
        n = min(len(obs_data[loc]), len(sim_interp))
        m = compute_all_metrics(obs_data[loc][:n], sim_interp[:n])
        print(f"\n  Location x = {loc/1000:.1f} km:")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    print("\n✓ All tasks complete. Figures saved.")


if __name__ == '__main__':
    main()
