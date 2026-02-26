"""
config.py
=========
Central configuration: parameter dataclasses, default values, and
time-series input builders for the Pollutant Fate & Transport Simulator.

Every physical quantity is annotated with SI units unless noted otherwise.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class DomainConfig:
    """Spatial and temporal discretisation."""
    L: float = 10_000.0        # reach length [m]
    nx: int = 201               # spatial nodes
    T_total: float = 3 * 86400  # simulation horizon [s] (3 days)
    dt: float = 30.0            # baseline time-step [s]
    dt_min: float = 1.0         # adaptive lower bound [s]
    dt_max: float = 120.0       # adaptive upper bound [s]
    cfl_target: float = 0.8     # target CFL for adaptive control

    # ---- derived (set in __post_init__) ----
    dx: float = field(init=False)
    x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = self.L / (self.nx - 1)
        self.x = np.linspace(0, self.L, self.nx)


@dataclass
class PhysicalParams:
    """Hydrodynamic / physical variables (can be arrays over time or space)."""
    velocity: float = 0.4               # u  [m/s]
    depth: float = 2.5                   # H  [m]
    temperature: float = 18.0            # T  [°C]
    spm: float = 40.0                    # suspended particulate matter [mg/L]
    dispersion_coeff: float = 15.0       # D  [m²/s]
    turbulence_intensity: float = 0.08   # TI [-]
    bed_slope: float = 0.0005            # S0 [-]
    manning_n: float = 0.035             # Manning n [s/m^{1/3}]
    rho_water: float = 1000.0            # water density [kg/m³]


@dataclass
class ChemicalParams:
    """Water-chemistry variables."""
    pH: float = 7.5
    dissolved_oxygen: float = 8.0      # DO  [mg/L]
    orp: float = 220.0                 # Eh  [mV]
    alkalinity: float = 120.0          # [mg/L as CaCO3]
    ionic_strength: float = 0.01       # [mol/L]
    light_intensity: float = 500.0     # I0  [W/m²] (PAR at surface)


@dataclass
class BiologicalParams:
    """Biological state variables."""
    microbial_biomass: float = 8.0     # X  [mg/L]
    chlorophyll_a: float = 5.0         # [µg/L]
    bod: float = 15.0                  # BOD [mg/L]
    nitrification_rate: float = 0.08   # [1/day]


@dataclass
class KineticParams:
    """All reaction-rate constants and equilibrium parameters."""
    # ---- Biodegradation (Monod + Arrhenius) ----
    k_bio_max: float = 0.4            # [1/day]
    K_s: float = 5.0                  # substrate half-sat [mg/L]
    K_X: float = 10.0                 # biomass half-sat [mg/L]
    theta_bio: float = 1.047          # Arrhenius θ

    # ---- Photolysis ----
    k_photo_ref: float = 0.02         # reference photolysis rate [1/day]
    k_extinction: float = 0.5         # light extinction coeff [1/m]

    # ---- Hydrolysis (3-term pH model) ----
    k_hyd_acid: float = 1.0e-3        # acid-catalysed [L/(mol·day)]
    k_hyd_neutral: float = 5.0e-4     # neutral [1/day]
    k_hyd_base: float = 5.0e-2        # base-catalysed [L/(mol·day)]
    theta_hyd: float = 1.04           # Arrhenius θ

    # ---- Volatilisation ----
    K_henry: float = 1.0e-4           # Henry constant [atm·m³/mol]
    k_vol_mass_transfer: float = 0.5  # overall K_L a  [m/day]

    # ---- Settling / resuspension ----
    v_settling: float = 2.0           # [m/day]
    tau_critical: float = 0.15        # critical shear stress [Pa]
    E_resuspension: float = 5.0e-4    # erosion rate constant [kg/(m²·s)]

    # ---- Redox ----
    k_redox_aer: float = 0.3          # aerobic rate [1/day]
    k_redox_anaer: float = 0.04       # anaerobic rate [1/day]
    orp_threshold: float = 50.0       # E_h* [mV]
    orp_beta: float = 0.05            # steepness of sigmoid [1/mV]

    # ---- Sorption ----
    K_d: float = 12.0                 # distribution coeff [L/kg]
    q_max: float = 80.0               # Langmuir capacity [mg/kg]
    K_L: float = 0.04                 # Langmuir affinity [L/mg]
    K_F: float = 4.0                  # Freundlich coeff [(mg/kg)/(mg/L)^(1/n)]
    n_F: float = 0.75                 # Freundlich exponent
    k_ads: float = 2.0                # adsorption rate [1/day]
    k_des: float = 0.2                # desorption rate [1/day]
    sorption_model: str = "langmuir"  # "langmuir" | "freundlich"


@dataclass
class SedimentParams:
    """Benthic-sediment and hyporheic-zone properties."""
    porosity: float = 0.40             # ε  [-]
    sed_depth: float = 0.10            # active-layer thickness [m]
    D_pore: float = 1.0e-5            # pore-water molecular diffusion [m²/s]
    sed_density: float = 2650.0        # grain density [kg/m³]
    D_eff: float = 5.0e-6             # effective diff across interface [m²/s]
    delta_bl: float = 1.0e-3           # diffusive boundary-layer thickness [m]

    # ---- Hyporheic zone ----
    hyp_depth: float = 0.30           # hyporheic thickness [m]
    alpha_hyp: float = 5.0e-6         # exchange coeff [1/s]
    k_bio_hyp: float = 0.05           # biodeg rate in hyporheic [1/day]


@dataclass
class SourceConfig:
    """Pollutant source specification."""
    # Upstream boundary (Dirichlet)
    C_upstream: float = 0.0            # background [mg/L]

    # Point-source injection
    inject_loc: float = 500.0          # [m] from upstream
    inject_mass_rate: float = 0.0      # continuous rate [mg/s]

    # Pulse (shock) loading
    pulse_loc: float = 500.0           # [m]
    pulse_mass: float = 5000.0         # total mass [mg]
    pulse_time: float = 0.0            # injection time [s]
    pulse_duration: float = 300.0      # duration [s]

    # Seasonal / periodic discharge
    seasonal_amplitude: float = 0.0    # amplitude of velocity variation [m/s]
    seasonal_period: float = 86400.0   # period [s]


@dataclass
class ModelConfig:
    """Top-level container aggregating all sub-configurations."""
    domain: DomainConfig = field(default_factory=DomainConfig)
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    chemical: ChemicalParams = field(default_factory=ChemicalParams)
    biological: BiologicalParams = field(default_factory=BiologicalParams)
    kinetic: KineticParams = field(default_factory=KineticParams)
    sediment: SedimentParams = field(default_factory=SedimentParams)
    source: SourceConfig = field(default_factory=SourceConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten all parameters into a single dictionary."""
        import dataclasses
        out = {}
        for fld in dataclasses.fields(self):
            sub = getattr(self, fld.name)
            if dataclasses.is_dataclass(sub):
                for sf in dataclasses.fields(sub):
                    val = getattr(sub, sf.name)
                    if not isinstance(val, np.ndarray):
                        out[f"{fld.name}.{sf.name}"] = val
        return out


def build_time_varying_params(cfg: ModelConfig, t_array: np.ndarray) -> dict:
    """
    Construct arrays of time-varying environmental parameters.
    Returns dict of arrays, each shaped (nt,).

    Includes: seasonal velocity variation, diel temperature cycle,
    and diel light cycle.
    """
    nt = len(t_array)
    p = {}

    # --- Velocity with optional seasonal pulse ---
    u_base = cfg.physical.velocity
    u_seas = cfg.source.seasonal_amplitude
    T_seas = cfg.source.seasonal_period
    p['velocity'] = u_base + u_seas * np.sin(2 * np.pi * t_array / T_seas)

    # --- Diel temperature cycle (±3 °C) ---
    T_base = cfg.physical.temperature
    p['temperature'] = T_base + 3.0 * np.sin(2 * np.pi * t_array / 86400 - np.pi / 2)

    # --- Diel light (half-sinusoid for daytime) ---
    hour = (t_array % 86400) / 3600.0
    daylight = np.where((hour >= 6) & (hour <= 18),
                        np.sin(np.pi * (hour - 6) / 12), 0.0)
    p['light'] = cfg.chemical.light_intensity * daylight

    # --- Other parameters (constant in default, but ready for arrays) ---
    p['depth'] = np.full(nt, cfg.physical.depth)
    p['spm'] = np.full(nt, cfg.physical.spm)
    p['pH'] = np.full(nt, cfg.chemical.pH)
    p['DO'] = np.full(nt, cfg.chemical.dissolved_oxygen)
    p['orp'] = np.full(nt, cfg.chemical.orp)
    p['biomass'] = np.full(nt, cfg.biological.microbial_biomass)
    p['dispersion'] = np.full(nt, cfg.physical.dispersion_coeff)

    return p
