"""
src/config.py
=============
Default bioreactor parameters and regime-override tables for BioSentinel 2.0.

All physical constants use SI-consistent units where possible:
  - Concentrations : g/L
  - Time           : hours
  - Rates          : per hour (h⁻¹)
  - kLa            : h⁻¹
  - OUR/CER/OD600 : g/(L·h) or dimensionless where noted

No Streamlit or UI imports; pure data.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Master parameter set (healthy baseline)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    # --- Biomass / Monod kinetics ---
    "mu_max":   0.40,      # h⁻¹   maximum specific growth rate
    "K_s":      0.10,      # g/L   Monod substrate half-saturation constant
    "K_O":      0.01,      # g/L   oxygen half-saturation constant (dual-Monod)
    "Y_xs":     0.50,      # g/g   biomass yield on substrate
    "Y_xo":     0.10,      # g/g   biomass yield on oxygen (O2 consumed per g biomass)
    "m_s":      0.02,      # h⁻¹   substrate maintenance coefficient
    "m_o":      0.005,     # h⁻¹   oxygen maintenance coefficient

    # --- Initial conditions ---
    "X0":       0.10,      # g/L   initial biomass
    "S0":       10.0,      # g/L   initial substrate
    "CL0":      0.008,     # g/L   initial dissolved oxygen (≈ 8 mg/L at 25 °C)

    # --- Oxygen transfer ---
    "kLa":      180.0,     # h⁻¹   volumetric mass-transfer coefficient
    "CL_star":  0.0084,    # g/L   DO saturation concentration (25 °C, air, 1 atm)

    # --- Substrate feed (fed-batch top-up, constant) ---
    "F_s":      0.05,      # g/(L·h)  substrate feed rate (dilution × feed conc)

    # --- Bioreactor instruments (logged but not ODE states) ---
    "RPM":      200.0,     # rpm
    "airflow":  1.0,       # vvm  (volumes of air per volume per minute)
    "pressure": 1.0,       # atm

    # --- Simulation time ---
    "t_span":   (0.0, 24.0),  # hours
    "t_eval_n": 1440,          # number of evaluation points (1 per minute)

    # --- Sensor noise ---
    "noise_pct": 0.05,     # 5 % Gaussian noise on all measured signals
}

# ---------------------------------------------------------------------------
# Regime-override tables
# Entries here are *merged* on top of DEFAULT_PARAMS at simulation time.
# Keys that differ per regime or that are regime-specific (onset, severity)
# are described below.
# ---------------------------------------------------------------------------

REGIME_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Healthy": {
        # No overrides – pure baseline
        "t_onset":  None,   # no fault
        "severity": 0.0,
    },

    "kLa_Limitation": {
        # After t_onset, kLa is reduced by severity fraction
        # e.g. severity=0.7 → kLa drops to 30 % of nominal
        "t_onset":  8.0,    # hours into run when fouling begins
        "severity": 0.70,   # fraction of kLa to *remove* (0–1)
    },

    "Substrate_Overfeeding": {
        # After t_onset, substrate feed rate is multiplied by (1 + severity)
        # e.g. severity=3.0 → F_s × 4
        "t_onset":  6.0,
        "severity": 3.0,    # dimensionless multiplier above baseline
    },

    "Contamination": {
        # After t_onset a second microbial population grows with faster mu_max
        # and higher O2 demand, causing DO sag and RQ divergence.
        "t_onset":      5.0,
        "severity":     0.50,   # fraction of X that contaminant can reach at max
        "mu_max_cont":  0.80,   # h⁻¹ contaminant growth rate
        "K_s_cont":     0.05,   # g/L contaminant Ks
        "Y_xo_cont":    0.25,   # g/g higher O2 demand
        "X_cont0":      0.001,  # g/L initial contaminant inoculum
    },
}

# ---------------------------------------------------------------------------
# Recoverability Engine constants
# Source: Extended_Bioprocess_Model_Equations.pdf  (Equations 14–15)
# ---------------------------------------------------------------------------

STRESS_PARAMS = {
    # --- Stress sensitivity coefficients (Eq. 14) ---
    # λ = k_stress * |pH_opt - pH(t)| + k_O2 * |C_crit - CL|
    #
    # The PDF specifies these as tunable constants; the numeric values below
    # are HACKATHON IMPLEMENTATION ASSUMPTIONS — not biologically validated.
    "k_stress": 1.0,       # h⁻¹  stress weight for pH deviation
    "k_O2":     200.0,     # h⁻¹·(g/L)⁻¹  stress weight for DO deviation
    "pH_opt":   7.0,       # dimensionless  optimal pH (assumption: neutral)
    # C_crit is the nominal healthy operating DO (≈ 7 mg/L = 0.007 g/L).
    # Stress = k_O2 * |C_crit - CL|, so stress ≈ 0 at healthy DO and rises
    # when DO sags below nominal during a fault.
    # HACKATHON IMPLEMENTATION ASSUMPTION (set to match simulator healthy DO).
    "C_crit":   0.0070,    # g/L  nominal healthy dissolved oxygen target

    # --- Recoverability thresholds (Eq. 15) ---
    # R(t) = exp(-∫λ dt) ∈ (0, 1]
    #
    # PNR: Point of No Return — R drops below this → batch deemed unrecoverable.
    # Chosen as 0.20 (20 %) for the demo.
    # HACKATHON IMPLEMENTATION ASSUMPTION.
    "R_pnr":    0.20,

    # DIW: Decision Intervention Window — hours remaining before R hits R_pnr
    # given the current instantaneous stress rate.
    # HACKATHON IMPLEMENTATION ASSUMPTION: linear extrapolation of stress.
    "R_diw_floor": 0.0,    # hours  minimum DIW clamp (cannot be negative)

    # --- Cost-of-Delay parameters ---
    # Cost-of-Delay(Δt) = batch_value * (1 - R(t + Δt)) - intervention_cost
    # HACKATHON IMPLEMENTATION ASSUMPTION (monetary model).
    "batch_value":        10_000.0,  # USD  expected revenue per successful batch
    "intervention_cost":    500.0,   # USD  cost of a corrective intervention
}
