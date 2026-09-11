"""
src/counterfactual.py
=====================
BioSentinel 2.0 — Counterfactual Rollout Engine (Layer 3).

Given the *current* reactor state at time ``current_t``, simulate each
candidate intervention forward for a bounded horizon and rank them by the
ending recoverability R(current_t + horizon).

Public API
----------
    run_counterfactual(
        current_state, current_t, params, intervention, horizon=2.0
    ) -> CounterfactualResult

    rank_interventions(
        current_state, current_t, params, horizon=2.0
    ) -> list[CounterfactualResult]   # sorted best → worst by ending R

Supported interventions
-----------------------
    "no_action"       — baseline: continue with current (possibly degraded) params
    "+150_RPM"        — increase agitation by 150 RPM → kLa ∝ RPM^0.6 scaling
    "-30%_feed"       — reduce substrate feed to 0.70 × F_s
    "combined"        — +150 RPM AND -30 % feed simultaneously

kLa–RPM scaling
---------------
Power-law approximation for stirred-tank bioreactors:

    kLa_new / kLa_old ≈ (RPM_new / RPM_old)^0.6

This exponent (0.6) is the project-specified approximate kLa–agitation
scaling.  See: Nienow, A.W. (2006) "Hydrodynamics of stirred bioreactors".
HACKATHON IMPLEMENTATION ASSUMPTION: exponent calibrated to stirred-tank
data; actual value is reactor-geometry-dependent.

Feed reduction
--------------
    F_s_new = 0.70 × F_s_current

No Streamlit / UI / database / API / Docker imports.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.config import DEFAULT_PARAMS, STRESS_PARAMS
from src.regimes import get_regime, ContaminationRegime
from src.stress import compute_stress, compute_recoverability

__all__ = [
    "CounterfactualResult",
    "run_counterfactual",
    "rank_interventions",
    "INTERVENTIONS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# kLa–RPM power-law exponent for stirred-tank bioreactors.
# Project specification: "approximate kLa scaling with RPM".
# HACKATHON IMPLEMENTATION ASSUMPTION: exponent = 0.6 (Van't Riet correlation).
_KLA_RPM_EXPONENT: float = 0.6

# Feed reduction factor for "-30%_feed" intervention
_FEED_REDUCTION: float = 0.70   # F_s_new = 0.70 × F_s_current

# Horizon evaluation points: number of ODE time-steps within the rollout.
# Fewer points → faster; still adequate for stress accumulation trends.
_HORIZON_EVAL_DENSITY: int = 60   # points per horizon-hour (≈ 1 per minute)

# Supported intervention identifiers
INTERVENTIONS: List[str] = [
    "no_action",
    "+150_RPM",
    "-30%_feed",
    "combined",
]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Result for a single counterfactual rollout.

    Attributes
    ----------
    intervention : str
        Identifier of the intervention applied.
    ending_R : float
        Recoverability R at the end of the rollout horizon (0–1].
        Higher is better.
    trajectory : pd.DataFrame
        Full simulated trajectory with stable schema matching
        ``simulate_batch()`` output, plus ``lambda_t`` and ``R`` columns.
    params_applied : dict
        Effective parameter dict used during the rollout (after intervention).
    horizon : float
        Duration of the rollout in hours.
    pnr_crossed : bool
        Whether R dropped below R_pnr during the rollout.
    """
    intervention:  str
    ending_R:      float
    trajectory:    pd.DataFrame
    params_applied: Dict[str, Any]
    horizon:       float
    pnr_crossed:   bool = field(default=False)

    def __repr__(self) -> str:
        return (
            f"CounterfactualResult(intervention={self.intervention!r}, "
            f"ending_R={self.ending_R:.4f}, pnr_crossed={self.pnr_crossed})"
        )


# ---------------------------------------------------------------------------
# Intervention parameter builders
# ---------------------------------------------------------------------------

def _apply_intervention(params: Dict[str, Any], intervention: str) -> Dict[str, Any]:
    """Return a new param dict with the specified intervention applied.

    Parameters
    ----------
    params : dict
        Current effective reactor parameters (already merged with regime).
    intervention : str
        One of INTERVENTIONS.

    Returns
    -------
    dict
        Modified copy of params.
    """
    p = copy.deepcopy(params)

    if intervention == "no_action":
        pass  # no modifications

    elif intervention == "+150_RPM":
        rpm_old = float(p.get("RPM", DEFAULT_PARAMS["RPM"]))
        rpm_new = rpm_old + 150.0
        kla_old = float(p.get("kLa", DEFAULT_PARAMS["kLa"]))
        # kLa scales with RPM^0.6 (project-specified approximate scaling)
        kla_new = kla_old * (rpm_new / max(rpm_old, 1.0)) ** _KLA_RPM_EXPONENT
        p["RPM"] = rpm_new
        p["kLa"] = kla_new

    elif intervention == "-30%_feed":
        p["F_s"] = float(p.get("F_s", DEFAULT_PARAMS["F_s"])) * _FEED_REDUCTION

    elif intervention == "combined":
        # +150 RPM AND -30 % feed
        rpm_old = float(p.get("RPM", DEFAULT_PARAMS["RPM"]))
        rpm_new = rpm_old + 150.0
        kla_old = float(p.get("kLa", DEFAULT_PARAMS["kLa"]))
        kla_new = kla_old * (rpm_new / max(rpm_old, 1.0)) ** _KLA_RPM_EXPONENT
        p["RPM"] = rpm_new
        p["kLa"] = kla_new
        p["F_s"] = float(p.get("F_s", DEFAULT_PARAMS["F_s"])) * _FEED_REDUCTION

    else:
        raise ValueError(
            f"Unknown intervention '{intervention}'. "
            f"Valid options: {INTERVENTIONS}"
        )

    return p


# ---------------------------------------------------------------------------
# ODE helpers (reuse simulator dynamics without importing private functions)
# ---------------------------------------------------------------------------

def _build_rollout_rhs(regime_obj, params: Dict[str, Any]):
    """Build the ODE RHS for a counterfactual rollout.

    Replicates the physics from simulator._build_rhs but parameterised with
    the intervention-modified params.  The regime's own mutate() method is
    still called so existing fault dynamics remain active (the intervention
    works *against* the fault, not around it).
    """
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        X  = max(y[0], 0.0)
        S  = max(y[1], 0.0)
        CL = max(y[2], 0.0)

        # Merge: regime mutation on top of intervention-modified base params
        p = regime_obj.mutate(t, tuple(y), params)

        mu_max = p["mu_max"]
        K_s    = p["K_s"]
        K_O    = p["K_O"]
        Y_xs   = p["Y_xs"]
        Y_xo   = p["Y_xo"]
        m_s    = p["m_s"]
        m_o    = p["m_o"]
        kLa    = p["kLa"]
        CL_s   = p["CL_star"]
        F_s    = p["F_s"]

        mu = (
            mu_max
            * (S  / (K_s  + S  + 1e-12))
            * (CL / (K_O  + CL + 1e-12))
        )

        dX  = mu * X
        dS  = -((mu / Y_xs) + m_s) * X + F_s

        OUR = ((mu / Y_xo) + m_o) * X

        # Contaminant contribution (if active)
        dX_c    = 0.0
        OUR_c   = 0.0
        if isinstance(regime_obj, ContaminationRegime) and len(y) > 3:
            X_c = max(y[3], 0.0)
            (dX_c,) = regime_obj.extra_rhs(t, tuple(y), p)
            mu_c_eff = dX_c / (X_c + 1e-12)
            dS    -= (mu_c_eff / Y_xs + m_s) * X_c
            OUR_c  = (mu_c_eff / regime_obj.Y_xo_cont + m_o) * X_c

        dCL = kLa * (CL_s - CL) - (OUR + OUR_c)

        if isinstance(regime_obj, ContaminationRegime) and len(y) > 3:
            return np.array([dX, dS, dCL, dX_c])
        return np.array([dX, dS, dCL])

    return rhs


# ---------------------------------------------------------------------------
# Core rollout function
# ---------------------------------------------------------------------------

def run_counterfactual(
    current_state: Dict[str, float],
    current_t: float,
    params: Dict[str, Any],
    intervention: str,
    *,
    horizon: float = 2.0,
    regime: str = "Healthy",
) -> CounterfactualResult:
    """Simulate one counterfactual intervention forward for ``horizon`` hours.

    Parameters
    ----------
    current_state : dict
        Reactor state at the rollout start point.  Required keys::

            "X"   — biomass concentration (g/L)
            "S"   — substrate concentration (g/L)
            "DO"  — dissolved oxygen (g/L)
            "X_c" — contaminant biomass (g/L) [optional; for Contamination]

    current_t : float
        Time (hours) corresponding to ``current_state``.
    params : dict
        Current effective reactor parameters.  This should be the merged
        parameter dict (DEFAULT_PARAMS + any regime overrides already applied).
        The function deep-copies this dict before mutating it.
    intervention : str
        One of ``INTERVENTIONS``:
        ``"no_action"``, ``"+150_RPM"``, ``"-30%_feed"``, ``"combined"``.
    horizon : float
        Number of hours to simulate forward.  Kept short (≤ 4 h) for live
        demo responsiveness.  Default: 2.0 h.
    regime : str
        Regime label used to instantiate the regime mutator.
        Default: ``"Healthy"`` (use when regime is uncertain or for baseline).

    Returns
    -------
    CounterfactualResult
        See the dataclass docstring.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0; got {horizon}")
    if intervention not in INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention '{intervention}'. Valid: {INTERVENTIONS}"
        )

    # ---- 1. Build intervention-modified params --------------------------------
    p_int = _apply_intervention(params, intervention)

    # ---- 2. Instantiate regime object ----------------------------------------
    regime_obj = get_regime(regime, p_int)

    # ---- 3. Build initial state vector ---------------------------------------
    X0  = max(float(current_state.get("X",  p_int.get("X0",  0.1))), 0.0)
    S0  = max(float(current_state.get("S",  p_int.get("S0", 10.0))), 0.0)
    CL0 = max(float(current_state.get("DO", p_int.get("CL0", 0.008))), 0.0)

    y0 = [X0, S0, CL0]
    if isinstance(regime_obj, ContaminationRegime):
        Xc0 = max(float(current_state.get("X_c", p_int.get("X_cont0", 0.0))), 0.0)
        y0.append(Xc0)
    y0 = np.array(y0, dtype=float)

    # ---- 4. Build time grid --------------------------------------------------
    t_end  = current_t + horizon
    n_pts  = max(int(np.ceil(_HORIZON_EVAL_DENSITY * horizon)), 10)
    t_eval = np.linspace(current_t, t_end, n_pts)

    # ---- 5. Integrate ODE ----------------------------------------------------
    rhs = _build_rollout_rhs(regime_obj, p_int)
    max_step = horizon / 20.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(
            fun=rhs,
            t_span=(current_t, t_end),
            y0=y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-5,
            atol=1e-8,
            max_step=max_step,
        )

    if not sol.success:
        # Return a degraded result rather than raising — keeps the demo live
        warnings.warn(
            f"ODE solver failed for '{intervention}': {sol.message}. "
            "Returning NaN result.",
            RuntimeWarning,
        )
        nan_df = pd.DataFrame({"timestamp": t_eval})
        return CounterfactualResult(
            intervention=intervention,
            ending_R=float("nan"),
            trajectory=nan_df,
            params_applied=p_int,
            horizon=horizon,
            pnr_crossed=False,
        )

    t_arr  = sol.t
    X_arr  = np.clip(sol.y[0], 0.0, None)
    S_arr  = np.clip(sol.y[1], 0.0, None)
    CL_arr = np.clip(sol.y[2], 0.0, None)
    Xc_arr = np.clip(sol.y[3], 0.0, None) if sol.y.shape[0] > 3 else None

    # ---- 6. Compute stress and recoverability --------------------------------
    lambda_t = compute_stress(CL_arr)
    R_arr    = compute_recoverability(t_arr - t_arr[0], lambda_t)  # reset t to 0 for accumulation
    ending_R = float(R_arr[-1])

    # Check PNR crossing
    R_pnr       = float(STRESS_PARAMS["R_pnr"])
    pnr_crossed = bool(np.any(R_arr < R_pnr))

    # ---- 7. Compute derived signals for trajectory DataFrame -----------------
    OD600_FACTOR = 2.5
    _RQ_HEALTHY  = 1.0

    mu_max = p_int.get("mu_max", DEFAULT_PARAMS["mu_max"])
    K_s    = p_int.get("K_s",    DEFAULT_PARAMS["K_s"])
    K_O    = p_int.get("K_O",    DEFAULT_PARAMS["K_O"])
    Y_xo   = p_int.get("Y_xo",   DEFAULT_PARAMS["Y_xo"])
    m_o    = p_int.get("m_o",    DEFAULT_PARAMS["m_o"])

    OUR_arr = np.zeros(len(t_arr))
    RQ_arr  = np.full(len(t_arr), _RQ_HEALTHY)
    CER_arr = np.zeros(len(t_arr))

    for i in range(len(t_arr)):
        state_i = (X_arr[i], S_arr[i], CL_arr[i]) if Xc_arr is None \
                  else (X_arr[i], S_arr[i], CL_arr[i], Xc_arr[i])
        p_eff = regime_obj.mutate(t_arr[i], state_i, p_int)

        mu_i = (
            p_eff["mu_max"]
            * (max(S_arr[i], 0.0) / (p_eff["K_s"] + max(S_arr[i], 0.0) + 1e-12))
            * (max(CL_arr[i], 0.0) / (p_eff["K_O"] + max(CL_arr[i], 0.0) + 1e-12))
        )
        our_i = ((mu_i / p_eff["Y_xo"]) + p_eff["m_o"]) * max(X_arr[i], 0.0)

        if Xc_arr is not None and isinstance(regime_obj, ContaminationRegime):
            Xc_i  = max(Xc_arr[i], 0.0)
            mu_c  = (
                regime_obj.mu_max_cont
                * (max(S_arr[i], 0.0) / (regime_obj.K_s_cont + max(S_arr[i], 0.0) + 1e-12))
                * (max(CL_arr[i], 0.0) / (p_eff["K_O"] + max(CL_arr[i], 0.0) + 1e-12))
            )
            our_i += (mu_c / regime_obj.Y_xo_cont + p_eff["m_o"]) * Xc_i

        OUR_arr[i] = our_i
        CER_arr[i] = our_i * _RQ_HEALTHY * (44.0 / 32.0)

    OD600_arr = X_arr * OD600_FACTOR
    if Xc_arr is not None:
        OD600_arr = (X_arr + Xc_arr) * OD600_FACTOR

    rpm_val      = float(p_int.get("RPM",      DEFAULT_PARAMS["RPM"]))
    airflow_val  = float(p_int.get("airflow",  DEFAULT_PARAMS["airflow"]))
    pressure_val = float(p_int.get("pressure", DEFAULT_PARAMS["pressure"]))

    trajectory = pd.DataFrame({
        "timestamp":  t_arr,
        "DO":         CL_arr,
        "RQ":         RQ_arr,
        "OUR":        OUR_arr,
        "CER":        CER_arr,
        "pressure":   np.full(len(t_arr), pressure_val),
        "RPM":        np.full(len(t_arr), rpm_val),
        "airflow":    np.full(len(t_arr), airflow_val),
        "OD600":      OD600_arr,
        "X":          X_arr,
        "S":          S_arr,
        "lambda_t":   lambda_t,
        "R":          R_arr,
        "intervention": intervention,
    })

    return CounterfactualResult(
        intervention=intervention,
        ending_R=ending_R,
        trajectory=trajectory,
        params_applied=p_int,
        horizon=horizon,
        pnr_crossed=pnr_crossed,
    )


# ---------------------------------------------------------------------------
# Batch ranking
# ---------------------------------------------------------------------------

def rank_interventions(
    current_state: Dict[str, float],
    current_t: float,
    params: Dict[str, Any],
    *,
    horizon: float = 2.0,
    regime: str = "Healthy",
    interventions: Sequence[str] = INTERVENTIONS,
) -> List[CounterfactualResult]:
    """Run all interventions and return them ranked best → worst by ending R.

    Parameters
    ----------
    current_state : dict
        See ``run_counterfactual``.
    current_t : float
        Current batch time (hours).
    params : dict
        Current effective reactor parameters.
    horizon : float
        Rollout horizon in hours (default 2.0 h).  Keep ≤ 4 h for live demos.
    regime : str
        Active regime label for the rollout ODE.
    interventions : sequence of str
        Subset of ``INTERVENTIONS`` to evaluate (default: all four).

    Returns
    -------
    list[CounterfactualResult]
        Sorted by ``ending_R`` descending (best first).
        NaN results are placed last.
    """
    results: List[CounterfactualResult] = []
    for intv in interventions:
        result = run_counterfactual(
            current_state=current_state,
            current_t=current_t,
            params=params,
            intervention=intv,
            horizon=horizon,
            regime=regime,
        )
        results.append(result)

    # Sort descending by ending_R; NaN → -inf (placed last)
    results.sort(
        key=lambda r: r.ending_R if np.isfinite(r.ending_R) else -np.inf,
        reverse=True,
    )
    return results
