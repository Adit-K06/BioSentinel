"""
src/features.py
===============
BioSentinel 2.0 — Feature Engineering for the regime classifier.

Converts raw ``simulate_batch()`` DataFrames into a rich feature matrix
suitable for LightGBM multiclass classification.

Feature groups
--------------
Raw sensor signals
    DO, RQ, OUR, CER, pressure, RPM, airflow, OD600, X, S

Physics-informed derived signals
    dDO_dt          — d(DO)/dt via finite difference (g/L per h)
    dS_dt           — d(S)/dt via finite difference (substrate drift)
    dOUR_dt         — d(OUR)/dt (metabolic acceleration)
    dRQ_dt          — d(RQ)/dt (respiration shift rate)
    RPM_shear       — RPM² / 1e4 (agitation proxy)
    DO_deficit      — CL_star - DO (oxygen transfer driving force)
    kLa_eff         — inferred mass-transfer from DO balance:
                      (dDO_dt + OUR) / (DO_deficit + eps)
                      drops sharply under kLa limitation
    RQ_deviation    — |RQ - 1.0| (contamination/overflow flag)
    OUR_per_X       — OUR / (X + eps) (specific oxygen uptake rate,
                      elevated with contaminant biomass)
    elapsed_frac    — elapsed_t / 24h (batch phase 0–1)
    CUSUM_DO        — cumulative sum of DO residual vs healthy baseline

Rolling statistics  (6 windows: 30 s, 2 min, 5 min, 10 min, 30 min, 60 min)
    _mean, _std, _slope  for each feature in _ROLLING_BASE
    Larger windows (30 min, 60 min) capture the slow drifts that define
    each fault regime.

No Streamlit / UI / API / database imports.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "FEATURE_COLS",
    "RAW_SIGNALS",
    "build_features",
    "compute_healthy_baseline",
    "get_feature_cols",
    "WINDOW_SECONDS",
]

# ---------------------------------------------------------------------------
# Rolling window sizes (seconds) — adapted to simulator timestep at call time
# ---------------------------------------------------------------------------
# 30 s, 2 min, 5 min, 10 min, 30 min, 60 min
# Larger windows capture regime-specific slow drifts.
WINDOW_SECONDS = [30, 120, 300, 600, 1800, 3600]

# Nominal DO saturation concentration (25°C, air, 1 atm) — for DO_deficit
_CL_STAR_DEFAULT = 0.0084  # g/L

# ---------------------------------------------------------------------------
# Signal lists
# ---------------------------------------------------------------------------

RAW_SIGNALS: List[str] = [
    "DO", "RQ", "OUR", "CER", "pressure", "RPM", "airflow", "OD600", "X", "S",
]

# Base signals fed into rolling-statistics block
_ROLLING_BASE: List[str] = [
    "DO", "RQ", "OUR", "CER",
    "dDO_dt", "dS_dt", "kLa_eff",
    "RQ_deviation", "OUR_per_X",
]


# ---------------------------------------------------------------------------
# Healthy-baseline statistics (for CUSUM_DO)
# ---------------------------------------------------------------------------

def compute_healthy_baseline(
    healthy_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Compute the mean and std of DO across all healthy-regime rows.

    Parameters
    ----------
    healthy_df : pd.DataFrame
        One or more concatenated healthy-regime ``simulate_batch()`` outputs.

    Returns
    -------
    (do_mean, do_std) : (float, float)
    """
    do_mean = float(healthy_df["DO"].mean())
    do_std  = max(float(healthy_df["DO"].std()), 1e-9)
    return do_mean, do_std


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window_rows(window_seconds: int, dt_hours: float) -> int:
    """Convert window length (seconds) to number of DataFrame rows."""
    dt_seconds = dt_hours * 3600.0
    n = int(np.ceil(window_seconds / dt_seconds))
    return max(n, 2)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Per-window OLS slope (vectorised)."""
    def _slope(vals: np.ndarray) -> float:
        n = len(vals)
        if n < 2:
            return 0.0
        x  = np.arange(n, dtype=float)
        xm = x - x.mean()
        ym = vals - vals.mean()
        denom = (xm * xm).sum()
        return float((xm * ym).sum() / denom) if denom > 1e-15 else 0.0

    return series.rolling(window, min_periods=2).apply(_slope, raw=True)


def _gradient_safe(arr: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference gradient, robust to single-element arrays."""
    if len(arr) < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, dt)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    do_mean: float = 0.006,
    do_std:  float = 0.001,
    *,
    cl_star: float = _CL_STAR_DEFAULT,
    window_seconds: Sequence[int] = WINDOW_SECONDS,
    drop_nan_rows: bool = True,
) -> pd.DataFrame:
    """Build the full feature matrix from a single simulate_batch() DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``src.simulator.simulate_batch()``.
    do_mean, do_std : float
        Healthy-baseline DO statistics for CUSUM_DO.
    cl_star : float
        DO saturation concentration for DO_deficit and kLa_eff.
    window_seconds : sequence of int
        Rolling window sizes in seconds.
    drop_nan_rows : bool
        If True, drop rows with any NaN (rolling warm-up artefacts).
        If False, fill NaN with 0 (preserves row count).

    Returns
    -------
    pd.DataFrame
        Feature matrix.  Index reset.  ``regime``, ``t_onset`` metadata kept.
    """
    df = df.copy().reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("DataFrame must have ≥ 2 rows.")

    # --- Infer timestep (hours) ---
    dt_h = float(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
    if dt_h <= 0:
        dt_h = 1.0 / 60.0

    out = pd.DataFrame(index=df.index)

    # ----------------------------------------------------------------
    # 1. Raw signals
    # ----------------------------------------------------------------
    for col in RAW_SIGNALS:
        if col in df.columns:
            out[col] = df[col].values.copy()

    # ----------------------------------------------------------------
    # 2. Physics-informed derived signals
    # ----------------------------------------------------------------
    do_arr  = df["DO"].values.astype(float)
    s_arr   = df["S"].values.astype(float)
    our_arr = df["OUR"].values.astype(float)
    rq_arr  = df["RQ"].values.astype(float)
    x_arr   = df["X"].values.astype(float)
    rpm_arr = df["RPM"].values.astype(float)

    # Time derivatives
    ddo  = _gradient_safe(do_arr,  dt_h)
    ds   = _gradient_safe(s_arr,   dt_h)
    dour = _gradient_safe(our_arr, dt_h)
    drq  = _gradient_safe(rq_arr,  dt_h)

    out["dDO_dt"]   = ddo
    out["dS_dt"]    = ds
    out["dOUR_dt"]  = dour
    out["dRQ_dt"]   = drq

    # RPM-derived agitation shear proxy
    out["RPM_shear"] = rpm_arr ** 2 / 1e4

    # DO deficit (oxygen transfer driving force)
    do_deficit = cl_star - do_arr
    out["DO_deficit"] = do_deficit

    # Inferred kLa from DO mass balance:
    #   dCL/dt = kLa*(CL* - CL) - OUR  →  kLa_eff = (dCL/dt + OUR) / (CL* - CL)
    # Drops sharply when kLa is fouled.
    denom_kla = np.abs(do_deficit) + 1e-7
    out["kLa_eff"] = np.clip((ddo + our_arr) / denom_kla, -200.0, 2000.0)

    # RQ deviation from unity (contamination / overflow signal)
    out["RQ_deviation"] = np.abs(rq_arr - 1.0)

    # Specific OUR (OUR per unit biomass — elevated with contaminant)
    out["OUR_per_X"] = our_arr / (x_arr + 1e-7)

    # Elapsed batch time and normalised phase [0, 1]
    t_arr = df["timestamp"].values.astype(float)
    out["elapsed_t"]    = t_arr
    t_max = max(float(t_arr.max()), 1.0)
    out["elapsed_frac"] = t_arr / t_max

    # CUSUM of standardised DO residual against healthy baseline
    z = (do_arr - do_mean) / do_std
    cusum = np.zeros(len(z))
    for i in range(1, len(z)):
        cusum[i] = max(0.0, cusum[i - 1] + z[i])
    out["CUSUM_DO"] = cusum

    # ----------------------------------------------------------------
    # 3. Rolling statistics (6 windows × mean/std/slope × 9 base signals)
    # ----------------------------------------------------------------
    rolling_cols: dict = {}
    for w_sec in window_seconds:
        w     = _window_rows(w_sec, dt_h)
        label = f"w{w_sec}s"
        for sig in _ROLLING_BASE:
            if sig not in out.columns:
                continue
            series = pd.Series(out[sig].values.astype(float))
            rolling_cols[f"{sig}_{label}_mean"]  = series.rolling(w, min_periods=1).mean().values
            rolling_cols[f"{sig}_{label}_std"]   = series.rolling(w, min_periods=1).std().fillna(0).values
            rolling_cols[f"{sig}_{label}_slope"] = _rolling_slope(series, w).values

    # Batch-assign all rolling columns at once (avoids DataFrame fragmentation)
    rolling_df = pd.DataFrame(rolling_cols, index=out.index)
    out = pd.concat([out, rolling_df], axis=1)

    # ----------------------------------------------------------------
    # 4. Metadata (not used as model inputs)
    # ----------------------------------------------------------------
    out["regime"]       = df["regime"].values
    out["t_onset"]      = df["t_onset"].values
    out["elapsed_t_raw"] = t_arr

    # ----------------------------------------------------------------
    # 5. NaN / Inf cleanup
    # ----------------------------------------------------------------
    slope_cols = [c for c in out.columns if c.endswith("_slope")]
    std_cols   = [c for c in out.columns if c.endswith("_std")]
    out[slope_cols] = out[slope_cols].fillna(0.0)
    out[std_cols]   = out[std_cols].fillna(0.0)

    # Replace any remaining inf with large finite value, then handle NaN
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], [1e6, -1e6])

    if drop_nan_rows:
        out = out.dropna().reset_index(drop=True)
    else:
        out = out.fillna(0.0).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Feature column selector
# ---------------------------------------------------------------------------

def get_feature_cols(df_features: pd.DataFrame) -> List[str]:
    """Return numeric feature column names (excludes metadata)."""
    meta = {"regime", "t_onset", "elapsed_t_raw", "label", "batch_id"}
    return [
        c for c in df_features.columns
        if c not in meta and pd.api.types.is_numeric_dtype(df_features[c])
    ]


# Lazily populated placeholder — avoids circular imports at module load
FEATURE_COLS: List[str] = []
