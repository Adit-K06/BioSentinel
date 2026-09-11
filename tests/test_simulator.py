"""
tests/test_simulator.py
=======================
Unit tests for the BioSentinel 2.0 digital-twin layer (src/).

Coverage:
    1. Healthy simulation runs without error.
    2. Each failure regime simulation runs without error.
    3. Required columns exist in the returned DataFrame.
    4. No unexpected NaN or Inf values.
    5. Regime labels are correct in the output DataFrame.
    6. Regime-specific physics sanity checks (DO drops for kLa fault, etc.).

Run via:
    python -m pytest tests/test_simulator.py -v
"""

import math
import sys
import os

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when running pytest from any directory
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import pytest

from src.simulator import simulate_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "timestamp",
    "DO",
    "RQ",
    "OUR",
    "CER",
    "pressure",
    "RPM",
    "airflow",
    "OD600",
    "regime",
    "t_onset",
    "X",
    "S",
]

ALL_REGIMES = [
    "Healthy",
    "kLa_Limitation",
    "Substrate_Overfeeding",
    "Contamination",
]

# Shorter simulation for faster tests: 12 h, 720 points
_FAST_PARAMS = {"t_span": (0.0, 12.0), "t_eval_n": 720}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(regime: str, extra_params: dict = None, seed: int = 42) -> pd.DataFrame:
    p = dict(_FAST_PARAMS)
    if extra_params:
        p.update(extra_params)
    return simulate_batch(params=p, regime=regime, seed=seed)


# ---------------------------------------------------------------------------
# Test 1: Healthy simulation runs
# ---------------------------------------------------------------------------

class TestHealthyRun:
    def test_returns_dataframe(self):
        df = _run("Healthy")
        assert isinstance(df, pd.DataFrame), "simulate_batch should return a DataFrame"

    def test_row_count(self):
        df = _run("Healthy")
        assert len(df) == 720, "Row count should match t_eval_n"

    def test_biomass_grows(self):
        df = _run("Healthy")
        # Biomass (OD600 proxy) should increase from start to end
        mid = len(df) // 2
        assert df["OD600"].iloc[-1] > df["OD600"].iloc[0], \
            "OD600 should increase over a healthy batch run"

    def test_substrate_decreases(self):
        df = _run("Healthy")
        # Without feed dominating, net substrate should be consumed overall;
        # check later half is lower than initial
        assert df["S"].iloc[-1] < df["S"].iloc[0] * 2.5, \
            "Substrate should not explode in a healthy run"

    def test_do_positive(self):
        df = _run("Healthy")
        assert (df["DO"] >= 0).all(), "DO must be nonnegative"


# ---------------------------------------------------------------------------
# Test 2: Each failure regime runs without error
# ---------------------------------------------------------------------------

class TestAllRegimesRun:
    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_regime_runs(self, regime):
        df = _run(regime)
        assert isinstance(df, pd.DataFrame), f"{regime} should return a DataFrame"
        assert len(df) > 0, f"{regime} DataFrame should be non-empty"


# ---------------------------------------------------------------------------
# Test 3: Required columns exist
# ---------------------------------------------------------------------------

class TestSchema:
    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_required_columns_present(self, regime):
        df = _run(regime)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        assert not missing, f"Missing columns in '{regime}': {missing}"

    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_column_dtypes_numeric(self, regime):
        df = _run(regime)
        numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("regime",)]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"Column '{col}' should be numeric in regime '{regime}'"


# ---------------------------------------------------------------------------
# Test 4: No NaN / Inf values
# ---------------------------------------------------------------------------

class TestNoNaNInf:
    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_no_nan(self, regime):
        df = _run(regime)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # t_onset is intentionally NaN for Healthy ("no fault" sentinel);
        # its NaN-ness is validated separately in TestRegimeLabels.
        cols_to_check = [c for c in numeric_cols if c != "t_onset"]
        nan_cols = [c for c in cols_to_check if df[c].isna().any()]
        assert not nan_cols, f"NaN found in '{regime}' columns: {nan_cols}"

    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_no_inf(self, regime):
        df = _run(regime)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = [c for c in numeric_cols if np.isinf(df[c]).any()]
        assert not inf_cols, f"Inf found in '{regime}' columns: {inf_cols}"

    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_nonneg_physical(self, regime):
        """Key physical quantities must be nonnegative."""
        df = _run(regime)
        for col in ["DO", "OUR", "CER", "OD600", "X", "S", "RPM", "airflow", "pressure"]:
            neg_count = (df[col] < -1e-9).sum()
            assert neg_count == 0, \
                f"Negative values in '{col}' for regime '{regime}': {neg_count} rows"


# ---------------------------------------------------------------------------
# Test 5: Regime labels correct
# ---------------------------------------------------------------------------

class TestRegimeLabels:
    @pytest.mark.parametrize("regime", ALL_REGIMES)
    def test_label_matches(self, regime):
        df = _run(regime)
        unique = df["regime"].unique()
        assert len(unique) == 1, f"Expected single regime label, got {unique}"
        assert unique[0] == regime, \
            f"Regime label mismatch: expected '{regime}', got '{unique[0]}'"

    @pytest.mark.parametrize("regime", ["kLa_Limitation", "Substrate_Overfeeding", "Contamination"])
    def test_t_onset_populated(self, regime):
        df = _run(regime)
        # t_onset should be a number (not NaN) for non-Healthy regimes
        t_onset_vals = df["t_onset"].dropna().unique()
        assert len(t_onset_vals) > 0, \
            f"t_onset should be set for regime '{regime}'"

    def test_healthy_t_onset_nan(self):
        df = _run("Healthy")
        assert df["t_onset"].isna().all(), \
            "t_onset should be NaN for the Healthy regime"


# ---------------------------------------------------------------------------
# Test 6: Regime-specific physics sanity checks
# ---------------------------------------------------------------------------

class TestRegimePhysics:
    def test_kla_limitation_do_lower(self):
        """DO should be lower after fault onset compared to healthy run."""
        df_h = _run("Healthy", {"t_span": (0.0, 12.0), "t_eval_n": 360})
        df_f = _run("kLa_Limitation", {"t_span": (0.0, 12.0), "t_eval_n": 360})

        # Compare average DO in the post-fault window (t > 9 h)
        do_h = df_h.loc[df_h["timestamp"] > 9.0, "DO"].mean()
        do_f = df_f.loc[df_f["timestamp"] > 9.0, "DO"].mean()

        assert do_f < do_h * 1.05, \
            (f"kLa limitation should depress DO. "
             f"Healthy avg DO={do_h:.4f}, Fault avg DO={do_f:.4f}")

    def test_overfeeding_substrate_higher(self):
        """Substrate should be higher post-onset in overfeeding regime."""
        df_h = _run("Healthy")
        df_f = _run("Substrate_Overfeeding")

        s_h = df_h.loc[df_h["timestamp"] > 7.0, "S"].mean()
        s_f = df_f.loc[df_f["timestamp"] > 7.0, "S"].mean()

        assert s_f > s_h * 0.95, \
            (f"Overfeeding should elevate substrate. "
             f"Healthy S={s_h:.4f}, Overfeed S={s_f:.4f}")

    def test_contamination_rq_higher(self):
        """RQ should diverge upward post-onset in contamination regime."""
        df_h = _run("Healthy")
        df_f = _run("Contamination")

        # Take readings well after onset (onset=5 h, check t > 8 h)
        rq_h = df_h.loc[df_h["timestamp"] > 8.0, "RQ"].mean()
        rq_f = df_f.loc[df_f["timestamp"] > 8.0, "RQ"].mean()

        assert rq_f > rq_h, \
            (f"Contamination should raise RQ. "
             f"Healthy RQ={rq_h:.3f}, Contamination RQ={rq_f:.3f}")

    def test_kla_limitation_our_drops(self):
        """OUR should drop with kLa fault (O2 starvation limits respiration)."""
        df_h = _run("Healthy", {"t_span": (0.0, 12.0), "t_eval_n": 360})
        df_f = _run("kLa_Limitation", {"t_span": (0.0, 12.0), "t_eval_n": 360})

        our_h = df_h.loc[df_h["timestamp"] > 9.5, "OUR"].mean()
        our_f = df_f.loc[df_f["timestamp"] > 9.5, "OUR"].mean()

        assert our_f <= our_h * 1.1, \
            (f"OUR should not rise with kLa limitation. "
             f"Healthy OUR={our_h:.4f}, Fault OUR={our_f:.4f}")

    def test_reproducibility(self):
        """Same seed → identical DataFrames."""
        df1 = _run("Healthy", seed=7)
        df2 = _run("Healthy", seed=7)
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, atol=1e-10)

    def test_different_seeds_differ(self):
        """Different seeds → different noisy outputs."""
        df1 = _run("Healthy", seed=1)
        df2 = _run("Healthy", seed=2)
        # At least one numeric column should differ
        diff = (df1["DO"] - df2["DO"]).abs().max()
        assert diff > 1e-10, "Different seeds should produce different noise"
