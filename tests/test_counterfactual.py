"""
tests/test_counterfactual.py
============================
Unit tests for the BioSentinel 2.0 Counterfactual Rollout Engine.

Coverage
--------
1.  no_action is a valid intervention (returns CounterfactualResult).
2.  All four interventions return trajectories (DataFrames with ≥ 1 row).
3.  Ending recovery values are finite for all interventions.
4.  Trajectory DataFrames contain the required columns.
5.  DO and X in trajectories are non-negative.
6.  R values in trajectory are in (0, 1].
7.  lambda_t in trajectory is non-negative.
8.  Ranking is deterministic — same state+params → same ranked order.
9.  Ranking is sorted best → worst (no element violates order).
10. +150_RPM raises kLa and RPM versus no_action.
11. -30%_feed reduces F_s versus no_action.
12. combined intervention applies both +RPM and -feed.
13. horizon parameter is respected (trajectory ends at current_t + horizon).
14. run_counterfactual raises on unknown intervention string.
15. rank_interventions returns all four interventions.
16. Healthy baseline run: no_action ending_R close to 1.0 (low stress).
17. kLa-limited state: +150_RPM improves ending_R vs no_action.
18. rank_interventions works with a Contamination regime state.
19. Short horizon (0.5 h) returns quickly (performance cap test).
20. run_counterfactual raises ValueError for non-positive horizon.

Run via:
    python -m pytest tests/test_counterfactual.py -v
"""

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_PARAMS, STRESS_PARAMS
from src.counterfactual import (
    INTERVENTIONS,
    CounterfactualResult,
    rank_interventions,
    run_counterfactual,
)
from src.simulator import simulate_batch

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_BASE_PARAMS = {k: v for k, v in DEFAULT_PARAMS.items()}

def _healthy_state(t: float = 8.0) -> dict:
    """Snapshot a healthy batch at time t and return a current_state dict."""
    df = simulate_batch(
        params={"t_span": (0.0, 24.0), "t_eval_n": 720},
        regime="Healthy",
        seed=42,
    )
    row = df[df["timestamp"] <= t].iloc[-1]
    return {"X": float(row["X"]), "S": float(row["S"]), "DO": float(row["DO"])}


def _kla_state(t: float = 12.0) -> dict:
    """Snapshot a kLa-limited batch (post-onset, DO sag visible)."""
    df = simulate_batch(
        params={"t_span": (0.0, 24.0), "t_eval_n": 720, "t_onset": 6.0, "severity": 0.75},
        regime="kLa_Limitation",
        seed=7,
    )
    row = df[df["timestamp"] <= t].iloc[-1]
    return {"X": float(row["X"]), "S": float(row["S"]), "DO": float(row["DO"])}


def _cont_state(t: float = 10.0) -> dict:
    """Snapshot a Contamination batch (post-onset)."""
    df = simulate_batch(
        params={"t_span": (0.0, 24.0), "t_eval_n": 720, "t_onset": 5.0, "severity": 0.4},
        regime="Contamination",
        seed=3,
    )
    row = df[df["timestamp"] <= t].iloc[-1]
    return {"X": float(row["X"]), "S": float(row["S"]), "DO": float(row["DO"])}


_CURRENT_T = 8.0
_HORIZON   = 1.0   # short, so tests run fast


# ===========================================================================
# Basic validity tests
# ===========================================================================

class TestNoAction:
    def test_no_action_returns_result(self):
        """no_action is a valid intervention."""
        state = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=_HORIZON
        )
        assert isinstance(result, CounterfactualResult)
        assert result.intervention == "no_action"

    def test_no_action_ending_r_finite(self):
        state = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=_HORIZON
        )
        assert np.isfinite(result.ending_R), "no_action ending_R must be finite"

    def test_no_action_r_in_range(self):
        state = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=_HORIZON
        )
        assert 0.0 < result.ending_R <= 1.0


class TestAllInterventionsReturnTrajectories:
    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_trajectory_is_dataframe(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert isinstance(result.trajectory, pd.DataFrame), \
            f"Trajectory for '{intv}' should be a DataFrame"

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_trajectory_has_rows(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert len(result.trajectory) >= 1, \
            f"Trajectory for '{intv}' should have ≥ 1 row"

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_ending_r_finite(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert np.isfinite(result.ending_R), \
            f"ending_R for '{intv}' must be finite"


class TestTrajectorySchema:
    required_cols = [
        "timestamp", "DO", "RQ", "OUR", "CER", "pressure",
        "RPM", "airflow", "OD600", "X", "S", "lambda_t", "R", "intervention"
    ]

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_required_columns_present(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        traj = result.trajectory
        for col in self.required_cols:
            assert col in traj.columns, \
                f"Column '{col}' missing from '{intv}' trajectory"

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_do_nonneg(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert (result.trajectory["DO"] >= 0).all()

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_x_nonneg(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert (result.trajectory["X"] >= 0).all()

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_r_in_valid_range(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        R = result.trajectory["R"]
        assert (R > 0).all() and (R <= 1.0).all(), \
            f"R must be in (0, 1]; got min={R.min():.4f}, max={R.max():.4f}"

    @pytest.mark.parametrize("intv", INTERVENTIONS)
    def test_lambda_nonneg(self, intv):
        state  = _healthy_state()
        result = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, intv, horizon=_HORIZON
        )
        assert (result.trajectory["lambda_t"] >= 0).all()


# ===========================================================================
# Ranking determinism
# ===========================================================================

class TestRankingDeterminism:
    def test_rank_deterministic_same_call(self):
        """Calling rank_interventions twice with same inputs gives same order."""
        state = _healthy_state()
        r1 = rank_interventions(state, _CURRENT_T, _BASE_PARAMS, horizon=_HORIZON)
        r2 = rank_interventions(state, _CURRENT_T, _BASE_PARAMS, horizon=_HORIZON)
        assert [r.intervention for r in r1] == [r.intervention for r in r2], \
            "Ranking must be deterministic for fixed inputs"

    def test_rank_sorted_descending(self):
        """Results are sorted best → worst (ending_R descending)."""
        state   = _healthy_state()
        results = rank_interventions(state, _CURRENT_T, _BASE_PARAMS, horizon=_HORIZON)
        r_vals  = [r.ending_R for r in results if np.isfinite(r.ending_R)]
        assert r_vals == sorted(r_vals, reverse=True), \
            "rank_interventions should return results sorted by ending_R desc"

    def test_rank_returns_all_interventions(self):
        state   = _healthy_state()
        results = rank_interventions(state, _CURRENT_T, _BASE_PARAMS, horizon=_HORIZON)
        returned_intvs = {r.intervention for r in results}
        assert returned_intvs == set(INTERVENTIONS), \
            "rank_interventions should return one result per intervention"


# ===========================================================================
# Intervention parameter correctness
# ===========================================================================

class TestInterventionParamChanges:
    def test_rpm_intervention_raises_kla(self):
        """+150_RPM params_applied has higher kLa than no_action."""
        state = _healthy_state()
        r_no   = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action",  horizon=_HORIZON)
        r_rpm  = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "+150_RPM",   horizon=_HORIZON)
        assert r_rpm.params_applied["kLa"] > r_no.params_applied["kLa"], \
            "+150_RPM should increase kLa"

    def test_rpm_intervention_raises_rpm(self):
        state = _healthy_state()
        r_no  = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=_HORIZON)
        r_rpm = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "+150_RPM",  horizon=_HORIZON)
        assert r_rpm.params_applied["RPM"] == r_no.params_applied["RPM"] + 150.0

    def test_feed_intervention_reduces_fs(self):
        """-30%_feed reduces F_s to 0.70 × baseline."""
        state = _healthy_state()
        r_no   = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action",  horizon=_HORIZON)
        r_feed = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "-30%_feed",  horizon=_HORIZON)
        assert abs(r_feed.params_applied["F_s"] - 0.70 * r_no.params_applied["F_s"]) < 1e-9, \
            "-30%_feed should reduce F_s to 0.70 × baseline"

    def test_combined_applies_both(self):
        """combined intervention applies both +RPM and -feed."""
        state = _healthy_state()
        r_no  = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=_HORIZON)
        r_comb = run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "combined",  horizon=_HORIZON)
        assert r_comb.params_applied["RPM"]  > r_no.params_applied["RPM"], \
            "combined: RPM should increase"
        assert r_comb.params_applied["kLa"]  > r_no.params_applied["kLa"], \
            "combined: kLa should increase"
        assert r_comb.params_applied["F_s"]  < r_no.params_applied["F_s"], \
            "combined: F_s should decrease"


# ===========================================================================
# Horizon parameter
# ===========================================================================

class TestHorizon:
    def test_trajectory_ends_at_current_plus_horizon(self):
        """Trajectory timestamps span approximately [current_t, current_t+horizon]."""
        state   = _healthy_state()
        horizon = 1.5
        result  = run_counterfactual(
            state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=horizon
        )
        t_end   = result.trajectory["timestamp"].iloc[-1]
        assert abs(t_end - (_CURRENT_T + horizon)) < 0.01, \
            f"Trajectory should end at {_CURRENT_T + horizon:.2f}, got {t_end:.4f}"

    def test_zero_horizon_raises(self):
        """Non-positive horizon raises ValueError."""
        state = _healthy_state()
        with pytest.raises(ValueError, match="horizon"):
            run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=0.0)

    def test_negative_horizon_raises(self):
        state = _healthy_state()
        with pytest.raises(ValueError, match="horizon"):
            run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "no_action", horizon=-1.0)


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:
    def test_unknown_intervention_raises(self):
        state = _healthy_state()
        with pytest.raises(ValueError, match="Unknown intervention"):
            run_counterfactual(state, _CURRENT_T, _BASE_PARAMS, "turbo_boost")

    def test_unknown_intervention_in_rank_raises(self):
        state = _healthy_state()
        with pytest.raises(ValueError):
            rank_interventions(
                state, _CURRENT_T, _BASE_PARAMS,
                interventions=["no_action", "invalid_op"],
                horizon=_HORIZON,
            )


# ===========================================================================
# Physics plausibility (regime-specific)
# ===========================================================================

class TestPhysicsPlausibility:
    def test_healthy_no_action_r_near_one(self):
        """Healthy state: no_action ending_R should be high (low stress)."""
        state  = _healthy_state(t=4.0)  # early in healthy run — high DO
        result = run_counterfactual(
            state, 4.0, _BASE_PARAMS, "no_action", horizon=1.0
        )
        assert result.ending_R > 0.70, \
            f"Healthy no-action R should be > 0.70; got {result.ending_R:.4f}"

    def test_kla_fault_rpm_improves_do(self):
        """After kLa fault: +150_RPM trajectory has higher mean DO than no_action."""
        state   = _kla_state(t=12.0)
        params  = {**_BASE_PARAMS, "kLa": 60.0}   # already degraded kLa
        t_now   = 12.0
        r_no    = run_counterfactual(state, t_now, params, "no_action",  horizon=_HORIZON)
        r_rpm   = run_counterfactual(state, t_now, params, "+150_RPM",   horizon=_HORIZON)
        mean_do_no  = r_no.trajectory["DO"].mean()
        mean_do_rpm = r_rpm.trajectory["DO"].mean()
        assert mean_do_rpm >= mean_do_no, \
            "+150_RPM should not decrease mean DO vs no_action under kLa fault"

    def test_contamination_regime_runs(self):
        """Counterfactual engine works for Contamination regime state."""
        state = _cont_state(t=10.0)
        for intv in INTERVENTIONS:
            result = run_counterfactual(
                state, 10.0, _BASE_PARAMS, intv,
                horizon=_HORIZON, regime="Contamination"
            )
            assert isinstance(result, CounterfactualResult)
            assert np.isfinite(result.ending_R), \
                f"Contamination/{intv}: ending_R should be finite"


# ===========================================================================
# Performance cap (live demo requirement)
# ===========================================================================

class TestPerformanceCap:
    def test_short_horizon_returns_quickly(self):
        """rank_interventions with 0.5 h horizon must complete in < 10 s."""
        state = _healthy_state()
        t0    = time.perf_counter()
        rank_interventions(state, _CURRENT_T, _BASE_PARAMS, horizon=0.5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, \
            f"rank_interventions (0.5 h horizon) took {elapsed:.2f} s — too slow for live demo"
