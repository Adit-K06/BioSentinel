"""
tests/test_features.py
======================
Unit tests for the BioSentinel 2.0 ML feature engineering and model loading.

Coverage
--------
Feature generation (src/features.py):
    1.  build_features returns a DataFrame.
    2.  All expected feature groups are present (raw, derived, rolling, context).
    3.  No NaN values remain after build_features (drop_nan_rows=False path).
    4.  CUSUM_DO is non-negative.
    5.  dDO_dt has the correct shape and type.
    6.  RPM_shear = RPM² / 1e4.
    7.  Rolling window columns exist for all three window sizes.
    8.  build_features works on each of the four regime outputs.
    9.  compute_healthy_baseline returns sensible (mean, std).
    10. Row count is preserved with drop_nan_rows=False.
    11. Row count may decrease with drop_nan_rows=True.

Model artefact tests (src/predict.py):
    12. load_model raises FileNotFoundError when model file is absent.
    13. After training, model file and feature_cols.json exist.
    14. Loaded model can predict on a new batch.
    15. predict_batch returns required keys.
    16. predict_batch predicted_class is one of the four regime labels.
    17. predict_batch probabilities sum to ≈ 1.0.
    18. predict_row works for a single dict of sensor values.

Run via:
    python -m pytest tests/test_features.py -v
"""

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.simulator import simulate_batch
from src.features import (
    build_features,
    compute_healthy_baseline,
    get_feature_cols,
    RAW_SIGNALS,
    WINDOW_SECONDS,
)
from src.predict import CLASSES, load_model, predict_batch, predict_row

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR        = _ROOT / "models"
MODEL_PATH        = MODELS_DIR / "regime_classifier.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SIM_PARAMS = {"t_span": (0.0, 12.0), "t_eval_n": 720}

def _sim(regime="Healthy", seed=0):
    return simulate_batch(params=_SIM_PARAMS, regime=regime, seed=seed)

def _feat(regime="Healthy", seed=0, **kw):
    df = _sim(regime, seed)
    return build_features(df, **kw)


# ===========================================================================
# Feature generation tests
# ===========================================================================

class TestBuildFeaturesBasic:
    def test_returns_dataframe(self):
        out = _feat()
        assert isinstance(out, pd.DataFrame), "build_features should return DataFrame"

    def test_raw_signals_present(self):
        out = _feat()
        for col in RAW_SIGNALS:
            assert col in out.columns, f"Raw signal '{col}' missing"

    def test_derived_signals_present(self):
        out = _feat()
        # Core derived signals (original)
        for col in ["dDO_dt", "RPM_shear", "CUSUM_DO", "elapsed_t"]:
            assert col in out.columns, f"Derived signal '{col}' missing"
        # Physics-informed signals (new)
        for col in ["dS_dt", "DO_deficit", "kLa_eff", "RQ_deviation",
                    "OUR_per_X", "elapsed_frac"]:
            assert col in out.columns, f"Physics feature '{col}' missing"

    def test_rolling_columns_present(self):
        out = _feat()
        for w_sec in WINDOW_SECONDS:
            label = f"w{w_sec}s"
            # At least one rolling stat column per window
            matching = [c for c in out.columns if label in c]
            assert len(matching) > 0, f"No rolling columns for window '{label}'"

    def test_rolling_stats_all_three_metrics(self):
        """Each window should have _mean, _std, _slope variants."""
        out = _feat()
        for w_sec in WINDOW_SECONDS:
            label = f"w{w_sec}s"
            for stat in ("mean", "std", "slope"):
                matching = [c for c in out.columns if label in c and c.endswith(stat)]
                assert len(matching) > 0, \
                    f"No '{stat}' rolling column for window '{label}'"


class TestNoNaNValues:
    def test_no_nan_drop_false(self):
        """With drop_nan_rows=False, all NaN must be filled."""
        out = _feat(drop_nan_rows=False)
        nan_cols = [c for c in out.columns if out[c].isna().any()]
        assert not nan_cols, f"NaN remaining in columns: {nan_cols}"

    def test_no_nan_drop_true(self):
        """With drop_nan_rows=True, remaining rows also have no NaN."""
        out = _feat(drop_nan_rows=True)
        nan_cols = [c for c in out.select_dtypes(include=[np.number]).columns
                    if out[c].isna().any()]
        assert not nan_cols, f"NaN in numeric columns after drop: {nan_cols}"


class TestDerivedSignalCorrectness:
    def test_cusum_do_nonneg(self):
        out = _feat(drop_nan_rows=False)
        assert (out["CUSUM_DO"] >= 0).all(), "CUSUM_DO must be non-negative"

    def test_rpm_shear_formula(self):
        """RPM_shear = RPM² / 1e4."""
        df  = _sim()
        out = build_features(df, drop_nan_rows=False)
        rpm = df["RPM"].values
        expected = rpm ** 2 / 1e4
        # After feature engineering, RPM may be noisy but proportional
        np.testing.assert_allclose(
            out["RPM_shear"].values, expected[:len(out)],
            rtol=0.01,
            err_msg="RPM_shear should equal RPM² / 1e4"
        )

    def test_ddo_dt_shape(self):
        out = _feat(drop_nan_rows=False)
        assert "dDO_dt" in out.columns
        assert len(out["dDO_dt"]) == len(out)

    def test_elapsed_t_monotonic(self):
        out = _feat(drop_nan_rows=False)
        diffs = np.diff(out["elapsed_t"].values)
        assert np.all(diffs >= 0), "elapsed_t must be non-decreasing"


class TestRowPreservation:
    def test_drop_false_preserves_row_count(self):
        df  = _sim()
        out = build_features(df, drop_nan_rows=False)
        assert len(out) == len(df), \
            f"Row count should be preserved; got {len(out)} vs {len(df)}"

    def test_drop_true_may_reduce_rows(self):
        df  = _sim()
        out_f = build_features(df, drop_nan_rows=False)
        out_t = build_features(df, drop_nan_rows=True)
        assert len(out_t) <= len(out_f), \
            "drop_nan_rows=True should yield ≤ rows than drop_nan_rows=False"


class TestAllRegimes:
    @pytest.mark.parametrize("regime", ["Healthy", "kLa_Limitation",
                                        "Substrate_Overfeeding", "Contamination"])
    def test_feature_build_succeeds(self, regime):
        out = _feat(regime=regime, drop_nan_rows=False)
        assert isinstance(out, pd.DataFrame)
        assert len(out) > 0

    @pytest.mark.parametrize("regime", ["Healthy", "kLa_Limitation",
                                        "Substrate_Overfeeding", "Contamination"])
    def test_feature_col_count_consistent(self, regime):
        """All regimes should produce the same set of feature columns."""
        out_h = _feat(regime="Healthy",     drop_nan_rows=False)
        out_r = _feat(regime=regime,        drop_nan_rows=False)
        cols_h = set(out_h.columns)
        cols_r = set(out_r.columns)
        assert cols_h == cols_r, \
            f"Column mismatch for '{regime}': {cols_r - cols_h} extra, {cols_h - cols_r} missing"


class TestHealthyBaseline:
    def test_compute_healthy_baseline_types(self):
        df = _sim("Healthy")
        mean, std = compute_healthy_baseline(df)
        assert isinstance(mean, float)
        assert isinstance(std,  float)

    def test_healthy_baseline_positive_std(self):
        df = _sim("Healthy")
        _, std = compute_healthy_baseline(df)
        assert std > 0, "Healthy baseline std should be positive"

    def test_healthy_baseline_do_range(self):
        df = _sim("Healthy")
        mean, _ = compute_healthy_baseline(df)
        assert 0.0 < mean < 0.02, \
            f"Healthy DO mean={mean:.5f} should be in physiological range (0–20 mg/L)"


# ===========================================================================
# Model artefact tests
# ===========================================================================

class TestLoadModelErrors:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(
                model_path=tmp_path / "nonexistent.pkl",
                cols_path=tmp_path / "cols.json",
            )


class TestModelArtifactsExist:
    """These tests require the training pipeline to have been run first.
    They are skipped gracefully if the model has not yet been trained.
    """

    def _skip_if_no_model(self):
        if not MODEL_PATH.exists():
            pytest.skip("Model not trained yet — run `python src/train_model.py` first.")

    def test_model_file_exists(self):
        self._skip_if_no_model()
        assert MODEL_PATH.exists()

    def test_feature_cols_file_exists(self):
        self._skip_if_no_model()
        assert FEATURE_COLS_PATH.exists()

    def test_feature_cols_is_nonempty_list(self):
        self._skip_if_no_model()
        with open(FEATURE_COLS_PATH) as f:
            cols = json.load(f)
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_load_model_returns_three_items(self):
        self._skip_if_no_model()
        result = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        assert len(result) == 3, "load_model should return (model, feature_cols, classes)"

    def test_predict_batch_keys(self):
        self._skip_if_no_model()
        model, feature_cols, classes = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        df = _sim("Healthy")
        result = predict_batch(df, model, feature_cols, classes)
        for key in ["predicted_class", "class_index", "probabilities",
                    "row_predictions", "n_rows"]:
            assert key in result, f"Key '{key}' missing from predict_batch output"

    def test_predict_batch_class_is_valid(self):
        self._skip_if_no_model()
        model, feature_cols, classes = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        for regime in CLASSES:
            df  = _sim(regime)
            out = predict_batch(df, model, feature_cols, classes)
            assert out["predicted_class"] in CLASSES, \
                f"Predicted class '{out['predicted_class']}' not in CLASSES"

    def test_predict_batch_probabilities_sum_to_one(self):
        self._skip_if_no_model()
        model, feature_cols, classes = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        df  = _sim("Healthy")
        out = predict_batch(df, model, feature_cols, classes)
        total = sum(out["probabilities"].values())
        assert abs(total - 1.0) < 1e-4, \
            f"Probabilities should sum to 1.0; got {total:.6f}"

    def test_predict_row_returns_class_and_probs(self):
        self._skip_if_no_model()
        model, feature_cols, classes = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        df  = _sim("Healthy")
        row = df.iloc[100].to_dict()
        out = predict_row(row, model, feature_cols, classes)
        assert "predicted_class" in out
        assert "class_index"     in out
        assert "probabilities"   in out
        assert out["predicted_class"] in CLASSES

    def test_predict_row_probs_sum_to_one(self):
        self._skip_if_no_model()
        model, feature_cols, classes = load_model(MODEL_PATH, FEATURE_COLS_PATH)
        df  = _sim("Healthy")
        row = df.iloc[200].to_dict()
        out = predict_row(row, model, feature_cols, classes)
        total = sum(out["probabilities"].values())
        assert abs(total - 1.0) < 1e-4, \
            f"Row probabilities should sum to 1.0; got {total:.6f}"
