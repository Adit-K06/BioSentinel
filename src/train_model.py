"""
src/train_model.py
==================
BioSentinel 2.0 — Offline ML Training Pipeline.

Generates synthetic batches from the digital twin, engineers features,
trains a LightGBM multiclass classifier, and saves the artefact.

Usage (run from repo root):
    python src/train_model.py

Key design decisions that drive accuracy
-----------------------------------------
Pre-onset re-labeling
    Rows that occur BEFORE a fault's t_onset are physically indistinguishable
    from Healthy rows.  Labelling them with the fault class injects ~25–32 %
    label noise and is the single largest contributor to low F1.
    Fix: all pre-onset rows in fault batches are re-labelled as Healthy (0).

Physics-informed features
    kLa_eff (inferred from DO mass balance) directly flags kLa limitation;
    dS_dt flags substrate accumulation; RQ_deviation and OUR_per_X flag
    contamination.  Rolling windows up to 60 min capture slow drifts.

Whole-batch group holdout
    Never split rows — entire batches are held out for validation.

Classes
-------
    0  Healthy
    1  kLa_Limitation
    2  Substrate_Overfeeding
    3  Contamination

No Streamlit / UI / API / database imports.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Ensure src/ is importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.simulator import simulate_batch
from src.features import (
    build_features,
    compute_healthy_baseline,
    get_feature_cols,
    WINDOW_SECONDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS_DIR = _ROOT / "models"
DATA_DIR   = _ROOT / "data" / "synthetic"

CLASSES   = ["Healthy", "kLa_Limitation", "Substrate_Overfeeding", "Contamination"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# Synthetic data generation
# SIM_POINTS = 720  →  one point every 2 min over 24 h.
# Larger windows (30 min = 15 pts, 60 min = 30 pts) are well-supported.
N_BATCHES_PER_CLASS  = 25   # 100 total
N_VALIDATION_BATCHES =  5   # held-out per class
SIM_HOURS   = 24.0
SIM_POINTS  = 720           # one point per 2 minutes

REGIME_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "kLa_Limitation": {
        "t_onset":  (4.0, 12.0),
        "severity": (0.40, 0.85),
    },
    "Substrate_Overfeeding": {
        "t_onset":  (3.0, 10.0),
        "severity": (1.5,  5.0),
    },
    "Contamination": {
        "t_onset":  (3.0,  8.0),
        "severity": (0.20, 0.70),
    },
}

# LightGBM hyperparameters — tuned for this dataset size
LGBM_PARAMS = {
    "objective":         "multiclass",
    "num_class":         len(CLASSES),
    "metric":            "multi_logloss",
    "boosting_type":     "gbdt",
    "n_estimators":      1000,
    "learning_rate":     0.03,
    "max_depth":         8,
    "num_leaves":        127,
    "min_child_samples": 30,
    "subsample":         0.80,
    "subsample_freq":    1,
    "colsample_bytree":  0.80,
    "reg_alpha":         0.05,
    "reg_lambda":        0.5,
    "is_unbalance":      True,   # handles healthy-class dominance after re-labeling
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def _random_params(regime: str, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    if regime == "Healthy":
        return {}
    p: Dict = {}
    for key, (lo, hi) in REGIME_RANGES[regime].items():
        p[key] = float(rng.uniform(lo, hi))
    return p


def generate_batches(verbose: bool = True) -> pd.DataFrame:
    """Generate all synthetic training batches → single concatenated DataFrame."""
    frames: List[pd.DataFrame] = []
    batch_id = 0

    for cls in CLASSES:
        for i in range(N_BATCHES_PER_CLASS):
            seed  = batch_id * 17 + 3
            extra = _random_params(cls, seed=seed + 500)
            params = {
                "t_span":   (0.0, SIM_HOURS),
                "t_eval_n": SIM_POINTS,
                **extra,
            }
            try:
                df = simulate_batch(params=params, regime=cls, seed=seed)
            except Exception as exc:
                warnings.warn(f"Batch {batch_id} ({cls}) failed: {exc}")
                batch_id += 1
                continue

            df["batch_id"] = batch_id
            df["label"]    = LABEL_MAP[cls]
            frames.append(df)
            batch_id += 1

        if verbose:
            print(f"  Generated {N_BATCHES_PER_CLASS} batches for '{cls}'")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Pre-onset re-labeling
# ---------------------------------------------------------------------------

def relabel_pre_onset(raw: pd.DataFrame) -> pd.DataFrame:
    """Re-label pre-onset rows in fault batches as Healthy.

    Before a fault occurs, the bioreactor state is physically identical to a
    healthy batch.  Labelling those rows with the fault class introduces up to
    ~32 % label noise and is the primary cause of low classifier F1.

    Strategy
    --------
    For every fault batch (regime != Healthy):
      - rows where timestamp < t_onset → label = 0 (Healthy)
      - rows where timestamp >= t_onset → keep original fault label

    Parameters
    ----------
    raw : pd.DataFrame
        Output of ``generate_batches()``.

    Returns
    -------
    pd.DataFrame
        Same rows, labels updated in-place.
    """
    out = raw.copy()

    # healthy rows — no change needed
    mask_fault = out["regime"] != "Healthy"

    for bid in out.loc[mask_fault, "batch_id"].unique():
        bmask = out["batch_id"] == bid
        t_onset_vals = out.loc[bmask, "t_onset"].dropna().unique()
        if len(t_onset_vals) == 0:
            continue
        t_onset = float(t_onset_vals[0])
        # Pre-onset → Healthy label
        pre_onset_mask = bmask & (out["timestamp"] < t_onset)
        out.loc[pre_onset_mask, "label"] = LABEL_MAP["Healthy"]

    return out


# ---------------------------------------------------------------------------
# Feature engineering (batch-by-batch to preserve group structure)
# ---------------------------------------------------------------------------

def build_all_features(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Feature-engineer every batch independently."""
    healthy_rows = raw[raw["regime"] == "Healthy"]
    do_mean, do_std = compute_healthy_baseline(healthy_rows)
    if verbose:
        print(f"  Healthy baseline: DO_mean={do_mean:.5f}, DO_std={do_std:.5f}")

    feat_frames: List[pd.DataFrame] = []
    for bid, grp in raw.groupby("batch_id"):
        label_val  = grp["label"].iloc[0]   # original regime label (before re-labeling)
        regime_val = grp["regime"].iloc[0]
        try:
            feat = build_features(
                grp.reset_index(drop=True), do_mean, do_std, drop_nan_rows=False
            )
        except Exception as exc:
            warnings.warn(f"Feature build failed for batch {bid}: {exc}")
            continue

        # Preserve the *per-row* labels (already re-labeled in raw)
        feat["label"]    = grp["label"].values
        feat["batch_id"] = bid
        feat_frames.append(feat)

    result = pd.concat(feat_frames, ignore_index=True)
    if verbose:
        print(f"  Feature matrix shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Train / validation split (whole-batch groups, never split rows)
# ---------------------------------------------------------------------------

def split_batches(
    feat: pd.DataFrame,
    n_val_per_class: int = N_VALIDATION_BATCHES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out entire batches by original regime label (before re-labeling)."""
    rng = np.random.default_rng(random_state)
    val_batch_ids: set = set()

    # Split by original regime, not re-labeled class
    for regime in CLASSES:
        ids = sorted(
            feat.loc[feat["regime"] == regime, "batch_id"].unique()
        )
        n = min(n_val_per_class, max(len(ids) - 1, 1))
        chosen = rng.choice(ids, size=n, replace=False)
        val_batch_ids.update(chosen.tolist())

    val_mask = feat["batch_id"].isin(val_batch_ids)
    return feat[~val_mask].copy(), feat[val_mask].copy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(verbose: bool = True) -> Tuple[lgb.LGBMClassifier, List[str], dict]:
    """Run the full training pipeline."""
    t0 = time.time()

    if verbose:
        print("\n[1/6] Generating synthetic batches ...")
    raw = generate_batches(verbose=verbose)

    if verbose:
        print("\n[2/6] Re-labelling pre-onset rows as Healthy ...")
    raw_relabeled = relabel_pre_onset(raw)

    # Report label distribution after re-labeling
    if verbose:
        for cls_idx, cls in enumerate(CLASSES):
            n = (raw_relabeled["label"] == cls_idx).sum()
            print(f"  {cls:<25}: {n:>7,} rows")

    if verbose:
        print("\n[3/6] Building features ...")
    feat = build_all_features(raw_relabeled, verbose=verbose)

    if verbose:
        print("\n[4/6] Splitting into train / validation (whole-batch holdout) ...")
    train_df, val_df = split_batches(feat)

    meta_cols = {"label", "batch_id", "regime", "t_onset", "elapsed_t_raw"}
    feature_cols = [
        c for c in get_feature_cols(feat) if c not in meta_cols
    ]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["label"].values.astype(int)

    if verbose:
        print(f"  Train rows: {len(X_train):,}   Val rows: {len(X_val):,}")
        print(f"  Feature columns: {len(feature_cols)}")

    if verbose:
        print("\n[5/6] Training LightGBM classifier (early stopping at 50) ...")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

    if verbose:
        best_it = getattr(model, "best_iteration_", LGBM_PARAMS["n_estimators"])
        print(f"  Best iteration: {best_it}")

    if verbose:
        print("\n[6/6] Evaluating on held-out validation batches ...")

    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    acc      = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    report   = classification_report(y_val, y_pred, target_names=CLASSES, digits=4)
    cm       = confusion_matrix(y_val, y_pred)

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Accuracy       : {acc*100:.2f}%")
        print(f"  Macro F1       : {macro_f1*100:.2f}%")
        print(f"\n  Classification Report:\n{report}")
        print(f"  Confusion Matrix (rows=true, cols=pred):")
        cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
        print(cm_df.to_string())
        print(f"{'='*62}")

        spec_lo, spec_hi = 89.0, 91.0
        if macro_f1 * 100 < spec_lo:
            print(
                f"\n  NOTE: Actual macro F1={macro_f1*100:.2f}% is below the "
                f"spec's indicative range ({spec_lo}-{spec_hi}%). "
                "This is the real, unmodified result."
            )
        elif macro_f1 * 100 > spec_hi:
            print(
                f"\n  Macro F1={macro_f1*100:.2f}% exceeds the spec range "
                f"({spec_lo}-{spec_hi}%) — great result."
            )
        else:
            print(f"\n  Macro F1 is within the target {spec_lo}-{spec_hi}% range.")

        elapsed = time.time() - t0
        print(f"\n  Total training time: {elapsed:.1f} s")

    best_iter = getattr(model, "best_iteration_", LGBM_PARAMS["n_estimators"])
    metrics = {
        "accuracy":          round(float(acc),      6),
        "macro_f1":          round(float(macro_f1), 6),
        "n_train_rows":      int(len(X_train)),
        "n_val_rows":        int(len(X_val)),
        "n_feature_cols":    int(len(feature_cols)),
        "n_estimators_used": int(best_iter) if best_iter else LGBM_PARAMS["n_estimators"],
        "classes":           CLASSES,
    }

    return model, feature_cols, metrics


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    model: lgb.LGBMClassifier,
    feature_cols: List[str],
    metrics: dict,
    models_dir: Path = MODELS_DIR,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "regime_classifier.pkl"
    joblib.dump(model, model_path)

    cols_path = models_dir / "feature_cols.json"
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    meta_path = models_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Model saved     -> {model_path}")
    print(f"  Feature cols    -> {cols_path}")
    print(f"  Training meta   -> {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BioSentinel 2.0 -- Regime Classifier Training")
    print("=" * 62)
    model, feature_cols, metrics = train(verbose=True)
    save_artefacts(model, feature_cols, metrics)
    print("\nDone.")
