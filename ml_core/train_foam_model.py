# FILE: ml_core/train_foam_model.py
#
# PURPOSE:
# Trains "Job 1" — the foam regressor. This is one of the two core ML
# models in the project.
#
# INPUTS: RQ, RPM, airflow, pressure, CV-derived dh/dt, rolling trend
# features (from features.py).
#
# PREDICTS: V_foam (predicted foam volume in litres) and T_overflow
# (predicted time-to-crest in minutes).
#
# WHAT IT WILL CONTAIN:
# - Code to load a simulated batch CSV from data/synthetic/.
# - Code to build features via ml_core/features.py.
# - Code to train two LightGBM regressors (one for V_foam, one for
#   T_overflow), with a train/validation split.
# - Code to save the trained models into ml_core/models/.
#
# WHAT IT WILL DO:
# Produces the early-warning model that lets the dashboard flag a
# foam overflow 5-15 minutes before it actually happens.
