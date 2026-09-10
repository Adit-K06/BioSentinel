# FILE: ml_core/features.py
#
# PURPOSE:
# Shared feature engineering used by both Job 1 (foam) and Job 2 (gas)
# models, so the same rolling-trend logic isn't duplicated in both
# training scripts.
#
# WHAT IT WILL CONTAIN:
# - A function that takes a dataframe of raw simulated telemetry
#   (RQ, RPM, airflow, pressure, dh/dt, gas %) and adds rolling-window
#   features: rolling mean and rolling slope (trend) for each column.
#
# WHAT IT WILL DO:
# Turns raw per-timestep readings into features that capture recent
# trend direction, which is what lets the models forecast a few
# minutes ahead instead of just reacting to the current instant.
