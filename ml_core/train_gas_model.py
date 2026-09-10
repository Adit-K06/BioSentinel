# FILE: ml_core/train_gas_model.py
#
# PURPOSE:
# Trains "Job 2" — the gas regressor(s). The second of the two core ML
# models in the project.
#
# INPUTS: RQ, RPM, airflow, gas history, rolling trend features (from
# features.py).
#
# PREDICTS: CH4 %, CO2 %, O2 % a few minutes ahead.
#
# SAFE THRESHOLDS (for reference, used by the alert logic elsewhere):
#   CO2 max 5%   -> carbonic acid / pH stress on methanogens
#   O2  max 0.5% -> oxygen poisoning of strict anaerobes
#   CH4 min 65%  -> minimum usable biogas purity
#
# WHAT IT WILL CONTAIN:
# - Code to load a simulated batch CSV from data/synthetic/.
# - Code to build features via ml_core/features.py.
# - Code to train LightGBM regressors for each gas target, with a
#   train/validation split.
# - Code to save the trained models into ml_core/models/.
#
# WHAT IT WILL DO:
# Produces the model that predicts gas composition drift toward unsafe
# thresholds, which triggers alerts and the virtual membrane response.
