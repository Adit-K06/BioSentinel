# FILE: simulator/foam_model.py
#
# PURPOSE:
# Simulates foam height inside the bioreactor over time — the core
# physical quantity that Job 1 (foam overflow prediction) is trying
# to forecast.
#
# WHAT IT WILL CONTAIN:
# - A single-step (Euler integration) update function implementing:
#   dh/dt = k1 * (growth_rate) * (agitation_factor) * (airflow_factor) - k2 * h
#   i.e. foam grows with metabolic activity/agitation/airflow, and
#   decays through natural drainage.
#
# WHAT IT WILL DO:
# Given the current foam height and current process conditions
# (growth rate, RPM-derived agitation factor, airflow factor), return
# the updated foam height for the next timestep. Called repeatedly by
# the batch simulator loop to build the full foam-height time series.
#
# OWNER: teammate doing formulas/research — exact constants (k1, k2)
# should come from data/formulas/ or be tuned against reference data.
