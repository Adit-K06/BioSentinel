# FILE: simulator/growth_model.py
#
# PURPOSE:
# Models biomass growth over time inside the simulated bioreactor.
# This is the "engine" that drives how much biomass exists at any
# timestep, which in turn feeds the off-gas model and foam model.
#
# WHAT IT WILL CONTAIN:
# - A logistic growth curve function: X(t) = X_max / (1 + e^(-r*(t - t_mid)))
#   for a simple S-shaped growth curve over a batch run.
# - A Monod substrate-limited growth function (mu = mu_max * S / (K_s + S))
#   for a more realistic, substrate-dependent growth rate.
#
# WHAT IT WILL DO:
# Given a timestep (or substrate concentration), return the current
# biomass value / growth rate, which other simulator modules use to
# compute downstream quantities (gas production, foam growth, etc).
#
# OWNER: teammate doing formulas/research — exact constants (X_max, r,
# t_mid, mu_max, K_s) should come from data/formulas/.
