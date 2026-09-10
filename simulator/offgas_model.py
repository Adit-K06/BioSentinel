# FILE: simulator/offgas_model.py
#
# PURPOSE:
# Computes off-gas analysis values used to characterize fermenter/digester
# health: OUR (oxygen uptake rate), CER (CO2 evolution rate), and RQ
# (respiratory quotient = CER / OUR).
#
# WHAT IT WILL CONTAIN:
# - A function to calculate OUR from inlet/outlet O2 concentration and
#   gas flow rate.
# - A function to calculate CER from inlet/outlet CO2 concentration and
#   gas flow rate.
# - A function to calculate RQ = CER / OUR.
#
# WHAT IT WILL DO:
# Takes raw simulated gas concentration + flow readings for a timestep
# and returns OUR, CER, RQ — these become input features for the Job 2
# (gas) ML model.
#
# OWNER: teammate doing formulas/research — exact formulas should be
# taken from the MDPI Processes (2021) off-gas soft sensor methodology
# referenced in data/formulas/.
