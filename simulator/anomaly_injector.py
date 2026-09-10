# FILE: simulator/anomaly_injector.py
#
# PURPOSE:
# Introduces scripted "stress events" into the simulated batch run so
# the ML models and dashboard have real anomalies to detect and react
# to (otherwise the simulation would just be a smooth, boring curve).
#
# WHAT IT WILL CONTAIN:
# - A class/function that can randomly (or on-demand) trigger one of:
#     - an RPM step change (sudden agitation change)
#     - an airflow surge
#     - a simulated filter-wetting event
# - Each anomaly temporarily spikes foam growth and/or gas levels.
#
# WHAT IT WILL DO:
# Called by the batch simulator loop each timestep to decide whether
# to inject an anomaly. Also exposes a manual "force trigger" function
# so the Streamlit dashboard can have buttons that inject an anomaly
# live during the demo (so judges see the detect -> predict -> alert
# -> act loop happen in real time).
