# FILE: dashboard/app.py
#
# PURPOSE:
# Main Streamlit entrypoint — the live demo dashboard judges will watch.
# Streams the simulated bioreactor batch in real time rather than
# showing static slides.
#
# WHAT IT WILL CONTAIN:
# - Streamlit page setup (title, layout).
# - Live telemetry gauges: RQ, RPM, airflow, pressure, updating
#   continuously as the simulator runs.
# - A synthetic camera panel showing the generated foam-boundary frame
#   with the OpenCV contour overlay visible (from vision/).
# - A foam forecast panel: predicted V_foam and T_overflow, with an
#   early-warning banner when a crest is imminent (from ml_core/).
# - Per-gas monitor rows (CH4/CO2/O2): green if safe, red if breached,
#   with the AI explanation (from explanation_engine/) shown inline
#   when red.
# - A membrane/actuator state indicator (OPEN/CLOSED) reacting live to
#   AI decisions (from actuator/).
# - Anomaly trigger buttons so judges can inject a foam surge or gas
#   breach live and watch the full detect -> predict -> alert -> act
#   loop happen in front of them (from simulator/anomaly_injector.py).
#
# WHAT IT WILL DO:
# Ties every other module in this repo together into one running app,
# launched with: streamlit run dashboard/app.py
