# FILE: explanation_engine/templates.py
#
# PURPOSE:
# Offline fallback for breach explanations — pre-written, scientifically
# accurate explanation text for each gas (CO2, O2, CH4), with the live
# reading substituted in. This is the safety net that fires instantly
# if the LLM call in llm_explainer.py fails or there's no internet at
# the venue, so the live demo never breaks.
#
# WHAT IT WILL CONTAIN:
# - A dictionary/mapping of gas name -> template explanation string,
#   written in the same "what changed -> why it's harmful -> what
#   happens if ignored -> action taken" structure as the LLM output.
# - A function that fills in the live value/threshold into the right
#   template and returns the final explanation string.
#
# WHAT IT WILL DO:
# Guarantees every breach alert has *some* correct, mechanism-based
# explanation shown on the dashboard, even with zero connectivity.
