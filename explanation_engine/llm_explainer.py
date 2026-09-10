# FILE: explanation_engine/llm_explainer.py
#
# PURPOSE:
# Primary path for generating human-readable, mechanism-based
# explanations when a gas threshold is breached (e.g. why CO2 at 6.2%
# is dangerous for this specific process, not just "warning: high CO2").
#
# WHAT IT WILL CONTAIN:
# - A function that takes breach context (gas name, current value,
#   threshold, trend direction) and sends it to an LLM (Anthropic API)
#   with a tight prompt asking for a 3-4 sentence technical explanation.
# - Reads the API key from an environment variable (ANTHROPIC_API_KEY),
#   loaded via a .env file (not committed to git).
#
# WHAT IT WILL DO:
# Produces dynamic, context-specific reasoning each time a breach
# happens, following the structure: what changed -> why it's
# mechanistically harmful -> what happens if ignored -> what
# corrective action was taken.
#
# NOTE: If there's no internet at the venue or this call fails, the
# system should fall back to explanation_engine/templates.py instead,
# so the demo never breaks.
