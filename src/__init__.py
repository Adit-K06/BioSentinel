# src/__init__.py
# Package marker for the BioSentinel 2.0 digital-twin source modules.
# Core public API re-exported for convenience:
from src.simulator import simulate_batch   # noqa: F401
from src.features  import build_features   # noqa: F401
from src.predict   import load_model, predict_batch, predict_row  # noqa: F401
