"""
src/regimes.py
==============
Regime definitions and per-step mutation logic for BioSentinel 2.0.

Each regime is implemented as a *mutator* — a callable that takes the current
simulation state and time, and returns (possibly modified) effective parameter
values for that timestep.  This keeps the ODE right-hand side in simulator.py
clean and regime-agnostic.

Available regimes (case-insensitive lookup via ``get_regime``):
  - "Healthy"
  - "kLa_Limitation"
  - "Substrate_Overfeeding"
  - "Contamination"

No Streamlit / UI / database / API imports.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import math

__all__ = ["REGIME_NAMES", "get_regime", "RegimeBase"]

REGIME_NAMES = [
    "Healthy",
    "kLa_Limitation",
    "Substrate_Overfeeding",
    "Contamination",
]

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RegimeBase:
    """Abstract base for a batch-run regime.

    Sub-classes implement ``mutate(t, state, params)`` which returns the
    effective parameter dict for a given timestep.  They may also expose
    extra ODE state extensions (e.g. contaminant biomass).
    """

    name: str = "RegimeBase"
    n_extra_states: int = 0  # number of ODE states added beyond [X, S, CL]

    def __init__(self, t_onset: Optional[float], severity: float, **kwargs: Any) -> None:
        self.t_onset = t_onset    # hours; None → never triggered
        self.severity = severity

    # ------------------------------------------------------------------
    def mutate(
        self,
        t: float,
        state: Tuple[float, ...],  # (X, S, CL [, extra...])
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return effective params for this timestep (default: identity)."""
        return params

    # ------------------------------------------------------------------
    def extra_rhs(
        self,
        t: float,
        state: Tuple[float, ...],
        params: Dict[str, Any],
    ) -> Tuple[float, ...]:
        """Return derivatives for any extra ODE states.  Default: empty."""
        return ()

    # ------------------------------------------------------------------
    def initial_extra_states(self, params: Dict[str, Any]) -> Tuple[float, ...]:
        """Return initial values for extra ODE states.  Default: empty."""
        return ()


# ---------------------------------------------------------------------------
# Healthy (no-fault baseline)
# ---------------------------------------------------------------------------

class HealthyRegime(RegimeBase):
    """Pure baseline: no mutations."""

    name = "Healthy"

    def __init__(self, t_onset=None, severity=0.0, **kwargs):
        super().__init__(t_onset=t_onset, severity=severity)


# ---------------------------------------------------------------------------
# kLa Limitation
# ---------------------------------------------------------------------------

class KLaLimitationRegime(RegimeBase):
    """Simulate progressive membrane/sparger fouling.

    After ``t_onset`` the effective kLa declines linearly over 2 h to
    ``kLa * (1 - severity)``, then stays constant.  This causes DO sag,
    which depresses growth, lowering OUR and raising apparent RQ.
    """

    name = "kLa_Limitation"

    def __init__(self, t_onset=8.0, severity=0.70, **kwargs):
        super().__init__(t_onset=t_onset, severity=severity)
        self._ramp_duration = 2.0  # hours over which kLa ramps down

    def mutate(self, t, state, params):
        if self.t_onset is None or t < self.t_onset:
            return params
        p = dict(params)
        elapsed = t - self.t_onset
        frac = min(elapsed / self._ramp_duration, 1.0)
        reduction = self.severity * frac
        p["kLa"] = max(params["kLa"] * (1.0 - reduction), 1.0)  # floor at 1 h⁻¹
        return p


# ---------------------------------------------------------------------------
# Substrate Overfeeding
# ---------------------------------------------------------------------------

class SubstrateFeedingRegime(RegimeBase):
    """Simulate operator over-addition of substrate feed.

    After ``t_onset`` the feed rate F_s is immediately multiplied by
    ``(1 + severity)``.  Excess substrate drives overflow metabolism,
    causing substrate accumulation, pH drop (not modelled explicitly),
    and elevated RQ.
    """

    name = "Substrate_Overfeeding"

    def __init__(self, t_onset=6.0, severity=3.0, **kwargs):
        super().__init__(t_onset=t_onset, severity=severity)

    def mutate(self, t, state, params):
        if self.t_onset is None or t < self.t_onset:
            return params
        p = dict(params)
        p["F_s"] = params["F_s"] * (1.0 + self.severity)
        return p


# ---------------------------------------------------------------------------
# Contamination
# ---------------------------------------------------------------------------

class ContaminationRegime(RegimeBase):
    """Introduce a second, faster-growing microbial population.

    The contaminant (X_c) is governed by its own Monod kinetics:
        dX_c/dt = mu_c(S, CL) * X_c

    It consumes the same substrate S and dissolved oxygen CL but with
    higher Y_xo (oxygen demand), causing DO to sag further than the
    process model would predict → RQ diverges from the healthy reference.

    Extra ODE state: [X_c]  (index 3 in the full state vector)
    """

    name = "Contamination"
    n_extra_states = 1

    def __init__(
        self,
        t_onset: float = 5.0,
        severity: float = 0.50,
        mu_max_cont: float = 0.80,
        K_s_cont: float = 0.05,
        Y_xo_cont: float = 0.25,
        X_cont0: float = 0.001,
        **kwargs,
    ):
        super().__init__(t_onset=t_onset, severity=severity)
        self.mu_max_cont = mu_max_cont
        self.K_s_cont = K_s_cont
        self.Y_xo_cont = Y_xo_cont
        self.X_cont0 = X_cont0

    # Extra initial conditions: [X_c]
    def initial_extra_states(self, params):
        return (0.0,)  # dormant until onset

    def extra_rhs(self, t, state, params):
        """Returns (dX_c/dt,) — contaminant growth term."""
        X, S, CL = state[0], state[1], state[2]
        X_c = state[3] if len(state) > 3 else 0.0

        if self.t_onset is not None and t < self.t_onset:
            return (0.0,)

        # Seed the contaminant if it has not yet been introduced
        X_c = max(X_c, self.X_cont0 if t >= (self.t_onset or 0.0) else 0.0)

        S_ = max(S, 0.0)
        CL_ = max(CL, 0.0)
        mu_c = (
            self.mu_max_cont
            * (S_ / (self.K_s_cont + S_ + 1e-12))
            * (CL_ / (params["K_O"] + CL_ + 1e-12))
        )
        return (mu_c * X_c,)

    def mutate(self, t, state, params):
        """Adjust effective S and CL consumption to account for contaminant."""
        # The contaminant's contribution to OUR is handled in simulator.py
        # directly via extra_rhs; here we only expose active status.
        return params


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {
    "healthy":               HealthyRegime,
    "kla_limitation":        KLaLimitationRegime,
    "substrate_overfeeding": SubstrateFeedingRegime,
    "contamination":         ContaminationRegime,
}


def get_regime(name: str, overrides: Dict[str, Any]) -> RegimeBase:
    """Return an instantiated regime object.

    Parameters
    ----------
    name : str
        Regime name (case-insensitive, spaces→underscores).
    overrides : dict
        Merged parameter dict (from config.DEFAULT_PARAMS +
        config.REGIME_OVERRIDES[name]).  Regime constructor kwargs
        are extracted automatically.

    Returns
    -------
    RegimeBase subclass instance.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown regime '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[key]
    # Extract keyword arguments the constructor accepts
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in overrides.items() if k in valid_keys}
    return cls(**kwargs)
