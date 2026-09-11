from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


REGIME_ALIASES = {
    "healthy": "Healthy",
    "normal": "Healthy",
    "baseline": "Healthy",
    "kla limitation": "kLa Limitation",
    "kla": "kLa Limitation",
    "klalimit": "kLa Limitation",
    "kla_limit": "kLa Limitation",
    "substrate overfeeding": "Substrate Overfeeding",
    "overfeed": "Substrate Overfeeding",
    "substrate_overfeeding": "Substrate Overfeeding",
    "feeding overload": "Substrate Overfeeding",
    "contamination": "Contamination",
    "contam": "Contamination",
    "contaminated": "Contamination",
}


def _normalize_scenario(scenario: str) -> str:
    if scenario is None:
        return "Healthy"
    key = str(scenario).strip().lower().replace("_", " ")
    return REGIME_ALIASES.get(key, str(scenario).strip())


def _base_profile(scenario: str, intervention_applied: Any) -> Dict[str, Any]:
    regime = _normalize_scenario(scenario)

    if regime == "Healthy":
        profile = {
            "DO": 2.9,
            "RQ": 0.19,
            "OUR": 46,
            "CER": 39,
            "pressure": 9.4,
            "RPM": 820,
            "recovery": 95,
            "diw_remaining": 12,
            "healthy_prob": 0.88,
            "kla_prob": 0.06,
            "overfeed_prob": 0.03,
            "contam_prob": 0.03,
            "actuator_state": "Nominal",
        }
    elif regime == "kLa Limitation":
        profile = {
            "DO": 1.2,
            "RQ": 0.29,
            "OUR": 61,
            "CER": 52,
            "pressure": 12.7,
            "RPM": 930,
            "recovery": 78,
            "diw_remaining": 6,
            "healthy_prob": 0.12,
            "kla_prob": 0.74,
            "overfeed_prob": 0.08,
            "contam_prob": 0.06,
            "actuator_state": "O2 transfer limited",
        }
    elif regime == "Substrate Overfeeding":
        profile = {
            "DO": 1.7,
            "RQ": 0.41,
            "OUR": 74,
            "CER": 66,
            "pressure": 13.9,
            "RPM": 1010,
            "recovery": 71,
            "diw_remaining": 5,
            "healthy_prob": 0.10,
            "kla_prob": 0.09,
            "overfeed_prob": 0.74,
            "contam_prob": 0.07,
            "actuator_state": "Feed trim recommended",
        }
    elif regime == "Contamination":
        profile = {
            "DO": 1.4,
            "RQ": 0.24,
            "OUR": 58,
            "CER": 57,
            "pressure": 14.6,
            "RPM": 970,
            "recovery": 68,
            "diw_remaining": 4,
            "healthy_prob": 0.06,
            "kla_prob": 0.11,
            "overfeed_prob": 0.12,
            "contam_prob": 0.71,
            "actuator_state": "Bleed + clean cycle",
        }
    else:
        profile = {
            "DO": 2.5,
            "RQ": 0.21,
            "OUR": 50,
            "CER": 43,
            "pressure": 10.0,
            "RPM": 860,
            "recovery": 88,
            "diw_remaining": 10,
            "healthy_prob": 0.60,
            "kla_prob": 0.18,
            "overfeed_prob": 0.12,
            "contam_prob": 0.10,
            "actuator_state": "Nominal",
        }

    intervention_name = str(intervention_applied).lower() if intervention_applied is not None else ""
    if "kla" in intervention_name:
        profile["actuator_state"] = "O2 transfer boost active"
        profile["DO"] = max(profile["DO"] * 1.15, 1.2)
        profile["RQ"] = max(profile["RQ"] * 0.92, 0.18)
        profile["recovery"] = min(profile["recovery"] + 6, 98)
        profile["diw_remaining"] = min(profile["diw_remaining"] + 2, 15)
    elif "overfeed" in intervention_name:
        profile["actuator_state"] = "Feed trim active"
        profile["DO"] = max(profile["DO"] * 1.10, 1.4)
        profile["RQ"] = max(profile["RQ"] * 0.96, 0.20)
        profile["recovery"] = min(profile["recovery"] + 8, 99)
    elif "contam" in intervention_name:
        profile["actuator_state"] = "Contaminant isolation active"
        profile["DO"] = max(profile["DO"] * 1.12, 1.3)
        profile["RQ"] = max(profile["RQ"] * 0.95, 0.18)
        profile["recovery"] = min(profile["recovery"] + 7, 98)
    elif "reset" in intervention_name or "normal" in intervention_name:
        profile = _base_profile("Healthy", None)

    return profile


def get_live_state(scenario: str, intervention_applied: Any) -> Dict[str, Any]:
    """Return the current operational snapshot for a given regime and intervention state."""
    profile = _base_profile(scenario, intervention_applied)
    return {
        "DO": round(profile["DO"], 2),
        "RQ": round(profile["RQ"], 3),
        "OUR": int(profile["OUR"]),
        "CER": int(profile["CER"]),
        "pressure": round(profile["pressure"], 1),
        "RPM": int(profile["RPM"]),
        "recovery": int(profile["recovery"]),
        "diw_remaining": int(profile["diw_remaining"]),
        "healthy_prob": round(profile["healthy_prob"], 2),
        "kla_prob": round(profile["kla_prob"], 2),
        "overfeed_prob": round(profile["overfeed_prob"], 2),
        "contam_prob": round(profile["contam_prob"], 2),
        "actuator_state": profile["actuator_state"],
    }


def _trajectory_profile(scenario: str, intervention_applied: Any) -> Dict[str, List[float]]:
    regime = _normalize_scenario(scenario)
    hours = np.linspace(0, 15, 16)

    if regime == "Healthy":
        do = np.linspace(2.9, 2.7, len(hours))
        rq = np.linspace(0.19, 0.20, len(hours))
    elif regime == "kLa Limitation":
        do = np.linspace(2.8, 1.1, len(hours))
        rq = np.linspace(0.20, 0.34, len(hours))
    elif regime == "Substrate Overfeeding":
        do = np.linspace(2.6, 1.4, len(hours))
        rq = np.linspace(0.22, 0.42, len(hours))
    elif regime == "Contamination":
        do = np.linspace(2.5, 1.3, len(hours))
        rq = np.linspace(0.21, 0.30, len(hours))
    else:
        do = np.linspace(2.7, 2.3, len(hours))
        rq = np.linspace(0.20, 0.25, len(hours))

    intervention_name = str(intervention_applied).lower() if intervention_applied is not None else ""
    if "kla" in intervention_name:
        do = do + np.linspace(0.0, 0.8, len(hours))
        rq = rq - np.linspace(0.0, 0.08, len(hours))
    elif "overfeed" in intervention_name:
        do = do + np.linspace(0.0, 0.5, len(hours))
        rq = rq - np.linspace(0.0, 0.04, len(hours))
    elif "contam" in intervention_name:
        do = do + np.linspace(0.0, 0.6, len(hours))
        rq = rq - np.linspace(0.0, 0.06, len(hours))
    elif "reset" in intervention_name or "normal" in intervention_name:
        do = np.linspace(2.9, 2.7, len(hours))
        rq = np.linspace(0.19, 0.20, len(hours))

    return {
        "time_hours": [round(float(v), 2) for v in hours],
        "DO": [round(float(v), 2) for v in do],
        "RQ": [round(float(v), 3) for v in rq],
    }


def get_trajectory_data(scenario: str, intervention_applied: Any) -> Dict[str, List[float]]:
    """Return a 15-hour trajectory for dissolved oxygen and respiratory quotient."""
    return _trajectory_profile(scenario, intervention_applied)


def get_cost_of_delay(initial_recovery: float) -> Dict[str, List[float]]:
    """Return a 0-15 minute delay curve for a recovery percentage level."""
    minutes = list(range(0, 16))
    start = float(initial_recovery)
    decay = np.linspace(start, max(start - 25, 45), len(minutes))
    return {
        "time_min": minutes,
        "recovery_pct": [round(float(v), 1) for v in decay],
    }


def get_counterfactual_actions(scenario: str) -> List[Dict[str, Any]]:
    regime = _normalize_scenario(scenario)

    if regime == "Healthy":
        return [
            {"rank": 1, "action": "Maintain current feed setpoint", "expected_recovery_gain": 0, "confidence": 0.97, "target_regime": "Healthy"},
            {"rank": 2, "action": "Increase agitation check", "expected_recovery_gain": 2, "confidence": 0.73, "target_regime": "Healthy"},
            {"rank": 3, "action": "Reduce feed variance", "expected_recovery_gain": 1, "confidence": 0.68, "target_regime": "Healthy"},
        ]
    if regime == "kLa Limitation":
        return [
            {"rank": 1, "action": "Increase oxygen transfer / boost kLa", "expected_recovery_gain": 12, "confidence": 0.91, "target_regime": "Healthy"},
            {"rank": 2, "action": "Increase agitation and sparge rate", "expected_recovery_gain": 8, "confidence": 0.84, "target_regime": "Healthy"},
            {"rank": 3, "action": "Reduce biomass loading transiently", "expected_recovery_gain": 6, "confidence": 0.77, "target_regime": "Healthy"},
        ]
    if regime == "Substrate Overfeeding":
        return [
            {"rank": 1, "action": "Cut feed pulse and dilute substrate", "expected_recovery_gain": 14, "confidence": 0.92, "target_regime": "Healthy"},
            {"rank": 2, "action": "Lower feed concentration by 15%", "expected_recovery_gain": 10, "confidence": 0.86, "target_regime": "Healthy"},
            {"rank": 3, "action": "Reduce recirculation to moderate RQ", "expected_recovery_gain": 7, "confidence": 0.80, "target_regime": "Healthy"},
        ]
    if regime == "Contamination":
        return [
            {"rank": 1, "action": "Isolate feed line and trigger clean-in-place", "expected_recovery_gain": 16, "confidence": 0.94, "target_regime": "Healthy"},
            {"rank": 2, "action": "Increase bleed and reduce residence time", "expected_recovery_gain": 11, "confidence": 0.88, "target_regime": "Healthy"},
            {"rank": 3, "action": "Hold substrate feed for 30 minutes", "expected_recovery_gain": 9, "confidence": 0.82, "target_regime": "Healthy"},
        ]
    return [
        {"rank": 1, "action": "Return to nominal operating envelope", "expected_recovery_gain": 10, "confidence": 0.8, "target_regime": "Healthy"},
        {"rank": 2, "action": "Check aeration and feed balance", "expected_recovery_gain": 7, "confidence": 0.74, "target_regime": "Healthy"},
        {"rank": 3, "action": "Run diagnostic scan for contamination", "expected_recovery_gain": 5, "confidence": 0.69, "target_regime": "Healthy"},
    ]


def get_explanation(scenario: str) -> Dict[str, str]:
    regime = _normalize_scenario(scenario)

    if regime == "Healthy":
        return {
            "what_changed": "Gas transfer and feed balance are stable; oxygen uptake and carbon evolution are tracking the expected setpoint.",
            "mechanism": "Aeration remains sufficient to match cellular oxygen demand while the mass-balance remains in the targeted operating window.",
            "consequence": "This preserves oxidative efficiency and limits risk of over-aeration or run-away metabolic stress.",
            "action": "Keep the current feed and agitation controls in place and continue routine health monitoring.",
        }
    if regime == "kLa Limitation":
        return {
            "what_changed": "Dissolved oxygen has fallen below the expected range while respiration continues to rise.",
            "mechanism": "The oxygen transfer coefficient is insufficient for the current cell density, so gas-liquid exchange cannot keep pace with OUR.",
            "consequence": "This drives a DO deficit that suppresses oxidative metabolism and reduces recovery margin before the process enters PNR.",
            "action": "Increase oxygen transfer, boost agitation/sparge intensity, and target the kLa deficit before DO deteriorates further.",
        }
    if regime == "Substrate Overfeeding":
        return {
            "what_changed": "The feed rate has exceeded the metabolic capacity of the culture, pushing RQ upward and creating a high carbon load.",
            "mechanism": "Excess substrate increases carbon overflow and respiratory demand; oxygen can no longer keep up with the resulting metabolic acceleration.",
            "consequence": "High RQ and rising pressure indicate a metabolic bottleneck that threatens productivity and can trigger foam or instability.",
            "action": "Trim the feed pulse, lower substrate concentration, and restore the metabolic balance to reduce carbon overload.",
        }
    if regime == "Contamination":
        return {
            "what_changed": "The process is showing a mixed signal: DO and RQ drift despite stable feed targets, suggesting an external process upset.",
            "mechanism": "A contaminant or off-spec feed is perturbing the bioprocess, changing oxygen demand and undermining the expected control response.",
            "consequence": "The culture loses predictability, recovery decays faster, and the process approaches the intervention threshold with little margin.",
            "action": "Isolate the feed source, trigger clean-in-place, and enforce a controlled recovery to re-establish the healthy envelope.",
        }
    return {
        "what_changed": "Process conditions are drifting from the normal operating envelope.",
        "mechanism": "A change in aeration, substrate supply, or contamination state is affecting the culture balance.",
        "consequence": "Recovery margin is narrowing and the system is approaching a risk threshold.",
        "action": "Review the active control loop, confirm the dominant process upset, and apply the corrective intervention quickly.",
    }


__all__ = [
    "get_live_state",
    "get_trajectory_data",
    "get_cost_of_delay",
    "get_counterfactual_actions",
    "get_explanation",
]
