from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.mock_data import (
    get_cost_of_delay,
    get_counterfactual_actions,
    get_explanation,
    get_live_state,
    get_trajectory_data,
)


st.set_page_config(page_title="BioSentinel Cockpit", layout="wide")

REGIME_LABELS = {
    "Normal": "Healthy",
    "kLa Limit": "kLa Limitation",
    "Overfeed": "Substrate Overfeeding",
    "Contam": "Contamination",
}


def apply_scenario(label: str) -> None:
    st.session_state.scenario = REGIME_LABELS.get(label, label)
    st.session_state.intervention_applied = label


if "scenario" not in st.session_state:
    st.session_state.scenario = "Healthy"
if "intervention_applied" not in st.session_state:
    st.session_state.intervention_applied = "Reset Normal"


scenario = st.session_state.scenario
intervention_applied = st.session_state.intervention_applied
state = get_live_state(scenario, intervention_applied)
trajectory = get_trajectory_data(scenario, intervention_applied)
trajectory_df = pd.DataFrame(trajectory)
cost_curve = get_cost_of_delay(state["recovery"])
root_cause = get_explanation(scenario)
counterfactuals = get_counterfactual_actions(scenario)


st.sidebar.title("Intervention Controls")
st.sidebar.caption("Mock control layer for demo scenarios")

sidebar_buttons = [
    "Inject kLa Limit",
    "Inject Overfeed",
    "Inject Contam",
    "Reset Normal",
]

for label in sidebar_buttons:
    if st.sidebar.button(label):
        apply_scenario(label.replace("Inject ", "").replace("Reset ", ""))

if st.sidebar.button("Apply Recommended Intervention Now"):
    recommended_action = counterfactuals[0]
    apply_scenario(recommended_action["target_regime"])


regime_color = {
    "Healthy": "#2ecc71",
    "kLa Limitation": "#f39c12",
    "Substrate Overfeeding": "#e67e22",
    "Contamination": "#e74c3c",
}

banner_bg = regime_color.get(scenario, "#2ecc71")

st.markdown(
    f"""
    <div style="background:{banner_bg};padding:16px;border-radius:12px;color:white;margin-bottom:18px;">
        <strong>Regime shift:</strong> {scenario}<br>
        <strong>Dynamic Intervention Window (DIW):</strong> {state['diw_remaining']} min remaining to PNR
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.metric("DO", f"{state['DO']} mg/L")
with kpi_cols[1]:
    st.metric("RQ", f"{state['RQ']}")
with kpi_cols[2]:
    st.metric("RPM", f"{state['RPM']}")
with kpi_cols[3]:
    st.metric("Recovery", f"{state['recovery']}%")
with kpi_cols[4]:
    st.metric("DIW", f"{state['diw_remaining']} min")

left_col, right_col = st.columns([1.55, 1])

with left_col:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trajectory_df["time_hours"],
            y=trajectory_df["DO"],
            mode="lines+markers",
            name="DO",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory_df["time_hours"],
            y=trajectory_df["RQ"],
            mode="lines+markers",
            name="RQ",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="DO and RQ trajectory",
        template="plotly_white",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Time (h)",
        yaxis_title="DO (mg/L)",
        yaxis2=dict(title="RQ", overlaying="y", side="right"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    action_df = pd.DataFrame(counterfactuals)[["rank", "action", "expected_recovery_gain", "confidence"]]
    action_df = action_df.rename(
        columns={
            "rank": "Rank",
            "action": "Intervention",
            "expected_recovery_gain": "Recovery Gain (%)",
            "confidence": "Confidence",
        }
    )
    st.dataframe(action_df, use_container_width=True, hide_index=True)

    if st.button("Apply Recommended Intervention Now"):
        chosen = counterfactuals[0]
        apply_scenario(chosen["target_regime"])

with right_col:
    prob_cols = [
        ("Healthy", state["healthy_prob"]),
        ("kLa Limitation", state["kla_prob"]),
        ("Substrate Overfeeding", state["overfeed_prob"]),
        ("Contamination", state["contam_prob"]),
    ]
    st.subheader("Regime probability")
    for label, value in prob_cols:
        st.write(f"{label}: {value:.0%}")
        st.progress(min(1.0, max(0.0, value)))

    cost_df = pd.DataFrame(cost_curve)
    cost_fig = go.Figure()
    cost_fig.add_trace(
        go.Scatter(
            x=cost_df["time_min"],
            y=cost_df["recovery_pct"],
            mode="lines+markers",
            line=dict(color="#d62728", width=3),
            marker=dict(size=6),
        )
    )
    cost_fig.update_layout(
        title="Cost of delay",
        template="plotly_white",
        height=320,
        xaxis_title="Minutes",
        yaxis_title="Recovery (%)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(cost_fig, use_container_width=True)

bottom_left, bottom_right = st.columns([1.8, 1])

with bottom_left:
    st.subheader("Mechanistic root-cause explanation")
    st.markdown(
        f"""
        <div style="background-color:#1e293b;border:1px solid #334155;border-radius:8px;padding:16px;line-height:1.6;color:#f1f5f9;">
            <div style="color:#38bdf8;font-weight:700;display:inline;">What changed:</div> {root_cause['what_changed']}<br><br>
            <div style="color:#38bdf8;font-weight:700;display:inline;">Mechanism:</div> {root_cause['mechanism']}<br><br>
            <div style="color:#38bdf8;font-weight:700;display:inline;">Consequence:</div> {root_cause['consequence']}<br><br>
            <div style="color:#38bdf8;font-weight:700;display:inline;">Action:</div> {root_cause['action']}
        </div>
        """,
        unsafe_allow_html=True,
    )

with bottom_right:
    st.subheader("Virtual actuator status")
    actuator_style = {
        "Nominal": "background:#2ecc71;color:white",
        "O2 transfer limited": "background:#f39c12;color:white",
        "Feed trim recommended": "background:#e67e22;color:white",
        "Bleed + clean cycle": "background:#e74c3c;color:white",
        "O2 transfer boost active": "background:#2ecc71;color:white",
        "Feed trim active": "background:#16a085;color:white",
        "Contaminant isolation active": "background:#c0392b;color:white",
    }
    st.markdown(
        f"""
        <div style="{actuator_style.get(state['actuator_state'], 'background:#666;color:white')};padding:18px;border-radius:12px;text-align:center;font-weight:600;">
            {state['actuator_state']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Current control loop: live mock actuator state")
    st.write(f"Pressure: {state['pressure']} bar")
    st.write(f"OUR: {state['OUR']} mmol/L/h")
    st.write(f"CER: {state['CER']} mmol/L/h")

st.caption("Standalone demo layer; no backend ML model required for UI operation.")
