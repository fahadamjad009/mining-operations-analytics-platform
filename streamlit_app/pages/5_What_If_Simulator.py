from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="What-If Simulator", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "models" / "downtime_risk_model.joblib"
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("🎛️ What-If Simulator")
st.caption(
    "Adjust feature values and watch the downtime risk score change in real time. "
    "Useful for **sensitivity testing**, **intervention modeling** ('what if maintenance reduces vibration?'), "
    "and **stress testing** ('what if temperature spikes?'). Demonstrates the model is operationalizable, not just a black box."
)

# ---------- Guards ----------
missing = [p for p in [MODEL_PATH, DAILY_PATH] if not p.exists()]
if missing:
    st.error("Missing model or daily summary outputs.")
    st.markdown("➡️ **If you are on Streamlit Cloud:** go to **Home** and click **Generate demo data now**.")
    st.markdown("**If running locally:** run:")
    st.code(
        "python -m src.miningops.generate_data\n"
        "python -m src.miningops.kpis\n"
        "python -m src.miningops.train",
        language="text",
    )
    st.stop()


# ---------- Load ----------
@st.cache_resource(show_spinner=False)
def _load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _load_data(path):
    return pd.read_parquet(path)


model = _load_model(MODEL_PATH)
df = _load_data(DAILY_PATH)


def _expected_features(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return ["avg_temp", "avg_vib", "avg_fuel", "utilization_rate", "downtime_rate", "work_orders"]


features = [f for f in _expected_features(model) if f in df.columns]

if not features:
    st.error("Model features not found in dataset.")
    st.stop()


# ---------- Compute fleet baseline ----------
@st.cache_data(show_spinner=False)
def _fleet_baseline(_df, feature_list, _model):
    X = _df[feature_list].copy()
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = _model.predict_proba(X)[:, 1]
    return X, proba


X_full, proba_full = _fleet_baseline(df, features, model)


# ---------- Presets ----------
def _preset_values(label):
    if label == "Low-risk profile (bottom 10%)":
        thresh = np.percentile(proba_full, 10)
        return X_full[proba_full <= thresh].median().to_dict()
    if label == "High-risk profile (top 10%)":
        thresh = np.percentile(proba_full, 90)
        return X_full[proba_full >= thresh].median().to_dict()
    return X_full.median().to_dict()


preset_label = st.selectbox(
    "Load preset values (optional starting point):",
    ["Median / typical asset", "Low-risk profile (bottom 10%)", "High-risk profile (top 10%)"],
    index=0,
)

preset_vals = _preset_values(preset_label)

# Reset slider state when preset changes
if "preset_key" not in st.session_state:
    st.session_state.preset_key = preset_label
if st.session_state.preset_key != preset_label:
    st.session_state.preset_key = preset_label
    for f in features:
        st.session_state.pop(f"slider_{f}", None)
    st.rerun()

# ---------- Slider bounds (p1-p99 ±20% for exploration headroom) ----------
feature_ranges = {}
for f in features:
    s = X_full[f]
    lo = float(s.quantile(0.01))
    hi = float(s.quantile(0.99))
    span = max(hi - lo, 1e-6)
    feature_ranges[f] = (lo - 0.2 * span, hi + 0.2 * span)

# ---------- Layout ----------
col_inputs, col_results = st.columns([1, 1])

with col_inputs:
    st.markdown("### Feature inputs")
    user_inputs = {}
    for f in features:
        lo, hi = feature_ranges[f]
        default = float(preset_vals.get(f, X_full[f].median()))
        default = max(min(default, hi), lo)
        step = (hi - lo) / 100
        user_inputs[f] = st.slider(
            f,
            min_value=lo,
            max_value=hi,
            value=default,
            step=step,
            key=f"slider_{f}",
        )

# ---------- Predict ----------
input_row = pd.DataFrame([user_inputs])[features]
risk_score = float(model.predict_proba(input_row)[:, 1][0])
percentile = float(np.mean(proba_full <= risk_score) * 100)

if risk_score < 0.10:
    band, band_color = "🟢 LOW", "#27ae60"
elif risk_score < 0.20:
    band, band_color = "🟡 MODERATE", "#f39c12"
elif risk_score < 0.30:
    band, band_color = "🟠 ELEVATED", "#e67e22"
else:
    band, band_color = "🔴 HIGH", "#e74c3c"

with col_results:
    st.markdown("### Predicted risk score")
    st.markdown(
        f"<div style='background:{band_color};color:white;padding:24px;border-radius:8px;text-align:center;'>"
        f"<h1 style='margin:0;color:white;font-size:48px;'>{risk_score:.4f}</h1>"
        f"<p style='margin:8px 0 0 0;font-size:20px;'>{band}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"This configuration sits at the **{percentile:.1f}th percentile** of risk across "
        f"{len(proba_full):,} historical fleet observations."
    )

    fig_dist = px.histogram(
        x=proba_full, nbins=40,
        labels={"x": "Risk score across fleet", "y": "Count"},
        title="Where does this score sit in the fleet?",
    )
    fig_dist.add_vline(
        x=risk_score, line_color="red", line_width=3,
        annotation_text=f"Simulated: {risk_score:.3f}",
        annotation_position="top",
    )
    fig_dist.update_layout(height=320, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

st.divider()

# ---------- SHAP local explanation (opt-in via button to keep UI responsive) ----------
if SHAP_AVAILABLE:
    st.markdown("### Why this score? (SHAP local explanation)")
    st.caption(
        "SHAP shows how much each feature value contributes to pushing this prediction above or below the "
        "model's baseline expectation. Click to compute — takes ~20-40s."
    )

    if st.button("🔍 Explain this prediction", type="primary"):
        with st.spinner("Computing SHAP explanation..."):
            try:
                @st.cache_resource(show_spinner=False)
                def _make_explainer(_model, _features, _X_full_values):
                    bg = _X_full_values[
                        np.random.RandomState(42).choice(len(_X_full_values), size=min(20, len(_X_full_values)), replace=False)
                    ]

                    def predict_fn(X_arr):
                        return _model.predict_proba(pd.DataFrame(X_arr, columns=_features))

                    return shap.KernelExplainer(predict_fn, bg)

                explainer = _make_explainer(model, features, X_full.values)
                sv = explainer.shap_values(input_row.values, nsamples=30, silent=True)

                if isinstance(sv, list):
                    sv = sv[1]
                elif hasattr(sv, "ndim") and sv.ndim == 3:
                    sv = sv[:, :, 1]
                sv = np.asarray(sv).flatten()

                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = float(np.atleast_1d(ev)[-1])
                else:
                    ev = float(ev)

                wf = pd.DataFrame({
                    "feature": features,
                    "value": [user_inputs[f] for f in features],
                    "shap": sv,
                }).sort_values("shap", key=abs, ascending=True)

                fig_wf = go.Figure(go.Bar(
                    x=wf["shap"],
                    y=[f"{f} = {v:.3f}" for f, v in zip(wf["feature"], wf["value"])],
                    orientation="h",
                    marker_color=["#e74c3c" if v > 0 else "#3498db" for v in wf["shap"]],
                    text=[f"{v:+.4f}" for v in wf["shap"]],
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title=f"SHAP contributions — base {ev:.4f} + sum {wf['shap'].sum():+.4f} = {ev + wf['shap'].sum():.4f}",
                    xaxis_title="SHAP value (impact on risk probability)",
                    height=400,
                )
                st.plotly_chart(fig_wf, use_container_width=True)
                st.caption(
                    "**Red bars** push risk UP from baseline; **blue bars** pull it DOWN. "
                    "Magnitude = how much each feature value moves the prediction."
                )
            except Exception as e:
                st.warning(f"SHAP explanation failed: {e}")
else:
    st.info("Install `shap` to enable per-point SHAP explanations.")

st.divider()

with st.expander("ℹ️ How to use this simulator"):
    st.markdown(
        """
        **Example workflows:**
        - **Sensitivity testing:** start from "Median asset" preset, move one slider at a time, observe which feature has the biggest impact on risk
        - **Failure mode exploration:** load "High-risk profile" and probe which combinations push risk into the 🔴 HIGH band
        - **Intervention modeling:** if a maintenance action reduces vibration from 45 → 25, how much does risk drop?
        - **Threshold finding:** identify the exact feature values where risk crosses from 🟡 MODERATE to 🟠 ELEVATED

        **Slider bounds:** range from the 1st to 99th percentile of training data, extended ±20% for exploration.
        Values outside this range are extrapolations the model wasn't trained on — interpret with caution.
        """
    )