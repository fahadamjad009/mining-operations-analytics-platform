from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Optional SHAP import — page degrades gracefully if missing
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="Model Performance", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "models" / "downtime_risk_model.joblib"
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("🧪 Model Performance")
st.caption("ROC · threshold tuning · calibration · lift · SHAP explainability — full evaluation surface for the downtime-risk classifier.")

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
    st.write("Missing files:")
    st.code("\n".join(str(p) for p in missing), language="text")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_parquet(DAILY_PATH)
df["date"] = pd.to_datetime(df.get("date"), errors="coerce")


def _get_expected_feature_names(m):
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):
        for step in reversed(list(m.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


# ---- Label ----
if "high_sev" not in df.columns:
    st.error("Column 'high_sev' not found in daily summary.")
    st.stop()

y = pd.Series(df["high_sev"]).fillna(0).astype(int).clip(0, 1).values
unique_y = np.unique(y)

if unique_y.size < 2:
    st.warning(
        f"Only one class present in y (high_sev): {unique_y.tolist()}. "
        "ROC/AUC cannot be computed."
    )

expected = _get_expected_feature_names(model)
if expected is None:
    expected = ["avg_temp", "avg_vib", "avg_fuel", "utilization_rate", "downtime_rate", "work_orders"]

missing_cols = [c for c in expected if c not in df.columns]
if missing_cols:
    st.error("Model expects columns missing from the daily summary:")
    st.code("\n".join(missing_cols), language="text")
    st.stop()

X = df[expected].copy()
for c in X.columns:
    if X[c].dtype == "bool":
        X[c] = X[c].astype(int)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

if "high_sev" in expected:
    st.warning("⚠️ Model was trained with 'high_sev' as input — target leakage. Re-train without it for realistic eval.")

if not hasattr(model, "predict_proba"):
    st.error("Loaded model does not support predict_proba().")
    st.stop()

proba_all = model.predict_proba(X)
if getattr(proba_all, "ndim", None) != 2 or proba_all.shape[1] < 2:
    st.error(f"Unexpected predict_proba output shape: {getattr(proba_all, 'shape', None)}.")
    st.stop()

proba = proba_all[:, 1]

# ---- Risk score summary ----
p_min = _safe_float(np.min(proba))
p_max = _safe_float(np.max(proba))
p_mean = _safe_float(np.mean(proba))
p_p50 = _safe_float(np.percentile(proba, 50))
p_p90 = _safe_float(np.percentile(proba, 90))
p_p95 = _safe_float(np.percentile(proba, 95))
p_p99 = _safe_float(np.percentile(proba, 99))
prevalence = float(np.mean(y)) if len(y) else 0.0

auc_val = None
roc_df = None
if unique_y.size == 2:
    try:
        auc_val = float(roc_auc_score(y, proba))
        fpr, tpr, _ = roc_curve(y, proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    except Exception as e:
        st.warning(f"Could not compute ROC/AUC: {e}")

# ---- Hero metrics (above tabs) ----
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC AUC", f"{auc_val:.3f}" if auc_val is not None else "—")
m2.metric("Positives (high_sev=1)", f"{int(y.sum()):,}")
m3.metric("Total rows", f"{len(y):,}")
m4.metric("Risk score range", f"{p_min:.3f} → {p_max:.3f}")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Risk mean", f"{p_mean:.3f}")
s2.metric("p90 / p95", f"{p_p90:.3f} / {p_p95:.3f}")
s3.metric("p99", f"{p_p99:.3f}")
s4.metric("Label prevalence", f"{prevalence:.3%}")

with st.expander("Risk score distribution"):
    hist_df = pd.DataFrame({"risk_score": proba})
    fig_hist = px.histogram(hist_df, x="risk_score", nbins=40, title="Risk score histogram")
    fig_hist.update_xaxes(title="Predicted probability")
    fig_hist.update_yaxes(title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ============================================================
# TAB STRUCTURE
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 ROC & Threshold",
    "🎯 Calibration",
    "📊 Lift & Gains",
    "🔍 SHAP Explainability",
])

# ============================================================
# TAB 1: ROC & THRESHOLD
# ============================================================
with tab1:
    left, right = st.columns([2, 1])

    with left:
        if roc_df is not None:
            fig_roc = px.line(roc_df, x="fpr", y="tpr", title="ROC curve (binary)")
            fig_roc.update_xaxes(title="False Positive Rate")
            fig_roc.update_yaxes(title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("ROC curve not available.")

    with right:
        st.subheader("Decision threshold")
        slider_min = 0.05
        slider_max = float(np.clip(max(p_max, 0.20), 0.20, 0.95))
        default_thr = float(np.clip(p_p95, slider_min, min(0.50, slider_max)))
        thr = st.slider(
            "Threshold",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=float(default_thr),
            step=0.01,
            help=f"Risk mean={p_mean:.3f}, max={p_max:.3f}.",
        )
        yhat = (proba >= thr).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        st.metric("Predicted positives", f"{int(yhat.sum()):,}")
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall", f"{rec:.3f}")
        st.metric("F1", f"{f1:.3f}")
        st.write("Confusion matrix:")
        cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
        st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("Top predicted downtime-risk alerts")
    base_cols = ["date", "site", "equipment_id", "equipment_type"]
    optional_cols = [c for c in ["downtime_minutes", "work_orders"] if c in df.columns]
    show_cols = [c for c in base_cols if c in df.columns] + optional_cols
    alerts = df[show_cols].copy()
    alerts["risk_score"] = proba
    alerts["predicted_high_risk"] = (proba >= thr).astype(int)
    top_alerts = alerts.sort_values("risk_score", ascending=False).head(30)
    st.dataframe(top_alerts, use_container_width=True, hide_index=True)

# ============================================================
# TAB 2: CALIBRATION
# ============================================================
with tab2:
    st.subheader("Calibration Curve — does '70% confident' actually mean 70%?")
    st.caption(
        "Plots predicted probability vs actual fraction of positives. "
        "Perfect calibration = points on the diagonal. Above diagonal = model under-confident. "
        "Below diagonal = over-confident. Critical for risk decisions where probability magnitude matters."
    )

    if unique_y.size < 2:
        st.warning("Calibration requires both classes present in y.")
    else:
        try:
            prob_true, prob_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
            cal_df = pd.DataFrame({
                "Mean predicted probability": prob_pred,
                "Fraction of positives": prob_true,
            })
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect calibration",
            ))
            fig_cal.add_trace(go.Scatter(
                x=cal_df["Mean predicted probability"],
                y=cal_df["Fraction of positives"],
                mode="lines+markers",
                line=dict(color="#3498db", width=3),
                marker=dict(size=10),
                name="Model",
            ))
            fig_cal.update_layout(
                title="Reliability Diagram (10 quantile bins)",
                xaxis_title="Mean predicted probability",
                yaxis_title="Observed fraction of positives",
                xaxis=dict(range=[0, max(cal_df["Mean predicted probability"].max() * 1.1, 0.1)]),
                yaxis=dict(range=[0, max(cal_df["Fraction of positives"].max() * 1.1, 0.1)]),
                height=500,
            )
            st.plotly_chart(fig_cal, use_container_width=True)

            brier = float(np.mean((proba - y) ** 2))
            st.metric("Brier score", f"{brier:.4f}",
                      help="Lower = better. Perfect model = 0. Random baseline at this prevalence ≈ p(1-p).")

            with st.expander("Calibration table"):
                st.dataframe(cal_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Could not compute calibration: {e}")

# ============================================================
# TAB 3: LIFT & GAINS
# ============================================================
with tab3:
    st.subheader("Lift & Cumulative Gains — operational decision support")
    st.caption(
        "**Gains chart:** if I act on the top X% of predicted-risk assets, what % of actual failures do I catch? "
        "**Lift chart:** how much better than random selection? Translates ML quality into operational action."
    )

    if unique_y.size < 2 or y.sum() == 0:
        st.warning("Lift/gains require both classes present and positive cases.")
    else:
        sorted_idx = np.argsort(-proba)
        y_sorted = y[sorted_idx]
        n = len(y_sorted)
        total_positives = y_sorted.sum()
        cumulative_pos = np.cumsum(y_sorted)
        percentile = np.arange(1, n + 1) / n * 100
        gain_pct = cumulative_pos / total_positives * 100
        lift = gain_pct / percentile

        lg_df = pd.DataFrame({
            "Top X% of predictions": percentile,
            "Gains %": gain_pct,
            "Lift": lift,
        })

        col_g, col_l = st.columns(2)

        with col_g:
            st.markdown("**Cumulative Gains Chart**")
            fig_gain = go.Figure()
            fig_gain.add_trace(go.Scatter(
                x=lg_df["Top X% of predictions"], y=lg_df["Gains %"],
                mode="lines", line=dict(color="#27ae60", width=3),
                name="Model",
            ))
            fig_gain.add_trace(go.Scatter(
                x=[0, 100], y=[0, 100], mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Random baseline",
            ))
            fig_gain.update_layout(
                xaxis_title="Top X% of predictions (acted on)",
                yaxis_title="% of actual failures captured",
                height=420,
            )
            st.plotly_chart(fig_gain, use_container_width=True)

        with col_l:
            st.markdown("**Lift Chart**")
            fig_lift = go.Figure()
            fig_lift.add_trace(go.Scatter(
                x=lg_df["Top X% of predictions"], y=lg_df["Lift"],
                mode="lines", line=dict(color="#e67e22", width=3),
                name="Model lift",
            ))
            fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray",
                              annotation_text="Random baseline (lift = 1.0)")
            fig_lift.update_layout(
                xaxis_title="Top X% of predictions",
                yaxis_title="Lift (multiple over random)",
                height=420,
            )
            st.plotly_chart(fig_lift, use_container_width=True)

        st.divider()
        st.markdown("**Key operational insights**")
        d1, d2, d3 = st.columns(3)
        idx_10 = max(int(n * 0.10) - 1, 0)
        idx_20 = max(int(n * 0.20) - 1, 0)
        idx_30 = max(int(n * 0.30) - 1, 0)
        d1.metric("Top 10% → % failures caught", f"{gain_pct[idx_10]:.1f}%",
                  delta=f"{lift[idx_10]:.2f}× lift", delta_color="off")
        d2.metric("Top 20% → % failures caught", f"{gain_pct[idx_20]:.1f}%",
                  delta=f"{lift[idx_20]:.2f}× lift", delta_color="off")
        d3.metric("Top 30% → % failures caught", f"{gain_pct[idx_30]:.1f}%",
                  delta=f"{lift[idx_30]:.2f}× lift", delta_color="off")

# ============================================================
# TAB 4: SHAP EXPLAINABILITY
# ============================================================
with tab4:
    st.subheader("SHAP Explainability — why does the model predict what it predicts?")
    st.caption(
        "**Global view:** which features drive risk overall? **Per-asset waterfall:** "
        "for a specific high-risk asset, exactly how each feature's value contributes to its score."
    )

    if not SHAP_AVAILABLE:
        st.error(
            "SHAP library not installed. Add `shap==0.46.0` to requirements.txt and reinstall."
        )
    else:
        @st.cache_data(show_spinner="Computing SHAP values (30-60s first run, cached after)...")
        def compute_shap_values(_model, X_sample_values, feature_names):
            """Returns (shap_values, expected_value) for positive class.
            Uses KernelExplainer with a function wrapper. The wrapper bypasses SHAP's
            attempt to set attributes on sklearn Pipelines (which raises 'no setter' errors)."""
            feature_names = list(feature_names)
            sample_arr = np.asarray(X_sample_values)

            def predict_fn(X_arr):
                X_df = pd.DataFrame(X_arr, columns=feature_names)
                return _model.predict_proba(X_df)

            bg_size = min(20, len(sample_arr))
            bg_idx = np.random.RandomState(42).choice(len(sample_arr), size=bg_size, replace=False)
            bg = sample_arr[bg_idx]

            try:
                explainer = shap.KernelExplainer(predict_fn, bg)
                sv = explainer.shap_values(sample_arr, nsamples=30, silent=True)
                # Handle different SHAP output shapes for binary classification
                if isinstance(sv, list):
                    sv = sv[1]  # older SHAP: list of [class_0_shap, class_1_shap]
                elif hasattr(sv, "ndim") and sv.ndim == 3:
                    sv = sv[:, :, 1]  # newer SHAP: (n_samples, n_features, n_classes)
                sv = np.asarray(sv)
                ev = explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    ev = float(np.atleast_1d(ev)[-1])
                return sv, float(ev), None
            except Exception as e:
                return None, None, f"SHAP computation failed: {e}"

        # Sample for SHAP (80 rows — KernelExplainer is the only universal option for sklearn Pipelines)
        sample_size = min(80, len(X))
        sample_idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[sample_idx].reset_index(drop=True)
        df_sample = df.iloc[sample_idx].reset_index(drop=True)
        proba_sample = proba[sample_idx]

        shap_vals, expected_value, note = compute_shap_values(
            model, X_sample.values, list(X_sample.columns)
        )

        if shap_vals is None:
            st.error(note)
        else:
            if note:
                st.info(note)

            # Handle potentially mismatched lengths
            if len(shap_vals) < len(X_sample):
                X_sample = X_sample.iloc[:len(shap_vals)].reset_index(drop=True)
                df_sample = df_sample.iloc[:len(shap_vals)].reset_index(drop=True)
                proba_sample = proba_sample[:len(shap_vals)]

            # --- Global feature importance ---
            st.markdown("### Global Feature Importance (mean |SHAP|)")
            global_imp = pd.DataFrame({
                "feature": list(X_sample.columns),
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0).flatten(),
            }).sort_values("mean_abs_shap", ascending=True)

            fig_gi = px.bar(global_imp, x="mean_abs_shap", y="feature", orientation="h",
                           color="mean_abs_shap", color_continuous_scale="Reds",
                           title=f"Mean |SHAP| across {len(shap_vals)} sampled assets")
            fig_gi.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_gi, use_container_width=True)

            st.divider()

            # --- Per-asset waterfall ---
            st.markdown("### Per-Asset Waterfall — Why is this asset flagged?")

            df_sample_with_risk = df_sample.copy()
            df_sample_with_risk["risk_score"] = proba_sample
            df_sample_with_risk["asset_label"] = df_sample_with_risk.apply(
                lambda r: f"{r.get('equipment_id', '?')} @ {r.get('site', '?')} "
                          f"({r.get('date').date() if pd.notna(r.get('date')) else '?'}) "
                          f"— risk {r['risk_score']:.3f}",
                axis=1,
            )
            top_for_select = (
                df_sample_with_risk.sort_values("risk_score", ascending=False)
                .head(30)
                .reset_index()
            )

            selected_label = st.selectbox(
                "Pick a high-risk asset/date to explain:",
                options=top_for_select["asset_label"].tolist(),
                index=0,
            )
            selected_row = top_for_select[top_for_select["asset_label"] == selected_label].iloc[0]
            selected_sample_idx = int(selected_row["index"])

            asset_shap = np.asarray(shap_vals[selected_sample_idx]).flatten()
            asset_features = X_sample.iloc[selected_sample_idx]

            waterfall_df = pd.DataFrame({
                "feature": list(X_sample.columns),
                "value": asset_features.values,
                "shap": asset_shap,
            }).sort_values("shap", key=abs, ascending=True)

            fig_wf = go.Figure(go.Bar(
                x=waterfall_df["shap"],
                y=[f"{f} = {v:.3f}" for f, v in zip(waterfall_df["feature"], waterfall_df["value"])],
                orientation="h",
                marker_color=["#e74c3c" if v > 0 else "#3498db" for v in waterfall_df["shap"]],
                text=[f"{v:+.4f}" for v in waterfall_df["shap"]],
                textposition="outside",
            ))
            fig_wf.update_layout(
                title=f"SHAP contributions for selected asset (base value: {expected_value:.4f}) — red increases risk, blue reduces",
                xaxis_title="SHAP value (impact on risk probability)",
                height=400,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            shap_sum = expected_value + asset_shap.sum()
            actual_proba = float(proba_sample[selected_sample_idx])
            st.caption(
                f"SHAP reconstruction: base ({expected_value:.4f}) + sum(SHAP) ({asset_shap.sum():+.4f}) "
                f"= {shap_sum:.4f}. Model's actual risk score for this asset: {actual_proba:.4f}."
            )