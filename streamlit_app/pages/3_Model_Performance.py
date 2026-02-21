from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="Model Performance", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "processed" / "models" / "downtime_risk_model.joblib"
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("Model Performance")
st.caption("Evaluate downtime-risk classifier with ROC/AUC, thresholding, confusion matrix, and top alerts.")

missing = [p for p in [MODEL_PATH, DAILY_PATH] if not p.exists()]
if missing:
    st.error("Missing model or daily summary outputs. Run:")
    st.code(
        "python -m src.miningops.generate_data\n"
        "python -m src.miningops.kpis\n"
        "python -m src.miningops.train"
    )
    st.write("Missing files:")
    st.code("\n".join(str(p) for p in missing))
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_parquet(DAILY_PATH)
df["date"] = pd.to_datetime(df.get("date"), errors="coerce")


def _get_expected_feature_names(m):
    """
    Try to discover which columns the model was trained on.
    Works for sklearn estimators and Pipelines (feature_names_in_).
    """
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)

    # Pipeline: search steps for feature_names_in_
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


# ---- Label (y) ----
if "high_sev" not in df.columns:
    st.error("Column 'high_sev' not found in daily summary. Check src.miningops.kpis output schema.")
    st.stop()

# strict binary {0,1}
y = (
    pd.Series(df["high_sev"])
    .fillna(0)
    .astype(int)
    .clip(0, 1)
    .values
)
unique_y = np.unique(y)

if unique_y.size < 2:
    st.warning(
        f"Only one class present in y (high_sev): {unique_y.tolist()}. "
        "ROC/AUC cannot be computed. The rest of the dashboard will still work."
    )

# ---- Features (X) must match what the model saw at fit time ----
expected = _get_expected_feature_names(model)

if expected is None:
    st.warning(
        "Could not detect feature_names_in_ from the saved model. "
        "Falling back to a default feature list (may fail)."
    )
    expected = [
        "avg_temp",
        "avg_vib",
        "avg_fuel",
        "utilization_rate",
        "downtime_rate",
        "work_orders",
    ]

missing_cols = [c for c in expected if c not in df.columns]
if missing_cols:
    st.error("This model expects columns that are missing from the daily summary:")
    st.code("\n".join(missing_cols))
    st.info(
        "Fix: ensure your training and KPI generation produce the same columns, "
        "then re-run: python -m src.miningops.kpis and python -m src.miningops.train"
    )
    st.stop()

X = df[expected].copy()

# Basic cleanup
for c in X.columns:
    if X[c].dtype == "bool":
        X[c] = X[c].astype(int)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Warn if leakage feature is present (but still allow it so the page runs)
if "high_sev" in expected:
    st.warning(
        "⚠️ Your saved model appears to have been trained with 'high_sev' as an INPUT feature. "
        "That is target leakage. The dashboard will run, but you should re-train the model "
        "without 'high_sev' in X for a realistic evaluation."
    )

# ---- Predict probabilities ----
if not hasattr(model, "predict_proba"):
    st.error("Loaded model does not support predict_proba(). Re-train with a probabilistic classifier.")
    st.stop()

proba_all = model.predict_proba(X)

# Expect binary classifier: proba_all should be (n,2)
if getattr(proba_all, "ndim", None) != 2 or proba_all.shape[1] < 2:
    st.error(
        f"Unexpected predict_proba output shape: {getattr(proba_all, 'shape', None)}. "
        "This dashboard expects a binary classifier with predict_proba -> (n, 2)."
    )
    st.stop()

# Positive-class probability
proba = proba_all[:, 1]

# ---- Risk score summary (helps pick thresholds) ----
p_min = _safe_float(np.min(proba))
p_max = _safe_float(np.max(proba))
p_mean = _safe_float(np.mean(proba))
p_p50 = _safe_float(np.percentile(proba, 50))
p_p90 = _safe_float(np.percentile(proba, 90))
p_p95 = _safe_float(np.percentile(proba, 95))
p_p99 = _safe_float(np.percentile(proba, 99))

prevalence = float(np.mean(y)) if len(y) else 0.0

# ---- ROC / AUC (safe) ----
auc_val = None
roc_df = None

if unique_y.size == 2:
    try:
        auc_val = float(roc_auc_score(y, proba))
        fpr, tpr, _ = roc_curve(y, proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    except Exception as e:
        st.warning(f"Could not compute ROC/AUC: {e}")

# Top metrics row (includes risk range)
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC AUC", f"{auc_val:.3f}" if auc_val is not None else "—")
m2.metric("Positives (high_sev=1)", f"{int(y.sum()):,}")
m3.metric("Total rows", f"{len(y):,}")
m4.metric("Risk score range", f"{p_min:.3f} → {p_max:.3f}")

# Second context row (optional but useful)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Risk mean", f"{p_mean:.3f}")
s2.metric("p90 / p95", f"{p_p90:.3f} / {p_p95:.3f}")
s3.metric("p99", f"{p_p99:.3f}")
s4.metric("Label prevalence", f"{prevalence:.3%}")

with st.expander("Risk score distribution", expanded=False):
    hist_df = pd.DataFrame({"risk_score": proba})
    fig_hist = px.histogram(hist_df, x="risk_score", nbins=40, title="Risk score histogram")
    fig_hist.update_xaxes(title="Predicted probability (risk_score)")
    fig_hist.update_yaxes(title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        "Tip: probabilities are compressed, so thresholds like 0.50 may yield zero alerts. "
        "Try around p90/p95/p99 depending on how many alerts you want."
    )

st.divider()

left, right = st.columns([2, 1])

with left:
    if roc_df is not None:
        fig_roc = px.line(roc_df, x="fpr", y="tpr", title="ROC curve (binary)")
        fig_roc.update_xaxes(title="False Positive Rate")
        fig_roc.update_yaxes(title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ROC curve not available (needs both classes in y and a valid probability score).")

with right:
    st.subheader("Decision threshold")

    # Model probabilities are compressed; pick a smart default and cap slider appropriately.
    # Use p95 as a good starting point; clamp into a sane range for UI.
    slider_min = 0.05
    slider_max = float(np.clip(max(p_max, 0.20), 0.20, 0.95))  # ensure max isn't below min-ish
    default_thr = float(np.clip(p_p95, slider_min, min(0.50, slider_max)))

    thr = st.slider(
        "Threshold",
        min_value=float(slider_min),
        max_value=float(slider_max),
        value=float(default_thr),
        step=0.01,
        help=f"Risk mean={p_mean:.3f}, max={p_max:.3f}. Threshold above max produces zero alerts.",
    )

    yhat = (proba >= thr).astype(int)

    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, yhat, average="binary", zero_division=0
    )

    st.metric("Predicted positives", f"{int(yhat.sum()):,}")
    st.metric("Precision", f"{prec:.3f}")
    st.metric("Recall", f"{rec:.3f}")
    st.metric("F1", f"{f1:.3f}")

    st.write("Confusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    st.dataframe(cm_df, use_container_width=True)

st.divider()

# ---- Top alerts table ----
st.subheader("Top predicted downtime-risk alerts")

base_cols = ["date", "site", "equipment_id", "equipment_type"]
optional_cols = [c for c in ["downtime_minutes", "work_orders"] if c in df.columns]
show_cols = [c for c in base_cols if c in df.columns] + optional_cols

alerts = df[show_cols].copy()
alerts["risk_score"] = proba
alerts["predicted_high_risk"] = (proba >= thr).astype(int)

top_alerts = alerts.sort_values("risk_score", ascending=False).head(30)

st.dataframe(top_alerts, use_container_width=True, hide_index=True)

st.info(
    "Tip: Move the threshold slider to control alert volume. "
    "Higher threshold = fewer alerts (higher precision). Lower threshold = more alerts (higher recall). "
    "If you set threshold above the model’s max score, you’ll get zero alerts."
)