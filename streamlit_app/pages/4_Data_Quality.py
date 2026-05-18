from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Data Quality", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("Data Quality")
st.caption("Monitor missingness, row volume, basic range checks, and distribution drift for the mining KPI dataset.")

# -------------------------------------------------
# Load data
# -------------------------------------------------
if not DAILY_PATH.exists():
    st.error("Missing daily summary (pipeline output).")
    st.markdown("➡️ **If you are on Streamlit Cloud:** go to **Home** and click **Generate demo data now**.")
    st.markdown("**If running locally:** run pipeline first:")
    st.code(
        "python -m src.miningops.generate_data\n"
        "python -m src.miningops.kpis",
        language="text",
    )
    st.markdown("**Expected file:**")
    st.code(str(DAILY_PATH), language="text")
    st.stop()

df = pd.read_parquet(DAILY_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -------------------------------------------------
# Top KPI cards
# -------------------------------------------------
total_rows = int(len(df))
total_cols = int(df.shape[1])
missing_cells = int(df.isna().sum().sum())
missing_pct = (missing_cells / (df.shape[0] * df.shape[1])) if total_rows > 0 else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{total_rows:,}")
c2.metric("Columns", f"{total_cols:,}")
c3.metric("Missing cells", f"{missing_pct:.2%}")

st.divider()

# ============================================================
# DRIFT DETECTION (Tier 3 — NEW)
# ============================================================
st.subheader("📉 Distribution Drift Detection (Kolmogorov-Smirnov test)")
st.caption(
    "Compares feature distributions between a **reference window** (earlier dates) and a **current window** (later dates). "
    "KS test detects shifts in mean, variance, or shape. **p < 0.05 → statistically significant drift** — investigate "
    "before trusting model predictions on the newer data, since the model was fit on the reference period's behavior."
)

if "date" not in df.columns or df["date"].isna().all():
    st.info("Drift detection requires a date column with valid timestamps.")
else:
    dates_sorted = df["date"].dropna().sort_values()
    if len(dates_sorted) < 10:
        st.info("Not enough date coverage for drift detection (need ≥10 rows with valid dates).")
    else:
        min_date = dates_sorted.min().to_pydatetime()
        max_date = dates_sorted.max().to_pydatetime()
        default_split = dates_sorted.iloc[len(dates_sorted) // 2].to_pydatetime()

        split_date = st.slider(
            "Drift split point (rows before this date = reference; rows on/after = current):",
            min_value=min_date,
            max_value=max_date,
            value=default_split,
            help="Default = median date. Slide left/right to test different baselines.",
        )

        ref = df[df["date"] < pd.to_datetime(split_date)]
        cur = df[df["date"] >= pd.to_datetime(split_date)]

        candidate_features = [
            c for c in ["avg_temp", "avg_vib", "avg_fuel", "utilization_rate",
                        "downtime_rate", "work_orders", "downtime_minutes"]
            if c in df.columns
        ]

        if not candidate_features:
            st.warning("No numeric features available for drift testing.")
        elif len(ref) < 5 or len(cur) < 5:
            st.warning(f"Each window needs ≥5 rows. Current split: reference={len(ref)}, current={len(cur)}.")
        else:
            wc1, wc2 = st.columns(2)
            wc1.metric(
                "Reference window",
                f"{len(ref):,} rows",
                help=f"{ref['date'].min().date()} → {(pd.to_datetime(split_date) - pd.Timedelta(days=1)).date()}",
            )
            wc2.metric(
                "Current window",
                f"{len(cur):,} rows",
                help=f"{pd.to_datetime(split_date).date()} → {cur['date'].max().date()}",
            )

            drift_results = []
            for feat in candidate_features:
                ref_vals = ref[feat].dropna().values
                cur_vals = cur[feat].dropna().values
                if len(ref_vals) < 5 or len(cur_vals) < 5:
                    continue
                ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
                ref_mean = float(np.mean(ref_vals))
                cur_mean = float(np.mean(cur_vals))
                pct_change = (cur_mean - ref_mean) / (abs(ref_mean) + 1e-10) * 100
                drift_results.append({
                    "feature": feat,
                    "ks_statistic": float(ks_stat),
                    "p_value": float(p_val),
                    "ref_mean": ref_mean,
                    "cur_mean": cur_mean,
                    "pct_change": float(pct_change),
                    "drift_flagged": p_val < 0.05,
                })

            if not drift_results:
                st.warning("Insufficient data per feature for KS testing.")
            else:
                drift_df = pd.DataFrame(drift_results).sort_values("ks_statistic", ascending=False)
                drifted_count = int(drift_df["drift_flagged"].sum())

                if drifted_count > 0:
                    st.error(
                        f"⚠️ Drift flagged on **{drifted_count} of {len(drift_df)}** features (p < 0.05). "
                        "See table below and drill into individual features."
                    )
                else:
                    st.success(
                        f"✅ All {len(drift_df)} features stable (no significant drift, p ≥ 0.05 across the board)."
                    )

                display_df = drift_df.copy()
                display_df["status"] = display_df["drift_flagged"].apply(
                    lambda x: "⚠️ Drift" if x else "✅ Stable"
                )
                display_df["ks_statistic"] = display_df["ks_statistic"].apply(lambda x: f"{x:.4f}")
                display_df["p_value"] = display_df["p_value"].apply(lambda x: f"{x:.4g}")
                display_df["ref_mean"] = display_df["ref_mean"].apply(lambda x: f"{x:.3f}")
                display_df["cur_mean"] = display_df["cur_mean"].apply(lambda x: f"{x:.3f}")
                display_df["pct_change"] = display_df["pct_change"].apply(lambda x: f"{x:+.2f}%")
                display_df = display_df[[
                    "feature", "status", "ks_statistic", "p_value",
                    "ref_mean", "cur_mean", "pct_change"
                ]]
                display_df.columns = [
                    "Feature", "Status", "KS stat", "p-value",
                    "Reference mean", "Current mean", "Mean change"
                ]
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.markdown("**Drill into a feature: overlay reference vs current distributions**")
                selected_feat = st.selectbox(
                    "Feature:",
                    options=drift_df["feature"].tolist(),
                    index=0,
                )

                ref_vals = ref[selected_feat].dropna().values
                cur_vals = cur[selected_feat].dropna().values

                overlay_df = pd.DataFrame({
                    "value": np.concatenate([ref_vals, cur_vals]),
                    "window": ["Reference"] * len(ref_vals) + ["Current"] * len(cur_vals),
                })

                fig_overlay = px.histogram(
                    overlay_df,
                    x="value",
                    color="window",
                    nbins=40,
                    barmode="overlay",
                    opacity=0.6,
                    color_discrete_map={"Reference": "#3498db", "Current": "#e74c3c"},
                    title=f"Distribution overlay — {selected_feat}",
                )
                fig_overlay.update_layout(height=400, xaxis_title=selected_feat, yaxis_title="Count")
                st.plotly_chart(fig_overlay, use_container_width=True)

                row = drift_df[drift_df["feature"] == selected_feat].iloc[0]
                sf1, sf2, sf3, sf4 = st.columns(4)
                sf1.metric("KS statistic", f"{row['ks_statistic']:.4f}")
                sf2.metric("p-value", f"{row['p_value']:.4g}")
                sf3.metric("Reference mean", f"{row['ref_mean']:.3f}")
                sf4.metric(
                    "Current mean",
                    f"{row['cur_mean']:.3f}",
                    delta=f"{row['pct_change']:+.2f}%",
                )

st.divider()

# -------------------------------------------------
# Missingness by column
# -------------------------------------------------
st.subheader("Missingness by column")

miss = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
miss.columns = ["column", "missing_rate"]

fig_miss = px.bar(
    miss,
    x="missing_rate",
    y="column",
    orientation="h",
    title="Missing value rate by column",
)
fig_miss.update_xaxes(range=[0, 1])
st.plotly_chart(fig_miss, use_container_width=True)

st.divider()

# -------------------------------------------------
# Row volume over time
# -------------------------------------------------
st.subheader("Row volume over time")

if "date" in df.columns:
    vol = df.groupby("date").size().reset_index(name="rows")
    fig_vol = px.line(
        vol,
        x="date",
        y="rows",
        title="Daily row volume",
    )
    st.plotly_chart(fig_vol, use_container_width=True)
else:
    st.info("No date column available for row volume chart.")

st.divider()

# -------------------------------------------------
# Basic range checks
# -------------------------------------------------
st.subheader("Basic range checks")

checks = []


def add_check(name, series, low=None, high=None):
    if name not in df.columns:
        return
    s = df[name]
    bad = 0
    if low is not None:
        bad += int((s < low).sum())
    if high is not None:
        bad += int((s > high).sum())
    checks.append(
        {
            "column": name,
            "min": float(np.nanmin(s)) if len(s) else np.nan,
            "max": float(np.nanmax(s)) if len(s) else np.nan,
            "out_of_range_rows": bad,
        }
    )


add_check("utilization_rate", df.get("utilization_rate"), 0, 1)
add_check("downtime_rate", df.get("downtime_rate"), 0, 1)
add_check("avg_temp", df.get("avg_temp"), -50, 200)
add_check("avg_vib", df.get("avg_vib"), 0, None)

if checks:
    st.dataframe(pd.DataFrame(checks), use_container_width=True)
else:
    st.info("No numeric columns available for range checks.")

st.divider()

st.success("✅ Data quality checks complete.")