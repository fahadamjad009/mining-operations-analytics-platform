from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Operations Drilldown", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "models" / "downtime_risk_model.joblib"

st.title("Operations Drilldown")
st.caption("Filter by site / equipment type / asset and explore utilization, downtime, and risk patterns.")

if not DAILY_PATH.exists():
    st.error("Missing daily summary (pipeline output).")
    st.markdown("➡️ **If you are on Streamlit Cloud:** go to **Home** and click **Generate demo data now**.")
    st.markdown("**If running locally:** run:")
    st.code("python -m src.miningops.kpis", language="text")
    st.markdown("**Expected file:**")
    st.code(str(DAILY_PATH), language="text")
    st.stop()

df = pd.read_parquet(DAILY_PATH)
df["date"] = pd.to_datetime(df["date"])

# ---- Sidebar filters ----
st.sidebar.header("Filters")

sites = ["All"] + sorted(df["site"].dropna().unique().tolist())
types = ["All"] + sorted(df["equipment_type"].dropna().unique().tolist())

site = st.sidebar.selectbox("Site", sites, index=0)
etype = st.sidebar.selectbox("Equipment type", types, index=0)

filtered = df.copy()
if site != "All":
    filtered = filtered[filtered["site"] == site]
if etype != "All":
    filtered = filtered[filtered["equipment_type"] == etype]

assets = ["All"] + sorted(filtered["equipment_id"].dropna().unique().tolist())
asset = st.sidebar.selectbox("Asset", assets, index=0)

if asset != "All":
    filtered = filtered[filtered["equipment_id"] == asset]

min_date = filtered["date"].min()
max_date = filtered["date"].max()
date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
)
filtered = filtered[
    (filtered["date"] >= pd.to_datetime(date_range[0]))
    & (filtered["date"] <= pd.to_datetime(date_range[1]))
]

# ---- Summary row ----
c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(filtered):,}")
c2.metric("Assets", f"{filtered['equipment_id'].nunique():,}")
c3.metric("Avg utilization", f"{filtered['utilization_rate'].mean():.2%}")
c4.metric("Avg downtime", f"{filtered['downtime_rate'].mean():.2%}")

st.divider()

# ============================================================
# RISK HEATMAP (Tier 3 — NEW)
# ============================================================
st.subheader("🗺️ Risk Heatmap — where is downtime risk concentrated?")
st.caption(
    "Cells colored by **mean predicted risk score** (model probability of high-severity downtime). "
    "Uses the selected date range; **site/type sidebar filters are ignored here** to show the full picture across the fleet."
)

if not MODEL_PATH.exists():
    st.info(
        "Risk heatmap requires a trained model. "
        "Go to **Home → Generate demo data now** (cloud) or run `python -m src.miningops.train` (local)."
    )
else:
    @st.cache_resource(show_spinner=False)
    def _load_model(path):
        return joblib.load(path)

    def _expected_features(_model):
        if hasattr(_model, "feature_names_in_"):
            return list(_model.feature_names_in_)
        if hasattr(_model, "named_steps"):
            for step in reversed(list(_model.named_steps.values())):
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
        return ["avg_temp", "avg_vib", "avg_fuel", "utilization_rate", "downtime_rate", "work_orders"]

    @st.cache_data(show_spinner=False)
    def _compute_risk(df_in: pd.DataFrame, feature_cols: list):
        missing = [c for c in feature_cols if c not in df_in.columns]
        if missing:
            return None, missing
        X = df_in[feature_cols].copy()
        for c in X.columns:
            if X[c].dtype == "bool":
                X[c] = X[c].astype(int)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X, None

    try:
        model_obj = _load_model(MODEL_PATH)
        feature_cols = _expected_features(model_obj)

        # Date-filtered subset (ignore site/type filters)
        date_filtered = df[
            (df["date"] >= pd.to_datetime(date_range[0]))
            & (df["date"] <= pd.to_datetime(date_range[1]))
        ].copy()

        X_heat, missing_feats = _compute_risk(date_filtered, feature_cols)

        if X_heat is None:
            st.warning(f"Model expects features missing from data: {missing_feats}")
        elif len(date_filtered) == 0:
            st.info("No data in selected date range.")
        else:
            proba = model_obj.predict_proba(X_heat)[:, 1]
            date_filtered["risk_score"] = proba

            heatmap_data = (
                date_filtered.groupby(["site", "equipment_type"], as_index=False)
                .agg(
                    mean_risk=("risk_score", "mean"),
                    asset_count=("equipment_id", "nunique"),
                    rows=("date", "count"),
                )
            )

            pivot_risk = heatmap_data.pivot(index="site", columns="equipment_type", values="mean_risk")

            fig_heat = px.imshow(
                pivot_risk,
                color_continuous_scale="Reds",
                aspect="auto",
                text_auto=".3f",
                labels=dict(color="Mean risk"),
                title="Mean predicted risk score by site × equipment type",
            )
            fig_heat.update_layout(height=450, xaxis_title="Equipment type", yaxis_title="Site")
            fig_heat.update_xaxes(side="bottom")
            st.plotly_chart(fig_heat, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Top sites by mean risk**")
                site_risk = (
                    date_filtered.groupby("site")["risk_score"].mean()
                    .sort_values(ascending=False).head(5).reset_index()
                )
                site_risk.columns = ["Site", "Mean risk"]
                site_risk["Mean risk"] = site_risk["Mean risk"].round(4)
                st.dataframe(site_risk, use_container_width=True, hide_index=True)

            with col_b:
                st.markdown("**Top equipment types by mean risk**")
                etype_risk = (
                    date_filtered.groupby("equipment_type")["risk_score"].mean()
                    .sort_values(ascending=False).head(5).reset_index()
                )
                etype_risk.columns = ["Equipment type", "Mean risk"]
                etype_risk["Mean risk"] = etype_risk["Mean risk"].round(4)
                st.dataframe(etype_risk, use_container_width=True, hide_index=True)

            with col_c:
                st.markdown("**Hottest cells (site × type)**")
                hot_cells = (
                    heatmap_data.sort_values("mean_risk", ascending=False)
                    .head(5)[["site", "equipment_type", "mean_risk", "asset_count"]]
                    .copy()
                )
                hot_cells.columns = ["Site", "Type", "Mean risk", "Assets"]
                hot_cells["Mean risk"] = hot_cells["Mean risk"].round(4)
                st.dataframe(hot_cells, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Could not compute risk heatmap: {e}")

st.divider()

# ---- Trends (daily totals / means) ----
st.subheader("Trends over time")

trend = (
    filtered.groupby("date", as_index=False)
    .agg(
        utilization_rate=("utilization_rate", "mean"),
        downtime_rate=("downtime_rate", "mean"),
        downtime_minutes=("downtime_minutes", "sum"),
        work_orders=("work_orders", "sum"),
    )
    .sort_values("date")
)

left, right = st.columns(2)

with left:
    fig_u = px.line(
        trend,
        x="date",
        y="utilization_rate",
        markers=True,
        title="Utilization rate (avg)",
    )
    fig_u.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_u, use_container_width=True)

with right:
    fig_d = px.line(
        trend,
        x="date",
        y="downtime_rate",
        markers=True,
        title="Downtime rate (avg)",
    )
    fig_d.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_d, use_container_width=True)

st.divider()

# ---- Site comparison (if multiple sites remain) ----
st.subheader("Site comparison")
site_comp = (
    filtered.groupby("site", as_index=False)
    .agg(
        avg_util=("utilization_rate", "mean"),
        avg_down=("downtime_rate", "mean"),
        total_down_min=("downtime_minutes", "sum"),
        total_work_orders=("work_orders", "sum"),
        assets=("equipment_id", "nunique"),
    )
    .sort_values("total_down_min", ascending=False)
)

fig_site = px.bar(
    site_comp,
    x="site",
    y="total_down_min",
    hover_data=["assets", "avg_util", "avg_down", "total_work_orders"],
    title="Total downtime minutes by site",
)
st.plotly_chart(fig_site, use_container_width=True)

st.divider()

# ---- Top downtime assets ----
st.subheader("Top assets by downtime minutes (filtered)")
asset_table = (
    filtered.groupby(["site", "equipment_id", "equipment_type"], as_index=False)
    .agg(
        total_down_min=("downtime_minutes", "sum"),
        avg_risk=("high_sev", "mean"),
        avg_util=("utilization_rate", "mean"),
        days=("date", "nunique"),
    )
    .sort_values("total_down_min", ascending=False)
    .head(20)
)

st.dataframe(asset_table, use_container_width=True, hide_index=True)