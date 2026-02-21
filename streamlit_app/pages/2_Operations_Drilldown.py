from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Operations Drilldown", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("Operations Drilldown")
st.caption("Filter by site / equipment type / asset and explore utilization & downtime patterns.")

# ✅ UPDATED: Cloud-friendly guidance + keep local CLI guidance
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