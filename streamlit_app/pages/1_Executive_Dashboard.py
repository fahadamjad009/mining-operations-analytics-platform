import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Executive Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[2]  # repo root (..../streamlit_app/pages -> repo)
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.title("Executive Dashboard")
st.caption("Daily KPIs, downtime trend, and top risky assets from the pipeline outputs.")

# ---------- Helpers ----------
def read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    return pd.read_csv(p)

def read_parquet_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        return None
    return pd.read_parquet(p)

# ---------- Load artifacts ----------
kpi_snapshot_path = PROCESSED / "kpi_snapshot.csv"
top_risky_path = PROCESSED / "top_risky_assets.csv"
daily_summary_path = PROCESSED / "daily_equipment_summary.parquet"
html_report_path = REPORTS / "mining_ops_report.html"

kpi = read_csv_safe(kpi_snapshot_path)
top_risky = read_csv_safe(top_risky_path)
daily = read_parquet_safe(daily_summary_path)

missing = []
for p in [kpi_snapshot_path, top_risky_path, daily_summary_path]:
    if not p.exists():
        missing.append(str(p))

# ✅ UPDATED: Cloud-friendly guidance + keep local CLI guidance
if missing:
    st.error("Missing pipeline outputs.")

    # Streamlit Cloud-friendly path
    st.markdown("➡️ **If you are on Streamlit Cloud:** go to **Home** and click **Generate demo data now**.")

    # Local dev path (your original instructions, kept)
    st.markdown("**If running locally:** run the pipeline:")
    st.code(
        "python -m src.miningops.generate_data\n"
        "python -m src.miningops.kpis\n"
        "python -m src.miningops.train\n"
        "python -m src.miningops.kpi_snapshot",
        language="text",
    )

    st.markdown("**Missing files:**")
    st.code("\n".join(missing), language="text")
    st.stop()

# ---------- KPI Cards ----------
row = kpi.iloc[0].to_dict() if len(kpi) else {}
c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Days covered", int(row.get("days_covered", 0)))
c2.metric("Assets covered", int(row.get("assets_covered", 0)))
c3.metric("Avg utilization", f"{float(row.get('avg_utilization_rate', 0.0)):.2%}")
c4.metric("Avg downtime", f"{float(row.get('avg_downtime_rate', 0.0)):.2%}")
c5.metric("Total downtime (min)", f"{int(row.get('total_downtime_minutes', 0)):,}")
c6.metric("Total work orders", f"{int(row.get('total_work_orders', 0)):,}")

st.divider()

# ---------- Downtime trend ----------
st.subheader("Downtime trend (daily)")
trend = (
    daily.groupby("date", as_index=False)["downtime_minutes"]
    .sum()
    .sort_values("date")
)

fig_trend = px.line(trend, x="date", y="downtime_minutes", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------- Top risky assets ----------
st.subheader("Top risky assets (avg risk)")
left, right = st.columns([2, 1])

with left:
    # Ensure sort
    top_risky_sorted = top_risky.sort_values("avg_risk", ascending=False).head(15)
    fig_risk = px.bar(
        top_risky_sorted,
        x="avg_risk",
        y="equipment_id",
        orientation="h",
        hover_data=["site", "equipment_type", "avg_downtime", "days"],
    )
    fig_risk.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_risk, use_container_width=True)

with right:
    st.markdown("### Table (Top 10)")
    st.dataframe(
        top_risky.sort_values("avg_risk", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ---------- Link to static HTML report ----------
st.subheader("Static HTML report (from pipeline)")
if html_report_path.exists():
    st.write("Open the full HTML report locally:")
    st.code(str(html_report_path))
    st.write("Or open the generated figures folder:")
    st.code(str(FIGURES))
else:
    st.warning("Report not found. Run: `python -m src.miningops.report`")