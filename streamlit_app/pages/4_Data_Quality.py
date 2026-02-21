from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Quality", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "data" / "processed" / "daily_equipment_summary.parquet"

st.title("Data Quality")
st.caption("Monitor missingness, row volume, and basic range checks for the mining KPI dataset.")

# -------------------------------------------------
# Load data
# -------------------------------------------------
if not DAILY_PATH.exists():
    st.error("Missing daily summary. Run pipeline first:")
    st.code(
        "python -m src.miningops.generate_data\n"
        "python -m src.miningops.kpis"
    )
    st.stop()

df = pd.read_parquet(DAILY_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# -------------------------------------------------
# Top KPI cards (polish)
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

# ✅ Fix axis range to look sane (0 → 1)
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


# sensible domain checks
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