import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Executive Dashboard", layout="wide")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.title("📊 Executive Dashboard")
st.caption("Daily KPIs · OEE · reliability metrics · downtime Pareto · cost impact — all derived from the pipeline outputs.")

# ---------- Helpers ----------
def read_csv_safe(p: Path):
    if not p.exists():
        return None
    return pd.read_csv(p)

def read_parquet_safe(p: Path):
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

if missing:
    st.error("Missing pipeline outputs.")
    st.markdown("➡️ **If you are on Streamlit Cloud:** go to **Home** and click **Generate demo data now**.")
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

# ============================================================
# TAB STRUCTURE
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏭 OEE & Reliability",
    "📉 Downtime Pareto",
    "💰 Cost Impact",
    "📄 Reports",
])

# ============================================================
# TAB 1: OVERVIEW (existing content preserved)
# ============================================================
with tab1:
    row = kpi.iloc[0].to_dict() if len(kpi) else {}
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Days covered", int(row.get("days_covered", 0)))
    c2.metric("Assets covered", int(row.get("assets_covered", 0)))
    c3.metric("Avg utilization", f"{float(row.get('avg_utilization_rate', 0.0)):.2%}")
    c4.metric("Avg downtime", f"{float(row.get('avg_downtime_rate', 0.0)):.2%}")
    c5.metric("Total downtime (min)", f"{int(row.get('total_downtime_minutes', 0)):,}")
    c6.metric("Total work orders", f"{int(row.get('total_work_orders', 0)):,}")

    st.divider()

    st.subheader("Downtime trend (daily)")
    trend = (
        daily.groupby("date", as_index=False)["downtime_minutes"]
        .sum()
        .sort_values("date")
    )
    fig_trend = px.line(trend, x="date", y="downtime_minutes", markers=True,
                        title="Total daily downtime across fleet")
    fig_trend.update_layout(height=380)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Top risky assets (avg risk)")
    left, right = st.columns([2, 1])
    with left:
        top_risky_sorted = top_risky.sort_values("avg_risk", ascending=False).head(15)
        fig_risk = px.bar(
            top_risky_sorted, x="avg_risk", y="equipment_id", orientation="h",
            hover_data=["site", "equipment_type", "avg_downtime", "days"],
        )
        fig_risk.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig_risk, use_container_width=True)
    with right:
        st.markdown("### Table (Top 10)")
        st.dataframe(
            top_risky.sort_values("avg_risk", ascending=False).head(10),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# TAB 2: OEE & RELIABILITY (NEW)
# ============================================================
with tab2:
    st.subheader("Overall Equipment Effectiveness (OEE)")
    st.caption(
        "OEE = Availability × Performance × Quality — the gold-standard industrial KPI. "
        "Quality is assumed 100% (no defect data in this benchmark)."
    )

    daily_oee = daily.copy()
    daily_oee["availability"] = 1 - daily_oee["downtime_rate"]
    daily_oee["performance"] = daily_oee["utilization_rate"]
    daily_oee["quality"] = 1.0
    daily_oee["oee"] = daily_oee["availability"] * daily_oee["performance"] * daily_oee["quality"]

    fleet_oee = daily_oee["oee"].mean()
    fleet_availability = daily_oee["availability"].mean()
    fleet_performance = daily_oee["performance"].mean()

    oee_c1, oee_c2, oee_c3, oee_c4 = st.columns(4)
    oee_c1.metric("Fleet OEE", f"{fleet_oee:.1%}",
                  delta=f"{(fleet_oee - 0.85)*100:+.1f} pts vs world-class",
                  delta_color="off")
    oee_c2.metric("Availability", f"{fleet_availability:.1%}")
    oee_c3.metric("Performance", f"{fleet_performance:.1%}")
    oee_c4.metric("Quality", "100.0%",
                  help="No defect/quality data in benchmark; assumed perfect.")

    if fleet_oee >= 0.85:
        st.success("🏆 **World-class OEE** (≥85%) — top decile in mining benchmarks.")
    elif fleet_oee >= 0.60:
        st.info("✅ **Typical OEE** (60-85%) — average industry performance.")
    else:
        st.warning("⚠️ **Below-average OEE** (<60%) — significant improvement opportunity.")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**OEE by Site**")
        site_oee = daily_oee.groupby("site").agg(
            oee=("oee", "mean"),
            availability=("availability", "mean"),
            performance=("performance", "mean"),
        ).reset_index().sort_values("oee", ascending=False)
        fig_site = px.bar(site_oee, x="site", y="oee", color="oee",
                          color_continuous_scale="RdYlGn",
                          hover_data=["availability", "performance"])
        fig_site.add_hline(y=0.85, line_dash="dash", line_color="gold",
                          annotation_text="World-class (85%)")
        fig_site.update_layout(yaxis_tickformat=".0%", height=380, showlegend=False)
        st.plotly_chart(fig_site, use_container_width=True)

    with col_b:
        st.markdown("**OEE by Equipment Type**")
        type_oee = daily_oee.groupby("equipment_type").agg(
            oee=("oee", "mean"),
            availability=("availability", "mean"),
            performance=("performance", "mean"),
        ).reset_index().sort_values("oee", ascending=False)
        fig_type = px.bar(type_oee, x="equipment_type", y="oee", color="oee",
                         color_continuous_scale="RdYlGn",
                         hover_data=["availability", "performance"])
        fig_type.add_hline(y=0.85, line_dash="dash", line_color="gold",
                          annotation_text="World-class (85%)")
        fig_type.update_layout(yaxis_tickformat=".0%", height=380, showlegend=False)
        st.plotly_chart(fig_type, use_container_width=True)

    st.divider()

    st.subheader("Reliability Metrics — MTBF & MTTR")
    st.caption(
        "**MTBF** (Mean Time Between Failures): higher = better reliability. "
        "**MTTR** (Mean Time To Repair): lower = faster recovery."
    )

    failures_per_asset = daily.groupby("equipment_id").agg(
        days_observed=("date", "nunique"),
        total_failures=("work_orders", "sum"),
        total_downtime=("downtime_minutes", "sum"),
        equipment_type=("equipment_type", "first"),
    ).reset_index()
    failures_per_asset["mtbf_days"] = (
        failures_per_asset["days_observed"]
        / failures_per_asset["total_failures"].replace(0, np.nan)
    )
    failures_per_asset["mttr_minutes"] = (
        failures_per_asset["total_downtime"]
        / failures_per_asset["total_failures"].replace(0, np.nan)
    )

    mtbf_by_type = failures_per_asset.groupby("equipment_type").agg(
        mtbf_days=("mtbf_days", "mean"),
        mttr_minutes=("mttr_minutes", "mean"),
    ).reset_index()

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("**MTBF by Equipment Type (days)**")
        fig_mtbf = px.bar(
            mtbf_by_type.sort_values("mtbf_days", ascending=False),
            x="equipment_type", y="mtbf_days",
            color="mtbf_days", color_continuous_scale="Greens",
            labels={"mtbf_days": "MTBF (days)", "equipment_type": "Type"},
        )
        fig_mtbf.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_mtbf, use_container_width=True)
    with m_col2:
        st.markdown("**MTTR by Equipment Type (minutes)**")
        fig_mttr = px.bar(
            mtbf_by_type.sort_values("mttr_minutes", ascending=True),
            x="equipment_type", y="mttr_minutes",
            color="mttr_minutes", color_continuous_scale="Reds",
            labels={"mttr_minutes": "MTTR (min)", "equipment_type": "Type"},
        )
        fig_mttr.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_mttr, use_container_width=True)

# ============================================================
# TAB 3: DOWNTIME PARETO (NEW)
# ============================================================
with tab3:
    st.subheader("Downtime Pareto — 80/20 Analysis")
    st.caption(
        "Which assets cause most of the downtime? Classic Pareto distribution: "
        "typically a small share of assets accounts for the majority of failures."
    )

    pareto = daily.groupby("equipment_id").agg(
        total_downtime=("downtime_minutes", "sum"),
        equipment_type=("equipment_type", "first"),
        site=("site", "first"),
    ).reset_index().sort_values("total_downtime", ascending=False)
    pareto["cumulative_pct"] = (
        pareto["total_downtime"].cumsum() / pareto["total_downtime"].sum() * 100
    )
    pareto["rank"] = range(1, len(pareto) + 1)

    pareto_80_idx = (pareto["cumulative_pct"] <= 80).sum()
    pareto_80_pct = pareto_80_idx / len(pareto) * 100 if len(pareto) else 0

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Total assets", len(pareto))
    p_col2.metric("Assets causing 80% of downtime", f"{pareto_80_idx}")
    p_col3.metric("That's only…", f"{pareto_80_pct:.1f}% of fleet")

    top_n = min(50, len(pareto))
    pareto_show = pareto.head(top_n).copy()

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=pareto_show["equipment_id"], y=pareto_show["total_downtime"],
        name="Downtime (min)", marker_color="#e74c3c",
        hovertext=pareto_show.apply(
            lambda r: f"{r['equipment_type']} @ {r['site']}", axis=1
        ),
    ))
    fig_pareto.add_trace(go.Scatter(
        x=pareto_show["equipment_id"], y=pareto_show["cumulative_pct"],
        name="Cumulative %", yaxis="y2", mode="lines+markers",
        line=dict(color="#3498db", width=3),
    ))
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="orange", yref="y2",
                         annotation_text="80% threshold",
                         annotation_position="top right")
    fig_pareto.update_layout(
        title=f"Top {top_n} Assets by Total Downtime",
        xaxis=dict(title="Asset", tickangle=-45),
        yaxis=dict(title="Downtime (minutes)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0, 105]),
        legend=dict(x=0.7, y=0.95),
        height=500,
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("**Top 20 downtime contributors**")
    st.dataframe(
        pareto.head(20)[
            ["rank", "equipment_id", "equipment_type", "site",
             "total_downtime", "cumulative_pct"]
        ],
        use_container_width=True, hide_index=True,
    )

# ============================================================
# TAB 4: COST IMPACT (NEW)
# ============================================================
with tab4:
    st.subheader("Cost Impact Analysis")
    st.caption(
        "Translate technical metrics into business impact. "
        "Adjust the hourly cost and model catch rate to model scenarios."
    )

    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        hourly_cost = st.slider(
            "Estimated cost per hour of downtime ($)",
            min_value=500, max_value=20000, value=5000, step=500,
            help="Industry estimates for mining equipment: $3k-$15k/hr depending on asset class.",
        )
    with cost_col2:
        catch_rate = st.slider(
            "Model failure-catch rate (%)",
            min_value=0, max_value=100, value=70, step=5,
            help="What % of failures the model would catch in advance, based on recall.",
        )

    total_downtime_minutes = daily["downtime_minutes"].sum()
    total_downtime_hours = total_downtime_minutes / 60
    total_cost = total_downtime_hours * hourly_cost

    days_observed = daily["date"].nunique()
    annualization_factor = 365 / days_observed if days_observed > 0 else 1
    annualized_cost = total_cost * annualization_factor
    avoidable_cost = annualized_cost * (catch_rate / 100)

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Observed downtime", f"{total_downtime_hours:,.0f} hrs")
    cc2.metric("Observed-period cost", f"${total_cost:,.0f}")
    cc3.metric("Annualized cost", f"${annualized_cost:,.0f}")
    cc4.metric(f"Avoidable @ {catch_rate}% catch",
               f"${avoidable_cost:,.0f}")

    st.divider()

    st.markdown("**Annualized cost by Equipment Type**")
    type_cost = daily.groupby("equipment_type").agg(
        downtime_hours=("downtime_minutes", lambda x: x.sum() / 60),
    ).reset_index()
    type_cost["annualized_cost"] = (
        type_cost["downtime_hours"] * hourly_cost * annualization_factor
    )
    type_cost = type_cost.sort_values("annualized_cost", ascending=False)

    fig_cost = px.bar(
        type_cost, x="equipment_type", y="annualized_cost",
        color="annualized_cost", color_continuous_scale="Reds",
        labels={"annualized_cost": "Annualized cost ($)", "equipment_type": "Type"},
    )
    fig_cost.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_cost, use_container_width=True)

    st.markdown("**Top 10 most-expensive assets (annualized)**")
    asset_cost = daily.groupby(
        ["equipment_id", "equipment_type", "site"]
    ).agg(
        downtime_hours=("downtime_minutes", lambda x: x.sum() / 60),
    ).reset_index()
    asset_cost["annualized_cost"] = (
        asset_cost["downtime_hours"] * hourly_cost * annualization_factor
    )
    asset_cost = asset_cost.sort_values("annualized_cost", ascending=False).head(10)
    asset_cost["annualized_cost"] = asset_cost["annualized_cost"].apply(lambda x: f"${x:,.0f}")
    asset_cost["downtime_hours"] = asset_cost["downtime_hours"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(asset_cost, use_container_width=True, hide_index=True)

# ============================================================
# TAB 5: REPORTS (existing)
# ============================================================
with tab5:
    st.subheader("Static HTML report (from pipeline)")
    if html_report_path.exists():
        st.write("Open the full HTML report locally:")
        st.code(str(html_report_path))
        st.write("Or open the generated figures folder:")
        st.code(str(FIGURES))
    else:
        st.warning("Report not found. Run: `python -m src.miningops.report`")