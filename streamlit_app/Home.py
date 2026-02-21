from __future__ import annotations

import streamlit as st

# Bootstrap helpers (new)
from streamlit_app.bootstrap import missing_artifacts, generate_all_pipeline_outputs

st.set_page_config(page_title="Mining Ops Analytics", layout="wide")

st.title("Mining Operations Analytics Platform")
st.caption("Predictive Maintenance MVP — KPIs • Operations • Model • Data Quality")

# --- NEW: Cloud-safe pipeline artifact check + one-click demo bootstrap ---
missing = missing_artifacts()

if missing:
    st.warning("Pipeline outputs are missing in this Streamlit Cloud environment.")
    st.markdown(
        "To make the dashboards work here, generate the demo pipeline outputs inside the cloud container."
    )

    with st.expander("Missing files (click to view)", expanded=False):
        st.code("\n".join(str(p) for p in missing), language="text")

    cols = st.columns([1, 2, 1])
    with cols[0]:
        if st.button("Generate demo data now (1–2 mins)", type="primary"):
            with st.spinner("Generating pipeline outputs…"):
                try:
                    generate_all_pipeline_outputs()
                except Exception as e:
                    st.error("Demo data generation failed.")
                    st.exception(e)
                else:
                    st.success("Demo data generated. Reloading…")
                    st.rerun()

    st.info(
        "If this is running locally and you want to generate data manually, you can also run:\n\n"
        "`python -m src.miningops.generate_data`\n"
        "`python -m src.miningops.kpis`\n"
        "`python -m src.miningops.train`\n"
        "`python -m src.miningops.kpi_snapshot`"
    )
else:
    st.success("Pipeline outputs detected ✅ Dashboards are ready.")

# --- Your original content (kept intact) ---
st.markdown(
    """
### Pages
- **Executive Dashboard**: KPI cards + downtime trend + top risky assets  
- **Operations Drilldown**: filters, asset trends, site heatmap  
- **Model Performance**: ROC/PR, thresholding, confusion matrix  
- **Data Quality**: missingness, row volume, range checks  

Run locally:
```bash
streamlit run streamlit_app/Home.py


```

""")