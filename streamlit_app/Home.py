from __future__ import annotations

import streamlit as st

# ✅ IMPORTANT: local import (Streamlit runs with streamlit_app/ as sys.path)
from bootstrap import missing_artifacts, generate_all_pipeline_outputs

st.set_page_config(page_title="Mining Ops Analytics", layout="wide")

st.title("Mining Operations Analytics Platform")
st.caption("Predictive Maintenance MVP — KPIs • Operations • Model • Data Quality")

# --- Demo bootstrap (Cloud-safe) ---
missing = missing_artifacts()

if missing:
    st.warning("Pipeline outputs are missing in this Streamlit Cloud environment.")
    st.markdown(
        "Click the button below to generate demo pipeline outputs inside the cloud container."
    )

    with st.expander("Missing files (click to view)", expanded=False):
        st.code("\n".join(str(p) for p in missing), language="text")

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
else:
    st.success("Pipeline outputs detected ✅ Dashboards are ready.")

# --- Your original content (unchanged) ---
st.markdown("""
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