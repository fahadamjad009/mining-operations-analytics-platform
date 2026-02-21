import streamlit as st

st.set_page_config(page_title="Mining Ops Analytics", layout="wide")

st.title("Mining Operations Analytics Platform")
st.caption("Predictive Maintenance MVP — KPIs • Operations • Model • Data Quality")

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