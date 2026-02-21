##Mining Operations Analytics Platform
#Predictive Maintenance Analytics & Downtime Risk Modeling

A portfolio-grade data engineering + analytics + machine learning system designed to simulate real-world mining operations and predictive maintenance workflows.

This project demonstrates how telemetry and maintenance logs can be transformed into analytics-ready KPIs and operational risk predictions — delivered through:

 Streamlit Executive Dashboard

 Downtime Risk ML Model

 Model Evaluation & Monitoring

 Static HTML Report (GitHub Pages)

 FastAPI Scoring Endpoint

 Live Access
 Interactive HTML Report (GitHub Pages)

https://fahadamjad009.github.io/mining-operations-analytics-platform/

 Streamlit Dashboard (Cloud Deployment)

👉 Deployable via Streamlit Community Cloud
Main file:

streamlit_app/Home.py

Once deployed, add your public URL here:

https://your-app-name.streamlit.app

GitHub Pages hosts the static report.
Streamlit must be deployed on Streamlit Cloud to be clickable without running locally.

🎯 Project Objective

Simulate an end-to-end mining analytics pipeline:

Generate telemetry + maintenance data

Engineer daily KPI summaries

Train a downtime-risk classifier

Deliver operational dashboards

Provide scoring API endpoint

Publish static executive report

This mirrors a real predictive maintenance system in heavy industry.

🏗 Architecture
Raw Telemetry + Maintenance Logs
            ↓
Data Validation & Cleaning
            ↓
Daily KPI Table (Analytics-Ready)
            ↓
Feature Engineering
            ↓
Downtime Risk Model (Logistic Regression)
            ↓
├── Streamlit Dashboard
├── FastAPI Scoring Endpoint
└── Static HTML Report (GitHub Pages)
📁 Repository Structure
mining-operations-analytics-platform/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── daily_equipment_summary.parquet
│       ├── kpi_snapshot.csv
│       ├── top_risky_assets.csv
│       └── models/
│           └── downtime_risk_model.joblib
│
├── src/miningops/
│   ├── generate_data.py
│   ├── kpis.py
│   ├── train.py
│   ├── kpi_snapshot.py
│   ├── report.py
│   └── api.py
│
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── Executive_Dashboard.py
│       ├── Operations_Drilldown.py
│       ├── Model_Performance.py
│       └── Data_Quality.py
│
├── reports/
│   └── mining_ops_report.html
│
├── docs/
│   └── figures/
│
└── README.md
📊 Streamlit Dashboard
1️⃣ Executive Dashboard

Utilization KPI

Downtime KPI

Work order metrics

Top risky assets

2️⃣ Operations Drilldown

Site-level filtering

Equipment-level trend analysis

Risk distribution view

3️⃣ Model Performance

ROC Curve

AUC

Threshold tuning

Confusion matrix

Precision / Recall / F1

4️⃣ Data Quality Monitoring

Missingness dashboard

Row volume over time

Basic domain range checks

🤖 Machine Learning

Model: Logistic Regression (baseline)
Objective: Predict high downtime severity next day

Evaluation (example run)

ROC AUC: ~0.67

31 days simulated

100 assets

Mean risk score ≈ 0.18

Downtime rate ≈ 3%

Threshold control implemented inside dashboard for operational tuning.

📈 Static Executive Report

Automatically generated HTML report:

reports/mining_ops_report.html

Contains:

Utilization by site

Downtime trends

Top risky assets

NASA benchmark comparison

Model evaluation plots

Hosted via GitHub Pages.

⚡ API Endpoint

Run locally:

uvicorn src.miningops.api:app --reload

Provides:

JSON scoring endpoint

Risk prediction on new KPI inputs

Production-style inference example

🧰 Tech Stack

Python 3.11

Pandas / NumPy

PyArrow

Scikit-learn

Plotly

Streamlit

FastAPI

Uvicorn

🚀 Local Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Generate pipeline:

python -m src.miningops.generate_data
python -m src.miningops.kpis
python -m src.miningops.train
python -m src.miningops.kpi_snapshot
python -m src.miningops.report

Run dashboard:

streamlit run streamlit_app/Home.py
🏆 What This Project Signals

Production-style data pipeline thinking

KPI engineering for industrial systems

Model evaluation transparency

Dashboard delivery for stakeholders

End-to-end reproducibility

Clean documentation and architecture
