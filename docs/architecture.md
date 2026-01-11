Mining Operations Analytics Platform - Architecture

Business Problem
Mining operations run heavy equipment such as haul trucks, drills and excavators.
Unplanned downtime is expensive and can create safety risks.
The goal of this system is to predict which assets are likely to experience high downtime one day in advance so maintenance teams can intervene.

Datasets (Two Modes)
This project supports two data modes:

1) Mining Story Mode (Synthetic)
- Synthetic telemetry + maintenance logs generated to resemble mining operations
- Used for mining-specific KPIs (utilization, downtime rate) and a simple downtime-risk baseline

2) Public Benchmark Mode (NASA C-MAPSS)
- Uses NASA's C-MAPSS turbofan engine degradation dataset (public benchmark)
- Used for credible predictive maintenance modelling and reporting
- Allows reproducible results without proprietary mining datasets

Pipeline Overview
Raw Data
-> Validation
-> Daily Aggregation (analytics-ready table)
-> Feature Engineering
-> Model Training + Evaluation
-> API for Scoring
-> Report generation (plots + HTML)

Storage Layers
Raw Layer:
- data/raw/telemetry.parquet
- data/raw/maintenance.parquet
- data/raw/nasa_cmapss/ (optional)

Processed Layer:
- data/processed/daily_equipment_summary.parquet
- data/processed/kpi_snapshot.csv
- data/processed/top_risky_assets.csv

Model Layer:
- data/processed/models/

API
FastAPI exposes:
- /health
- /score

Reporting (Proof of Work)
The project generates charts and an HTML report for GitHub visibility:
- KPI charts (utilization, downtime)
- Top risky assets table/chart
- ROC curve
- Feature importance / coefficients

Future Extensions
- Real-time streaming ingestion (Kafka)
- Dashboards (Power BI / Superset)
- Model monitoring + scheduled retraining
