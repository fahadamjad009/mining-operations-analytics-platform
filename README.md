# Mining Operations Analytics Platform (Predictive Maintenance MVP)

Portfolio-grade data engineering + analytics + ML project focused on mining operations.
This repo simulates equipment telemetry and maintenance logs, builds an analytics-ready daily KPI table, and trains a baseline model to predict **high downtime risk** for the next day.

## What this demonstrates

* **Data Engineering:** ingestion, validation, transformations, analytics-ready tables (medallion-style)
* **Analytics:** utilization, downtime rate, work-order volume, top risky assets
* **Machine Learning:** baseline downtime risk model with evaluation and an API scoring endpoint
* **Delivery:** reproducible runs, clear outputs, documentation structure

## Architecture (MVP)

Raw data → Validation → Daily KPI table → Feature engineering → Model training → API scoring

* data/raw/telemetry.parquet
* data/raw/maintenance.parquet
* data/processed/daily\_equipment\_summary.parquet
* data/processed/kpi\_snapshot.csv
* data/processed/top\_risky\_assets.csv
* data/processed/models/downtime\_risk\_model.joblib

## Tech stack

Python, Pandas, NumPy, PyArrow, Pydantic, Scikit-learn, FastAPI, Uvicorn

## Quickstart (Windows / PowerShell)

From repo root:

`powershell
python -m venv .venv
..venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

@"

# Mining Operations Analytics Platform (Predictive Maintenance MVP)

Portfolio-grade data engineering + analytics + ML project focused on mining operations.
This repo simulates equipment telemetry and maintenance logs, builds an analytics-ready daily KPI table, and trains a baseline model to predict **high downtime risk** for the next day.

## What this demonstrates

* **Data Engineering:** ingestion, validation, transformations, analytics-ready tables (medallion-style)
* **Analytics:** utilization, downtime rate, work-order volume, top risky assets
* **Machine Learning:** baseline downtime risk model with evaluation and an API scoring endpoint
* **Delivery:** reproducible runs, clear outputs, documentation structure

## Architecture (MVP)

Raw data → Validation → Daily KPI table → Feature engineering → Model training → API scoring

* data/raw/telemetry.parquet
* data/raw/maintenance.parquet
* data/processed/daily\_equipment\_summary.parquet
* data/processed/kpi\_snapshot.csv
* data/processed/top\_risky\_assets.csv
* data/processed/models/downtime\_risk\_model.joblib

## Tech stack

Python, Pandas, NumPy, PyArrow, Pydantic, Scikit-learn, FastAPI, Uvicorn

## Quickstart (Windows / PowerShell)

From repo root:

python -m venv .venv  
..venv\\Scripts\\Activate.ps1  
pip install -r requirements.txt

### Generate synthetic mining data

python -m src.miningops.generate\_data

### Build daily KPI table

python -m src.miningops.kpis

### Train downtime risk model

python -m src.miningops.train

### Generate executive KPI snapshot

python -m src.miningops.kpi\_snapshot

### Run API

uvicorn src.miningops.api:app --reload

## Example outputs

\## Proof of Work (Charts)



\### Mining Mode (Synthetic)

!\[Mining Utilization by Site](reports/figures/mining\_utilization\_by\_site.png)



!\[Mining Top 10 Risky Assets](reports/figures/mining\_top10\_risky\_assets.png)



\### NASA Benchmark Mode (C-MAPSS FD001)

!\[NASA ROC Curve (Fail within 30 cycles)](reports/figures/nasa\_roc\_curve.png)



\## Interactive Report

Open the full interactive HTML report locally:

\- reports/mining\_ops\_report.html



* Days covered: 31
* Assets covered: 100
* Avg utilization rate: ~0.88
* Avg downtime rate: ~0.03
* Baseline model ROC AUC: ~0.61

## License

MIT

