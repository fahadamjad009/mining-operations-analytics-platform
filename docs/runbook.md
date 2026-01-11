Runbook - Mining Operations Analytics Platform



This runbook describes how to run the full pipeline from raw data to predictions.



--------------------------------

SETUP

--------------------------------

1\. Create and activate virtual environment

2\. Install dependencies from requirements.txt



--------------------------------

DATA GENERATION (MINING MODE)

--------------------------------

python -m src.miningops.generate\_data



--------------------------------

BUILD KPI TABLE

--------------------------------

python -m src.miningops.kpis



--------------------------------

TRAIN MODEL

--------------------------------

python -m src.miningops.train



--------------------------------

GENERATE KPI SNAPSHOT

--------------------------------

python -m src.miningops.kpi\_snapshot



--------------------------------

RUN API

--------------------------------

uvicorn src.miningops.api:app --reload



--------------------------------

HEALTH CHECK

--------------------------------

GET http://127.0.0.1:8000/health



--------------------------------

PREDICTION

--------------------------------

POST http://127.0.0.1:8000/score



