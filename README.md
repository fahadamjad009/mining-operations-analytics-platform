# Mining Operations Analytics Platform

Predictive Maintenance MVP for industrial mining equipment.  
This project demonstrates a production-style analytics workflow where telemetry and maintenance signals are transformed into operational KPIs, downtime-risk scoring, executive dashboards, and automated data quality monitoring — delivered via a cloud-deployed Streamlit application.

---

## **1. Live Demo**

**Streamlit App**  
https://fahadamjad009-mining-operations-analyt-streamlit-apphome-iuilko.streamlit.app

---

## **2. Screenshots**

### **2.1 Home**

![Home](docs/screenshots/home_page.png)

### **2.2 Executive Dashboard**

![Executive Dashboard](docs/screenshots/executive_dashboard.png)

### **2.3 Operations Drilldown**

![Operations Drilldown](docs/screenshots/operations_drilldown.png)

### **2.4 Model Performance**

![Model Performance](docs/screenshots/model_performance.png)

### **2.5 Data Quality**

![Data Quality](docs/screenshots/data_quality.png)

---

## **3. Objectives**

1. Convert equipment telemetry and maintenance activity into daily operational KPIs.  
2. Score equipment downtime risk using an evaluation-first ML approach.  
3. Provide executive and operational dashboards for decision support.  
4. Monitor dataset health through automated data quality checks.  
5. Deploy a cloud-safe Streamlit application resilient to cold starts.

---

## **4. Architecture**

### **4.1 High-Level Flow**


Telemetry + Maintenance (synthetic/demo)
│
▼
ETL / Feature Engineering
│
┌───────┴────────┐
▼ ▼
Daily KPI Table Model Training
(Parquet) (Scikit-learn)
│ │
▼ ▼
KPI Snapshot CSV Risk Model (joblib)
└────────────┬────────────┘
▼
Streamlit Dashboards
▼
Streamlit Cloud


---

### **4.2 Cloud-Safe Behavior**

Streamlit Cloud runs in a clean container.  
This application includes a bootstrap mechanism that:

- detects missing pipeline artifacts  
- generates demo artifacts inside the container when needed  
- prevents broken dashboards after redeploy or cold start  
- ensures the app always renders successfully  


---

## **5. Dashboard Pages**

### **5.1 Executive Dashboard**

- KPI cards (coverage, utilization, downtime)  
- downtime trend  
- top risky assets ranking  
- executive summary metrics  

---

### **5.2 Operations Drilldown**

- dynamic filters (site, equipment type, asset, date range)  
- utilization trend  
- downtime trend  
- site comparison  
- top downtime assets  

---

### **5.3 Model Performance**

- ROC / AUC evaluation  
- risk score distribution  
- threshold tuning  
- confusion matrix  
- precision / recall / F1  
- top predicted alerts  

---

### **5.4 Data Quality**

- missingness by column  
- row volume over time  
- range and sanity checks  
- dataset health indicators  

---

## **6. Tech Stack**

- Python 3.11  
- Streamlit  
- Pandas, NumPy  
- Scikit-learn, Joblib  
- Plotly  
- PyArrow (Parquet)  

---

## **7. Repository Structure**


mining-operations-analytics-platform/
├── streamlit_app/
│ ├── Home.py
│ ├── bootstrap.py
│ └── pages/
│ ├── 1_Executive_Dashboard.py
│ ├── 2_Operations_Drilldown.py
│ ├── 3_Model_Performance.py
│ └── 4_Data_Quality.py
├── src/
│ └── miningops/
├── data/
├── docs/
│ └── screenshots/
├── requirements.txt
└── runtime.txt


---

## **8. Running Locally**

### **8.1 Setup**

```bash
git clone https://github.com/fahadamjad009/mining-operations-analytics-platform.git
cd mining-operations-analytics-platform
pip install -r requirements.txt

### **8.2 Generate Demo Data**

python -m src.miningops.generate_data
python -m src.miningops.kpis
python -m src.miningops.train

### **8.3 Start Streamlit**

streamlit run streamlit_app/Home.py

## **9. Deployment Stability Notes**

Python version pinned via runtime.txt

Dependencies pinned in requirements.txt

Bootstrap prevents cold-start failures

Streamlit Cloud compatible

Production-style project structure

Stable baseline tag:
savepoint-streamlit-stable-2026-02-22

## **10. Future Enhancements**

real telemetry ingestion

model monitoring and drift detection

alert routing (email / ops tools)

multi-site scaling

experiment tracking

role-based dashboards
