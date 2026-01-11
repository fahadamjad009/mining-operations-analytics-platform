from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from .config import settings
from .features import FEATURES

app = FastAPI(
    title="Mining Operations – Downtime Risk API",
    version="1.0.0",
    description="Predicts probability of high downtime for mining equipment on the next day"
)

model = None

class ScoreRequest(BaseModel):
    avg_temp: float
    avg_vib: float
    avg_fuel: float
    utilization_rate: float
    downtime_rate: float
    work_orders: int
    high_sev: int

@app.on_event("startup")
def load_model():
    global model
    model_path = settings.model_dir / "downtime_risk_model.joblib"
    model = joblib.load(model_path)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(req: ScoreRequest):
    X = [[getattr(req, f) for f in FEATURES]]
    p = float(model.predict_proba(X)[0][1])
    return {
        "downtime_risk_probability": round(p, 4),
        "risk_level": "HIGH" if p >= 0.5 else "LOW"
    }
