import pandas as pd
from .config import settings
from .schemas import TelemetryRow, MaintenanceRow

def _validate_df(df: pd.DataFrame, model):
    required = set(model.model_fields.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # validate a sample (fast, practical)
    sample = df.sample(min(500, len(df)), random_state=settings.random_seed)
    for _, row in sample.iterrows():
        model(**row.to_dict())

def load_raw():
    tele = pd.read_parquet(settings.data_raw / "telemetry.parquet")
    maint = pd.read_parquet(settings.data_raw / "maintenance.parquet")

    _validate_df(tele, TelemetryRow)
    _validate_df(maint, MaintenanceRow)
    return tele, maint
