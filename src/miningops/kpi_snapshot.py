import pandas as pd
import joblib

from .config import settings
from .features import FEATURES

def main():
    settings.data_processed.mkdir(parents=True, exist_ok=True)
    daily_path = settings.data_processed / "daily_equipment_summary.parquet"
    model_path = settings.model_dir / "downtime_risk_model.joblib"

    daily = pd.read_parquet(daily_path)
    model = joblib.load(model_path)

    # Predict risk for each row (day-level)
    X = daily[FEATURES]
    daily["downtime_risk_probability"] = model.predict_proba(X)[:, 1]

    # Executive KPIs (overall)
    kpis = {
        "days_covered": int(daily["date"].nunique()),
        "assets_covered": int(daily["equipment_id"].nunique()),
        "avg_utilization_rate": float(daily["utilization_rate"].mean()),
        "avg_downtime_rate": float(daily["downtime_rate"].mean()),
        "total_downtime_minutes": int(daily["downtime_minutes"].sum()),
        "total_work_orders": int(daily["work_orders"].sum()),
    }

    # Top risky assets (average risk across days)
    top_assets = (
        daily.groupby(["site", "equipment_id", "equipment_type"])
        .agg(avg_risk=("downtime_risk_probability", "mean"),
             avg_downtime=("downtime_minutes", "mean"),
             days=("date", "nunique"))
        .reset_index()
        .sort_values(["avg_risk", "avg_downtime"], ascending=False)
        .head(10)
    )

    # Create a snapshot table (1-row KPI + top assets)
    kpi_df = pd.DataFrame([kpis])
    out_csv = settings.data_processed / "kpi_snapshot.csv"
    out_top_csv = settings.data_processed / "top_risky_assets.csv"

    kpi_df.to_csv(out_csv, index=False)
    top_assets.to_csv(out_top_csv, index=False)

    print("Saved:")
    print(" -", out_csv)
    print(" -", out_top_csv)
    print("\nKPI Snapshot:")
    print(kpi_df.to_string(index=False))
    print("\nTop 10 Risky Assets:")
    print(top_assets.to_string(index=False))

if __name__ == "__main__":
    main()
