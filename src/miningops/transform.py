import pandas as pd

def build_daily_equipment_summary(telemetry: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    tele = telemetry.copy()
    tele["date"] = tele["ts"].dt.date

    daily = tele.groupby(["site","equipment_id","equipment_type","date"]).agg(
        points=("event","size"),
        running=("event", lambda s: (s=="running").sum()),
        idle=("event", lambda s: (s=="idle").sum()),
        down=("event", lambda s: (s=="down").sum()),
        avg_temp=("engine_temp_c","mean"),
        avg_vib=("vibration_mm_s","mean"),
        avg_fuel=("fuel_rate_lph","mean"),
    ).reset_index()

    daily["utilization_rate"] = daily["running"] / daily["points"]
    daily["downtime_rate"] = daily["down"] / daily["points"]

    m = maintenance.copy()
    m["date"] = m["ts"].dt.date
    md = m.groupby(["site","equipment_id","date"]).agg(
        work_orders=("work_order_id","nunique"),
        downtime_minutes=("downtime_minutes","sum"),
        high_sev=("severity", lambda s: (s=="high").sum()),
    ).reset_index()

    out = daily.merge(md, on=["site","equipment_id","date"], how="left")
    out["work_orders"] = out["work_orders"].fillna(0).astype(int)
    out["downtime_minutes"] = out["downtime_minutes"].fillna(0).astype(int)
    out["high_sev"] = out["high_sev"].fillna(0).astype(int)
    return out
