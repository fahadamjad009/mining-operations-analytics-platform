from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .config import settings

SITES = ["Pilbara_WA", "Hunter_NSW", "Bowen_QLD"]
EQUIPMENT = [
    ("haul_truck", 60),
    ("excavator", 20),
    ("drill", 20),
]

FAULTS = ["HYD_LEAK", "ENG_OVERHEAT", "ELEC_SENSOR", "BRAKE_WEAR", "TRANS_SLIP", "BEARING_FAIL"]

def _rng():
    return np.random.default_rng(settings.random_seed)

def generate(days: int = 30, freq_minutes: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = _rng()
    start = datetime.utcnow() - timedelta(days=days)
    timestamps = pd.date_range(start=start, end=datetime.utcnow(), freq=f"{freq_minutes}min", tz="UTC")

    # equipment registry
    eq_rows = []
    for eq_type, count in EQUIPMENT:
        for i in range(1, count + 1):
            eq_rows.append((eq_type, f"{eq_type[:2].upper()}-{i:04d}"))
    eq = pd.DataFrame(eq_rows, columns=["equipment_type", "equipment_id"])

    # telemetry base
    t = eq.merge(pd.DataFrame({"ts": timestamps}), how="cross")
    t["site"] = rng.choice(SITES, size=len(t), replace=True)
    t["shift"] = np.where(t["ts"].dt.hour.between(6, 17), "day", "night")

    base_temp = t["equipment_type"].map({"haul_truck": 85, "excavator": 78, "drill": 70}).astype(float)
    base_vib  = t["equipment_type"].map({"haul_truck": 12, "excavator": 10, "drill": 14}).astype(float)
    base_fuel = t["equipment_type"].map({"haul_truck": 220, "excavator": 160, "drill": 90}).astype(float)
    base_spd  = t["equipment_type"].map({"haul_truck": 35, "excavator": 5, "drill": 2}).astype(float)

    day_factor = np.where(t["shift"].eq("day"), 1.05, 0.98)
    temp = base_temp * day_factor + rng.normal(0, 4, size=len(t))
    vib = base_vib * day_factor + rng.normal(0, 2, size=len(t))
    fuel = base_fuel * day_factor + rng.normal(0, 15, size=len(t))
    spd = np.clip(base_spd * day_factor + rng.normal(0, 8, size=len(t)), 0, 120)

    # degradation drift per equipment
    eq_deg = (
        t[["equipment_id", "ts"]]
        .drop_duplicates()
        .sort_values(["equipment_id", "ts"])
    )
    eq_deg["deg"] = 0.0
    for eid in eq["equipment_id"].unique():
        mask = eq_deg["equipment_id"].eq(eid)
        drift = np.cumsum(rng.normal(0.0005, 0.001, size=mask.sum()))
        eq_deg.loc[mask, "deg"] = np.clip(drift + rng.uniform(0, 0.05), 0, 1.5)

    t = t.merge(eq_deg, on=["equipment_id", "ts"], how="left")

    t["engine_temp_c"] = np.clip(temp + (t["deg"] * 18), -20, 180)
    t["vibration_mm_s"] = np.clip(vib + (t["deg"] * 20), 0, 200)
    t["fuel_rate_lph"] = np.clip(fuel + (t["deg"] * 60), 0, 1000)
    t["speed_kmh"] = spd

    # down probability increases with degradation
    p_down = np.clip(0.002 + t["deg"] * 0.03, 0, 0.25)
    p_idle = np.clip(0.08 + (t["equipment_type"].eq("excavator").astype(int) * 0.06), 0, 0.35)
    u = rng.random(len(t))
    t["event"] = np.where(u < p_down, "down", np.where(u < (p_down + p_idle), "idle", "running"))

    telemetry = t[[
        "ts", "site", "equipment_id", "equipment_type", "shift",
        "engine_temp_c", "vibration_mm_s", "fuel_rate_lph", "speed_kmh", "event"
    ]].copy()

    # maintenance logs: subset of down events
    down = telemetry[telemetry["event"].eq("down")].sample(frac=0.25, random_state=settings.random_seed)
    maint = down[["ts", "site", "equipment_id"]].copy()
    maint["work_order_id"] = [f"WO-{i:07d}" for i in range(1, len(maint) + 1)]
    maint["fault_code"] = rng.choice(FAULTS, size=len(maint), replace=True)
    maint["severity"] = rng.choice(["low", "medium", "high"], p=[0.55, 0.32, 0.13], size=len(maint))
    base_dt = rng.integers(10, 180, size=len(maint))
    sev_boost = maint["severity"].map({"low": 1.0, "medium": 1.6, "high": 2.4}).astype(float)
    maint["downtime_minutes"] = np.clip((base_dt * sev_boost).astype(int), 0, 1440)

    return telemetry, maint

def main():
    settings.data_raw.mkdir(parents=True, exist_ok=True)
    telemetry, maint = generate(days=30, freq_minutes=10)

    telemetry_path = settings.data_raw / "telemetry.parquet"
    maint_path = settings.data_raw / "maintenance.parquet"
    telemetry.to_parquet(telemetry_path, index=False)
    maint.to_parquet(maint_path, index=False)

    print("Wrote:")
    print(" -", telemetry_path)
    print(" -", maint_path)
    print("Rows:", len(telemetry), "telemetry |", len(maint), "maintenance")

if __name__ == "__main__":
    main()
