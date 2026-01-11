import pandas as pd

FEATURES = [
    "avg_temp",
    "avg_vib",
    "avg_fuel",
    "utilization_rate",
    "downtime_rate",
    "work_orders",
    "high_sev",
]

def make_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["equipment_id", "date"]).copy()

    # label: high downtime the NEXT day (>= 120 minutes)
    df["downtime_next_day"] = df.groupby("equipment_id")["downtime_minutes"].shift(-1)
    df["label_high_downtime"] = (df["downtime_next_day"].fillna(0) >= 120).astype(int)

    # drop last day per equipment (no label available)
    df = df.dropna(subset=["downtime_next_day"])
    return df
