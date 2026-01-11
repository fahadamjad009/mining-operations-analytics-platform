from .config import settings
from .ingest import load_raw
from .transform import build_daily_equipment_summary

def main():
    settings.data_processed.mkdir(parents=True, exist_ok=True)

    telemetry, maintenance = load_raw()
    daily = build_daily_equipment_summary(telemetry, maintenance)

    out_path = settings.data_processed / "daily_equipment_summary.parquet"
    daily.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(daily))
    print("Columns:", list(daily.columns))

if __name__ == "__main__":
    main()
