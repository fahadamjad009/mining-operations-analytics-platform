from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


# ----------------------------
# Paths
# ----------------------------
def project_root() -> Path:
    # streamlit_app/bootstrap.py -> repo root is parent of streamlit_app
    return Path(__file__).resolve().parents[1]


def processed_dir() -> Path:
    return project_root() / "data" / "processed"


def models_dir() -> Path:
    return processed_dir() / "models"


def daily_summary_path() -> Path:
    return processed_dir() / "daily_equipment_summary.parquet"


def model_path() -> Path:
    return models_dir() / "downtime_risk_model.joblib"


def kpi_snapshot_path() -> Path:
    return processed_dir() / "kpi_snapshot.csv"


def top_risky_assets_path() -> Path:
    return processed_dir() / "top_risky_assets.csv"


def ensure_dirs() -> None:
    processed_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)


# ----------------------------
# Cloud-safe module runner
# ----------------------------
def run_module(module: str) -> None:
    """
    Run `python -m <module>` in repo root, capturing stdout/stderr for useful errors in Streamlit.
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [sys.executable, "-m", module]
    p = subprocess.run(
        cmd,
        cwd=str(project_root()),
        env=env,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        # Raise a helpful error including stdout/stderr
        msg = [
            f"Module failed: {module}",
            f"Command: {' '.join(cmd)}",
            "",
            "----- STDOUT -----",
            p.stdout or "(empty)",
            "",
            "----- STDERR -----",
            p.stderr or "(empty)",
        ]
        raise RuntimeError("\n".join(msg))


# ----------------------------
# What the dashboards need
# ----------------------------
def missing_artifacts() -> list[Path]:
    need = [
        daily_summary_path(),
        model_path(),
        kpi_snapshot_path(),
        top_risky_assets_path(),
    ]
    return [p for p in need if not p.exists()]


# ----------------------------
# Build the 2 CSVs ourselves (NO kpi_snapshot module)
# ----------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def generate_kpi_snapshot_and_risk_tables() -> None:
    """
    Generates:
      - data/processed/kpi_snapshot.csv
      - data/processed/top_risky_assets.csv

    Uses:
      - daily_equipment_summary.parquet
      - downtime_risk_model.joblib (if available)
    """
    ensure_dirs()

    ds = daily_summary_path()
    if not ds.exists():
        raise FileNotFoundError(f"Missing daily summary: {ds}")

    df = pd.read_parquet(ds)

    # Basic required cols (best-effort; don’t hard-crash if a column is missing)
    for col in ["site", "equipment_id", "equipment_type"]:
        if col not in df.columns:
            df[col] = "unknown"

    # Dates
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # ----------------------------
    # Risk score: prefer model predict_proba, else fallback heuristic
    # ----------------------------
    risk = None
    mp = model_path()

    if mp.exists():
        try:
            import joblib  # local import to avoid dependency issues if not installed
            model = joblib.load(mp)

            # Try to detect model feature columns
            expected = None
            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
            elif hasattr(model, "named_steps"):
                for step in reversed(list(model.named_steps.values())):
                    if hasattr(step, "feature_names_in_"):
                        expected = list(step.feature_names_in_)
                        break

            # Default feature set (only if model doesn’t expose names)
            if expected is None:
                expected = [
                    "avg_temp",
                    "avg_vib",
                    "avg_fuel",
                    "utilization_rate",
                    "downtime_rate",
                    "work_orders",
                ]

            # Build X with missing columns filled
            X = df.copy()
            for c in expected:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[expected].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if getattr(proba, "ndim", 0) == 2 and proba.shape[1] >= 2:
                    risk = proba[:, 1]
        except Exception:
            # Don’t crash the app; fallback below
            risk = None

    if risk is None:
        # Heuristic fallback (stable): scale downtime_rate into [0,1]
        if "downtime_rate" in df.columns:
            s = pd.to_numeric(df["downtime_rate"], errors="coerce").fillna(0.0)
            # clip into [0,1]
            s = s.clip(lower=0.0, upper=1.0)
            risk = s.values
        else:
            risk = pd.Series([0.0] * len(df)).values

    df["_risk_score"] = risk

    # ----------------------------
    # KPI Snapshot (single row)
    # ----------------------------
    days_covered = df["date"].nunique(dropna=True) if "date" in df.columns else 0
    assets_covered = df["equipment_id"].nunique(dropna=True)
    avg_util = _safe_float(pd.to_numeric(df.get("utilization_rate"), errors="coerce").mean(), 0.0)
    avg_down = _safe_float(pd.to_numeric(df.get("downtime_rate"), errors="coerce").mean(), 0.0)
    total_down_min = _safe_int(pd.to_numeric(df.get("downtime_minutes"), errors="coerce").fillna(0).sum(), 0)
    total_work_orders = _safe_int(pd.to_numeric(df.get("work_orders"), errors="coerce").fillna(0).sum(), 0)

    snap = pd.DataFrame([{
        "days_covered": int(days_covered),
        "assets_covered": int(assets_covered),
        "avg_utilization_rate": float(avg_util),
        "avg_downtime_rate": float(avg_down),
        "total_downtime_minutes": int(total_down_min),
        "total_work_orders": int(total_work_orders),
    }])
    snap.to_csv(kpi_snapshot_path(), index=False)

    # ----------------------------
    # Top risky assets table
    # ----------------------------
    # Summarize risk by asset across days
    grp_cols = ["site", "equipment_type", "equipment_id"]
    agg = {
        "_risk_score": "mean",
        "date": "nunique",
    }
    if "downtime_rate" in df.columns:
        agg["downtime_rate"] = "mean"
    if "downtime_minutes" in df.columns:
        agg["downtime_minutes"] = "mean"

    out = (
        df.groupby(grp_cols, as_index=False)
          .agg(agg)
          .rename(columns={
              "_risk_score": "avg_risk",
              "date": "days",
              "downtime_rate": "avg_downtime",
              "downtime_minutes": "avg_downtime_minutes",
          })
          .sort_values("avg_risk", ascending=False)
    )

    # Make sure expected columns exist for your page hover/table
    if "avg_downtime" not in out.columns:
        out["avg_downtime"] = 0.0
    if "days" not in out.columns:
        out["days"] = 0

    out.to_csv(top_risky_assets_path(), index=False)


# ----------------------------
# Main entry used by Home button
# ----------------------------
def generate_all_pipeline_outputs() -> None:
    """
    Cloud-safe generation:
      1) run generate_data
      2) run kpis (produces daily_equipment_summary.parquet)
      3) run train (produces model)
      4) generate the two CSVs (NO kpi_snapshot module)
    """
    ensure_dirs()

    # Step 1 + 2: Ensure daily summary exists
    if not daily_summary_path().exists():
        run_module("src.miningops.generate_data")
        run_module("src.miningops.kpis")

    # Step 3: Ensure model exists (optional but preferred)
    if not model_path().exists():
        # If train fails, we still allow dashboards using heuristic risk
        try:
            run_module("src.miningops.train")
        except Exception:
            pass

    # Step 4: Always generate the CSVs the Streamlit pages expect
    generate_kpi_snapshot_and_risk_tables()