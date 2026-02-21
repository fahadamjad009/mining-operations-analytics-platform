from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    daily_summary: Path
    kpi_snapshot: Path
    top_risky_assets: Path
    model_path: Path


def project_root() -> Path:
    # streamlit_app/ is inside repo root
    return Path(__file__).resolve().parents[1]


def artifacts() -> ArtifactPaths:
    root = project_root()
    processed = root / "data" / "processed"
    return ArtifactPaths(
        root=root,
        daily_summary=processed / "daily_equipment_summary.parquet",
        kpi_snapshot=processed / "kpi_snapshot.csv",
        top_risky_assets=processed / "top_risky_assets.csv",
        model_path=processed / "models" / "downtime_risk_model.joblib",
    )


def missing_artifacts() -> List[Path]:
    a = artifacts()
    required = [a.daily_summary, a.kpi_snapshot, a.top_risky_assets, a.model_path]
    return [p for p in required if not p.exists()]


def run_module(module: str) -> None:
    # Runs: python -m <module> inside Streamlit container
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-m", module]
    subprocess.check_call(cmd, cwd=str(project_root()), env=env)


def generate_all_pipeline_outputs() -> None:
    """
    Order matters.
    Adjust module names below ONLY if your filenames differ.
    """
    run_module("src.miningops.generate_data")
    run_module("src.miningops.kpis")
    run_module("src.miningops.train")
    run_module("src.miningops.kpi_snapshot")