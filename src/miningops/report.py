from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, roc_curve

from .config import settings
from .features import FEATURES as MINING_FEATURES

REPORT_DIR = settings.project_root / "reports"
FIG_DIR = REPORT_DIR / "figures"


def _ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_expected_feature_names(model) -> list[str] | None:
    """
    Discover which columns the model was trained on (sklearn feature_names_in_),
    including when the model is a Pipeline.
    """
    # Estimator directly
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            return None

    # Pipeline: check steps in reverse order
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                try:
                    return list(step.feature_names_in_)
                except Exception:
                    return None

    return None


def _make_X_for_model(model, df: pd.DataFrame, fallback_features: list[str]) -> pd.DataFrame:
    """
    Build X that matches the trained model's expected feature names and order.
    Uses model.feature_names_in_ when available; otherwise uses fallback_features.
    Ensures bool->int and fills NaNs to keep sklearn happy.

    NOTE: We do NOT force-remove leakage columns from df here because the correct
    authority is the trained model. If the model expects a leakage feature, the
    report will still run (but training should be fixed separately).
    """
    expected = _get_expected_feature_names(model)
    if expected is None:
        expected = list(fallback_features)

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "Report: daily summary missing expected feature columns:\n"
            + "\n".join(missing)
            + "\n\nFix: re-run `python -m src.miningops.kpis` and `python -m src.miningops.train` "
            "so the KPI schema matches training."
        )

    X = df[expected].copy()

    # Basic cleanup for sklearn
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)

    X = X.fillna(0.0)
    return X


def _save_placeholder(fig_path: Path, message: str) -> None:
    """
    Save a small placeholder PNG instead of crashing the report when
    optional columns are missing.
    """
    plt.figure()
    plt.text(0.01, 0.5, message, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def mining_plots() -> list[str]:
    daily_path = settings.data_processed / "daily_equipment_summary.parquet"
    daily = pd.read_parquet(daily_path)

    # Ensure date is datetime for trend grouping/plotting
    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"])

    # 1) Utilization by site
    out1 = FIG_DIR / "mining_utilization_by_site.png"
    if "site" in daily.columns and "utilization_rate" in daily.columns:
        util_by_site = daily.groupby("site")["utilization_rate"].mean().sort_values(ascending=False)
        plt.figure()
        util_by_site.plot(kind="bar")
        plt.title("Mining - Average Utilization Rate by Site")
        plt.ylabel("Utilization rate")
        plt.tight_layout()
        plt.savefig(out1, dpi=160)
        plt.close()
    else:
        _save_placeholder(out1, "Missing columns for utilization_by_site plot")

    # 2) Downtime trend over time
    out2 = FIG_DIR / "mining_downtime_trend.png"
    if "date" in daily.columns and "downtime_rate" in daily.columns:
        trend = daily.groupby("date")["downtime_rate"].mean()
        plt.figure()
        trend.plot()
        plt.title("Mining - Average Downtime Rate Over Time")
        plt.ylabel("Downtime rate")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(out2, dpi=160)
        plt.close()
    else:
        _save_placeholder(out2, "Missing columns for downtime trend plot")

    # 3) Top risky assets (from mining model)
    model_path = settings.model_dir / "downtime_risk_model.joblib"
    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Report: downtime_risk_model.joblib does not support predict_proba().")

    # IMPORTANT: Build X to match model.feature_names_in_ (avoids 'high_sev' mismatch)
    X = _make_X_for_model(model, daily, fallback_features=list(MINING_FEATURES))

    # Predict risk and plot top assets
    out3 = FIG_DIR / "mining_top10_risky_assets.png"
    daily2 = daily.copy()
    daily2["risk"] = model.predict_proba(X)[:, 1]

    if "equipment_id" in daily2.columns:
        top_assets = (
            daily2.groupby("equipment_id")["risk"].mean()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        plt.figure()
        top_assets.plot(kind="barh")
        plt.title("Mining - Top 10 Risky Assets (Avg Predicted Risk)")
        plt.xlabel("Risk probability")
        plt.tight_layout()
        plt.savefig(out3, dpi=160)
        plt.close()
    else:
        _save_placeholder(out3, "Missing equipment_id for risky assets plot")

    return [out1.name, out2.name, out3.name]


def nasa_plots() -> list[str]:
    df = pd.read_parquet(settings.data_processed / "nasa_fd001_train.parquet")

    # 1) RUL distribution
    out1 = FIG_DIR / "nasa_rul_distribution.png"
    plt.figure()
    df["rul"].plot(kind="hist", bins=50)
    plt.title("NASA FD001 - RUL Distribution")
    plt.xlabel("Remaining Useful Life (cycles)")
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) ROC curve for failure within 30 cycles model
    model = joblib.load(settings.model_dir / "nasa_fd001_failure30_model.joblib")

    FAIL_WITHIN = 30
    df2 = df.copy()
    df2["label"] = (df2["rul"] <= FAIL_WITHIN).astype(int)

    features = [c for c in df2.columns if c.startswith("op_setting_") or c.startswith("sensor_")]
    X = df2[features]
    y = df2["label"]

    if not hasattr(model, "predict_proba"):
        raise ValueError("Report: nasa_fd001_failure30_model.joblib does not support predict_proba().")

    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    out2 = FIG_DIR / "nasa_roc_curve.png"
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("NASA FD001 - ROC Curve (Fail within 30 cycles)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()

    # 3) Top coefficients (logistic regression)
    out3 = FIG_DIR / "nasa_top_coefficients.png"
    if hasattr(model, "named_steps") and "model" in getattr(model, "named_steps", {}):
        lr = model.named_steps["model"]
        if hasattr(lr, "coef_"):
            coefs = pd.Series(lr.coef_[0], index=features).sort_values()
            top = pd.concat([coefs.head(10), coefs.tail(10)])

            plt.figure()
            top.plot(kind="barh")
            plt.title("NASA FD001 - Logistic Regression Coefficients (Top +/-)")
            plt.xlabel("Coefficient weight")
            plt.tight_layout()
            plt.savefig(out3, dpi=160)
            plt.close()
        else:
            _save_placeholder(out3, "NASA model has no coef_ to plot (not a linear model).")
    else:
        _save_placeholder(out3, "NASA model is not a Pipeline with a 'model' step; cannot plot coefficients.")

    return [out1.name, out2.name, out3.name]


def write_html(mining_imgs: list[str], nasa_imgs: list[str]) -> None:
    html = f"""
<html>
<head>
  <meta charset="utf-8">
  <title>Mining Ops Analytics Report</title>
</head>
<body style="font-family: Arial; margin: 24px;">
  <h1>Mining Operations Analytics Platform - Report</h1>

  <h2>Mining Mode (Synthetic)</h2>
  <ul>
    <li>Utilization by site</li>
    <li>Downtime trend</li>
    <li>Top risky assets</li>
  </ul>
  {''.join([f'<img src="figures/{img}" style="max-width: 980px; width: 100%; margin-bottom: 16px;" />' for img in mining_imgs])}

  <h2>NASA Mode (C-MAPSS FD001)</h2>
  <ul>
    <li>RUL distribution</li>
    <li>ROC curve (fail within 30 cycles)</li>
    <li>Top model coefficients</li>
  </ul>
  {''.join([f'<img src="figures/{img}" style="max-width: 980px; width: 100%; margin-bottom: 16px;" />' for img in nasa_imgs])}

  <p><em>Generated locally from the project pipeline outputs.</em></p>
</body>
</html>
"""
    out = REPORT_DIR / "mining_ops_report.html"
    out.write_text(html, encoding="utf-8")
    print("Saved report:", out)


def main() -> None:
    _ensure_dirs()
    mining_imgs = mining_plots()
    nasa_imgs = nasa_plots()
    write_html(mining_imgs, nasa_imgs)
    print("Saved figures to:", FIG_DIR)


if __name__ == "__main__":
    main()