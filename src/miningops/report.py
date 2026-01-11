from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc

from .config import settings
from .features import FEATURES as MINING_FEATURES

REPORT_DIR = settings.project_root / "reports"
FIG_DIR = REPORT_DIR / "figures"

def _ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def mining_plots():
    daily = pd.read_parquet(settings.data_processed / "daily_equipment_summary.parquet")

    # 1) Utilization by site
    util_by_site = daily.groupby("site")["utilization_rate"].mean().sort_values(ascending=False)
    plt.figure()
    util_by_site.plot(kind="bar")
    plt.title("Mining - Average Utilization Rate by Site")
    plt.ylabel("Utilization rate")
    plt.tight_layout()
    out1 = FIG_DIR / "mining_utilization_by_site.png"
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) Downtime trend over time
    trend = daily.groupby("date")["downtime_rate"].mean()
    plt.figure()
    trend.plot()
    plt.title("Mining - Average Downtime Rate Over Time")
    plt.ylabel("Downtime rate")
    plt.xlabel("Date")
    plt.tight_layout()
    out2 = FIG_DIR / "mining_downtime_trend.png"
    plt.savefig(out2, dpi=160)
    plt.close()

    # 3) Top risky assets (from mining model)
    model = joblib.load(settings.model_dir / "downtime_risk_model.joblib")
    X = daily[MINING_FEATURES]
    daily["risk"] = model.predict_proba(X)[:, 1]
    top_assets = (
        daily.groupby("equipment_id")["risk"].mean()
        .sort_values(ascending=False)
        .head(10)
        .sort_values()
    )

    plt.figure()
    top_assets.plot(kind="barh")
    plt.title("Mining - Top 10 Risky Assets (Avg Predicted Risk)")
    plt.xlabel("Risk probability")
    plt.tight_layout()
    out3 = FIG_DIR / "mining_top10_risky_assets.png"
    plt.savefig(out3, dpi=160)
    plt.close()

    return [out1.name, out2.name, out3.name]

def nasa_plots():
    df = pd.read_parquet(settings.data_processed / "nasa_fd001_train.parquet")

    # 1) RUL distribution
    plt.figure()
    df["rul"].plot(kind="hist", bins=50)
    plt.title("NASA FD001 - RUL Distribution")
    plt.xlabel("Remaining Useful Life (cycles)")
    plt.tight_layout()
    out1 = FIG_DIR / "nasa_rul_distribution.png"
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) ROC curve for failure within 30 cycles model
    model = joblib.load(settings.model_dir / "nasa_fd001_failure30_model.joblib")

    FAIL_WITHIN = 30
    df["label"] = (df["rul"] <= FAIL_WITHIN).astype(int)

    features = [c for c in df.columns if c.startswith("op_setting_") or c.startswith("sensor_")]
    X = df[features]
    y = df["label"]

    # Use model probabilities on full dataset for a simple ROC plot (reporting view)
    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("NASA FD001 - ROC Curve (Fail within 30 cycles)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out2 = FIG_DIR / "nasa_roc_curve.png"
    plt.savefig(out2, dpi=160)
    plt.close()

    # 3) Top coefficients (logistic regression)
    lr = model.named_steps["model"]
    coefs = pd.Series(lr.coef_[0], index=features).sort_values()
    top = pd.concat([coefs.head(10), coefs.tail(10)])

    plt.figure()
    top.plot(kind="barh")
    plt.title("NASA FD001 - Logistic Regression Coefficients (Top +/-)")
    plt.xlabel("Coefficient weight")
    plt.tight_layout()
    out3 = FIG_DIR / "nasa_top_coefficients.png"
    plt.savefig(out3, dpi=160)
    plt.close()

    return [out1.name, out2.name, out3.name]

def write_html(mining_imgs, nasa_imgs):
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

def main():
    _ensure_dirs()
    mining_imgs = mining_plots()
    nasa_imgs = nasa_plots()
    write_html(mining_imgs, nasa_imgs)
    print("Saved figures to:", FIG_DIR)

if __name__ == "__main__":
    main()
