import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .config import settings
import pandas as pd
from .features import make_training_frame, FEATURES

def main():
    settings.data_processed.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    daily_path = settings.data_processed / "daily_equipment_summary.parquet"
    daily = pd.read_parquet(daily_path)

    df = make_training_frame(daily)

    X = df[FEATURES]
    y = df["label_high_downtime"]

    # stratify if possible (helps when imbalance exists)
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.random_seed, stratify=stratify
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict_proba(X_test)[:, 1]

    # AUC requires both classes present
    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, preds)
        print(f"ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: N/A (only one class in test split)")

    model_path = settings.model_dir / "downtime_risk_model.joblib"
    joblib.dump(pipe, model_path)
    print("Saved model:", model_path)

if __name__ == "__main__":
    main()
