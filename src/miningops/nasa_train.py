import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import settings

FAIL_WITHIN = 30

SENSOR_FEATURES = [f"sensor_{i}" for i in range(1, 22)]
OP_FEATURES = [f"op_setting_{i}" for i in range(1, 4)]
FEATURES = OP_FEATURES + SENSOR_FEATURES

def main():
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(settings.data_processed / "nasa_fd001_train.parquet")

    # Label: 1 if RUL <= FAIL_WITHIN
    df["label_fail_within"] = (df["rul"] <= FAIL_WITHIN).astype(int)

    X = df[FEATURES]
    y = df["label_fail_within"]
    groups = df["unit"]  # prevent leakage: keep whole units together

    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=settings.random_seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, probs)
        print(f"NASA FD001 - Fail within {FAIL_WITHIN} cycles - ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: N/A (only one class in test split)")

    out_path = settings.model_dir / f"nasa_fd001_failure{FAIL_WITHIN}_model.joblib"
    joblib.dump(pipe, out_path)
    print("Saved model:", out_path)

if __name__ == "__main__":
    main()
