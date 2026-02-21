import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import settings
from .features import make_training_frame, FEATURES


# Any column names that must NEVER be used as model inputs (target/leakage)
LEAKAGE_HINTS = (
    "label",        # label_high_downtime, label_*, etc.
    "high_sev",     # target used in KPIs / dashboard
    "target",       # common naming
    "y",            # common shorthand
    "downtime_minutes",  # can be target-adjacent depending on label definition
)


def _safe_feature_list(feature_list: list[str]) -> list[str]:
    """
    Filter out obviously-leaky columns from FEATURES, and warn loudly.
    This keeps the pipeline robust even if FEATURES is accidentally edited later.
    """
    bad = []
    safe = []

    for c in feature_list:
        lc = c.lower()
        if any(h in lc for h in LEAKAGE_HINTS):
            bad.append(c)
        else:
            safe.append(c)

    if bad:
        print("WARNING: Removing potential leakage features from training inputs:")
        for b in bad:
            print(f" - {b}")
        print("Fix your FEATURES list if this was unintentional.\n")

    return safe


def main():
    settings.data_processed.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    daily_path = settings.data_processed / "daily_equipment_summary.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing {daily_path}. Run:\n"
            "  python -m src.miningops.generate_data\n"
            "  python -m src.miningops.kpis\n"
        )

    daily = pd.read_parquet(daily_path)

    # Build training frame (should include features + label column)
    df = make_training_frame(daily)

    # ---- Label ----
    label_col = "label_high_downtime"
    if label_col not in df.columns:
        raise KeyError(
            f"Training label column '{label_col}' not found. "
            "Check make_training_frame() output."
        )

    y = df[label_col].astype(int)

    # ---- Features (X) ----
    # Use FEATURES from features.py, but enforce "no leakage" safety filter
    safe_features = _safe_feature_list(list(FEATURES))

    missing_features = [c for c in safe_features if c not in df.columns]
    if missing_features:
        raise KeyError(
            "These FEATURES are missing from the training frame:\n"
            + "\n".join(f" - {c}" for c in missing_features)
            + "\n\nFix: ensure make_training_frame() produces these columns, "
              "or update FEATURES accordingly."
        )

    X = df[safe_features].copy()

    # Convert boolean columns to int for sklearn + fill NaNs
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype(int)
    X = X.fillna(0.0)

    # Stratify if possible
    stratify = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_seed,
        stratify=stratify,
    )

    # Note: StandardScaler is OK because all features are numeric
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X_train, y_train)

    # Predict probabilities
    if hasattr(pipe, "predict_proba"):
        preds = pipe.predict_proba(X_test)[:, 1]
    else:
        raise TypeError("Model does not support predict_proba(); use a probabilistic classifier.")

    # AUC requires both classes present
    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, preds)
        print(f"ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: N/A (only one class in test split)")

    model_path = settings.model_dir / "downtime_risk_model.joblib"
    joblib.dump(pipe, model_path)
    print("Saved model:", model_path)

    # Helpful debug print: shows exactly what the model expects later in Streamlit
    try:
        expected = list(pipe.feature_names_in_)
        print("Model expects features (feature_names_in_):")
        for c in expected:
            print(f" - {c}")
    except Exception:
        # Not fatal; just informational
        print("Note: feature_names_in_ not available on this sklearn version/pipeline.")


if __name__ == "__main__":
    main()