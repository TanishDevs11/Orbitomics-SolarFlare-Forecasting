"""
Calibrated probabilistic flare forecasting model (Concept 3).

Uses RandomForestClassifier with class_weight='balanced' to handle rare-event
imbalance. Outputs predict_proba() — never thresholded to a hard binary label
before display. Evaluated with TSS and HSS, not plain accuracy.

TSS (True Skill Statistic) = POD - POFD: ranges -1 to 1, 0 = no skill.
HSS (Heidke Skill Score)   = (H - E) / (total - E): ranges -∞ to 1, 0 = chance.
Both are standard solar flare forecast metrics (see CLAUDE.md references).
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from .features import get_feature_columns

MODEL_PATH = Path(__file__).parent.parent / "data" / "forecast_model.pkl"


def _tss(y_true, y_pred) -> float:
    """True Skill Statistic: TP/(TP+FN) - FP/(FP+TN)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    pod = tp / max(tp + fn, 1)
    pofd = fp / max(fp + tn, 1)
    return float(pod - pofd)


def _hss(y_true, y_pred) -> float:
    """Heidke Skill Score."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return 0.0
    tn, fp, fn, tp = cm.ravel()
    n = tn + fp + fn + tp
    expected = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / max(n, 1)
    correct = tp + tn
    return float((correct - expected) / max(n - expected, 1e-9))


def build_feature_matrix(df: pd.DataFrame, labels: pd.Series) -> tuple:
    """
    Build feature matrix X and label vector y, dropping NaN rows.
    Feature window: current row only (could extend to multi-row windows later).
    """
    feat_cols = get_feature_columns()
    available = [c for c in feat_cols if c in df.columns]
    combined = df[available].copy()
    combined["label"] = labels
    combined = combined.dropna()

    X = combined[available].values
    y = combined["label"].values
    return X, y, available


def train_model(df: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Train RandomForest on the labeled data. Saves model to data/forecast_model.pkl.
    Returns evaluation metrics dict.
    """
    X, y, feat_cols = build_feature_matrix(df, labels)

    if len(np.unique(y)) < 2:
        print("[ForecastModel] Only one class in labels — cannot train. Check nowcaster.")
        return _dummy_metrics()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # temporal split: no leakage
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",   # handles severe class imbalance
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Use threshold=0.3 for imbalanced data (lower than default 0.5 increases recall)
    y_pred_thresh = (y_prob >= 0.3).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, y_pred_thresh, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred_thresh, zero_division=0)),
        "tss":       float(_tss(y_test, y_pred_thresh)),
        "hss":       float(_hss(y_test, y_pred_thresh)),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "pos_rate":  float(y_train.mean()),
        "feature_cols": feat_cols,
    }

    # Feature importances for explainability
    rf = model.named_steps["rf"]
    importances = dict(zip(feat_cols, rf.feature_importances_))
    metrics["feature_importances"] = importances

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "feat_cols": feat_cols}, f)

    print(f"[ForecastModel] Trained. TSS={metrics['tss']:.3f}  HSS={metrics['hss']:.3f}  "
          f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}")
    return metrics


def load_model():
    """Load saved model from disk."""
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feat_cols"]


def predict_proba_latest(model, feat_cols: list, df: pd.DataFrame) -> float:
    """
    Predict flare probability for the most recent row in df.
    Returns probability [0, 1] or -1 if model unavailable.
    """
    if model is None:
        return -1.0
    available = [c for c in feat_cols if c in df.columns]
    row = df[available].dropna().tail(1)
    if row.empty:
        return -1.0
    prob = model.predict_proba(row.values)[0][1]
    return float(prob)


def _dummy_metrics() -> dict:
    return {
        "precision": 0.0, "recall": 0.0, "tss": 0.0, "hss": 0.0,
        "n_train": 0, "n_test": 0, "pos_rate": 0.0,
        "feature_cols": get_feature_columns(),
        "feature_importances": {},
    }
