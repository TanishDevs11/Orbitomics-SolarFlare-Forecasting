"""
Builds binary training labels for the forecasting model.
Label = 1 if a flare onset occurs within the next HORIZON minutes, else 0.
Uses nowcasted events so we don't need an external event list.
"""
import numpy as np
import pandas as pd

HORIZON_MINUTES = 30   # forecast window: predict flares in next 30 min
ONSET_CLASS = "C"      # minimum class to count as a positive event


def build_labels(df: pd.DataFrame, horizon_min: int = HORIZON_MINUTES) -> pd.Series:
    """
    Label each row: 1 if a C+/M/X-class onset starts within `horizon_min` minutes.
    Uses the nowcast_state column populated by run_nowcast_batch().

    Returns a binary pd.Series aligned with df.index.
    """
    if "nowcast_state" not in df.columns:
        raise ValueError("Run run_nowcast_batch() first to populate nowcast_state.")

    # Onset rows are where nowcast_state transitions into 'ONSET' or 'PEAK'
    is_onset = (df["nowcast_state"].isin(["ONSET", "PEAK"])).astype(int)
    # Only count significant flares (SXR above C threshold at some point during event)
    if "sxr_flux" in df.columns:
        is_significant = (df["sxr_flux"] >= 1e-6).astype(int)
        is_onset = (is_onset & is_significant).astype(int)

    labels = np.zeros(len(df), dtype=int)
    onset_indices = np.where(is_onset.values > 0)[0]

    for idx in onset_indices:
        # Label all rows in the preceding horizon_min minutes as positive
        start = max(0, idx - horizon_min)
        labels[start:idx] = 1

    return pd.Series(labels, index=df.index, name="label")


def class_balance_report(labels: pd.Series) -> dict:
    """Return label balance statistics for logging."""
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    total = len(labels)
    return {
        "total": total,
        "positive": n_pos,
        "negative": n_neg,
        "positive_rate": round(n_pos / total, 4) if total > 0 else 0.0,
        "imbalance_ratio": round(n_neg / max(n_pos, 1), 1),
    }
