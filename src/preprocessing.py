"""
Cleans and aligns GOES X-ray data for downstream analysis.
Handles: NaN gaps, irregular cadence, negative flux values (instrument artifacts).
"""
import pandas as pd
import numpy as np


def preprocess(df: pd.DataFrame, freq: str = "1min", max_gap_fill: int = 10) -> pd.DataFrame:
    """
    Resample to uniform 1-minute grid, interpolate short gaps, clip negatives.

    Args:
        df: Raw DataFrame with [sxr_flux, hxr_flux], DatetimeIndex
        freq: Target cadence (default '1min')
        max_gap_fill: Max consecutive NaN minutes to interpolate (longer gaps left as NaN)

    Returns:
        Cleaned DataFrame on uniform grid.
    """
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Resample to uniform cadence (mean handles sub-minute duplicates)
    df = df.resample(freq).mean()

    # Clip physically impossible negatives and zero-flux (instrument saturation artifact)
    # Minimum meaningful GOES flux is ~1e-9 W/m²
    for col in ["sxr_flux", "hxr_flux"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=1e-9)

    # Interpolate short gaps; leave long gaps as NaN (do not fabricate data)
    df = df.interpolate(method="time", limit=max_gap_fill)

    # Forward-fill at edges (handles leading/trailing NaN after resample)
    df = df.ffill(limit=3).bfill(limit=3)

    # Ensure HXR < SXR (short channel flux is always lower than long channel)
    # If inverted, it suggests channel swap — correct silently
    if "hxr_flux" in df.columns and "sxr_flux" in df.columns:
        swap_mask = df["hxr_flux"] > df["sxr_flux"] * 2
        if swap_mask.sum() > len(df) * 0.3:
            df["sxr_flux"], df["hxr_flux"] = df["hxr_flux"].copy(), df["sxr_flux"].copy()

    return df


def validate(df: pd.DataFrame) -> dict:
    """Return quality stats for logging/UI display."""
    return {
        "n_rows": len(df),
        "n_nan_sxr": int(df["sxr_flux"].isna().sum()),
        "n_nan_hxr": int(df["hxr_flux"].isna().sum()),
        "sxr_min": float(df["sxr_flux"].min()),
        "sxr_max": float(df["sxr_flux"].max()),
        "date_start": str(df.index.min()),
        "date_end": str(df.index.max()),
    }
