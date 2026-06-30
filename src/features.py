"""
Physics-informed feature engineering for solar flare forecasting.
Feature names match the team's research narrative exactly (Concept 4).

Features computed:
  - hardness_ratio    : HXR / SXR — rises during impulsive non-thermal energy release
  - log_hardness_ratio: log10(HXR/SXR) — more stable for ML
  - sxr_rolling_mean  : rolling baseline of soft X-ray flux
  - hxr_rolling_mean  : rolling baseline of hard X-ray flux
  - sxr_rolling_std   : variability measure (pre-flare turbulence proxy)
  - hxr_rolling_std   : HXR variability
  - sxr_rolling_slope : rate of change in SXR (rising = onset candidate)
  - hxr_rolling_slope : rate of change in HXR
  - sxr_above_baseline: how many std above quiet background
  - hxr_above_baseline: same for HXR
  - impulsiveness     : peak HXR / rise_time proxy (high = sharp impulsive event)
  - soft_hard_lag_est : estimated lead of HXR over SXR via rolling cross-correlation

Rise time and decay constant are computed in nowcast.py (event-level features).
"""
import numpy as np
import pandas as pd
from scipy.signal import correlate


# Window sizes — justified by flare timescales:
# 60-min baseline captures quiet pre-flare background
# 10-min slope is sensitive to onset rise rates (typical C-class rise ~5-20 min)
BASELINE_WIN = 60   # minutes
SLOPE_WIN = 10      # minutes
STD_WIN = 15        # minutes


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Least-squares slope of log10(flux) over a rolling window.
    Log-space slope is more physically meaningful (flux is lognormal).
    Units: decades per minute.
    """
    log_s = np.log10(series.clip(lower=1e-12))

    def slope(x):
        if x.isna().sum() > len(x) // 2:
            return np.nan
        t = np.arange(len(x))
        valid = ~np.isnan(x.values)
        if valid.sum() < 3:
            return np.nan
        return float(np.polyfit(t[valid], x.values[valid], 1)[0])

    return log_s.rolling(window, min_periods=max(3, window // 2)).apply(slope, raw=False)


def _soft_hard_lag(sxr: np.ndarray, hxr: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate HXR lead over SXR via cross-correlation on a window.
    Positive lag = HXR leads (physically expected during impulsive phase).
    Returns lag in minutes.
    """
    if len(sxr) < 10:
        return 0.0
    s = (sxr - sxr.mean()) / (sxr.std() + 1e-30)
    h = (hxr - hxr.mean()) / (hxr.std() + 1e-30)
    corr = correlate(s, h, mode="full")
    lags = np.arange(-(len(s) - 1), len(s))
    best_lag = lags[np.argmax(corr)]
    return float(np.clip(best_lag, -max_lag, max_lag))


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all physics-informed features. Returns df with new feature columns appended.
    Input must have columns: sxr_flux, hxr_flux, DatetimeIndex at 1-min cadence.
    """
    out = df.copy()

    # --- Hardness ratio (Concept 4 core feature) ---
    # Rising ratio = non-thermal particle acceleration dominating (impulsive phase)
    ratio = out["hxr_flux"] / out["sxr_flux"].clip(lower=1e-12)
    out["hardness_ratio"] = ratio.clip(lower=1e-4, upper=1e4)
    out["log_hardness_ratio"] = np.log10(out["hardness_ratio"].clip(lower=1e-6))

    # --- Rolling statistics (baseline + variability) ---
    out["sxr_rolling_mean"] = out["sxr_flux"].rolling(BASELINE_WIN, min_periods=10).mean()
    out["hxr_rolling_mean"] = out["hxr_flux"].rolling(BASELINE_WIN, min_periods=10).mean()
    out["sxr_rolling_std"]  = out["sxr_flux"].rolling(STD_WIN, min_periods=5).std()
    out["hxr_rolling_std"]  = out["hxr_flux"].rolling(STD_WIN, min_periods=5).std()

    # --- Deviation above baseline (sigma above quiet sun) ---
    baseline_std = out["sxr_flux"].rolling(BASELINE_WIN, min_periods=10).std().clip(lower=1e-12)
    out["sxr_above_baseline"] = (out["sxr_flux"] - out["sxr_rolling_mean"]) / baseline_std
    hxr_base_std = out["hxr_flux"].rolling(BASELINE_WIN, min_periods=10).std().clip(lower=1e-12)
    out["hxr_above_baseline"] = (out["hxr_flux"] - out["hxr_rolling_mean"]) / hxr_base_std

    # --- Rolling slope (onset detection signal) ---
    out["sxr_rolling_slope"] = _rolling_slope(out["sxr_flux"], SLOPE_WIN)
    out["hxr_rolling_slope"] = _rolling_slope(out["hxr_flux"], SLOPE_WIN)

    # --- Impulsiveness proxy: HXR flux / baseline (sharpness of spike relative to quiet) ---
    # True impulsiveness = peak / rise_time, computed per-event in nowcast.py
    # Here we compute a rolling proxy usable in ML feature windows
    out["impulsiveness"] = out["hxr_flux"] / out["hxr_rolling_mean"].clip(lower=1e-12)

    # --- Soft-hard lag estimate (rolling window cross-correlation) ---
    # Computed over 30-min windows; positive = HXR leads SXR (expected)
    lag_win = 30
    lag_vals = np.full(len(out), np.nan)
    for i in range(lag_win, len(out)):
        s_win = out["sxr_flux"].values[i - lag_win:i]
        h_win = out["hxr_flux"].values[i - lag_win:i]
        lag_vals[i] = _soft_hard_lag(s_win, h_win)
    out["soft_hard_lag_est"] = lag_vals

    # Forward-fill the lag estimate to avoid NaN in the first 30 rows
    out["soft_hard_lag_est"] = out["soft_hard_lag_est"].ffill().bfill().fillna(0.0)

    return out


def get_feature_columns() -> list:
    """Return the ML feature column names (subset safe for model input)."""
    return [
        "log_hardness_ratio",
        "sxr_rolling_mean",
        "hxr_rolling_mean",
        "sxr_rolling_std",
        "hxr_rolling_std",
        "sxr_above_baseline",
        "hxr_above_baseline",
        "sxr_rolling_slope",
        "hxr_rolling_slope",
        "impulsiveness",
        "soft_hard_lag_est",
    ]
