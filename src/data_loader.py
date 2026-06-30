"""
Loads GOES X-ray flux data from NOAA SWPC (or synthetic fallback).

GOES XRS channel mapping (treating as Aditya-L1 proxies):
  Short (0.05-0.4 nm / 0.5-4 Å) -> HXR proxy (hard X-ray, impulsive phase)
  Long  (0.10-0.80 nm / 1-8 Å)  -> SXR proxy (soft X-ray, thermal phase,
                                     standard NOAA flare classification channel)

Swapping to real SoLEXS/HEL1OS data requires only changing this file.
"""
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "goes_xray_sample.csv"

# NOAA SWPC primary endpoints (try in order)
SWPC_URLS = [
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json",
    "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
]


def _parse_swpc_json(records: list) -> pd.DataFrame:
    """
    Parse NOAA SWPC xrays JSON. Records alternate short/long band per timestamp,
    or may contain separate 'energy' field to identify the band.
    Returns DataFrame with columns [sxr_flux, hxr_flux], index=timestamp.
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    df["time_tag"] = pd.to_datetime(df["time_tag"])
    df = df.sort_values("time_tag").reset_index(drop=True)

    # Strategy 1: explicit energy/band column
    if "energy" in df.columns:
        long_mask = df["energy"].astype(str).str.contains("0.1|1-8|long", case=False, na=False)
        short_mask = ~long_mask
        sxr = df[long_mask][["time_tag", "flux"]].rename(columns={"flux": "sxr_flux"})
        hxr = df[short_mask][["time_tag", "flux"]].rename(columns={"flux": "hxr_flux"})
        out = pd.merge(sxr, hxr, on="time_tag", how="inner")
        out = out.set_index("time_tag").sort_index()
        return out

    # Strategy 2: two records per timestamp (alternating short/long)
    dupes = df["time_tag"].duplicated(keep=False)
    if dupes.any():
        # Group by timestamp, assign higher flux = SXR (long channel), lower = HXR (short)
        def split_channels(group):
            fluxes = group["flux"].values
            if len(fluxes) >= 2:
                return pd.Series({
                    "sxr_flux": float(np.max(fluxes)),
                    "hxr_flux": float(np.min(fluxes)),
                })
            return pd.Series({"sxr_flux": float(fluxes[0]), "hxr_flux": float(fluxes[0]) * 0.3})

        out = df.groupby("time_tag").apply(split_channels)
        out.index.name = "timestamp"
        return out.sort_index()

    # Strategy 3: single channel only; synthesize HXR as 0.3x SXR (scaled proxy)
    # This is a rough approximation — clearly labelled in UI
    out = df[["time_tag", "flux"]].copy()
    out.columns = ["timestamp", "sxr_flux"]
    out["hxr_flux"] = out["sxr_flux"] * 0.3
    return out.set_index("timestamp").sort_index()


def _fetch_live() -> pd.DataFrame:
    """Try SWPC endpoints in order; return parsed DataFrame."""
    for url in SWPC_URLS:
        try:
            resp = requests.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            # Stream the response to avoid memory issues with large JSON
            content = resp.content
            records = __import__("json").loads(content)
            df = _parse_swpc_json(records)
            if len(df) > 10:
                print(f"[DataLoader] Fetched {len(df)} rows from {url}")
                return df
        except Exception as e:
            print(f"[DataLoader] Failed {url}: {e}")
    return pd.DataFrame()


def generate_synthetic_data(days: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic GOES-like data with embedded C/M-class flares.
    Used when network unavailable. Clearly labeled as SYNTHETIC in UI.

    Flare profiles: exponential rise + exponential decay, matching typical GOES shapes.
    HXR peaks ~3 min before SXR (impulsive phase leads thermal response).
    """
    np.random.seed(seed)
    end = pd.Timestamp("2024-05-13 12:00:00")  # fixed anchor for reproducibility
    start = end - pd.Timedelta(days=days)
    idx = pd.date_range(start, end, freq="1min")
    n = len(idx)

    # Quiet background with slow drift (background solar wind variation)
    background = 5e-8 * np.exp(0.3 * np.sin(np.linspace(0, 4 * np.pi, n)))
    noise = np.random.lognormal(0, 0.15, n) * 5e-9
    sxr = background + noise

    def embed_flare(arr, peak_i, peak_flux, rise_min, decay_min, hxr_lead=3):
        """Embed a flare into sxr array; return (sxr_updated, hxr_burst)."""
        # Initialize HXR burst to zeros — np.maximum() with the base HXR array fills
        # in background levels everywhere outside the flare region. Using arr.copy()
        # (the SXR array) was a bug that made HXR ≈ SXR during quiet background,
        # inflating hxr_sigma and log_hardness_ratio and causing constant false alarms.
        hxr = np.zeros_like(arr)
        # Rise: linear in log-space (exponential in flux)
        for i in range(rise_min):
            t = i / rise_min
            j = peak_i - rise_min + i
            if 0 <= j < len(arr):
                arr[j] = max(arr[j], arr[max(0, j-1)] + (peak_flux - arr[max(0, peak_i - rise_min)]) * t / rise_min)
        arr[peak_i] = max(arr[peak_i], peak_flux)
        # Decay: exponential
        tau = decay_min / 3.0
        for i in range(1, decay_min * 3):
            j = peak_i + i
            if j < len(arr):
                val = peak_flux * np.exp(-i / tau) + background[j]
                arr[j] = max(arr[j], val)
        # HXR peaks hxr_lead minutes earlier, sharper impulsive spike
        hxr_peak = max(0, peak_i - hxr_lead)
        hxr_rise = max(1, rise_min // 2)
        hxr_peak_flux = peak_flux * 0.35
        for i in range(hxr_rise):
            j = hxr_peak - hxr_rise + i
            if 0 <= j < len(hxr):
                hxr[j] = max(hxr[j], hxr_peak_flux * i / hxr_rise)
        hxr[hxr_peak] = max(hxr[hxr_peak], hxr_peak_flux)
        hxr_decay = max(1, decay_min // 3)
        for i in range(1, hxr_decay * 4):
            j = hxr_peak + i
            if j < len(hxr):
                hxr[j] = max(hxr[j], hxr_peak_flux * np.exp(-i / (hxr_decay / 2.0)) + background[j] * 0.3)
        return arr, hxr

    hxr = background * 0.3 + noise * 0.3

    # Flares: (peak_index_fraction, peak_flux, rise_min, decay_min)
    flare_defs = [
        (0.18, 4e-6, 8, 35),    # C4 flare
        (0.40, 3e-5, 12, 90),   # M3 flare (main demo event)
        (0.65, 8e-6, 6, 30),    # C8 flare
        (0.85, 2e-5, 10, 55),   # M2 flare
    ]
    for frac, peak_flux, rise_min, decay_min in flare_defs:
        peak_i = int(frac * n)
        sxr, hxr_burst = embed_flare(sxr, peak_i, peak_flux, rise_min, decay_min)
        hxr = np.maximum(hxr, hxr_burst)

    return pd.DataFrame({"sxr_flux": sxr, "hxr_flux": hxr}, index=idx)


def load_goes_data(force_synthetic: bool = False, cache_max_age_hours: float = 6.0) -> pd.DataFrame:
    """
    Main entry: load GOES X-ray data.
    Order: (1) fresh cache, (2) live SWPC fetch, (3) stale cache, (4) synthetic.
    Returns DataFrame[sxr_flux, hxr_flux] with DatetimeIndex.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_synthetic and CACHE_PATH.exists():
        age_s = (pd.Timestamp.now() - pd.Timestamp(CACHE_PATH.stat().st_mtime, unit="s")).total_seconds()
        if age_s < cache_max_age_hours * 3600:
            df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            df.index.name = "timestamp"
            if "sxr_flux" in df.columns and len(df) > 100:
                print(f"[DataLoader] Loaded cache ({len(df)} rows, {df.index[0]} to {df.index[-1]})")
                return df

    if not force_synthetic:
        live = _fetch_live()
        if not live.empty:
            live.index.name = "timestamp"
            live.to_csv(CACHE_PATH)
            print(f"[DataLoader] Saved {len(live)} rows to cache.")
            return live
        # Try stale cache before synthetic
        if CACHE_PATH.exists():
            df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            if "sxr_flux" in df.columns and len(df) > 100:
                print("[DataLoader] Using stale cache (live fetch failed).")
                return df

    df = generate_synthetic_data(days=3)
    df.index.name = "timestamp"
    df.to_csv(CACHE_PATH)
    print(f"[DataLoader] Using SYNTHETIC data ({len(df)} rows). Label as demo mode in UI.")
    return df


def load_training_data() -> pd.DataFrame:
    """
    Returns data suitable for ML model training.
    Always uses synthetic data (3 days, 4 embedded flares) for reproducibility.
    The dashboard visualizes live data; the model trains on this synthetic set.
    This is clearly disclosed in the UI — the pipeline architecture is the same.
    """
    return generate_synthetic_data(days=3)


if __name__ == "__main__":
    df = load_goes_data()
    print("\n=== Data Summary ===")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.describe().to_string())
