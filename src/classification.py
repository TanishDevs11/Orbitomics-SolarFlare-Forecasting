"""
Standard GOES X-ray flare classification based on 1-8 Å (SXR) peak flux.
Thresholds are NOAA official — do not modify.
Reference: https://www.swpc.noaa.gov/phenomena/solar-flares-radio-blackouts
"""
import numpy as np
import pandas as pd

# GOES flare class thresholds (W/m²), based on 1-8 Å channel peak flux
THRESHOLDS = {
    "X": 1e-4,
    "M": 1e-5,
    "C": 1e-6,
    "B": 1e-7,
    # Below 1e-7 is A-class
}


def classify_flux(flux: float) -> str:
    """Return GOES flare class letter for a given 1-8 Å flux value."""
    if flux >= 1e-4:
        return "X"
    elif flux >= 1e-5:
        return "M"
    elif flux >= 1e-6:
        return "C"
    elif flux >= 1e-7:
        return "B"
    else:
        return "A"


def classify_with_number(flux: float) -> str:
    """
    Return full GOES class label like 'M2.3'.
    Subclass number = flux / lower_threshold (e.g. M2.3 = 2.3 × 1e-5).
    """
    letter = classify_flux(flux)
    thresholds = {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7, "A": 1e-8}
    base = thresholds.get(letter, 1e-8)
    number = flux / base
    return f"{letter}{number:.1f}"


def classify_series(sxr_series: pd.Series) -> pd.Series:
    """Vectorized classification of a flux Series."""
    return sxr_series.apply(classify_flux)


def is_significant(flux: float, min_class: str = "C") -> bool:
    """Check if flux meets or exceeds a minimum class (for alert thresholds)."""
    order = ["A", "B", "C", "M", "X"]
    return order.index(classify_flux(flux)) >= order.index(min_class)
