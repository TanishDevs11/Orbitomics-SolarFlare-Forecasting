"""
Rule-based nowcaster: detects flare onset, peak, and decay in real-time.
Implements Concept 2: Explainable Nowcaster — every alert states WHICH signals
triggered it (not just the class label).

State machine: QUIET -> ONSET -> PEAK -> DECAY -> QUIET
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List
from .classification import classify_flux, classify_with_number

# --- Thresholds (tuned for GOES data; document WHY) ---
# SXR must rise > 3 sigma above baseline to trigger onset
ONSET_SIGMA = 3.0
# Rolling slope (log10 decades/min) must be positive and above this for onset
ONSET_SLOPE_THRESH = 0.015
# HXR deviation that flags impulsive spike (hard X-ray alarm independently)
HXR_SPIKE_SIGMA = 2.5
# Hardness ratio must be above this to flag non-thermal energy (log10 scale)
HARDNESS_RATIO_THRESH = -0.5   # log10(HXR/SXR) > -0.5 => HXR is 30%+ of SXR
# Flux must drop below this fraction of peak to declare decay end
DECAY_FRACTION = 0.3
# Minimum event duration before we call it a real flare (avoids noise spikes)
MIN_FLARE_MINUTES = 5


@dataclass
class FlareEvent:
    onset_time: pd.Timestamp
    peak_time: Optional[pd.Timestamp] = None
    end_time: Optional[pd.Timestamp] = None
    peak_sxr: float = 0.0
    peak_hxr: float = 0.0
    flare_class: str = "?"
    rise_time_min: float = 0.0
    decay_constant: float = 0.0      # e-folding time in minutes
    impulsiveness: float = 0.0       # peak_hxr / rise_time_min
    triggered_by: List[str] = field(default_factory=list)


class NowcastEngine:
    """
    Stateful event detector that processes one row at a time (streaming mode).
    Call update(row) for each new 1-minute observation.
    """

    def __init__(self):
        self.state = "QUIET"
        self.current_event: Optional[FlareEvent] = None
        self.completed_events: List[FlareEvent] = []
        self._onset_flux = None
        self._onset_time = None
        self._peak_flux = None
        self._peak_time = None
        self._decay_log_fluxes = []
        self._decay_times = []

    def update(self, row: pd.Series) -> dict:
        """
        Process one observation row. Returns a status dict for the dashboard:
        {state, flare_class, alert_text, triggers, current_event, probability_hint}
        """
        sxr = row.get("sxr_flux", np.nan)
        hxr = row.get("hxr_flux", np.nan)
        sxr_sigma = row.get("sxr_above_baseline", 0.0)
        hxr_sigma = row.get("hxr_above_baseline", 0.0)
        sxr_slope = row.get("sxr_rolling_slope", 0.0)
        hxr_slope = row.get("hxr_rolling_slope", 0.0)
        log_hr = row.get("log_hardness_ratio", -1.0)
        ts = row.name if hasattr(row, "name") else pd.Timestamp.now()

        if pd.isna(sxr) or pd.isna(hxr):
            return self._status()

        # --- State transitions ---
        if self.state == "QUIET":
            # Guard: background SXR is ~5e-8 W/m²; B-class starts at 1e-7.
            # Any trigger below B-class is background noise, not a flare onset.
            if sxr < 1e-7:
                return self._status()
            triggers = []
            if sxr_sigma > ONSET_SIGMA and sxr_slope > ONSET_SLOPE_THRESH:
                triggers.append(f"SXR rising {sxr_sigma:.1f}x above baseline (slope={sxr_slope:+.3f} dec/min)")
            if hxr_sigma > HXR_SPIKE_SIGMA:
                triggers.append(f"HXR impulsive spike {hxr_sigma:.1f}x above baseline")
            if log_hr > HARDNESS_RATIO_THRESH:
                triggers.append(f"Hardness ratio elevated (log HR={log_hr:.2f})")

            if len(triggers) >= 1:
                self.state = "ONSET"
                self._onset_flux = sxr
                self._onset_time = ts
                self._peak_flux = sxr
                self._peak_hxr = hxr
                self._peak_time = ts
                self.current_event = FlareEvent(
                    onset_time=ts,
                    peak_sxr=sxr,
                    peak_hxr=hxr,
                    triggered_by=triggers,
                )

        elif self.state == "ONSET":
            # Track peak
            if sxr > self._peak_flux:
                self._peak_flux = sxr
                self._peak_hxr = hxr
                self._peak_time = ts
                if self.current_event:
                    self.current_event.peak_sxr = sxr
                    self.current_event.peak_hxr = hxr
                    self.current_event.peak_time = ts

            # Transition to PEAK once flux is clearly declining.
            # The 10-min rolling slope lags the actual peak by several minutes, so
            # requiring sxr >= 95% of peak is too strict — the flux has already decayed.
            # Use a clearly-negative slope threshold instead (half the onset threshold).
            if sxr_slope < -ONSET_SLOPE_THRESH * 0.5:
                self.state = "PEAK"
                if self.current_event:
                    dt = (ts - self._onset_time).total_seconds() / 60.0
                    self.current_event.rise_time_min = max(1.0, dt)
                    self.current_event.impulsiveness = (
                        self.current_event.peak_hxr / max(1.0, self.current_event.rise_time_min)
                    )
                self._decay_log_fluxes = [np.log(max(sxr, 1e-12))]
                self._decay_times = [0.0]

        elif self.state == "PEAK":
            # Start decay tracking
            self._decay_log_fluxes.append(np.log(max(sxr, 1e-12)))
            elapsed = (ts - self._peak_time).total_seconds() / 60.0
            self._decay_times.append(elapsed)

            # Fit decay constant once we have enough points (5 min)
            if len(self._decay_times) >= 5:
                slope, _ = np.polyfit(self._decay_times, self._decay_log_fluxes, 1)
                tau = -1.0 / slope if slope < 0 else 999.0   # e-folding time in minutes
                if self.current_event:
                    self.current_event.decay_constant = tau

            # Check if we're back to QUIET
            baseline_flux = row.get("sxr_rolling_mean", self._onset_flux or sxr)
            if sxr < max(1e-8, (baseline_flux or 1e-8)) * (1 + DECAY_FRACTION):
                self.state = "DECAY"

        elif self.state == "DECAY":
            # Check recovery to QUIET
            baseline_flux = row.get("sxr_rolling_mean", self._onset_flux or sxr)
            event_dur = (ts - self._onset_time).total_seconds() / 60.0 if self._onset_time else 0
            if sxr < max(1e-8, (baseline_flux or 1e-8)) * 1.5 and event_dur > MIN_FLARE_MINUTES:
                # Finalize event
                if self.current_event:
                    self.current_event.end_time = ts
                    fc = classify_with_number(self.current_event.peak_sxr)
                    self.current_event.flare_class = fc
                    self.completed_events.append(self.current_event)
                self.current_event = None
                self.state = "QUIET"
            elif sxr_sigma > ONSET_SIGMA * 1.5 and sxr_slope > ONSET_SLOPE_THRESH:
                # Another flare starting during decay (sympathetic flare)
                self.state = "ONSET"

        return self._status()

    def _status(self) -> dict:
        state = self.state
        ev = self.current_event

        if state == "QUIET" or ev is None:
            return {
                "state": "QUIET",
                "flare_class": "—",
                "alert_text": "No flare activity detected.",
                "triggers": [],
                "current_event": None,
            }

        fc = classify_with_number(ev.peak_sxr) if ev.peak_sxr > 0 else "?"
        phase_labels = {
            "ONSET": "Onset / Rising Phase",
            "PEAK": "Peak Phase",
            "DECAY": "Decay Phase",
        }
        phase = phase_labels.get(state, state)

        alert = f"⚡ FLARE IN PROGRESS — Class {fc} | Phase: {phase}"
        return {
            "state": state,
            "flare_class": fc,
            "alert_text": alert,
            "triggers": ev.triggered_by,
            "current_event": ev,
        }


def run_nowcast_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run nowcaster over entire DataFrame (batch mode for analysis/training).
    Returns df with added columns: nowcast_state, nowcast_class.
    """
    engine = NowcastEngine()
    states = []
    classes = []
    for ts, row in df.iterrows():
        status = engine.update(row)
        states.append(status["state"])
        classes.append(status["flare_class"])
    df = df.copy()
    df["nowcast_state"] = states
    df["nowcast_class"] = classes
    return df, engine.completed_events
