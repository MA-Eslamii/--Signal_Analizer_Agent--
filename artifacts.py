"""Heuristic raw-waveform quality flags; these are advisory, not diagnoses."""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch


def detect_artifacts(frame: pd.DataFrame, sample_rate_hz: int = 125, window_seconds: int = 4):
    """Flag windows with multiple simple quality indicators for clinician review."""
    window = max(32, int(window_seconds * sample_rate_hz))
    leads = {}
    for lead in frame.columns:
        raw = frame[lead].astype(float).to_numpy()
        finite = np.isfinite(raw)
        filled = pd.Series(raw).interpolate(limit_direction="both").fillna(0).to_numpy()
        reports = []
        for start in range(0, len(filled), window):
            segment = filled[start:start + window]
            if len(segment) < max(16, window // 2):
                continue
            centered = segment - np.median(segment)
            scale = np.median(np.abs(centered)) * 1.4826
            diff = np.diff(segment)
            flatline = np.ptp(segment) < max(1e-8, 0.01 * max(np.ptp(filled), 1e-8))
            jump = np.percentile(np.abs(diff), 99) > max(5 * scale, 1e-6) if len(diff) else False
            try:
                lowpass = butter(2, 0.5, btype="lowpass", fs=sample_rate_hz, output="sos")
                baseline = np.ptp(sosfiltfilt(lowpass, segment)) > max(3 * scale, 1e-6)
            except ValueError:
                baseline = False
            freqs, power = welch(centered, fs=sample_rate_hz, nperseg=min(256, len(centered)))
            total = power[(freqs >= 0.5) & (freqs <= min(40, sample_rate_hz / 2))].sum()
            high = power[freqs >= 25].sum() / max(total, 1e-12) > 0.35
            flags = {"flatline": bool(flatline), "abrupt_motion": bool(jump),
                     "baseline_wander": bool(baseline), "high_frequency_noise": bool(high)}
            score = sum(flags.values())
            reports.append({"start_sec": round(start / sample_rate_hz, 2),
                            "end_sec": round(min(start + window, len(filled)) / sample_rate_hz, 2),
                            "flags": flags, "flag_count": score, "review": score >= 2})
        leads[lead] = {"missing_fraction": float(1 - finite.mean()), "windows": reports}
    return {"method": "heuristic window flags; needs raw-trace review", "leads": leads}

