"""Signal filtering helpers for ECG and slower vital-sign series."""

import pandas as pd
from scipy.signal import butter, sosfiltfilt


def filter_ecg(series: pd.Series, sample_rate_hz: int = 125) -> pd.Series:
    """Apply a zero-phase 0.5–40 Hz ECG band-pass filter, preserving gaps."""
    if sample_rate_hz <= 80:
        raise ValueError("sample_rate_hz must exceed 80 Hz for the 40 Hz upper cutoff")
    values = series.astype(float)
    filled = values.interpolate(limit_direction="both")
    if filled.isna().all():
        return values.copy()
    sos = butter(4, [0.5, 40], btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, filled.to_numpy())
    return pd.Series(filtered, index=series.index, name=series.name).where(values.notna())


def filter_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    """Reduce isolated spikes with a centered rolling median."""
    result = vitals.copy()
    if "HR" in result:
        result["HR"] = result["HR"].rolling(5, center=True, min_periods=1).median()
    if "SpO2" in result:
        result["SpO2"] = result["SpO2"].rolling(3, center=True, min_periods=1).median()
    return result

