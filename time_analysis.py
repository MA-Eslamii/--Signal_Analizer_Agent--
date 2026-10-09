"""Split each data source into 15-minute blocks and derive ECG-based HRV."""

import numpy as np
import pandas as pd
import neurokit2 as nk

WINDOW_SECONDS = 15 * 60


def _time_as_seconds(index):
    """Return relative seconds and a readable origin for numeric or datetime indexes."""
    if isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index.dtype):
        times = pd.DatetimeIndex(index)
        return (times - times[0]).total_seconds().to_numpy(), times[0].isoformat()
    values = pd.to_numeric(pd.Index(index), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Time index must contain numeric seconds or datetimes")
    return values - values[0], f"{float(values[0]):.3f} sec (source time)"


def _parameter_trends(frame):
    """Summarize within-parameter changes without exposing device-scale values."""
    result = {}
    for column in frame.columns:
        if str(column).startswith("_"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            states = frame[column].dropna().astype(str)
            if len(states):
                result[str(column)] = {"observations": int(len(states)),
                                       "pattern": "state changed" if states.nunique() > 1 else "state unchanged",
                                       "observed_states": list(dict.fromkeys(states.tolist())),
                                       "state_changes": int(states.ne(states.shift()).sum() - 1)}
            continue
        quarter = max(1, len(values) // 4)
        early = float(values.iloc[:quarter].median())
        late = float(values.iloc[-quarter:].median())
        if abs(early) > 1e-12:
            relative_change = late / early
            direction = "relatively rising" if relative_change > 1.05 else "relatively falling" if relative_change < 0.95 else "relatively stable"
        else:
            direction = "relatively rising" if late > early else "relatively falling" if late < early else "relatively stable"
        early_scale = max(abs(early), 1e-12)
        late_scale = max(abs(late), 1e-12)
        early_spread = float(np.median(np.abs(values.iloc[:quarter] - early))) / early_scale
        late_spread = float(np.median(np.abs(values.iloc[-quarter:] - late))) / late_scale
        spread_pattern = "relative variation increases" if late_spread > early_spread * 1.2 else "relative variation decreases" if late_spread < early_spread * 0.8 else "relative variation is broadly similar"
        result[str(column)] = {"observations": int(len(values)),
                               "relative_direction": direction,
                               "relative_variability": spread_pattern,
                               "pattern": "variable" if values.nunique() > 1 else "unchanged",
                               "missing_or_unreadable": int(frame[column].isna().sum())}
    return result


def _parameter_co_movement(frame):
    """Describe same-time parameter relationships using unitless rank correlation."""
    columns = [column for column in frame.columns if not str(column).startswith("_")]
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    relationships = []
    for left_index, left in enumerate(numeric.columns):
        for right in numeric.columns[left_index + 1:]:
            paired = numeric[[left, right]].dropna()
            if len(paired) < 4:
                continue
            correlation = paired.corr(method="spearman").iloc[0, 1]
            if not np.isfinite(correlation):
                continue
            pattern = "tend to rise and fall together" if correlation >= 0.6 else "tend to move in opposite directions" if correlation <= -0.6 else "no consistent same-time relationship"
            relationships.append({"parameters": [str(left), str(right)], "pattern": pattern})
    return relationships


def split_monitor_windows(parameters: pd.DataFrame, window_seconds=WINDOW_SECONDS):
    """Make independent 15-minute summaries using only parameters actually present."""
    if parameters.empty:
        return []
    seconds, origin = _time_as_seconds(parameters.index)
    frame = parameters.copy()
    frame["_elapsed_sec"] = seconds
    result = []
    for start in np.arange(0, max(float(seconds.max()), 0) + 1, window_seconds):
        block = frame.loc[(frame["_elapsed_sec"] >= start) & (frame["_elapsed_sec"] < start + window_seconds)]
        if block.empty:
            continue
        result.append({"window_start_sec_from_source": round(float(start), 2),
                       "window_end_sec_from_source": round(float(min(start + window_seconds, seconds.max())), 2),
                       "source_time_origin": origin,
                       "observed_time_start": str(block.index.min()),
                       "observed_time_end": str(block.index.max()),
                       "parameters_available": _parameter_trends(block.drop(columns="_elapsed_sec")),
                       "same_time_parameter_relationships": _parameter_co_movement(block),
                       "waveforms_seen_on_monitor": sorted({v for v in block.get("_waveforms_present", pd.Series(dtype=str)).dropna().astype(str) if v}),
                       "ocr_uncertain_fields": sorted({v for v in block.get("_uncertain_fields", pd.Series(dtype=str)).dropna().astype(str) if v}),
                       "coverage": f"{len(block)} OCR/monitor observations in this block"})
    return result


def _relative_cycle_profile(intervals, interval_times, start, end, origin, source_channel=None):
    """Describe beat timing with within-window ratios, excluding waveform gain."""
    if len(intervals) < 3:
        return None
    typical = float(np.median(intervals))
    if not np.isfinite(typical) or typical <= 0:
        return None

    # Each 30-second segment is compared with this same window's typical cycle length.
    segment_edges = np.arange(start, end + 30, 30)
    relative_cycle = []
    variability = []
    for left, right in zip(segment_edges[:-1], segment_edges[1:]):
        local = intervals[(interval_times >= left) & (interval_times < right)]
        if len(local) < 2:
            continue
        ratio = float(np.median(local) / typical)
        relative_cycle.append("relatively shorter" if ratio < 0.95 else "relatively longer" if ratio > 1.05 else "near the window pattern")
        local_cv = float(np.median(np.abs(local - np.median(local))) / max(float(np.median(local)), 1e-9))
        variability.append(local_cv)

    quarter = max(1, len(intervals) // 4)
    early = float(np.median(intervals[:quarter]))
    late = float(np.median(intervals[-quarter:]))
    relative_change = late / max(early, 1e-9)
    direction = "cycles relatively lengthen toward the end" if relative_change > 1.05 else "cycles relatively shorten toward the end" if relative_change < 0.95 else "no clear directional cycle change"
    if len(variability) >= 4:
        half = max(1, len(variability) // 2)
        first_variability = float(np.median(variability[:half]))
        last_variability = float(np.median(variability[half:]))
        variability_trend = "relative cycle variability increases" if last_variability > first_variability * 1.1 else "relative cycle variability decreases" if last_variability < first_variability * 0.9 else "relative cycle variability is broadly similar"
    else:
        variability_trend = "not enough time segments to compare variability"

    relative_intervals = intervals / typical
    short_cycles = relative_intervals < 0.8
    long_cycles = relative_intervals > 1.5
    short_long_pairs = int(np.sum(short_cycles[:-1] & long_cycles[1:]))
    irregular_fraction = float(np.mean(np.abs(relative_intervals - 1.0) > 0.15))
    pattern_flags = []
    if irregular_fraction <= 0.2:
        pattern_flags.append("cycle timing is predominantly regular relative to this window")
    else:
        pattern_flags.append("cycle timing is intermittently or persistently irregular")
    if short_long_pairs:
        pattern_flags.append("short-long cycle sequences occur; these can reflect ectopic-like timing or beat-detection error")
    if np.any(long_cycles):
        pattern_flags.append("relative long-cycle gaps occur; review the original trace for pauses or missed detections")

    profile = {"window_start_sec_from_source": round(float(start), 2),
               "window_end_sec_from_source": round(float(end), 2),
               "source_time_origin": origin,
               "hrv_available": True,
               "timing_basis": "gain-independent beat-to-beat interval ratios; ECG amplitude values excluded",
               "relative_cycle_direction": direction,
               "relative_variability_trend": variability_trend,
               "timing_pattern_evidence": pattern_flags,
               "relative_cycle_pattern_by_30s": relative_cycle}
    if source_channel is not None:
        profile["source_interval_channel"] = str(source_channel)
    return profile


def ecg_hrv_windows(signals: pd.DataFrame, sample_rate_hz=125, window_seconds=WINDOW_SECONDS):
    """Describe gain-independent HRV timing from ECG R peaks or a true RR/IBI stream."""
    rr_columns = [name for name in signals.columns if str(name).strip().lower() in {"rr", "rr_ms", "rri", "ibi", "nn", "nn_ms"}]
    if rr_columns:
        column = rr_columns[0]
        seconds, origin = _time_as_seconds(signals.index)
        intervals = pd.to_numeric(signals[column], errors="coerce").to_numpy(dtype=float)
        finite_values = intervals[np.isfinite(intervals)]
        if len(finite_values) and np.median(finite_values) < 10:
            intervals *= 1000
        output = []
        for start in np.arange(0, max(float(seconds[-1]), 0) + 1, window_seconds):
            valid = ((seconds >= start) & (seconds < start + window_seconds)
                     & np.isfinite(intervals) & (intervals > 0))
            selected = intervals[valid]
            selected_times = seconds[valid]
            block_end = min(float(start + window_seconds), float(seconds[-1]))
            profile = _relative_cycle_profile(selected, selected_times, float(start), block_end, origin, column)
            if profile is not None:
                output.append(profile)
        if output:
            return output

    leads = [name for name in ("II", "II ECG", "ECG II") if name in signals.columns]
    if not leads:
        leads = [name for name in signals.columns if "ecg" in str(name).lower()]
    lead = leads[0] if leads else ("II" if "II" in signals.columns else None)
    if lead is None:
        return []
    series = pd.to_numeric(signals[lead], errors="coerce")
    cleaned = nk.ecg_clean(series.interpolate(limit_direction="both"), sampling_rate=sample_rate_hz, method="neurokit")
    _, info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate_hz, method="neurokit")
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    elapsed, origin = _time_as_seconds(signals.index)
    peak_times = elapsed[np.clip(peaks, 0, max(len(elapsed) - 1, 0))] if len(peaks) else np.array([])
    if len(peak_times) < 3:
        return [{"source_time_origin": origin, "hrv_available": False,
                 "reason": "Too few R peaks detected to estimate HRV."}]
    rr_ms = np.diff(peak_times) * 1000
    rr_time = peak_times[1:]
    output = []
    starts = np.arange(0, max(float(elapsed[-1]), 0) + 1, window_seconds)
    for start in starts:
        selected = rr_ms[(rr_time >= start) & (rr_time < start + window_seconds)]
        if len(selected) < 3:
            continue
        profile = _relative_cycle_profile(selected, rr_time[(rr_time >= start) & (rr_time < start + window_seconds)], float(start), float(min(start + window_seconds, elapsed[-1])), origin)
        if profile is not None:
            output.append(profile)
    return output or [{"source_time_origin": origin, "hrv_available": False,
                       "reason": "Recording is shorter than a 15-minute HRV block or has too few detected RR intervals."}]

