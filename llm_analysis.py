"""Build one-window-at-a-time prompts and request cautious English narratives."""

import json
from datetime import datetime
from config import LLM_BASE_URL, LLM_MODEL, get_api_key


def analysis_segments(evidence: dict):
    """Pair same-order 15-minute blocks while retaining their separate time origins."""
    monitor = evidence.get("monitor_parameter_windows", [])
    hrv = evidence.get("ecg_hrv_windows", [])
    count = max(len(monitor), len(hrv), 1)
    segments = []
    for index in range(count):
        segment = {"period_number": index + 1,
                   "window_duration_minutes": 15,
                   "available_parameters": evidence.get("available_parameter_names", []),
                   "input_mode": evidence.get("input_mode", "unknown")}
        for field in ("available_signal_channels", "ecg_channels", "signal_coverage", "ecg_sample_rate_hz"):
            if field in evidence:
                segment[field] = evidence[field]
        if "record_metadata" in evidence:
            segment["record_metadata"] = evidence["record_metadata"]
        if index < len(monitor):
            segment["monitor_parameter_window"] = monitor[index]
        else:
            segment["monitor_parameter_window"] = {"parameters_available": {}, "coverage": "No monitor observations in this period."}
        if index < len(hrv):
            segment["relative_hrv_window"] = hrv[index]
        else:
            segment["relative_hrv_window"] = {"hrv_available": False, "reason": "No sampled ECG/RR data for this period."}

        # Keep artifact types and times inside this block; one flag is still worth reporting.
        quality = evidence.get("ecg_quality_flags", {})
        hrv_window = segment["relative_hrv_window"]
        start = hrv_window.get("window_start_sec_from_source")
        end = hrv_window.get("window_end_sec_from_source")
        if start is not None and end is not None:
            current_flags = {}
            for lead, report in quality.get("leads", {}).items():
                flagged = []
                for window in report.get("windows", []):
                    if window.get("end_sec", -1) <= start or window.get("start_sec", end + 1) >= end:
                        continue
                    kinds = [name for name, active in window.get("flags", {}).items() if active]
                    if kinds:
                        flagged.append({"start_sec": window["start_sec"], "end_sec": window["end_sec"],
                                        "artifact_types": kinds, "review_threshold_met": window.get("review", False)})
                if flagged:
                    current_flags[lead] = flagged
            segment["quality_assessment"] = (
                {"status": "heuristic_flags_present", "lead_intervals": current_flags}
                if current_flags else
                {"status": "no_heuristic_flags_detected", "note": "This is not evidence that the ECG is normal."}
            )
        else:
            segment["quality_assessment"] = {"status": "not_assessed", "note": "No sampled ECG interval is available."}
        segments.append(segment)
    return segments


def analysis_period_label(segment: dict) -> str:
    """Build a selector label with the real time coverage and partial-window status."""
    number = segment.get("period_number", 1)
    monitor = segment.get("monitor_parameter_window", {})
    hrv = segment.get("relative_hrv_window", {})
    start = hrv.get("window_start_sec_from_source", monitor.get("window_start_sec_from_source"))
    end = hrv.get("window_end_sec_from_source", monitor.get("window_end_sec_from_source"))
    observed_start = monitor.get("observed_time_start") or hrv.get("observed_time_start")
    observed_end = monitor.get("observed_time_end") or hrv.get("observed_time_end")
    if observed_start and observed_end:
        coverage = f"{observed_start} – {observed_end}"
        try:
            duration = (datetime.fromisoformat(observed_end) - datetime.fromisoformat(observed_start)).total_seconds()
        except (ValueError, TypeError):
            try:
                duration = float(observed_end) - float(observed_start)
            except (ValueError, TypeError):
                duration = (float(end) - float(start)) if start is not None and end is not None else None
    elif start is not None and end is not None:
        coverage = f"{start:g}–{end:g} sec from source start"
        duration = float(end) - float(start)
    else:
        coverage, duration = "available time coverage", None
    partial = " — partial" if duration is not None and duration < 15 * 60 else ""
    return f"Period {number}: {coverage}{partial}"


def analyze_segment(segment: dict) -> str:
    """Send exactly one segment to the model; later windows require user consent."""
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model=LLM_MODEL, api_key=get_api_key(), base_url=LLM_BASE_URL)
    system = """You are a research assistant describing time-bounded vital-monitor patterns and cautious possible ECG rhythm explanations, not diagnosing a patient.
This request contains exactly one 15-minute period (or a clearly shorter available portion). Analyze only this period; do not predict later periods.
Explain what physiological pattern is observable in this interval: relative cycle timing and variability, available parameter trends, and co-occurrence only if the source timestamps actually align. Keep the ECG and monitor time origins distinct when they differ.
ECG gain varies between devices. Absolute waveform amplitudes and absolute HRV values are intentionally excluded. Use only the supplied dimensionless within-window relationships: relative cycle timing across time bins and relative variability changes. Do not reconstruct or invent physical amplitudes, rates, or diagnoses.
If the ECG timing provides enough evidence, give at most one or two cautious possible rhythm-pattern explanations (for example, regular timing, ectopic-like short-long sequences, or irregular timing that merits review). State the evidence and uncertainty. RR timing alone cannot confirm a specific arrhythmia, conduction disorder, ischemia, or structural heart disease; do not claim those diagnoses. If morphology or P-wave evidence is unavailable, say so and keep the inference limited to timing. If fewer than 12 leads are available, explicitly state that the pattern is preliminary and needs review on a diagnostic ECG.
If metadata warns that normalized signal or parameter labels differ from original labels, preserve that ambiguity and do not assign a specific physiological meaning or lead identity to the disputed channels.
For monitor parameters, describe direction, persistence, variability, state changes, and relationships; do not focus on isolated readings. Missing parameters are absent, not zero or normal.
Always include an explicit ECG signal-quality sentence. If quality_assessment.status is heuristic_flags_present, state that the heuristic detector flagged the named artifact types, lead/channel, and time interval(s), and that rhythm interpretation there is less reliable. These are possible artifacts, not confirmed diagnoses. If status is no_heuristic_flags_detected, state exactly that no windows were flagged by the heuristic detector and explicitly say this does not establish a normal ECG. If not_assessed, state that ECG quality was not assessed because no sampled ECG interval was available. Then describe whether the ECG timing appears regular/irregular or is uninterpretable; do not call it clinically normal based only on absent flags. Treat OCR and automated quality flags as uncertain. Describe implications cautiously, never as a confirmed cause or diagnosis. If evidence is insufficient, say so.
Return one concise English paragraph. Include an "Observed pattern" and a "Possible explanation" in plain prose, with confidence described as low/moderate only when justified. Start with the period number and its observed time coverage."""
    user = "Describe the physiological pattern in this single period using relative relationships only:\n" + json.dumps(segment, ensure_ascii=False, allow_nan=False)
    return model.invoke([("system", system), ("human", user)]).content

