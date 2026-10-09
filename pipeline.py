"""Shared data preparation for the desktop interface and command-line runner."""

import pandas as pd
from artifacts import detect_artifacts
from data_loader import load_data, load_signals
from filtering import filter_vitals
from ocr_ingestion import extract_monitor_frame
from time_analysis import ecg_hrv_windows, split_monitor_windows

ECG_LABELS = {"ii", "v", "ecg", "ecg_i", "ecg_ii", "lead_i", "lead_ii", "ii_ecg"}
RR_LABELS = {"rr", "rr_ms", "rri", "ibi", "nn", "nn_ms"}


def _numeric_channels(signals: pd.DataFrame):
    """Select numeric waveform and interval channels from a signal workbook."""
    numeric = signals.select_dtypes(include="number")
    channels = [column for column in numeric if str(column).strip().lower().replace(" ", "_") in ECG_LABELS | RR_LABELS]
    return numeric[channels]


def build_evidence(parameters, signals=None, input_mode="workbook", sample_rate_hz=125):
    """Create time-separated summaries from all available monitor parameters."""
    evidence = {
        "window_size_minutes": 15,
        "monitor_parameter_windows": split_monitor_windows(parameters),
        "available_parameter_names": [str(column) for column in parameters.columns if not str(column).startswith("_")],
        "input_mode": input_mode,
    }
    record_metadata = parameters.attrs.get("record_metadata") or (signals.attrs.get("record_metadata") if signals is not None else None)
    if record_metadata:
        evidence["record_metadata"] = record_metadata
    if input_mode == "monitor screenshot OCR":
        evidence["ocr_notes"] = "Screenshots are sparse observations; extracted waveform labels do not contain continuous ECG samples."
    if signals is not None and not signals.empty:
        waveforms = _numeric_channels(signals)
        if not waveforms.empty:
            leads = waveforms[[column for column in waveforms if str(column).strip().lower().replace(" ", "_") in ECG_LABELS]]
            evidence["signal_coverage"] = [str(signals.index.min()), str(signals.index.max())]
            evidence["available_signal_channels"] = [str(column) for column in waveforms.columns]
            evidence["ecg_channels"] = [str(column) for column in leads.columns]
            metadata_rate = (record_metadata or {}).get("waveform_sample_rate_hz")
            actual_sample_rate = int(metadata_rate or sample_rate_hz)
            if not leads.empty:
                evidence["ecg_sample_rate_hz"] = actual_sample_rate
                evidence["ecg_quality_flags"] = detect_artifacts(leads, actual_sample_rate)
            evidence["ecg_hrv_windows"] = ecg_hrv_windows(waveforms, actual_sample_rate)
    if "ecg_hrv_windows" not in evidence:
        evidence["ecg_hrv_windows"] = [{"hrv_available": False,
                                         "reason": "No sampled ECG or RR/IBI interval stream was supplied; monitor HR values and screenshots cannot establish HRV."}]
    return evidence


def load_workbook_sources(signals_file, parameters_file, sample_rate_hz=125):
    """Load workbook channels and prepare plot-ready data plus model evidence."""
    signals, parameters = load_data(signals_file, parameters_file)
    record_metadata = parameters.attrs.get("record_metadata")
    parameters = filter_vitals(parameters)
    if record_metadata:
        parameters.attrs["record_metadata"] = record_metadata
    evidence = build_evidence(parameters, signals, "workbook", sample_rate_hz)
    return signals, parameters, evidence


def load_ocr_sources(image_paths, signals_file=None, parameters_file=None, sample_rate_hz=125):
    """OCR monitor snapshots and optionally pair them with sampled ECG."""
    parameters = extract_monitor_frame(image_paths)
    signals = None
    if signals_file:
        signals = load_signals(signals_file)
    evidence = build_evidence(parameters, signals, "monitor screenshot OCR", sample_rate_hz)
    return signals, parameters, evidence

