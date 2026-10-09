"""Load legacy workbooks and metadata-described combined BIDMC records."""

from pathlib import Path
import re
import pandas as pd


def _find_time_column(columns):
    """Find common time labels while leaving every data column intact."""
    aliases = {"time_sec", "time", "timestamp", "datetime", "date_time", "زمان", "تاریخ", "تاریخ_زمان"}
    for column in columns:
        if str(column).strip().lower().replace(" ", "_") in aliases:
            return column
    raise ValueError(f"No recognizable time column found. Columns: {list(columns)}")


def _index_by_time(frame, name):
    """Normalize a frame to an ordered timestamp index and reject duplicates."""
    time_column = _find_time_column(frame.columns)
    frame = frame.rename(columns={"%SpO2": "SpO2", "SpO₂": "SpO2"}).set_index(time_column).sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate timestamps")
    return frame


def _record_metadata(path):
    """Read the matching BIDMC record section from its companion metadata file."""
    path = Path(path)
    metadata_path = path.parent / "BIDMC_two_random_records_metadata.txt"
    record_match = re.search(r"_(\d+)$", path.stem)
    if not metadata_path.is_file() or not record_match:
        return None
    requested_id = int(record_match.group(1))
    text = metadata_path.read_text(encoding="utf-8", errors="replace")
    headings = list(re.finditer(r"(?m)^BIDMC RECORD:\s*(\d+)\s*$", text))
    selected = next((index for index, heading in enumerate(headings)
                     if int(heading.group(1)) == requested_id), None)
    if selected is None:
        raise ValueError(f"No metadata section for record {requested_id:02d} in {metadata_path.name}")
    start = headings[selected].end()
    end = headings[selected + 1].start() if selected + 1 < len(headings) else len(text)
    section = text[start:end]

    def value(label):
        match = re.search(rf"(?m)^{re.escape(label)}:\s*(.*?)\s*$", section)
        return match.group(1).strip() if match else None

    def number(label):
        match = re.search(r"[0-9]+(?:\.[0-9]+)?", value(label) or "")
        return float(match.group()) if match else None

    def items(label):
        raw = value(label)
        return [item.strip() for item in raw.split(",") if item.strip()] if raw else []

    original_section = section.split("Original Fix metadata:", 1)[-1]
    original = {}
    for label in ("Signals", "Numerics"):
        match = re.search(rf"(?m)^{label}:\s*(.*?)\s*$", original_section)
        if match:
            original[label.lower()] = [part.strip() for part in match.group(1).split(";") if part.strip()]
    metadata = {
        "record_id": f"BIDMC_{requested_id:02d}",
        "dataset": value("Dataset"),
        "source": value("Source"),
        "waveform_sample_rate_hz": number("Waveform sampling rate"),
        "parameter_sample_rate_hz": number("Numerics sampling rate"),
        "waveform_channels": items("Waveforms"),
        "parameter_channels": items("Parameters"),
        "age": value("Age"),
        "gender": value("Gender"),
        "care_location": value("Location"),
        "original_channel_metadata": original,
        "metadata_file": metadata_path.name,
    }
    warnings = []
    original_signals = metadata["original_channel_metadata"].get("signals", [])
    if original_signals and original_signals != metadata["waveform_channels"]:
        warnings.append(
            "Original Fix signal labels differ from the normalized workbook channels. "
            "The workbook does not identify the source ECG lead; do not infer missing leads."
        )
    original_parameters = metadata["original_channel_metadata"].get("numerics", [])
    if original_parameters and original_parameters != metadata["parameter_channels"]:
        warnings.append(
            "The normalized parameter labels differ from the Original Fix metadata labels. "
            "The workbook columns use the normalized Parameters list; verify RR/PR meanings before clinical use."
        )
    if warnings:
        metadata["metadata_warnings"] = warnings
    return metadata


def _load_combined_record(path):
    """Split one headerless BIDMC workbook into waveforms and 1 Hz parameters."""
    path = Path(path)
    metadata = _record_metadata(path)
    if metadata is None:
        raise ValueError(f"No companion metadata found for combined workbook: {path.name}")
    raw = pd.read_excel(path, sheet_name=0, header=None)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    expected = 1 + len(metadata["waveform_channels"]) + len(metadata["parameter_channels"])
    if raw.shape[1] != expected:
        raise ValueError(
            f"{path.name} has {raw.shape[1]} populated columns, but its metadata describes "
            f"{expected} columns (time + {len(metadata['waveform_channels'])} waveforms + "
            f"{len(metadata['parameter_channels'])} parameters). Check the workbook and metadata."
        )
    columns = ["Time_sec", *metadata["waveform_channels"], *metadata["parameter_channels"]]
    raw.columns = columns
    for column in columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["Time_sec"]).set_index("Time_sec").sort_index()
    if raw.index.has_duplicates:
        raw = raw.groupby(level=0).mean(numeric_only=True)

    signal_names = metadata["waveform_channels"]
    parameter_names = metadata["parameter_channels"]
    signals = raw[signal_names].copy()
    parameters = raw[parameter_names].copy()
    source_rate = metadata["waveform_sample_rate_hz"]
    parameter_rate = metadata["parameter_sample_rate_hz"]
    if source_rate and parameter_rate and source_rate > parameter_rate:
        elapsed = signals.index.to_numpy(dtype=float) - float(signals.index.min())
        buckets = (elapsed * parameter_rate + 1e-8).astype(int)
        downsampled = parameters.groupby(buckets, sort=True).first()
        downsampled.index = float(raw.index.min()) + downsampled.index.to_numpy(dtype=float) / parameter_rate
        parameters = downsampled

    signals.attrs["record_metadata"] = metadata
    parameters.attrs["record_metadata"] = metadata
    return signals, parameters


def _is_combined_record(path):
    """Check whether a workbook has the companion BIDMC metadata schema."""
    path = Path(path)
    return (path.parent / "BIDMC_two_random_records_metadata.txt").is_file() and re.search(r"_\d+$", path.stem) is not None


def load_data(signals_file, numerics_file):
    """Load combined metadata-described records or separate legacy workbooks."""
    signals_path, numerics_path = Path(signals_file), Path(numerics_file)
    if signals_path.resolve() == numerics_path.resolve() or _is_combined_record(signals_path):
        return _load_combined_record(signals_path)
    signals = pd.read_excel(signals_path, sheet_name="Signals")
    vitals = pd.read_excel(numerics_path, sheet_name="Sheet1")
    signals = _index_by_time(signals, "ECG")
    vitals = _index_by_time(vitals, "monitor parameters")
    metadata = {
        "record_id": signals_path.stem,
        "dataset": "Imported workbook",
        "source": "Workbook columns",
        "waveform_sample_rate_hz": None,
        "parameter_sample_rate_hz": None,
        "waveform_channels": [str(column) for column in signals.columns],
        "parameter_channels": [str(column) for column in vitals.columns],
        "metadata_file": None,
    }
    signals.attrs["record_metadata"] = metadata
    vitals.attrs["record_metadata"] = metadata
    return signals, vitals


def load_signals(signals_file):
    """Load signal channels from a combined BIDMC or legacy signal workbook."""
    if _is_combined_record(signals_file):
        return _load_combined_record(signals_file)[0]
    frame = pd.read_excel(Path(signals_file), sheet_name="Signals")
    return _index_by_time(frame, "ECG")
