"""Extract timestamped monitor parameters from screenshots using image input."""

import base64
import json
import mimetypes
from pathlib import Path
import re
import pandas as pd
from config import LLM_BASE_URL, LLM_MODEL, get_api_key


def _model():
    """Create the same OpenAI-compatible client used for narrative analysis."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, api_key=get_api_key(), base_url=LLM_BASE_URL)


def _parse_json(text):
    """Accept JSON responses with or without a Markdown code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def extract_monitor_frame(image_paths):
    """OCR one monitor screenshot per observation; reject missing timestamps."""
    model = _model()
    observations = []
    for image_path in image_paths:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError(f"Unsupported image type: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = ("Read this patient-monitor screenshot as OCR only. Return JSON only in this schema: "
                  '{"timestamp":"ISO-8601 timestamp exactly as displayed, else null",'
                  '"parameters":{"display label":number or string},'
                  '"waveforms_present":["labels visibly shown"],"uncertain_fields":["labels"]}. '
                  "Copy only visible values and labels; do not infer hidden values, units, timestamps, diagnoses, or trends. "
                  "Keep every numeric monitor parameter that is legible. If a value is unclear, omit it and name its label in uncertain_fields.")
        try:
            response = model.invoke([("human", [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ])])
        except Exception as error:
            raise RuntimeError("The configured API endpoint rejected image input. Check that its GPT-5.6 Luna route supports vision.") from error
        result = _parse_json(response.content)
        if not result.get("timestamp"):
            raise ValueError(f"OCR could not read the timestamp in {path.name}; add a visible timestamp or supply a time separately.")
        result["source_image"] = path.name
        observations.append(result)
    if not observations:
        raise ValueError("Choose at least one monitor screenshot")
    rows = []
    for obs in observations:
        # Preserve the monitor's displayed local clock; do not silently shift it to UTC.
        timestamp = pd.to_datetime(obs["timestamp"], errors="coerce")
        if pd.isna(timestamp):
            raise ValueError(f"Unreadable OCR timestamp: {obs['timestamp']!r}")
        row = {"timestamp": timestamp}
        row.update(obs.get("parameters") or {})
        row["_source_image"] = obs["source_image"]
        row["_uncertain_fields"] = ", ".join(obs.get("uncertain_fields") or [])
        row["_waveforms_present"] = ", ".join(obs.get("waveforms_present") or [])
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
    frame.attrs["ocr_observations"] = observations
    return frame

