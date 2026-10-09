"""Project paths and model settings discovered from the project directory."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _default_data_dir():
    """Find the user's data folder without depending on its capitalization."""
    if os.getenv("ECG_DATA_DIR"):
        return Path(os.environ["ECG_DATA_DIR"])
    for candidate in (PROJECT_DIR / "data", PROJECT_DIR / "DATA"):
        if candidate.is_dir():
            return candidate
    return PROJECT_DIR


DATA_DIR = _default_data_dir()
_combined_default = DATA_DIR / "BIDMC_08.xlsx"
_has_combined_default = _combined_default.is_file()
SIGNALS_FILE = Path(os.getenv("ECG_SIGNALS_FILE", _combined_default if _has_combined_default else PROJECT_DIR / "3103017_signals.xlsx"))
NUMERICS_FILE = Path(os.getenv("ECG_NUMERICS_FILE", _combined_default if _has_combined_default else PROJECT_DIR / "3103017_Numerics.xlsx"))
SAMPLE_RATE_HZ = int(os.getenv("ECG_SAMPLE_RATE_HZ", "125"))
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.arvancloudai.ir/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
API_KEY_FILE = Path(os.getenv("OPENAI_API_KEY_FILE", PROJECT_DIR / "api.txt"))


def get_api_key():
    """Read the API credential from the environment or the configured key file."""
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    if API_KEY_FILE.is_file():
        file_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key
    raise RuntimeError(
        f"Set OPENAI_API_KEY or place the key in {API_KEY_FILE} "
        "(override the path with OPENAI_API_KEY_FILE)."
    )
