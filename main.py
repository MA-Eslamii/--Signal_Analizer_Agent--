"""Run monitor analysis from Excel workbooks or timestamped OCR screenshots."""

import argparse
import json
import sys
from pathlib import Path
from config import SIGNALS_FILE, NUMERICS_FILE, SAMPLE_RATE_HZ


def _arguments():
    """Define input sources and output location for the command-line workflow."""
    parser = argparse.ArgumentParser(description="Analyze monitor trends in separate 15-minute blocks")
    parser.add_argument("--cli", action="store_true", help="Use the command-line workflow instead of the desktop UI")
    parser.add_argument("--ocr", nargs="+", metavar="IMAGE", help="OCR timestamped monitor screenshots")
    parser.add_argument("--ecg-file", help="Optional ECG workbook paired with screenshot parameters")
    parser.add_argument("--signals-file", default=str(SIGNALS_FILE))
    parser.add_argument("--parameters-file", default=str(NUMERICS_FILE))
    parser.add_argument("--output", default="analysis_evidence.json")
    return parser.parse_args()


def main():
    """Open the desktop UI by default; use --cli to run the command-line workflow."""
    args = _arguments()
    if not args.cli:
        try:
            from app import main as run_desktop_app
            run_desktop_app()
        except ModuleNotFoundError as error:
            print(f"The desktop UI could not start because module '{error.name}' is missing. Install project packages with: {sys.executable} -m pip install -r requirements.txt")
            return
        return
    try:
        from pipeline import load_ocr_sources, load_workbook_sources
        from llm_analysis import analysis_segments, analyze_segment
    except ModuleNotFoundError as error:
        print(f"Missing dependency: {error.name}. Install project packages with: {sys.executable} -m pip install -r requirements.txt")
        return
    if args.ocr:
        _, _, evidence = load_ocr_sources(args.ocr, args.ecg_file, args.parameters_file, SAMPLE_RATE_HZ)
    else:
        _, _, evidence = load_workbook_sources(args.signals_file, args.parameters_file, SAMPLE_RATE_HZ)
    Path(args.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(f"Evidence saved to {args.output}")
    periods = analysis_segments(evidence)
    for index, period in enumerate(periods):
        try:
            print(f"\nPeriod {index + 1} of {len(periods)}")
            print(analyze_segment(period))
        except RuntimeError as error:
            print(error)
            return
        if index + 1 < len(periods):
            try:
                answer = input("Would you like to continue with the next 15-minute period? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in {"y", "yes"}:
                print("Stopped before sending the next period.")
                break


if __name__ == "__main__":
    main()

