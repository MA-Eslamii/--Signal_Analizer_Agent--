
# Demo_Video
https://drive.google.com/file/d/1Q2uyHHX9DBf5MdLInDMTx0MNjcwDO6P7/view?usp=sharing


# ECG and monitor analysis

Modular research workflow for monitor workbooks or timestamped monitor screenshots. The pipeline keeps measured data, gain-independent ECG timing patterns, heuristic quality flags, and optional LLM interpretation as distinct evidence. The model may suggest cautious rhythm-pattern possibilities, not diagnose a heart condition.

## Setup

Use Python 3.10 or newer. Install dependencies with `python -m pip install -r requirements.txt`. The application reads `OPENAI_API_KEY` from the environment first, then falls back to `api.txt` in the project directory. You can set `OPENAI_API_KEY_FILE` to use another file. Keep the credential file private; `api.txt` is ignored by Git. The default input is the combined `BIDMC_08.xlsx` workbook under the discovered `data`/`DATA` folder. Choose `BIDMC_41.xlsx` in the UI to inspect the other record. Legacy split workbooks remain supported.

The default data folder is discovered as either `data` or `DATA`. The combined `BIDMC_08.xlsx` and `BIDMC_41.xlsx` files are headerless: `BIDMC_two_random_records_metadata.txt` supplies their record-specific column names and sampling rates. These workbooks contain both signals and monitor parameters, so the desktop app starts with the same workbook selected for both source fields. It displays metadata in the Record Metadata tab. The normalized signal and parameter names differ from the original source labels in this metadata; the UI and analysis preserve those warnings rather than silently relabeling channels or inventing ECG lead identities.

## Run

Start the desktop interface with `python main.py` (or `python app.py`). It plots all signal channels over their full available time range, lists monitor parameter observations, and runs the English analysis without freezing the window. For command-line use, add `--cli`: `python main.py --cli` analyzes the Excel workbooks. Use `python main.py --cli --ocr monitor_1.jpg monitor_2.png` to OCR screenshot observations instead. The image must visibly include its timestamp; ambiguous fields are called out rather than guessed. To pair OCR parameters with the workbook ECG, add `--ecg-file 3103017_signals.xlsx`.

Both modes split parameter observations into separate 15-minute blocks. Only columns actually supplied are included; absent parameters are omitted. Each block includes its observed timestamps. Screenshot timestamps remain in the monitor's displayed timezone. A shorter recording is labeled by its actual coverage and is not presented as a complete 15-minute segment. The command-line workflow writes its evidence to `analysis_evidence.json`; the desktop workflow presents the English summary in the Analysis tab.

## OCR and model input

GPT-5.6 Luna accepts image input through the OpenAI API. The OCR mode sends each selected screenshot as an image plus a strict transcription request and asks the model to return only visible labels, values, and timestamps. It does not infer trends from one screenshot. The project currently defaults to an OpenAI-compatible Arvan endpoint; verify that the specific gateway/model route supports image inputs. If it rejects images, configure a compatible vision endpoint through `OPENAI_BASE_URL`.

An image of the monitor's displayed HR is a sparse numeric observation, not the beat-to-beat ECG signal. HRV is calculated only from sampled ECG R peaks or a real RR/IBI interval stream; it is never inferred from displayed HR, SpO2, or OCR alone. The LLM does not receive ECG amplitude or absolute HRV values: cycle timing is normalized to each window's own typical interval and summarized as relative changes. This makes the narrative less sensitive to device gain. It is still a cautious signal description, not a diagnosis.

The desktop UI lists available 15-minute periods with their observed time coverage (and marks shorter recordings as partial). Select a period and click “Show selected period analysis”; after its result, the UI asks before sending the next period. The CLI starts at the first period and asks before continuing. Each English response describes the observable physiological pattern and may cautiously suggest a timing-based rhythm pattern such as irregular or ectopic-like cycles. It also reports whether heuristic ECG-quality windows were flagged and names flagged types such as flatline, abrupt motion, baseline wander, or high-frequency noise. “No flags” does not mean “normal ECG.” The model cannot identify structural disease, ischemia, or confirm an arrhythmia from RR timing alone. A clinician must review the source tracing; rhythm diagnosis often needs review of the ECG waveform and appropriate leads. [AHA: common arrhythmia tests](https://www.heart.org/en/health-topics/arrhythmia/symptoms-diagnosis--monitoring-of-arrhythmia/common-tests-for-arrhythmia), [AHA: how AFib is diagnosed](https://www.heart.org/en/health-topics/atrial-fibrillation/treatment-and-prevention-of-atrial-fibrillation/afib-diagnosis). Separate source timelines are kept distinct unless their timestamps align.

## Modules

- `data_loader.py`: workbook reading and timestamp validation.
- `filtering.py`: ECG band-pass and vital-sign median filters.
- `ocr_ingestion.py`: timestamp-preserving OCR from monitor screenshots.
- `time_analysis.py`: 15-minute parameter windows and ECG/RR-based HRV.
- `artifacts.py`: advisory, multi-criterion waveform quality flags.
- `llm_analysis.py`: compact evidence preparation and optional LLM call.
- `pipeline.py`: shared source loading and evidence preparation.
- `app.py`: Tkinter desktop application.
- `main.py`: command-line orchestration.

## Interpretation limits

Automated R-peak detection, OCR, and artifact flags are fallible and need review against the original trace. OCR images cannot restore continuous waveform samples. Relative cycle patterns can suggest that a segment merits review, but missed beats, ectopy, gaps, and artifact can mimic rhythm changes. The new BIDMC traces are eight minutes long, so the last/only period is a partial 15-minute window. Only leads actually present are analyzed, and signal and parameter sources retain separate time coverage. The LLM does not receive ECG amplitude or absolute HRV values and cannot diagnose a definitive rhythm or heart condition.

