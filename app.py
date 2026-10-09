"""Tkinter desktop interface for loading, plotting, and analyzing monitor data."""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from config import NUMERICS_FILE, SAMPLE_RATE_HZ, SIGNALS_FILE
from llm_analysis import analysis_period_label, analysis_segments, analyze_segment
from pipeline import load_ocr_sources, load_workbook_sources


class MonitorAnalysisApp:
    """A compact desktop UI for the full signal view and English analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("Vital Monitor Signal Analysis")
        self.root.geometry("1380x900")
        self.root.minsize(1000, 700)
        self.signals = None
        self.parameters = None
        self.evidence = None
        self.analysis_periods = []
        self.analysis_period_index = 0
        self.period_labels = []
        self.screenshot_paths = []
        self.signals_path = tk.StringVar(value=str(SIGNALS_FILE))
        self.parameters_path = tk.StringVar(value=str(NUMERICS_FILE))
        self.use_screenshots = tk.BooleanVar(value=False)
        self.include_ecg = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Choose Excel workbooks or monitor screenshots to begin.")
        self._build_ui()

    def _build_ui(self):
        """Create source controls, chart tabs, parameter table, and report view."""
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Vital Monitor Signal Analysis", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Load all available channels, inspect the complete time series, then request a time-aware English summary.").pack(anchor="w", pady=(2, 10))

        sources = ttk.LabelFrame(outer, text="Data sources", padding=9)
        sources.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(sources, text="Use timestamped monitor screenshots (OCR) instead of the parameter workbook",
                        variable=self.use_screenshots, command=self._toggle_source_controls).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(sources, text="Signal / combined data workbook").grid(row=1, column=0, sticky="w")
        self.signal_entry = ttk.Entry(sources, textvariable=self.signals_path)
        self.signal_entry.grid(row=1, column=1, sticky="ew", padx=6)
        self.signal_browse = ttk.Button(sources, text="Browse…", command=self._choose_signal_file)
        self.signal_browse.grid(row=1, column=2)
        ttk.Label(sources, text="Separate parameter workbook (same file for combined data)").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.parameter_entry = ttk.Entry(sources, textvariable=self.parameters_path)
        self.parameter_entry.grid(row=2, column=1, sticky="ew", padx=6, pady=(5, 0))
        self.parameter_browse = ttk.Button(sources, text="Browse…", command=self._choose_parameter_file)
        self.parameter_browse.grid(row=2, column=2, pady=(5, 0))
        self.screenshot_button = ttk.Button(sources, text="Choose screenshots…", command=self._choose_screenshots)
        self.screenshot_button.grid(row=3, column=1, sticky="w", pady=(7, 0))
        self.screenshot_label = ttk.Label(sources, text="No screenshots selected")
        self.screenshot_label.grid(row=3, column=2, sticky="w", padx=7, pady=(7, 0))
        self.include_ecg_check = ttk.Checkbutton(sources, text="Include sampled ECG workbook for HRV", variable=self.include_ecg)
        self.include_ecg_check.grid(row=4, column=1, sticky="w", pady=(5, 0))
        sources.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 7))
        self.load_button = ttk.Button(actions, text="Load data and plot all channels", command=self._load_data)
        self.load_button.pack(side="left")
        ttk.Label(actions, text="15-minute period:").pack(side="left", padx=(12, 4))
        self.period_choice = tk.StringVar()
        self.period_selector = ttk.Combobox(actions, textvariable=self.period_choice, state="disabled", width=48)
        self.period_selector.pack(side="left")
        self.analyze_button = ttk.Button(actions, text="Show selected period analysis", command=self._analyze_data, state="disabled")
        self.analyze_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status).pack(side="left", padx=12)

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.signal_tab = ttk.Frame(self.tabs)
        self.parameter_tab = ttk.Frame(self.tabs)
        self.analysis_tab = ttk.Frame(self.tabs)
        self.metadata_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.signal_tab, text="Waveforms")
        self.tabs.add(self.parameter_tab, text="Monitor Parameters")
        self.tabs.add(self.analysis_tab, text="Analysis")
        self.tabs.add(self.metadata_tab, text="Record Metadata")
        self._build_table(self.parameter_tab)
        self.analysis_text = tk.Text(self.analysis_tab, wrap="word", font=("Segoe UI", 11), padx=12, pady=10)
        self.analysis_text.pack(fill="both", expand=True)
        self.analysis_text.insert("1.0", "Choose a 15-minute period above, then click Show selected period analysis.")
        self.analysis_text.configure(state="disabled")
        self.metadata_text = tk.Text(self.metadata_tab, wrap="word", font=("Segoe UI", 10), padx=12, pady=10)
        self.metadata_text.pack(fill="both", expand=True)
        self.metadata_text.insert("1.0", "Metadata from the selected workbook will appear here.")
        self.metadata_text.configure(state="disabled")
        self._toggle_source_controls()

    def _build_table(self, parent):
        """Show every monitor parameter observation in a scrollable table."""
        container = ttk.Frame(parent, padding=6)
        container.pack(fill="both", expand=True)
        cols = ("timestamp", "parameter", "value", "source", "ocr_notes")
        self.parameter_table = ttk.Treeview(container, columns=cols, show="headings")
        labels = {"timestamp": "Timestamp", "parameter": "Parameter", "value": "Observed value",
                  "source": "Source", "ocr_notes": "OCR notes"}
        widths = {"timestamp": 210, "parameter": 180, "value": 180, "source": 180, "ocr_notes": 260}
        for column in cols:
            self.parameter_table.heading(column, text=labels[column])
            self.parameter_table.column(column, width=widths[column], anchor="w")
        yscroll = ttk.Scrollbar(container, orient="vertical", command=self.parameter_table.yview)
        xscroll = ttk.Scrollbar(container, orient="horizontal", command=self.parameter_table.xview)
        self.parameter_table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.parameter_table.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

    def _toggle_source_controls(self):
        """Enable controls for the selected parameter input mode."""
        ocr = self.use_screenshots.get()
        for widget in (self.parameter_entry, self.parameter_browse):
            widget.configure(state="disabled" if ocr else "normal")
        self.screenshot_button.configure(state="normal" if ocr else "disabled")
        self.screenshot_label.configure(state="normal" if ocr else "disabled")
        self.include_ecg_check.configure(state="normal" if ocr else "disabled")

    def _choose_signal_file(self):
        """Select the workbook containing signal channels."""
        path = filedialog.askopenfilename(title="Choose signal workbook", filetypes=[("Excel workbooks", "*.xlsx *.xls"), ("All files", "*.*")])
        if path:
            self.signals_path.set(path)

    def _choose_parameter_file(self):
        """Select the workbook containing monitor parameters."""
        path = filedialog.askopenfilename(title="Choose parameter workbook", filetypes=[("Excel workbooks", "*.xlsx *.xls"), ("All files", "*.*")])
        if path:
            self.parameters_path.set(path)

    def _choose_screenshots(self):
        """Select monitor images with visible timestamps for OCR."""
        paths = filedialog.askopenfilenames(title="Choose timestamped monitor screenshots", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.gif"), ("All files", "*.*")])
        if paths:
            self.screenshot_paths = list(paths)
            self.screenshot_label.configure(text=f"{len(paths)} screenshot(s) selected")

    def _run_in_background(self, work, complete):
        """Run file parsing or network calls away from the Tk event loop."""
        def worker():
            try:
                result = work()
                self.root.after(0, lambda: complete(result, None))
            except Exception as error:
                self.root.after(0, lambda error=error: complete(None, error))
        threading.Thread(target=worker, daemon=True).start()

    def _load_data(self):
        """Load the selected workbooks or OCR screenshots, then draw every channel."""
        if self.use_screenshots.get() and not self.screenshot_paths:
            messagebox.showinfo("Screenshots required", "Choose at least one timestamped monitor screenshot.")
            return
        self._set_busy(True, "Loading data…")
        if self.use_screenshots.get():
            ecg_file = self.signals_path.get() if self.include_ecg.get() else None
            image_paths = tuple(self.screenshot_paths)
            def work():
                return load_ocr_sources(image_paths, ecg_file, None, SAMPLE_RATE_HZ)
        else:
            signals_file = self.signals_path.get()
            parameters_file = self.parameters_path.get()
            def work():
                return load_workbook_sources(signals_file, parameters_file, SAMPLE_RATE_HZ)
        self._run_in_background(work, self._loaded)

    def _loaded(self, result, error):
        """Display loaded channels and save the evidence for the analysis action."""
        self._set_busy(False)
        if error:
            self.status.set("Could not load the selected data.")
            messagebox.showerror("Load failed", str(error))
            return
        self.signals, self.parameters, self.evidence = result
        self.analysis_periods = analysis_segments(self.evidence)
        self.period_labels = [analysis_period_label(period) for period in self.analysis_periods]
        self.period_selector.configure(values=self.period_labels, state="readonly" if self.period_labels else "disabled")
        if self.period_labels:
            self.period_choice.set(self.period_labels[0])
        self._show_metadata(self.evidence.get("record_metadata", {}))
        self._draw_charts(self.signal_tab, self.signals, "Signal channels")
        self._draw_charts(self.parameter_tab, self.parameters, "Monitor parameters", table=True)
        channel_count = 0 if self.signals is None else len(self.signals.columns)
        parameter_count = len([c for c in self.parameters.columns if not str(c).startswith("_")])
        self.status.set(f"Loaded {channel_count} signal channel(s), {parameter_count} parameter(s); choose a period to analyze.")
        self.analyze_button.configure(state="normal" if self.analysis_periods else "disabled")
        self.tabs.select(self.signal_tab)

    def _show_metadata(self, metadata):
        """Display the workbook's data dictionary and sampling metadata."""
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", "end")
        if not metadata:
            self.metadata_text.insert("1.0", "No companion metadata was found. Column names were read from the workbook.")
        else:
            for key, value in metadata.items():
                self.metadata_text.insert("end", f"{key.replace('_', ' ').title()}: {value}\n")
        self.metadata_text.configure(state="disabled")

    def _draw_charts(self, tab, frame, title, table=False):
        """Render each supplied numeric series on its own complete-time axis."""
        for child in tab.winfo_children():
            child.destroy()
        if table:
            pages = ttk.Notebook(tab)
            pages.pack(fill="both", expand=True)
            chart_page = ttk.Frame(pages)
            value_page = ttk.Frame(pages)
            pages.add(chart_page, text="Trends")
            pages.add(value_page, text="All observations")
            self._build_table(value_page)
            self._fill_parameter_table(frame)
            tab = chart_page
        if frame is None or frame.empty:
            ttk.Label(tab, text=f"No {title.lower()} were supplied.", padding=18).pack(anchor="w")
            return
        visible = [column for column in frame.columns if not str(column).startswith("_")]
        if not visible:
            ttk.Label(tab, text=f"No displayable {title.lower()} were found.", padding=18).pack(anchor="w")
            return
        figure = Figure(figsize=(11, max(3.2, 2.8 * len(visible))), dpi=100, constrained_layout=True)
        axes = figure.subplots(len(visible), 1, squeeze=False).ravel()
        plotted = 0
        for axis, column in zip(axes, visible):
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().any():
                axis.plot(frame.index, series, linewidth=0.8)
                axis.set_ylabel(str(column))
                axis.grid(True, alpha=0.25)
                plotted += 1
            else:
                axis.text(0.5, 0.5, f"Categorical observations for {column} are listed in All observations.", ha="center", va="center", transform=axis.transAxes)
                axis.set_ylabel(str(column))
            axis.set_xlabel("Time")
        figure.suptitle(f"{title} — full available time range")
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        if not plotted:
            self.status.set(f"Loaded {title.lower()}; categorical data is available in the table.")

    def _fill_parameter_table(self, frame):
        """Insert each parameter value with its original timestamp and source."""
        if frame is None:
            return
        for timestamp, row in frame.iterrows():
            source = row.get("_source_image", "workbook")
            notes = row.get("_uncertain_fields", "")
            for parameter, value in row.items():
                if str(parameter).startswith("_") or pd.isna(value):
                    continue
                self.parameter_table.insert("", "end", values=(str(timestamp), str(parameter), str(value), str(source), str(notes)))

    def _analyze_data(self):
        """Analyze only the period explicitly selected by the user."""
        if self.evidence is None:
            return
        try:
            self.analysis_period_index = self.period_labels.index(self.period_choice.get())
        except ValueError:
            messagebox.showinfo("Choose a period", "Select an available 15-minute period first.")
            return
        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.configure(state="disabled")
        self._analyze_current_period()

    def _analyze_current_period(self):
        """Send only the selected 15-minute period to the LLM."""
        period = self.analysis_periods[self.analysis_period_index]
        number = self.analysis_period_index + 1
        total = len(self.analysis_periods)
        self._set_busy(True, f"Analyzing period {number} of {total}…")
        self._run_in_background(lambda: analyze_segment(period), self._analysis_done)

    def _analysis_done(self, result, error):
        """Show one English analysis, then ask before sending another period."""
        self._set_busy(False)
        self.tabs.select(self.analysis_tab)
        self.analysis_text.configure(state="normal")
        if error:
            self.analysis_text.insert("end", f"Analysis stopped: {error}\n")
            self.analysis_text.configure(state="disabled")
            self.status.set("Analysis failed.")
            return
        self.analysis_text.insert("end", f"Period {self.analysis_period_index + 1} of {len(self.analysis_periods)}\n{result}\n\n")
        self.analysis_text.configure(state="disabled")
        if self.analysis_period_index + 1 < len(self.analysis_periods):
            continue_analysis = messagebox.askyesno(
                "Continue analysis?",
                "The current 15-minute period is complete. Would you like to send the next period to the model?",
            )
            if continue_analysis:
                self.analysis_period_index += 1
                self.period_choice.set(self.period_labels[self.analysis_period_index])
                self._analyze_current_period()
                return
            self.status.set("Analysis stopped at your request.")
            return
        self.status.set("All available periods have been analyzed.")

    def _set_busy(self, busy, message=None):
        """Keep the UI responsive and prevent duplicate requests."""
        self.load_button.configure(state="disabled" if busy else "normal")
        if hasattr(self, "period_selector"):
            self.period_selector.configure(state="disabled" if busy or not self.period_labels else "readonly")
        if self.evidence is not None:
            self.analyze_button.configure(state="disabled" if busy else "normal")
        if message:
            self.status.set(message)


def main():
    """Start the desktop application."""
    root = tk.Tk()
    MonitorAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

