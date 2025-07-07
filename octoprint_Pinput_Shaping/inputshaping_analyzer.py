"""Input Shaping Analyzer for OctoPrint Plugin Pinput_Shaping"""

import logging
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

# Define a constant for maximum bytes for memory optimization in compute_psd
MAX_BYTES_32 = 2_000_000_000  # ~ 2 GiB


class InputShapingAnalyzer: # pylint: disable=too-many-instance-attributes
    """Class to analyze input shaping data from a CSV file.
    It loads the data, applies low-pass filtering, computes the Power Spectral Density (PSD),
    generates input shapers, applies them, and generates graphs.
    It also provides methods to get recommendations for input shaping commands.
    """

    def __init__( # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, save_dir: str, csv_path: str, damping: float = 0.5,
        cutoff_freq: int = 100, axis: Optional[str] = None, logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initializes the InputShapingAnalyzer with the given parameters.
        :param save_dir: Directory to save the results.
        :param csv_path: Path to the CSV file containing the raw acceleration data.
        :param damping: Damping factor for the input shapers (default is 0.5).
        :param cutoff_freq: Cutoff frequency for the low-pass filter (default is 100 Hz).
        :param axis: Axis to analyze (e.g., "X", "Y", "Z"). If None, it will be set to "X".
        :param logger: Optional logger for logging messages. If None, a default logger will be used.
        """

        # Initialize logger: use provided logger or create a new one
        self._plugin_logger = logger or logging.getLogger(
            "octoprint.plugins.Pinput_Shaping"
        )

        # Assign input parameters to instance attributes
        self.csv_path = csv_path
        self.damping = damping
        self.cutoff_freq = cutoff_freq

        # Ensure axis is a string and uppercase, defaulting to "X" if None
        if axis is None:
            self._plugin_logger.warning("Axis not specified, defaulting to 'X'.")
            self.axis = "X"
        else:
            self.axis = axis.upper() # Safe to call .upper() after the None check

        self.result_dir = save_dir # Directory where graphs and results will be saved

        # Initialize attributes that will be populated later in the analysis process
        self.best_shaper: Optional[str] = None
        self.base_freq: Optional[float] = None
        self.shaper_results: Dict[str, Dict[str, Any]] = {} # Store shaper analysis results
        self.time: Optional[np.ndarray] = None # Time data from CSV
        self.raw: Optional[np.ndarray] = None # Raw acceleration data from CSV
        self.filtered: Optional[np.ndarray] = None # Filtered acceleration data
        self.sampling_rate: Optional[float] = None # Determined from time data
        self.freqs: Optional[np.ndarray] = None # Frequencies from PSD
        self.psd: Optional[np.ndarray] = None # Power Spectral Density

    def load_data(self) -> None:
        """Loads the data from the CSV file and processes it.
        Raises ValueError if the specified axis column is not found or if data is invalid.
        """

        self._plugin_logger.info(
            f"Loading data from CSV file {self.csv_path} for axis {self.axis}"
        )
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError:
            self._plugin_logger.error(f"CSV file not found at {self.csv_path}")
            raise # Re-raise the error to be handled by the caller

        # Clean column names by stripping whitespace and converting to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Process 'time' column
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            raise ValueError("No valid time data found in CSV after cleaning.")

        # Process selected axis column
        axis_col = self.axis.lower()  # "x", "y", or "z"
        if axis_col not in df.columns:
            raise ValueError(f"Column '{axis_col}' not found in CSV file '{self.csv_path}'")

        df[axis_col] = pd.to_numeric(df[axis_col], errors="coerce")
        df = df.dropna(subset=[axis_col])
        if df.empty:
            raise ValueError(f"No valid data found for axis '{self.axis}' in CSV after cleaning.")

        self.time = df["time"].to_numpy(dtype=np.float64)
        self.raw = df[axis_col].to_numpy(dtype=np.float64)

        # Calculate sampling rate
        time_diffs = np.diff(self.time)
        if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
            raise ValueError("Cannot determine sampling rate: insufficient or constant time data.")
        self.sampling_rate = 1.0 / np.mean(time_diffs)

    def lowpass_filter(self, data: np.ndarray, order: int = 4) -> np.ndarray:
        """Applies a low-pass Butterworth filter to the data.
        Ensures sampling_rate is set before filtering.
        :param data: The input data to filter.
        :param order: The order of the Butterworth filter (default is 4).
        :return: The filtered data.
        Raises ValueError if sampling rate is not set or cutoff frequency is invalid.
        """
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Call load_data() first.")

        nyq = 0.5 * self.sampling_rate
        cutoff = min(self.cutoff_freq, nyq * 0.99) # Ensure cutoff doesn't exceed Nyquist
        norm_cutoff = cutoff / nyq

        self._plugin_logger.debug(
            "lowpass_filter: cutoff=%s, nyq=%s, norm_cutoff=%s, sampling_rate=%s",
            cutoff, nyq, norm_cutoff, self.sampling_rate
        )

        # Validate normalized cutoff frequency
        if not 0 < norm_cutoff < 1:
            self._plugin_logger.error(
                "Invalid normalized cutoff frequency: %s (cutoff=%s, nyq=%s)",
                norm_cutoff, cutoff, nyq
            )
            raise ValueError(
                f"Digital filter critical frequencies must be 0 < Wn < 1 "
                f"(got {norm_cutoff}, cutoff={cutoff}, nyq={nyq})"
            )
        b, a = butter(order, norm_cutoff, btype="low") # type: ignore
        return filtfilt(b, a, data)

    def generate_shapers(self, freq: float) -> dict:
        """Generates input shapers based on the given frequency.
        :param freq: The resonant frequency for which to generate shapers.
        :return: A dictionary of shaper types with their impulse sequences.
        """

        t = 1 / freq # Period of the vibration
        # Damping factor 'k' for the shaper equations
        k = np.exp(-self.damping * np.pi / np.sqrt(1 - self.damping**2))
        shapers = {}

        # Zero Vibration (ZV) shaper
        shapers["ZV"] = [(0, 1 / (1 + k)), (t, k / (1 + k))]

        # Modified ZV (MZV) shaper
        shapers["MZV"] = [
            (0,   1 / (1 + k + k**2)),
            (t,   k / (1 + k + k**2)),
            (2*t, k**2 / (1 + k + k**2)),
        ]

        # Extra Insensitive (EI) shaper
        shapers["EI"] = [
            (0,   1 / (1 + 3*k + 3*k**2 + k**3)),
            (t,   3*k / (1 + 3*k + 3*k**2 + k**3)),
            (2*t, 3*k**2 / (1 + 3*k + 3*k**2 + k**3)),
            (3*t, k**3 / (1 + 3*k + 3*k**2 + k**3)),
        ]

        # 2-Hump Extra Insensitive (2HUMP_EI) shaper
        shapers["2HUMP_EI"] = [
            (0,   1 / (1 + 4*k + 6*k**2 + 4*k**3 + k**4)),
            (t,   4*k / (1 + 4*k + 6*k**2 + 4*k**3 + k**4)),
            (2*t, 6*k**2 / (1 + 4*k + 6*k**2 + 4*k**3 + k**4)),
            (3*t, 4*k**3 / (1 + 4*k + 6*k**2 + 4*k**3 + k**4)),
            (4*t, k**4 / (1 + 4*k + 6*k**2 + 4*k**3 + k**4)),
        ]

        # 3-Hump Extra Insensitive (3HUMP_EI) shaper
        shapers["3HUMP_EI"] = [
            (0,   1 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (t,   6*k / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (2*t, 15*k**2 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (3*t, 20*k**3 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (4*t, 15*k**4 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (5*t, 6*k**5 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
            (6*t, k**6 / (1 + 6*k + 15*k**2 + 20*k**3 + 15*k**4 + 6*k**5 + k**6)),
        ]

        return shapers

    def apply_shaper(self, signal: np.ndarray, time_data: np.ndarray,
                     shaper: list[tuple[float, float]]) -> np.ndarray:
        """Applies the input shaper to the signal.
        :param signal: The input signal to shape.
        :param time_data: The time vector corresponding to the signal.
        :param shaper: The input shaper to apply, defined as a list of (delay, amplitude) tuples.
        :return: The shaped signal.
        """

        # Ensure time_data is not empty to avoid division by zero
        if len(time_data) < 2:
            raise ValueError("Time data must contain at least two points to calculate dt.")
        dt = np.mean(np.diff(time_data))
        if dt == 0:
            raise ValueError("Time difference (dt) is zero, cannot apply shaper.")

        n = len(signal)
        shaped = np.zeros(n)
        for delay, amp in shaper:
            # Calculate shift in samples
            shift = int(np.round(delay / dt))
            if shift < n:
                # Apply the impulse response by adding a delayed and scaled version of the signal
                shaped[shift:] += amp * signal[: n - shift]
        return shaped

    def compute_psd(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the Power Spectral Density (PSD) of the signal using Welch's method.
        Ensures sampling_rate is set before computing PSD.
        :param signal: The input signal to analyze.
        :return: A tuple containing the frequencies and the corresponding PSD values.
        """
        if self.sampling_rate is None:
            raise ValueError("Sampling rate is not set. Call load_data() first.")

        # Convert signal to float32 to manage memory, if not already
        sig = signal.astype(np.float32, copy=False)

        # Determine nperseg (number of samples per segment) dynamically for Welch's method
        # Start with a reasonable default, then adjust to stay within memory limits
        nperseg = min(4096, len(sig) // 8)
        nperseg = max(nperseg, 256) # Minimum segment length

        # Adaptive Welch that guarantees not exceeding the limit of 2 GiB
        while True:
            # Number of windows for Welch's method
            n_win = len(sig) - nperseg + 1
            # Estimated memory usage for the internal buffer of Welch's method
            est_mem = n_win * nperseg * sig.itemsize
            if est_mem < MAX_BYTES_32 or nperseg <= 256:
                # Break if memory is within limits or nperseg is at its minimum
                break
            nperseg //= 2  # Reduce nperseg by half and try again

        self._plugin_logger.debug(
            "Welch: nperseg=%s, windows=%s, est_mem=%s MB, len=%s",
            nperseg, n_win, est_mem / 1e6, len(sig)
        )

        return welch(sig, fs=self.sampling_rate, nperseg=nperseg)

    def analyze(self) -> str:
        """Analyzes the input shaping data and returns the best shaper.
        Populates self.filtered, self.freqs, self.psd, self.base_freq, self.shaper_results,
        self.best_shaper. Raises ValueError if data loading or filtering fails.
        """

        self.load_data() # This populates self.time, self.raw, self.sampling_rate

        # Ensure raw data is available before filtering
        if self.raw is None:
            raise ValueError("Raw data is not loaded. Cannot perform analysis.")

        try:
            self.filtered = self.lowpass_filter(self.raw)
        except ValueError as e:
            self._plugin_logger.error(f"Lowpass filter failed: {e}")
            raise # Re-raise the error for upstream handling

        # Ensure filtered data is available before computing PSD
        if self.filtered is None:
            raise ValueError("Filtered data is not available. Cannot compute PSD.")

        self.freqs, self.psd = self.compute_psd(self.filtered)

        # Ensure freqs and psd are available
        if self.freqs is None or self.psd is None:
            raise ValueError("PSD computation failed. Frequencies or PSD data missing.")

        # Find the base frequency (dominant frequency in a specific range)
        freq_range = (self.freqs > 20) & (self.freqs < 80) # Common range for printer resonances
        if not np.any(freq_range):
            self._plugin_logger.warning("No dominant frequency found in the 20-80 Hz range."
                                        "Using max PSD frequency."
            )
            # Fallback: use the overall max PSD frequency if no dominant freq in range
            self.base_freq = self.freqs[np.argmax(self.psd)]
        else:
            self.base_freq = self.freqs[freq_range][np.argmax(self.psd[freq_range])]

        # Ensure base_freq is a valid float
        if self.base_freq is None or not isinstance(self.base_freq, (int, float)):
            raise ValueError("Base frequency could not be determined for shaper generation.")

        shapers = self.generate_shapers(self.base_freq)

        # Analyze each shaper
        for name, shaper in shapers.items():
            # Ensure filtered data and time are available for applying shaper
            if self.filtered is None or self.time is None:
                raise ValueError("Filtered data or time data missing for shaper application.")

            shaped = self.apply_shaper(self.filtered, self.time, shaper)
            _, shaped_psd = self.compute_psd(shaped)
            vibr = np.sum(shaped_psd) # Total vibration (sum of PSD)
            # Calculate peak acceleration from the shaped signal's gradient
            accel = max(np.abs(np.gradient(shaped, np.mean(np.diff(self.time)))))
            self.shaper_results[name] = {
                "psd": shaped_psd,
                "vibr": vibr,
                "accel": accel,
            }

        # Determine the best shaper based on minimum vibration
        if not self.shaper_results:
            raise ValueError("No shaper results generated. Cannot determine best shaper.")
        self.best_shaper = min(
            self.shaper_results, key=lambda s: self.shaper_results[s]["vibr"]
        )
        return self.best_shaper

    def generate_graphs(self) -> tuple[str, str, Dict[str, Any], str, float]:
        """Generates graphs for the original and filtered signals, and the PSD with input shapers.
        Ensures all necessary data (time, raw, filtered, freqs, psd, shaper_results, base_freq,
        best_shaper) are populated before generating graphs.
        :return: A tuple containing the paths to the generated graphs and the shaper results.
        """
        # Ensure all required data is available
        if self.time is None or self.raw is None or self.filtered is None or \
           self.freqs is None or self.psd is None or not self.shaper_results or \
           self.best_shaper is None or self.base_freq is None:
            raise ValueError("Missing data for graph generation. Run analyze() first.")

        # Extract date from csv_path (e.g., "Raw_accel_values_AXIS_X_20250416T133919.csv")
        # Added a check to ensure csv_path is a string before splitting
        if not isinstance(self.csv_path, str):
            raise ValueError("csv_path is not a string. Cannot extract date for graph filename.")

        try:
            date_part = os.path.basename(self.csv_path).split("_")[-1]
            date = date_part.split(".")[0] if "." in date_part else date_part
        except IndexError:
            self._plugin_logger.warning(
                f"Could not parse date from CSV path: {self.csv_path}. Using 'unknown_date'."
            )
            date = "unknown_date"


        # Signal Graph
        signal_path = os.path.join(self.result_dir, f"{self.axis}_signal_{date}.png")
        plt.figure(figsize=(14, 5))
        plt.plot(
            self.time[::50], # Sample every 50th point for performance/readability
            self.raw[::50],
            label="Original",
            alpha=0.4,
            color="#007bff",
        )
        plt.plot(
            self.time[::50], # Sample every 50th point for performance/readability
            self.filtered[::50],
            label="Filtered",
            linewidth=2.0,
            color="#ff7f0e",
        )
        plt.title(f"Signal - Axis {self.axis}", fontsize=14)
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(signal_path, dpi=150)
        plt.close()

        # PSD Graph
        psd_path = os.path.join(self.result_dir, f"{self.axis}_psd_{date}.png")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.freqs, self.psd, label="Original", color="black", linewidth=1.5)

        for name, result in self.shaper_results.items():
            # Ensure 'psd' key exists and is a numpy array for plotting
            if "psd" in result and isinstance(result["psd"], np.ndarray):
                label = (
                    f"{name} ({self.base_freq:.1f} Hz)  "
                    f"vibr={result['vibr']:.2e}  "
                    f"accel={result['accel']:.1f}"
                )
                ax.plot(
                    self.freqs, result["psd"], linestyle="--", linewidth=1.2, label=label
                )
            else:
                self._plugin_logger.warning(f"Shaper '{name}' missing 'psd' data for plotting.")


        ax.set_title(f"PSD with Input Shapers - Axis {self.axis}", fontsize=14)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (PSD)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(0, 200)
        # Ensure max_psd is calculated safely
        max_psd = np.max(self.psd) if self.psd is not None and len(self.psd) > 0 else 1.0
        ax.set_ylim(0, max_psd * 1.1)
        ax.legend(loc="upper right", fontsize=8)
        # Adjust lower space
        plt.subplots_adjust(bottom=0.35)
        # Recommended text
        recommendation_text = (
            f"Recommended: {self.best_shaper} ({self.base_freq:.1f} Hz)\n"
            f"Marlin CMD: M593 F{self.base_freq:.1f} D{self.damping} S{self.best_shaper}"
        )

        # Add the box behind the text
        fig.text(
            0.5,
            0.08,
            recommendation_text,
            ha="right",
            va="bottom",
            fontsize=10,
            zorder=2,
            bbox={"facecolor": 'white', "edgecolor": 'gray', "boxstyle": 'round,pad=0.5'},
        )

        plt.tight_layout(rect=(0, 0.03, 1, 1))  # Leaves space for text at bottom
        plt.savefig(psd_path, dpi=150)
        plt.close()

        return (
            signal_path,
            psd_path,
            self.shaper_results,
            self.best_shaper,
            self.base_freq
        )

    def get_recommendation(self) -> str:
        """Generates a recommendation string for the best input shaper.
        Ensures base_freq, damping, and best_shaper are set.
        """
        if self.base_freq is None or self.best_shaper is None:
            raise ValueError("Base frequency or best shaper not determined yet."
                             "Run analyze() first."
            )
        return f"M593 F{self.base_freq:.1f} D{self.damping} S{self.best_shaper}"

    def get_plotly_data(self) -> dict:
        """Generates data for Plotly visualization.
        Ensures all necessary data is populated.
        """
        if self.time is None or self.raw is None or self.filtered is None or \
           self.freqs is None or self.psd is None or not self.shaper_results or \
           self.base_freq is None or self.best_shaper is None:
            raise ValueError("Missing data for Plotly visualization. Run analyze() first.")

        # Ensure all data is converted to standard Python types for JSON serialization
        return {
            "time": [float(t) for t in self.time[::5]],
            "raw": [float(r) for r in self.raw[::5]],
            "filtered": [float(f) for f in self.filtered[::5]],
            "freqs": [float(f) for f in self.freqs],
            "psd_original": [float(p) for p in self.psd],
            "shapers": {
                name: {
                    "psd": [float(p) for p in result["psd"]],
                    "vibr": round(float(result["vibr"]), 3),
                    "accel": round(float(result["accel"]), 2),
                }
                for name, result in self.shaper_results.items()
            },
            "base_freq": round(float(self.base_freq), 2),
            "best_shaper": str(self.best_shaper)
        }
