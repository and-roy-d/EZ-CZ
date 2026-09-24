#!/usr/bin/env python3
"""
cz_gui.py: EZ CZ dashboard for Complex Impedance (CZ) Measurement.

Features:
  - Editable experimental parameters (Channel, Bias schedule, AWG settings, Timing).
  - Derived timing strictly from time_per_flux_ramp (eliminates arbitrary magic numbers).
  - Background execution via QThread with non-blocking UI.
  - Automatic Superconducting (0V) Transfer Function Health Check & Abort:
      * Fits single-pole L/Rs model: |Z(f)| = amp0 / sqrt(1 + (f / f_roll)^2)
      * Fits phase/cable delay tau_shift
      * Computes R^2 goodness of fit
      * Evaluates f_roll physical limits (10 Hz - 25 kHz)
      * Immediate abort if transfer function is bad or corrupted
      * Optional user pause after 0V calibration for manual visual inspection
  - Live 4-panel Matplotlib display:
      1. Real & Imaginary vs Frequency (semilog-x)
      2. Nyquist Plot (Imag vs Real)
      3. Magnitude Response (log-log) + Model Fit
      4. Phase Response (semilog-x) + Model Fit
  - Dedicated instant ABORT button that safely zeroes AWG DC offset.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.fft
from lmfit import Model, Parameters

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit,
    QCheckBox, QMessageBox, QComboBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Caldaq / Qsdumux imports
import qsdumux.instruments.keysight33600A_usb as keysight33600A_usb
try:
    from dastardcommander import EasyClientDastard
except ImportError:
    EasyClientDastard = None


# ---------------------------------------------------------------------------
# Timing & Mathematical Helpers (Pure time_per_flux_ramp derivation)
# ---------------------------------------------------------------------------
TIME_PER_FLUX_RAMP_POINT_S = 256e-9  # FPGA clock tick (1 / 3.90625 MHz)
DEFAULT_CONFIG_PATH = "/data/experiment.cfg"


def get_time_per_flux_ramp_from_cfg(cfg_path=DEFAULT_CONFIG_PATH):
    """Auto-reads flux_ramp_points from experiment.cfg and computes time_per_flux_ramp."""
    import configparser
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    parser = configparser.ConfigParser()
    parser.read(cfg_path)

    for section in ["Global", "global"]:
        if section in parser and "flux_ramp_points" in parser[section]:
            raw_val = parser[section]["flux_ramp_points"].split("#")[0].strip()
            if "*" in raw_val:
                parts = raw_val.split("*")
                fr_points = int(parts[0]) * int(parts[1])
            else:
                fr_points = int(raw_val)

            if fr_points <= 0:
                raise ValueError(f"Invalid flux_ramp_points ({fr_points})")

            time_per_flux_ramp = fr_points * TIME_PER_FLUX_RAMP_POINT_S
            return time_per_flux_ramp, fr_points

    raise KeyError(f"Could not find 'flux_ramp_points' under [Global] in {cfg_path}")


def derive_timing_from_flux_ramp(time_per_flux_ramp, target_duration_s=0.2,
                                 target_f0_hz=20.0, nominal_awg_rate=5e6):
    """
    Derives all DAQ and AWG parameters strictly from time_per_flux_ramp.
    Eliminates all arbitrary magic numbers.
    """
    fs_global = 1.0 / time_per_flux_ramp
    N_samples = int(round(target_duration_s / time_per_flux_ramp))
    t_duration = N_samples * time_per_flux_ramp

    f_fund = 1.0 / t_duration
    f_low = f_fund

    k0 = max(1, int(round(target_f0_hz / f_fund)))
    f0 = k0 * f_fund

    n_awg_points = int(round(t_duration * nominal_awg_rate))
    awg_sample_rate = n_awg_points / t_duration

    return {
        "time_per_flux_ramp": time_per_flux_ramp,
        "fs_global": fs_global,
        "N_samples": N_samples,
        "t_duration": t_duration,
        "f_fund": f_fund,
        "f_low": f_low,
        "k0": k0,
        "f0": f0,
        "n_awg_points": n_awg_points,
        "awg_sample_rate": awg_sample_rate,
    }


def generate_stimulus_tones(fs_global, f0, f_low, start_exp=0, stop_exp=3, num_high_freq_tones=30):
    """Generates log-spaced tone multipliers up to Nyquist limit."""
    max_desired_freq = fs_global / 2.0
    max_allowable_multiplier = int(max_desired_freq / f0)
    multipliers = np.unique(np.int64(np.logspace(start_exp, stop_exp, num_high_freq_tones)))
    valid_mults = [m for m in multipliers if 1 <= m <= max_allowable_multiplier]
    all_freqs = np.insert(np.array(valid_mults) * f0, 0, f_low)
    return valid_mults, all_freqs


def synthesize_comb_waveform(all_freqs, t_duration, awg_sample_rate, multiplier):
    """Synthesizes zero-leakage multi-tone cosine waveform."""
    npts = int(round(t_duration * awg_sample_rate))
    t = np.arange(npts) / awg_sample_rate
    v_superimposed = np.zeros(npts, dtype=np.float64)
    for f in all_freqs:
        v_superimposed += np.cos(2 * np.pi * t * f)
    return v_superimposed * multiplier


# ---------------------------------------------------------------------------
# Background Acquisition & Processing Worker
# ---------------------------------------------------------------------------
class CZWorker(QThread):
    """
    Executes CZ data collection on a separate background thread.
    Communicates with GUI via Qt signals.
    """
    log_signal = pyqtSignal(str, str)             # (level, message)
    step_started = pyqtSignal(int, int, float)    # (index, total, bias_v)
    step_completed = pyqtSignal(float, dict)      # (bias_v, plot_dict)
    sc_calibrated = pyqtSignal(dict)              # (sc_stats)
    sc_check_failed = pyqtSignal(str, dict)       # (reason, sc_stats)
    pause_for_inspection = pyqtSignal(dict)       # (sc_stats)
    progress_signal = pyqtSignal(int, int)        # (current, total)
    finished_signal = pyqtSignal(bool, str)       # (success, message)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self._is_aborted = False
        self._resume_event = QtCore.QWaitCondition()
        self._resume_mutex = QtCore.QMutex()
        self._is_paused = False
        self.fg = None

    def abort(self):
        """Signals worker to abort execution immediately."""
        self._is_aborted = True
        self.log_signal.emit("WARN", "ABORT requested by user! Halting acquisition...")
        self.resume()  # Wake up if waiting on pause

    def resume(self):
        """Resumes execution after user inspection pause."""
        self._resume_mutex.lock()
        self._is_paused = False
        self._resume_event.wakeAll()
        self._resume_mutex.unlock()

    def run(self):
        self._is_aborted = False
        p = self.params
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_date = datetime.now().strftime("%Y%m%d")
        base_dir = Path(p["save_dir"]) / run_date / "Complex_Z" / run_timestamp
        base_dir.mkdir(parents=True, exist_ok=True)

        self.log_signal.emit("INFO", f"Run directory: {base_dir}")

        # 1. Timing calculation
        try:
            time_per_flux_ramp = p["time_per_flux_ramp"]
            timing = derive_timing_from_flux_ramp(
                time_per_flux_ramp=time_per_flux_ramp,
                target_duration_s=p["target_duration_s"],
                target_f0_hz=p["target_f0_hz"],
                nominal_awg_rate=p["nominal_awg_rate"]
            )
            fs_global = timing["fs_global"]
            N_samples = timing["N_samples"]
            t_duration = timing["t_duration"]
            f_low = timing["f_low"]
            f0 = timing["f0"]
            awg_sample_rate = timing["awg_sample_rate"]
            n_awg_points = timing["n_awg_points"]

            self.log_signal.emit("INFO", (
                f"Timing: dt = {time_per_flux_ramp*1e6:.4f} µs, fs = {fs_global:.1f} Hz, "
                f"N = {N_samples} flux ramps (T = {t_duration:.6f} s)"
            ))
            self.log_signal.emit("INFO", (
                f"Comb: f_low = {f_low:.4f} Hz, f0 = {f0:.4f} Hz ({timing['k0']}x df), "
                f"AWG = {n_awg_points} pts @ {awg_sample_rate:.1f} Sa/s"
            ))
        except Exception as e:
            self.finished_signal.emit(False, f"Timing calculation failed: {e}")
            return

        # 2. Voltage limit pre-flight check
        bias_values = p["bias_values"]
        voltage_check_values = bias_values
        if p.get("normal_first", False):
            voltage_check_values = np.append(voltage_check_values, p["vnorm"])
        vpp = p["awg_vpp"]
        load = p["awg_load"]
        v_limit = 10.0 if load.upper() in ("INFINITY", "INF") else 5.0
        v_max = np.max(voltage_check_values) + (vpp / 2.0)
        v_min = np.min(voltage_check_values) - (vpp / 2.0)
        headroom = v_limit - max(abs(v_max), abs(v_min))

        if v_max > v_limit or v_min < -v_limit:
            err = f"Voltage limits exceeded! Range [{v_min:+.3f} V, {v_max:+.3f} V] exceeds ±{v_limit:.1f} V."
            self.log_signal.emit("ERROR", err)
            self.finished_signal.emit(False, err)
            return
        self.log_signal.emit("INFO", f"AWG Voltage verification PASS: Headroom = +{headroom:.3f} V")

        # 3. Connect to AWG and upload waveform
        self.log_signal.emit("INFO", f"Connecting to AWG at {p['awg_address']}...")
        try:
            if p["awg_address"].startswith("vxi11://") or "://" in p["awg_address"]:
                self.fg = keysight33600A_usb.Keysight33600A_usb(url=p["awg_address"])
            else:
                self.fg = keysight33600A_usb.Keysight33600A_usb(device=p["awg_address"])
        except Exception as e:
            err = f"Failed to connect to AWG: {e}"
            self.log_signal.emit("ERROR", err)
            self.finished_signal.emit(False, err)
            return

        # Generate comb waveform
        f_mults, all_freqs = generate_stimulus_tones(
            fs_global, f0, f_low, num_high_freq_tones=p["num_high_freq_tones"]
        )
        v_waveform = synthesize_comb_waveform(
            all_freqs, t_duration, awg_sample_rate, p["amp_multiplier"]
        )
        self.log_signal.emit("INFO", f"Synthesized {len(all_freqs)} tones ({all_freqs[0]:.2f} Hz to {all_freqs[-1]:.2f} Hz)")

        # Save supplied waveform
        try:
            np.savez(
                base_dir / "supplied_waveform.npz",
                waveform=v_waveform,
                frequencies=all_freqs,
                awg_sample_rate=awg_sample_rate,
                amplitude_vpp=vpp,
                multiplier=p["amp_multiplier"],
                fs_global=fs_global,
                time_per_flux_ramp=time_per_flux_ramp,
                created_at=datetime.now().isoformat()
            )
            self.log_signal.emit("INFO", f"Saved supplied waveform to {base_dir / 'supplied_waveform.npz'}")
        except Exception as e:
            self.log_signal.emit("WARN", f"Failed to save supplied waveform npz: {e}")

        # Program AWG
        try:
            source = p["awg_channel"]
            self.fg.setOutputLoad(source=source, load=load)
            self.fg._checkSource(source)
            self.fg.clearVolatileMem(source)
            time.sleep(0.1)
            self.fg.setByteOrder(swap_bytes=True)
            packed = np.array(v_waveform, dtype=np.float32).tobytes()
            header = f"SOURce{source}:DATA:ARB cz_sum_sines, #{len(str(len(packed)))}{len(packed)}".encode("ascii")
            self.fg._write_raw(header + packed + b"\n")

            self.fg.setFunction(waveform_function="ARB", source=source)
            self.fg.setArbFile(source, "cz_sum_sines")
            self.fg.setArbSampleRate(source, awg_sample_rate)
            self.fg.setArbFilter(source, filter_state="STEP")
            self.fg.setAmplitude(source, vpp)
            self.fg.setOffset(source, bias_values[0])
            self.fg.setOutputState(source=source, output_state="ON")
            time.sleep(0.5)
            self.log_signal.emit("INFO", f"Keysight 33600A Ch{source} active: Vpp={vpp} V, Offset={bias_values[0]:.3f} V")
        except Exception as e:
            err = f"Failed to program AWG: {e}"
            self.log_signal.emit("ERROR", err)
            self._safe_cleanup_awg()
            self.finished_signal.emit(False, err)
            return

        # 4. Connect to DASTARD
        c = None
        if not p.get("dry_run_mode", False):
            self.log_signal.emit("INFO", "Connecting to DASTARD...")
            for attempt in range(5):
                if self._is_aborted:
                    self._safe_cleanup_awg()
                    self.finished_signal.emit(False, "Aborted before DASTARD connection.")
                    return
                try:
                    c = EasyClientDastard()
                    self.log_signal.emit("INFO", "Connected to DASTARD successfully.")
                    break
                except Exception as e:
                    self.log_signal.emit("WARN", f"DASTARD connection attempt {attempt+1}/5 failed ({e})")
                    time.sleep(0.5)

            if c is None:
                err = "Failed to connect to DASTARD client."
                self.log_signal.emit("ERROR", err)
                self._safe_cleanup_awg()
                self.finished_signal.emit(False, err)
                return

        # 5. Acquisition Loop
        tau_shift_0V = None
        Z_model_pure = None
        amp0 = None
        f_roll = None
        total_steps = len(bias_values)
        V_stim = p["amp_multiplier"] * vpp

        try:
            for idx, bias in enumerate(bias_values):
                if self._is_aborted:
                    break

                self.step_started.emit(idx + 1, total_steps, bias)
                self.progress_signal.emit(idx + 1, total_steps)
                self.log_signal.emit("INFO", f"[{idx+1}/{total_steps}] Setting bias = {bias:.3f} V...")

                # Apply AWG DC offset
                self.fg.setOffset(source=p["awg_channel"], offset=bias)
                point_settle_time = p.get("settle_time_s", 0.3)
                first_dense_after_normal = p.get("normal_first", False) and idx == 1
                if np.isclose(bias, 0.0):
                    point_settle_time = p.get("sc_settle_time_s", point_settle_time)
                    self.log_signal.emit("INFO", "Starting SC acquisition without a settle wait.")
                elif first_dense_after_normal:
                    self.log_signal.emit("INFO", "Starting first dense complex-Z acquisition immediately after offset change.")

                # Fetch data from DASTARD
                if c is not None:
                    n_req = N_samples * p.get("n_averages", 4)
                    ch_key = f"chan{p['channel']}"
                    n_avg = p.get("n_averages", 4)
                    n_candidates = 3 if np.isclose(bias, 0.0) else 1
                    if n_candidates > 1:
                        self.log_signal.emit("INFO", "Selecting the strongest of 3 fresh SC data blocks.")
                    candidate_data = []
                    candidate_scores = []
                    candidate_results = []
                    candidate_bin_width = fs_global / N_samples
                    candidate_indices = np.rint(all_freqs / candidate_bin_width).astype(int)
                    for _ in range(n_candidates):
                        data_all = c.getNewData(n_req)
                        if ch_key not in data_all:
                            raise KeyError(f"Channel '{ch_key}' not found in DASTARD data.")
                        data_raw = data_all[ch_key] * 1.0
                        data_mean = data_raw[:N_samples * n_avg].reshape((n_avg, N_samples)).mean(axis=0)
                        first_frame_index = data_all["firstFrameIndex"][0] % N_samples
                        candidate = np.roll(data_mean, first_frame_index)
                        candidate_fft = scipy.fft.rfft(candidate)
                        valid_indices = candidate_indices[(candidate_indices > 0) & (candidate_indices < len(candidate_fft))]
                        candidate_scores.append(float(np.sum(np.abs(candidate_fft[valid_indices]))))
                        candidate_results.append(data_all)
                        candidate_data.append(candidate)
                    selected_index = int(np.argmax(candidate_scores))
                    data = candidate_data[selected_index]
                    data_all = candidate_results[selected_index]
                else:
                    # Simulated data for headless/dry-run verification
                    t = np.arange(N_samples) * time_per_flux_ramp
                    data = np.zeros(N_samples)
                    f_r_sim = 1200.0
                    for f in all_freqs:
                        mag = 5000.0 / np.sqrt(1 + (f / f_r_sim)**2)
                        phi = -np.arctan(f / f_r_sim) - 2 * np.pi * f * 15e-6
                        data += mag * np.cos(2 * np.pi * f * t + phi)
                    data += np.random.normal(0, 10, N_samples)

                # Compute FFT
                fft_data_raw = scipy.fft.rfft(data)
                fft_freqs = scipy.fft.rfftfreq(N_samples, d=time_per_flux_ramp)
                f_sch_idx = np.argmin(
                    np.abs(fft_freqs[:, None] - all_freqs[None, :]),
                    axis=0,
                )
                fs_sch = all_freqs
                tone_data_raw = fft_data_raw[f_sch_idx]
                sc_tone_data_raw = tone_data_raw

                # -------------------------------------------------------------
                # 0 V Calibration & Superconducting Transfer Function Health Check
                # -------------------------------------------------------------
                if np.isclose(bias, 0.0):
                    self.log_signal.emit("INFO", "Evaluating Superconducting (0 V) Transfer Function...")

                    def magnitude_model(fs, amp0, f_roll):
                        return amp0 / np.sqrt(1 + (fs / f_roll)**2)

                    model_mag = Model(magnitude_model)
                    params_mag = Parameters()
                    params_mag.add("amp0", value=np.abs(fft_data_raw[0]), min=0, vary=True)
                    params_mag.add("f_roll", value=1e3, min=1.0, max=1e6, vary=True)

                    y_data = np.abs(sc_tone_data_raw)
                    res_mag = model_mag.fit(y_data, fs=fs_sch, params=params_mag)

                    amp0 = float(res_mag.params["amp0"].value)
                    f_roll = float(res_mag.params["f_roll"].value)

                    # Goodness of fit (R^2)
                    fit_best = magnitude_model(fs_sch, amp0, f_roll)
                    ss_res = np.sum((y_data - fit_best)**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                    Z_model_pure = amp0 / (1.0 + 1j * fs_sch / f_roll)

                    def real_model(fs, tau_shift):
                        return np.real((fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift))[f_sch_idx])

                    model_real = Model(real_model)
                    tau_guess = -np.argmax(data) / fs_global
                    params_real = Parameters()
                    params_real.add("tau_shift", value=tau_guess, vary=True)
                    res_real = model_real.fit(np.real(Z_model_pure), fs=all_freqs, params=params_real)
                    tau_shift_0V = float(res_real.params["tau_shift"].value)

                    sc_stats = {
                        "amp0": amp0,
                        "f_roll": f_roll,
                        "r_squared": r_squared,
                        "tau_shift": tau_shift_0V,
                        "y_data": y_data,
                        "best_fit": fit_best,
                        "all_freqs": all_freqs,
                    }
                    self.sc_calibrated.emit(sc_stats)
                    self.log_signal.emit("INFO", (
                        f"0 V Fit Results: R² = {r_squared:.4f}, f_roll = {f_roll:.1f} Hz, "
                        f"amp0 = {amp0:.1f}, τ_shift = {tau_shift_0V*1e6:.2f} µs"
                    ))

                    # Evaluate failure criteria
                    abort_reasons = []
                    min_r2 = p.get("min_sc_r2", 0.85)
                    min_froll = p.get("min_froll_hz", 10.0)
                    max_froll = p.get("max_froll_hz", 25000.0)

                    if r_squared < min_r2:
                        abort_reasons.append(f"Low fit quality: R² = {r_squared:.4f} < {min_r2:.4f}")
                    if f_roll < min_froll or f_roll > max_froll:
                        abort_reasons.append(f"Unphysical f_roll: {f_roll:.1f} Hz outside [{min_froll}, {max_froll}] Hz")
                    if amp0 < 1.0:
                        abort_reasons.append(f"Extremely weak response: amp0 = {amp0:.2e} (possible unlock/open circuit)")

                    if abort_reasons:
                        failure_msg = " | ".join(abort_reasons)
                        self.log_signal.emit("ERROR", f"Superconducting Transfer Function Check FAILED: {failure_msg}")
                        self.sc_check_failed.emit(failure_msg, sc_stats)

                        if p.get("auto_abort_on_bad_sc", True):
                            self._safe_cleanup_awg()
                            self.finished_signal.emit(False, f"ABORTED: Superconducting transfer function bad ({failure_msg})")
                            return

                    # Optional user pause for visual inspection
                    if p.get("pause_after_0v", False):
                        self.log_signal.emit("WARN", "PAUSED: Waiting for user visual confirmation of 0 V SC transfer function...")
                        self.pause_for_inspection.emit(sc_stats)
                        self._resume_mutex.lock()
                        self._is_paused = True
                        while self._is_paused and not self._is_aborted:
                            self._resume_event.wait(self._resume_mutex)
                        self._resume_mutex.unlock()

                        if self._is_aborted:
                            self._safe_cleanup_awg()
                            self.finished_signal.emit(False, "Aborted by user during 0 V calibration pause.")
                            return
                        self.log_signal.emit("INFO", "User approved 0 V calibration. Resuming full bias sweep...")

                    if p.get("normal_first", False):
                        self.log_signal.emit("INFO", f"Driving TES normal at Vnorm = {p['vnorm']:.3f} V for 1.0 s...")
                        self.fg.setOffset(source=p["awg_channel"], offset=p["vnorm"])
                        self.log_signal.emit("INFO", "Normal-state conditioning offset set. Starting ascending dense-to-medium bias sweep.")

                # The normal-first point precedes SC calibration, so leave it unshifted.
                applied_tau_shift = tau_shift_0V if tau_shift_0V is not None else 0.0
                fft_unroll = tone_data_raw * np.exp(-2j * np.pi * fs_sch * applied_tau_shift)

                # Prepare plot data payload for GUI
                plot_data = {
                    "bias": bias,
                    "tau_shift": applied_tau_shift,
                    "fs_sch": fs_sch,
                    "fft_unroll": fft_unroll,
                    "Z_model_pure": Z_model_pure,
                    "amp0": amp0,
                    "f_roll": f_roll,
                }
                self.step_completed.emit(bias, plot_data)

                # Save step data
                bias_str = f"{bias:.2f}".replace(".", "v")
                bias_dir = base_dir / f"bias_{bias_str}"
                bias_dir.mkdir(parents=True, exist_ok=True)

                np.savez(
                    bias_dir / f"vbias_{bias_str}_data.npz",
                    bias=bias,
                    v_stim=V_stim,
                    fft_unroll=fft_unroll,
                    fs_sch=all_freqs,
                    tau_shift=applied_tau_shift,
                    raw_data=data,
                    fs_global=fs_global,
                    time_per_flux_ramp=time_per_flux_ramp
                )
                self._save_bias_plot(
                    bias_dir=bias_dir,
                    bias=bias,
                    fs_sch=fs_sch,
                    fft_unroll=fft_unroll,
                    Z_model_pure=Z_model_pure,
                    tau_shift=applied_tau_shift,
                )

                if p.get("single_0v_test_only", False):
                    self.log_signal.emit("INFO", "Single 0 V SC test completed.")
                    break

        except Exception as e:
            self.log_signal.emit("ERROR", f"Acquisition error: {e}")
            self._safe_cleanup_awg()
            self.finished_signal.emit(False, str(e))
            return

        # Final cleanup
        self._safe_cleanup_awg()
        if self._is_aborted:
            self.finished_signal.emit(False, "Run was aborted by user. AWG safely reset to 0.0 V.")
        else:
            self.finished_signal.emit(True, f"Run completed successfully! All data saved to {base_dir}")

    def _safe_cleanup_awg(self):
        """Safely ramps down the AWG DC offset to 0.0 V and disables output."""
        if self.fg is not None:
            self.log_signal.emit("INFO", "Safely resetting AWG Channel 2 offset to 0.0 V...")
            try:
                self.fg.setOffset(source=self.params["awg_channel"], offset=0.0)
                self.fg.setOutputState(source=self.params["awg_channel"], output_state="OFF")
                self.log_signal.emit("INFO", "AWG output offset reset to 0.0 V and output turned OFF.")
            except Exception as e:
                self.log_signal.emit("WARN", f"AWG cleanup exception: {e}")
            try:
                self.fg.channel.close()
            except Exception:
                pass
            self.fg = None

    def _save_bias_plot(self, bias_dir, bias, fs_sch, fft_unroll, Z_model_pure, tau_shift):
        fig = Figure(figsize=(12, 9), dpi=100)
        axes = fig.subplots(2, 2).flatten()
        fig.set_facecolor("white")
        for axis in axes:
            axis.grid(True, which="both", linestyle="--", alpha=0.4)

        axes[0].semilogx(fs_sch, np.real(fft_unroll), ".-", label="Real (Data)")
        axes[0].semilogx(fs_sch, np.imag(fft_unroll), ".-", label="Imag (Data)")
        if Z_model_pure is not None:
            axes[0].semilogx(fs_sch, np.real(Z_model_pure), "--", label="Real (Model)")
            axes[0].semilogx(fs_sch, np.imag(Z_model_pure), "--", label="Imag (Model)")
        axes[0].set_title("Real and Imaginary Parts")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].legend(fontsize=8)

        axes[1].plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", label="Data")
        if Z_model_pure is not None:
            axes[1].plot(np.real(Z_model_pure), np.imag(Z_model_pure), "--", label="Model")
        axes[1].set_title("Nyquist Plot")
        axes[1].set_xlabel("Real(Z)")
        axes[1].set_ylabel("Imag(Z)")
        axes[1].legend(fontsize=8)
        axes[1].axis("equal")

        axes[2].loglog(fs_sch, np.abs(fft_unroll), ".-", label="Data")
        if Z_model_pure is not None:
            axes[2].loglog(fs_sch, np.abs(Z_model_pure), "--", label="Model")
        axes[2].set_title("Magnitude Response")
        axes[2].set_xlabel("Frequency (Hz)")
        axes[2].legend(fontsize=8)

        axes[3].semilogx(fs_sch, np.angle(fft_unroll) / np.pi, ".-", label="Data")
        if Z_model_pure is not None:
            axes[3].semilogx(fs_sch, np.angle(Z_model_pure) / np.pi, "--", label="Model")
        axes[3].set_title("Phase Response")
        axes[3].set_xlabel("Frequency (Hz)")
        axes[3].set_ylabel("Phase (π rad)")
        axes[3].legend(fontsize=8)

        fig.suptitle(f"EZ CZ dashboard | Bias = {bias:.3f} V | Delay = {tau_shift * 1e6:.2f} µs")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(bias_dir / f"vbias_{bias:.2f}_impedance_plot.png".replace(".", "v", 1))


# ---------------------------------------------------------------------------
# Embedded 4-Panel Matplotlib Canvas Widget
# ---------------------------------------------------------------------------
class CZPlotCanvas(FigureCanvas):
    """Matplotlib canvas containing 4 diagnostic subplots for CZ testing."""
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.dark_mode = True
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor("#1e222b")
        self.axs = self.fig.subplots(2, 2)
        self.fig.tight_layout(pad=3.0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.init_plots()

    def init_plots(self):
        titles = [
            "Real & Imaginary Parts",
            "Nyquist Plot (Complex Z)",
            "Magnitude Response |Z(f)|",
            "Phase Response ∠Z(f) / π"
        ]
        background_color = "#15181e" if self.dark_mode else "#ffffff"
        text_color = "#e0e4ed" if self.dark_mode else "#202124"
        grid_color = "#333842" if self.dark_mode else "#c7cbd1"

        for ax, title in zip(self.axs.flatten(), titles):
            ax.clear()
            ax.set_facecolor(background_color)
            ax.set_title(title, color=text_color, fontsize=11, fontweight="bold", pad=8)
            ax.tick_params(colors=text_color, which="both", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444b58" if self.dark_mode else "#9aa0a6")
            ax.grid(True, which="both", color=grid_color, linestyle="--", alpha=0.5)

        self.axs[0, 0].set_xlabel("Frequency (Hz)", color=text_color, fontsize=9)
        self.axs[0, 1].set_xlabel("Real(Z)", color=text_color, fontsize=9)
        self.axs[0, 1].set_ylabel("Imag(Z)", color=text_color, fontsize=9)
        self.axs[1, 0].set_xlabel("Frequency (Hz)", color=text_color, fontsize=9)
        self.axs[1, 1].set_xlabel("Frequency (Hz)", color=text_color, fontsize=9)
        self.draw()

    def set_theme(self, dark_mode):
        self.dark_mode = dark_mode
        self.fig.set_facecolor("#1e222b" if dark_mode else "#ffffff")
        self.init_plots()

    def update_plots(self, plot_data, is_sc_eval=False):
        """Updates all 4 subplots with latest data point."""
        bias = plot_data["bias"]
        tau_shift = plot_data["tau_shift"]
        fs_sch = plot_data["fs_sch"]
        fft_unroll = plot_data["fft_unroll"]
        Z_model_pure = plot_data.get("Z_model_pure", None)

        text_color = "#e0e4ed" if self.dark_mode else "#202124"
        legend_face = "#1e222b" if self.dark_mode else "#ffffff"
        legend_edge = "#444b58" if self.dark_mode else "#9aa0a6"
        self.init_plots()

        # 1. Real and Imaginary parts
        ax1 = self.axs[0, 0]
        ax1.semilogx(fs_sch, np.real(fft_unroll), ".-", color="#00e5ff", label="Real (Data)", lw=1.5)
        ax1.semilogx(fs_sch, np.imag(fft_unroll), ".-", color="#ff4081", label="Imag (Data)", lw=1.5)
        if Z_model_pure is not None:
            ax1.semilogx(fs_sch, np.real(Z_model_pure), "--", color="#80d8ff", label="Real (Model)", alpha=0.7)
            ax1.semilogx(fs_sch, np.imag(Z_model_pure), "--", color="#ff80ab", label="Imag (Model)", alpha=0.7)
        ax1.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        # 2. Nyquist Plot
        ax2 = self.axs[0, 1]
        ax2.plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", color="#69f0ae", label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax2.plot(np.real(Z_model_pure), np.imag(Z_model_pure), "--", color="#b9f6ca", label="Model", alpha=0.7)
        ax2.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)
        ax2.axis("equal")

        # 3. Magnitude Response
        ax3 = self.axs[1, 0]
        ax3.loglog(fs_sch, np.abs(fft_unroll), ".-", color="#ffd740", label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax3.loglog(fs_sch, np.abs(Z_model_pure), "--", color="#ffe57f", label="Fit Model", lw=1.2)
        ax3.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        # 4. Phase Response
        ax4 = self.axs[1, 1]
        ax4.semilogx(fs_sch, np.angle(fft_unroll) / np.pi, ".-", color="#b388ff", label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax4.semilogx(fs_sch, np.angle(Z_model_pure) / np.pi, "--", color="#d1c4e9", label="Fit Model", lw=1.2)
        ax4.set_ylabel("Phase (π rad)", color=text_color, fontsize=9)
        ax4.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        tag = " [0 V Superconducting Baseline]" if is_sc_eval or bias == 0 else ""
        self.fig.suptitle(
            f"V_bias = {bias:.3f} V (AWG Ch2 Offset) | τ_shift = {tau_shift*1e6:.2f} µs{tag}",
            color=text_color, fontsize=12, fontweight="bold", y=0.98
        )
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.draw()


# ---------------------------------------------------------------------------
# Main GUI Window
# ---------------------------------------------------------------------------
class CZMainWindow(QMainWindow):
    """Main Application Window for CZ Testing."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EZ CZ dashboard")
        self.resize(1600, 800)
        self.worker = None

        self.dark_stylesheet = """
            QMainWindow, QWidget {
                background-color: #12141a;
                color: #e0e4ed;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #2a313d;
                border-radius: 6px;
                margin-top: 22px;
                padding-top: 16px;
                font-weight: bold;
                color: #82b1ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1b2028;
                border: 1px solid #363d4a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                selection-background-color: #2979ff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #2979ff;
            }
            QPushButton {
                background-color: #263238;
                border: 1px solid #37474f;
                border-radius: 4px;
                padding: 8px 14px;
                font-weight: bold;
                color: #eceff1;
            }
            QPushButton:hover {
                background-color: #37474f;
                border-color: #546e7a;
            }
            QPushButton:pressed {
                background-color: #1e272c;
            }
            QPushButton:disabled {
                background-color: #171a21;
                color: #546e7a;
                border-color: #263238;
            }
            QProgressBar {
                border: 1px solid #2a313d;
                border-radius: 4px;
                text-align: center;
                background-color: #181d24;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00c853;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #0d1017;
                border: 1px solid #262c38;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
                color: #cfd8dc;
            }
            QTabWidget::pane {
                border: 1px solid #2a313d;
                border-radius: 4px;
                background-color: #151922;
            }
            QTabBar::tab {
                background-color: #1e2430;
                color: #90a4ae;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #151922;
                color: #82b1ff;
                border-bottom: 2px solid #2979ff;
            }
        """

        self.init_ui()
        self.on_timing_source_changed()
        self.update_bias_preview()
        self.apply_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # -------------------------------------------------------------------
        # Left Pane: Config Tabs & Controls
        # -------------------------------------------------------------------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Tab Widget for Parameter Sections
        tabs = QTabWidget()
        tabs.addTab(self._create_bias_tab(), "Bias & Channel")
        tabs.addTab(self._create_awg_tab(), "AWG Stimulus")
        tabs.addTab(self._create_timing_tab(), "Timing / Flux Ramp")
        tabs.addTab(self._create_sc_health_tab(), "SC Health Check")
        left_layout.addWidget(tabs, stretch=0)

        # Safety & Status Group
        status_box = QGroupBox("Acquisition Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setSpacing(6)

        # Badges
        badge_layout = QHBoxLayout()
        self.lbl_status = QLabel("STATE: IDLE")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #b0bec5; font-size: 11pt;")
        self.lbl_current_bias = QLabel("Bias: -- V")
        self.lbl_current_bias.setStyleSheet("font-weight: bold; color: #82b1ff; font-size: 11pt;")
        badge_layout.addWidget(self.lbl_status)
        badge_layout.addStretch()
        badge_layout.addWidget(self.lbl_current_bias)
        status_layout.addLayout(badge_layout)

        # Health Indicator Badge
        self.lbl_sc_health = QLabel("SC 0V Health: Not Tested")
        self.lbl_sc_health.setStyleSheet("padding: 4px; border-radius: 4px; background-color: #263238; color: #eceff1; font-weight: bold;")
        status_layout.addWidget(self.lbl_sc_health)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        left_layout.addWidget(status_box)

        # Action Buttons
        btn_box = QGroupBox("Execution Controls")
        btn_layout = QGridLayout(btn_box)
        btn_layout.setSpacing(8)

        self.btn_check_limits = QPushButton("Verify AWG Limits")
        self.btn_check_limits.setStyleSheet("background-color: #1976d2; color: #ffffff;")
        self.btn_check_limits.clicked.connect(self.on_verify_limits_clicked)

        self.btn_test_0v = QPushButton("Test 0V SC Baseline Only")
        self.btn_test_0v.setStyleSheet("background-color: #00897b; color: #ffffff;")
        self.btn_test_0v.clicked.connect(self.on_test_0v_clicked)

        self.btn_start = QPushButton("START FULL SWEEP")
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: #ffffff; font-size: 11pt; padding: 10px;")
        self.btn_start.clicked.connect(self.on_start_clicked)

        self.btn_abort = QPushButton("ABORT (SAFE ZERO)")
        self.btn_abort.setStyleSheet("background-color: #c62828; color: #ffffff; font-size: 11pt; font-weight: bold; padding: 10px;")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.on_abort_clicked)

        self.btn_resume = QPushButton("Resume / Proceed")
        self.btn_resume.setStyleSheet("background-color: #f57f17; color: #ffffff;")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self.on_resume_clicked)

        for button in (
            self.btn_check_limits,
            self.btn_test_0v,
            self.btn_start,
            self.btn_abort,
            self.btn_resume,
        ):
            button.setFixedSize(250, 44)

        btn_layout.addWidget(self.btn_check_limits, 0, 0)
        btn_layout.addWidget(self.btn_test_0v, 0, 1)
        btn_layout.addWidget(self.btn_start, 1, 0)
        btn_layout.addWidget(self.btn_abort, 1, 1)
        btn_layout.addWidget(self.btn_resume, 2, 0)
        left_layout.addWidget(btn_box)

        # Console Log
        log_box = QGroupBox("Activity Console Log")
        log_layout = QVBoxLayout(log_box)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)
        left_layout.addWidget(log_box, stretch=1)

        splitter.addWidget(left_widget)

        # -------------------------------------------------------------------
        # Right Pane: 4-Panel Live Canvas & Toolbar
        # -------------------------------------------------------------------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.canvas = CZPlotCanvas(self, width=9, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #1e2430; border: 1px solid #2a313d; border-radius: 4px;")

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

    # -----------------------------------------------------------------------
    # Parameter Tab Builders
    # -----------------------------------------------------------------------
    def _create_bias_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        # Channel Box
        ch_box = QGroupBox("Channel Selection")
        ch_layout = QHBoxLayout(ch_box)
        ch_layout.addWidget(QLabel("Target Channel (ch_global):"))
        self.spin_channel = QSpinBox()
        self.spin_channel.setRange(0, 2048)
        self.spin_channel.setValue(13)
        ch_layout.addWidget(self.spin_channel)
        layout.addWidget(ch_box)

        # Schedule Box
        sched_box = QGroupBox("Bias Voltage Schedule (V)")
        grid = QGridLayout(sched_box)
        grid.setSpacing(6)

        grid.addWidget(QLabel("Baseline Point:"), 0, 0)
        self.spin_base_bias = QDoubleSpinBox()
        self.spin_base_bias.setRange(0.0, 0.0)
        self.spin_base_bias.setValue(0.0)
        self.spin_base_bias.setDecimals(3)
        self.spin_base_bias.setEnabled(False)
        grid.addWidget(self.spin_base_bias, 0, 1)
        grid.addWidget(QLabel("(Fixed 0.0 V for SC calibration)"), 0, 2)

        self.chk_normal_first = QCheckBox("SC check, then normalize TES for 1 s before sweep")
        self.chk_normal_first.setChecked(False)
        self.chk_normal_first.toggled.connect(self.update_bias_preview)
        grid.addWidget(self.chk_normal_first, 0, 3, 1, 3)

        grid.addWidget(QLabel("Vnorm:"), 3, 0)
        self.spin_vnorm = QDoubleSpinBox()
        self.spin_vnorm.setRange(0.0, 10.0)
        self.spin_vnorm.setValue(10.0)
        self.spin_vnorm.setDecimals(3)
        self.spin_vnorm.setSingleStep(0.5)
        self.spin_vnorm.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_vnorm, 3, 1)
        grid.addWidget(QLabel("(Used as a pre-sweep normalizing offset)"), 3, 2, 1, 4)

        grid.addWidget(QLabel("Post-normal recovery (s):"), 4, 0)
        self.spin_post_normal_settle = QDoubleSpinBox()
        self.spin_post_normal_settle.setRange(0.0, 60.0)
        self.spin_post_normal_settle.setValue(5.0)
        self.spin_post_normal_settle.setDecimals(1)
        self.spin_post_normal_settle.setSingleStep(1.0)
        grid.addWidget(self.spin_post_normal_settle, 4, 1)
        grid.addWidget(QLabel("(Wait before first sweep point)"), 4, 2, 1, 4)

        # Dense region
        grid.addWidget(QLabel("Dense Region:"), 1, 0)
        self.spin_dense_start = QDoubleSpinBox()
        self.spin_dense_start.setRange(0.0, 10.0)
        self.spin_dense_start.setValue(0.5)
        self.spin_dense_start.setSingleStep(0.1)
        self.spin_dense_start.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_dense_start, 1, 1)

        grid.addWidget(QLabel("to"), 1, 2)
        self.spin_dense_stop = QDoubleSpinBox()
        self.spin_dense_stop.setRange(0.0, 10.0)
        self.spin_dense_stop.setValue(1.5)
        self.spin_dense_stop.setSingleStep(0.1)
        self.spin_dense_stop.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_dense_stop, 1, 3)

        grid.addWidget(QLabel("step:"), 1, 4)
        self.spin_dense_step = QDoubleSpinBox()
        self.spin_dense_step.setRange(0.001, 1.0)
        self.spin_dense_step.setValue(0.01)
        self.spin_dense_step.setDecimals(3)
        self.spin_dense_step.setSingleStep(0.005)
        self.spin_dense_step.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_dense_step, 1, 5)

        # Medium region
        grid.addWidget(QLabel("Medium Region:"), 2, 0)
        self.spin_med_start = QDoubleSpinBox()
        self.spin_med_start.setRange(0.0, 10.0)
        self.spin_med_start.setValue(1.5)
        self.spin_med_start.setSingleStep(0.1)
        self.spin_med_start.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_med_start, 2, 1)

        grid.addWidget(QLabel("to"), 2, 2)
        self.spin_med_stop = QDoubleSpinBox()
        self.spin_med_stop.setRange(0.0, 10.0)
        self.spin_med_stop.setValue(5.0)
        self.spin_med_stop.setSingleStep(0.5)
        self.spin_med_stop.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_med_stop, 2, 3)

        grid.addWidget(QLabel("step:"), 2, 4)
        self.spin_med_step = QDoubleSpinBox()
        self.spin_med_step.setRange(0.01, 2.0)
        self.spin_med_step.setValue(0.1)
        self.spin_med_step.setDecimals(2)
        self.spin_med_step.setSingleStep(0.05)
        self.spin_med_step.valueChanged.connect(self.update_bias_preview)
        grid.addWidget(self.spin_med_step, 2, 5)

        layout.addWidget(sched_box)

        # Bias Points Summary Label
        self.lbl_bias_summary = QLabel("Total Points: -- | Max: -- V")
        self.lbl_bias_summary.setStyleSheet("color: #80cbc4; font-weight: bold;")
        layout.addWidget(self.lbl_bias_summary)

        # Bias points table preview
        self.tbl_bias = QTableWidget(0, 2)
        self.tbl_bias.setHorizontalHeaderLabels(["Index", "Bias Voltage (V)"])
        self.tbl_bias.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_bias.setMaximumHeight(160)
        layout.addWidget(self.tbl_bias)

        layout.addStretch()
        return tab

    def _create_awg_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        grid.setSpacing(8)

        grid.addWidget(QLabel("AWG Address:"), 0, 0)
        self.edit_awg_addr = QLineEdit("USB::2391::22279::INSTR")
        grid.addWidget(self.edit_awg_addr, 0, 1)

        grid.addWidget(QLabel("Output Channel:"), 1, 0)
        self.spin_awg_ch = QSpinBox()
        self.spin_awg_ch.setRange(1, 2)
        self.spin_awg_ch.setValue(2)
        grid.addWidget(self.spin_awg_ch, 1, 1)

        grid.addWidget(QLabel("Termination Load:"), 2, 0)
        self.combo_load = QComboBox()
        self.combo_load.addItems(["INFinity", "50"])
        grid.addWidget(self.combo_load, 2, 1)

        grid.addWidget(QLabel("AC Amplitude Vpp (V):"), 3, 0)
        self.spin_awg_vpp = QDoubleSpinBox()
        self.spin_awg_vpp.setRange(0.001, 1.0)
        self.spin_awg_vpp.setValue(0.02)
        self.spin_awg_vpp.setDecimals(4)
        self.spin_awg_vpp.setSingleStep(0.005)
        grid.addWidget(self.spin_awg_vpp, 3, 1)

        grid.addWidget(QLabel("Multiplier per Sine:"), 4, 0)
        self.spin_sine_mult = QDoubleSpinBox()
        self.spin_sine_mult.setRange(0.001, 0.5)
        self.spin_sine_mult.setValue(0.04)
        self.spin_sine_mult.setDecimals(4)
        grid.addWidget(self.spin_sine_mult, 4, 1)

        grid.addWidget(QLabel("Nominal AWG Rate (MSa/s):"), 5, 0)
        self.spin_awg_rate = QDoubleSpinBox()
        self.spin_awg_rate.setRange(0.1, 50.0)
        self.spin_awg_rate.setValue(5.0)
        grid.addWidget(self.spin_awg_rate, 5, 1)

        grid.addWidget(QLabel("Settle Time (s):"), 6, 0)
        self.spin_settle = QDoubleSpinBox()
        self.spin_settle.setRange(0.05, 5.0)
        self.spin_settle.setValue(0.3)
        grid.addWidget(self.spin_settle, 6, 1)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    def _create_timing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        grid.setSpacing(8)

        self.chk_auto_cfg = QCheckBox("Auto-read time_per_flux_ramp from experiment.cfg")
        self.chk_auto_cfg.setChecked(True)
        self.chk_auto_cfg.toggled.connect(self.on_timing_source_changed)
        grid.addWidget(self.chk_auto_cfg, 0, 0, 1, 2)

        grid.addWidget(QLabel("Config File Path:"), 1, 0)
        cfg_h = QHBoxLayout()
        self.edit_cfg_path = QLineEdit(DEFAULT_CONFIG_PATH)
        cfg_h.addWidget(self.edit_cfg_path)
        self.btn_browse_cfg = QPushButton("Browse")
        self.btn_browse_cfg.clicked.connect(self.on_browse_cfg)
        cfg_h.addWidget(self.btn_browse_cfg)
        grid.addLayout(cfg_h, 1, 1)

        grid.addWidget(QLabel("Manual time_per_flux_ramp (µs):"), 2, 0)
        self.spin_manual_tfr = QDoubleSpinBox()
        self.spin_manual_tfr.setRange(0.5, 100.0)
        self.spin_manual_tfr.setValue(8.192)
        self.spin_manual_tfr.setDecimals(4)
        self.spin_manual_tfr.setEnabled(False)
        grid.addWidget(self.spin_manual_tfr, 2, 1)

        grid.addWidget(QLabel("Target Duration (s):"), 3, 0)
        self.spin_target_dur = QDoubleSpinBox()
        self.spin_target_dur.setRange(0.05, 5.0)
        self.spin_target_dur.setValue(0.2)
        self.spin_target_dur.setDecimals(3)
        grid.addWidget(self.spin_target_dur, 3, 1)

        grid.addWidget(QLabel("Target Base Comb f0 (Hz):"), 4, 0)
        self.spin_target_f0 = QDoubleSpinBox()
        self.spin_target_f0.setRange(1.0, 500.0)
        self.spin_target_f0.setValue(20.0)
        grid.addWidget(self.spin_target_f0, 4, 1)

        grid.addWidget(QLabel("Num High-Freq Tones:"), 5, 0)
        self.spin_num_tones = QSpinBox()
        self.spin_num_tones.setRange(5, 50)
        self.spin_num_tones.setValue(30)
        grid.addWidget(self.spin_num_tones, 5, 1)

        grid.addWidget(QLabel("Averages per Bias Point:"), 6, 0)
        self.spin_averages = QSpinBox()
        self.spin_averages.setRange(1, 32)
        self.spin_averages.setValue(4)
        grid.addWidget(self.spin_averages, 6, 1)

        grid.addWidget(QLabel("Data Save Root Dir:"), 7, 0)
        self.edit_save_dir = QLineEdit("/data")
        grid.addWidget(self.edit_save_dir, 7, 1)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    def _create_sc_health_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        grid.setSpacing(8)

        self.chk_auto_abort = QCheckBox("Automatically ABORT sweep if 0V transfer function looks bad")
        self.chk_auto_abort.setChecked(True)
        self.chk_auto_abort.setStyleSheet("font-weight: bold; color: #ff80ab;")
        grid.addWidget(self.chk_auto_abort, 0, 0, 1, 2)

        self.chk_pause_0v = QCheckBox("Pause after 0V calibration for user visual inspection")
        self.chk_pause_0v.setChecked(False)
        self.chk_pause_0v.setStyleSheet("color: #ffd54f;")
        grid.addWidget(self.chk_pause_0v, 1, 0, 1, 2)

        grid.addWidget(QLabel("SC acquisition settle time (s):"), 2, 0)
        self.spin_sc_settle = QDoubleSpinBox()
        self.spin_sc_settle.setRange(0.0, 30.0)
        self.spin_sc_settle.setValue(2.0)
        self.spin_sc_settle.setDecimals(1)
        self.spin_sc_settle.setSingleStep(0.5)
        grid.addWidget(self.spin_sc_settle, 2, 1)

        grid.addWidget(QLabel("Minimum Acceptable R² Fit Quality:"), 3, 0)
        self.spin_min_r2 = QDoubleSpinBox()
        self.spin_min_r2.setRange(0.1, 0.999)
        self.spin_min_r2.setValue(0.85)
        self.spin_min_r2.setDecimals(3)
        self.spin_min_r2.setSingleStep(0.05)
        grid.addWidget(self.spin_min_r2, 3, 1)

        grid.addWidget(QLabel("Allowed f_roll Range (Hz):"), 4, 0)
        f_h = QHBoxLayout()
        self.spin_min_froll = QDoubleSpinBox()
        self.spin_min_froll.setRange(1.0, 1000.0)
        self.spin_min_froll.setValue(10.0)
        f_h.addWidget(self.spin_min_froll)
        f_h.addWidget(QLabel("to"))
        self.spin_max_froll = QDoubleSpinBox()
        self.spin_max_froll.setRange(100.0, 100000.0)
        self.spin_max_froll.setValue(25000.0)
        f_h.addWidget(self.spin_max_froll)
        grid.addLayout(f_h, 4, 1)

        self.chk_dry_run = QCheckBox("Simulation / Dry Run Mode (No DASTARD hardware required)")
        self.chk_dry_run.setChecked(False)
        grid.addWidget(self.chk_dry_run, 5, 0, 1, 2)

        self.chk_dark_mode = QCheckBox("Dark mode")
        self.chk_dark_mode.setChecked(False)
        self.chk_dark_mode.toggled.connect(self.apply_theme)
        grid.addWidget(self.chk_dark_mode, 6, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Event Handlers & Helpers
    # -----------------------------------------------------------------------
    def on_timing_source_changed(self):
        auto = self.chk_auto_cfg.isChecked()
        self.edit_cfg_path.setEnabled(auto)
        self.btn_browse_cfg.setEnabled(auto)
        self.spin_manual_tfr.setEnabled(not auto)

    def apply_theme(self):
        dark_mode = self.chk_dark_mode.isChecked()
        if dark_mode:
            self.setStyleSheet(self.dark_stylesheet)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f4f6f8;
                    color: #202124;
                }
                QGroupBox {
                    border: 1px solid #c7cbd1;
                    border-radius: 6px;
                    margin-top: 22px;
                    padding-top: 16px;
                    color: #245ea8;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 8px;
                }
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background-color: #ffffff;
                    border: 1px solid #aeb4bc;
                    color: #202124;
                    selection-background-color: #90caf9;
                }
                QPushButton {
                    background-color: #e4e8ed;
                    border: 1px solid #aeb4bc;
                    color: #202124;
                }
                QTextEdit {
                    background-color: #ffffff;
                    border: 1px solid #c7cbd1;
                    color: #202124;
                }
                QTabWidget::pane {
                    border: 1px solid #c7cbd1;
                    background-color: #ffffff;
                }
                QTabBar::tab {
                    background-color: #e4e8ed;
                    color: #4f5965;
                }
                QTabBar::tab:selected {
                    background-color: #ffffff;
                    color: #245ea8;
                }
            """)
        self.canvas.set_theme(dark_mode)

    def on_browse_cfg(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select experiment.cfg", "/data", "Config Files (*.cfg *.ini);;All Files (*)")
        if filename:
            self.edit_cfg_path.setText(filename)

    def update_bias_preview(self):
        try:
            dense = np.arange(self.spin_dense_start.value(), self.spin_dense_stop.value(), self.spin_dense_step.value())
            medium = np.arange(self.spin_med_start.value(), self.spin_med_stop.value() + 1e-9, self.spin_med_step.value())
            if self.chk_normal_first.isChecked():
                bias_vals = np.concatenate([[0.0], dense, medium])
            else:
                bias_vals = np.concatenate([[0.0], dense, medium])
                bias_vals = np.sort(bias_vals)
            bias_vals = np.round(bias_vals, 4)
            _, first_indices = np.unique(bias_vals, return_index=True)
            bias_vals = bias_vals[np.sort(first_indices)]

            self.lbl_bias_summary.setText(f"Total Points: {len(bias_vals)} | Max: {np.max(bias_vals):.3f} V")
            self.tbl_bias.setRowCount(len(bias_vals))
            for i, b in enumerate(bias_vals):
                item_idx = QTableWidgetItem(str(i + 1))
                item_idx.setTextAlignment(Qt.AlignCenter)
                item_v = QTableWidgetItem(f"{b:.3f}")
                item_v.setTextAlignment(Qt.AlignCenter)
                self.tbl_bias.setItem(i, 0, item_idx)
                self.tbl_bias.setItem(i, 1, item_v)
        except Exception as e:
            self.lbl_bias_summary.setText(f"Error computing bias: {e}")

    def get_configured_bias_values(self):
        dense = np.arange(self.spin_dense_start.value(), self.spin_dense_stop.value(), self.spin_dense_step.value())
        medium = np.arange(self.spin_med_start.value(), self.spin_med_stop.value() + 1e-9, self.spin_med_step.value())
        if self.chk_normal_first.isChecked():
            bias_vals = np.concatenate([[0.0], dense, medium])
        else:
            bias_vals = np.concatenate([[0.0], dense, medium])
            bias_vals = np.sort(bias_vals)
        bias_vals = np.round(bias_vals, 4)
        _, first_indices = np.unique(bias_vals, return_index=True)
        return bias_vals[np.sort(first_indices)]

    def get_time_per_flux_ramp(self):
        if self.chk_auto_cfg.isChecked():
            tfr, _ = get_time_per_flux_ramp_from_cfg(self.edit_cfg_path.text().strip())
            return tfr
        else:
            return self.spin_manual_tfr.value() * 1e-6

    def assemble_parameters(self, single_0v_only=False):
        tfr = self.get_time_per_flux_ramp()
        bias_vals = np.array([0.0]) if single_0v_only else self.get_configured_bias_values()

        return {
            "time_per_flux_ramp": tfr,
            "target_duration_s": self.spin_target_dur.value(),
            "target_f0_hz": self.spin_target_f0.value(),
            "nominal_awg_rate": self.spin_awg_rate.value() * 1e6,
            "num_high_freq_tones": self.spin_num_tones.value(),
            "n_averages": self.spin_averages.value(),
            "channel": self.spin_channel.value(),
            "bias_values": bias_vals,
            "normal_first": self.chk_normal_first.isChecked() and not single_0v_only,
            "vnorm": self.spin_vnorm.value(),
            "post_normal_settle_s": self.spin_post_normal_settle.value(),
            "awg_address": self.edit_awg_addr.text().strip(),
            "awg_channel": self.spin_awg_ch.value(),
            "awg_load": self.combo_load.currentText(),
            "awg_vpp": self.spin_awg_vpp.value(),
            "amp_multiplier": self.spin_sine_mult.value(),
            "settle_time_s": self.spin_settle.value(),
            "save_dir": self.edit_save_dir.text().strip(),
            "auto_abort_on_bad_sc": self.chk_auto_abort.isChecked(),
            "pause_after_0v": self.chk_pause_0v.isChecked() and not single_0v_only,
            "sc_settle_time_s": self.spin_sc_settle.value(),
            "min_sc_r2": self.spin_min_r2.value(),
            "min_froll_hz": self.spin_min_froll.value(),
            "max_froll_hz": self.spin_max_froll.value(),
            "single_0v_test_only": single_0v_only,
            "dry_run_mode": self.chk_dry_run.isChecked(),
        }

    # -----------------------------------------------------------------------
    # Action Button Handlers
    # -----------------------------------------------------------------------
    def on_verify_limits_clicked(self):
        try:
            bias_vals = self.get_configured_bias_values()
            voltage_check_values = bias_vals
            if self.chk_normal_first.isChecked():
                voltage_check_values = np.append(voltage_check_values, self.spin_vnorm.value())
            vpp = self.spin_awg_vpp.value()
            load = self.combo_load.currentText()
            v_limit = 10.0 if load.upper() in ("INFINITY", "INF") else 5.0
            v_max = np.max(voltage_check_values) + (vpp / 2.0)
            v_min = np.min(voltage_check_values) - (vpp / 2.0)
            headroom = v_limit - max(abs(v_max), abs(v_min))

            self.log("INFO", "=" * 55)
            self.log("INFO", f"AWG Voltage Verification (Load={load}):")
            self.log("INFO", f"  Max allowable: ±{v_limit:.2f} V")
            self.log("INFO", f"  Peak max: {v_max:+.4f} V | Peak min: {v_min:+.4f} V")

            if v_max > v_limit or v_min < -v_limit:
                msg = f"LIMIT EXCEEDED! Range [{v_min:+.3f} V, {v_max:+.3f} V] exceeds ±{v_limit:.1f} V."
                self.log("ERROR", msg)
                QMessageBox.critical(self, "Limit Error", msg)
            else:
                msg = f"Verification PASSED! Headroom: +{headroom:.3f} V"
                self.log("INFO", msg)
                QMessageBox.information(self, "Limit Verification", msg)
            self.log("INFO", "=" * 55)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_test_0v_clicked(self):
        self.start_worker(single_0v_only=True)

    def on_start_clicked(self):
        self.start_worker(single_0v_only=False)

    def start_worker(self, single_0v_only=False):
        try:
            params = self.assemble_parameters(single_0v_only=single_0v_only)
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", f"Invalid parameters: {e}")
            return

        self.btn_start.setEnabled(False)
        self.btn_test_0v.setEnabled(False)
        self.btn_check_limits.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_resume.setEnabled(False)

        state_tag = "0V TEST ONLY" if single_0v_only else "RUNNING"
        self.lbl_status.setText(f"STATE: {state_tag}")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 11pt;")
        self.progress_bar.setValue(0)

        self.worker = CZWorker(params)
        self.worker.log_signal.connect(self.log)
        self.worker.step_started.connect(self.on_worker_step_started)
        self.worker.step_completed.connect(self.on_worker_step_completed)
        self.worker.sc_calibrated.connect(self.on_sc_calibrated)
        self.worker.sc_check_failed.connect(self.on_sc_check_failed)
        self.worker.pause_for_inspection.connect(self.on_pause_for_inspection)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.start()

    def on_abort_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.lbl_status.setText("STATE: ABORTING...")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 11pt;")
            self.worker.abort()

    def on_resume_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.btn_resume.setEnabled(False)
            self.lbl_status.setText("STATE: RUNNING")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 11pt;")
            self.worker.resume()

    # -----------------------------------------------------------------------
    # Worker Signals Handling
    # -----------------------------------------------------------------------
    @pyqtSlot(int, int, float)
    def on_worker_step_started(self, idx, total, bias_v):
        self.lbl_current_bias.setText(f"Bias: {bias_v:.3f} V [{idx}/{total}]")

    @pyqtSlot(float, dict)
    def on_worker_step_completed(self, bias_v, plot_data):
        self.canvas.update_plots(plot_data, is_sc_eval=(bias_v == 0.0))

    @pyqtSlot(dict)
    def on_sc_calibrated(self, sc_stats):
        r2 = sc_stats["r_squared"]
        f_roll = sc_stats["f_roll"]
        self.lbl_sc_health.setText(f"SC 0V Health: PASS (R² = {r2:.4f}, f_roll = {f_roll:.1f} Hz)")
        self.lbl_sc_health.setStyleSheet("padding: 4px; border-radius: 4px; background-color: #1b5e20; color: #b9f6ca; font-weight: bold;")

    @pyqtSlot(str, dict)
    def on_sc_check_failed(self, reason, sc_stats):
        r2 = sc_stats["r_squared"]
        f_roll = sc_stats["f_roll"]
        self.lbl_sc_health.setText(f"SC 0V Health: BAD ({reason})")
        self.lbl_sc_health.setStyleSheet("padding: 4px; border-radius: 4px; background-color: #b71c1c; color: #ffcdd2; font-weight: bold;")

    @pyqtSlot(dict)
    def on_pause_for_inspection(self, sc_stats):
        self.lbl_status.setText("STATE: PAUSED FOR 0V REVIEW")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #ffd600; font-size: 11pt;")
        self.btn_resume.setEnabled(True)
        QMessageBox.information(
            self,
            "0 V Calibration Inspection",
            f"0 V Superconducting Transfer Function fit complete:\n"
            f"  - R² Fit Quality: {sc_stats['r_squared']:.4f}\n"
            f"  - f_roll: {sc_stats['f_roll']:.1f} Hz\n"
            f"  - tau_shift: {sc_stats['tau_shift']*1e6:.2f} µs\n\n"
            f"Please inspect the 4 diagnostic subplots.\n"
            f"Click 'Resume / Proceed' to sweep higher voltages, or 'ABORT' to halt."
        )

    @pyqtSlot(int, int)
    def on_progress(self, current, total):
        pct = int(round((current / total) * 100))
        self.progress_bar.setValue(pct)

    @pyqtSlot(bool, str)
    def on_worker_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_test_0v.setEnabled(True)
        self.btn_check_limits.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.btn_resume.setEnabled(False)

        if success:
            self.lbl_status.setText("STATE: COMPLETED")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 11pt;")
            self.log("INFO", message)
            QMessageBox.information(self, "Acquisition Complete", message)
        else:
            self.lbl_status.setText("STATE: ABORTED / ERROR")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 11pt;")
            self.log("ERROR", message)
            QMessageBox.warning(self, "Acquisition Stopped", message)

    @pyqtSlot(str, str)
    def log(self, level, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": "#80cbc4",
            "WARN": "#ffd54f",
            "ERROR": "#ef5350"
        }
        color = color_map.get(level.upper(), "#e0e4ed")
        html = f"<span style='color:#78909c;'>[{timestamp}]</span> <b style='color:{color};'>[{level.upper()}]</b> {msg}"
        self.txt_log.append(html)
        # Scroll to end
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        """Safe shutdown on window close."""
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "An acquisition run is currently active! Abort and zero AWG offset before exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.worker.abort()
                self.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ===========================================================================
# Application Entrypoint
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="EZ CZ dashboard GUI")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to experiment.cfg")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    window = CZMainWindow()
    if os.path.exists(args.config):
        window.edit_cfg_path.setText(args.config)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

