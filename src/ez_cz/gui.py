#!/usr/bin/env python3
"""
gui.py: EZ-CZ Dashboard for Complex Impedance (CZ) Measurement.

Features:
  - Selectable DC Bias Source via rolldown menu:
      * "DC source: AWG" (Default): Direct DC offset on Keysight 33600A AWG Ch2.
      * "DC source: SRS900": External SRS SIM900/SIM928 DC voltage box.
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
  - Dedicated instant ABORT button that safely zeroes voltages on AWG and SRS900.
  - Simulation / Dry Run mode for verification and testing without hardware.
"""

import sys
import os
import time
import json
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import scipy.fft
from lmfit import Model, Parameters

logger = logging.getLogger(__name__)

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit,
    QCheckBox, QMessageBox, QComboBox, QDoubleSpinBox, QSpinBox, QSplitter,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QScrollArea, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from ez_cz.models import AVAILABLE_MODELS, get_model, ParameterDef, FitResult
from ez_cz.data_io import load_iv_curve, load_cz_sweep_dataset

# ---------------------------------------------------------------------------
# Hardware Imports with Mock Fallbacks for Simulation / Offline Use
# ---------------------------------------------------------------------------
try:
    import qsdumux
    import qsdumux.instruments.keysight33600A_usb as keysight33600A_usb
    import qsdumux.instruments.srs_sim900 as sim900
    import qsdumux.instruments.srs_sim928 as sim928
    HAS_HARDWARE_LIBS = True
except ImportError:
    HAS_HARDWARE_LIBS = False
    keysight33600A_usb = None
    sim900 = None
    sim928 = None

try:
    from dastardcommander import EasyClientDastard
    HAS_DASTARD_LIB = True
except ImportError:
    HAS_DASTARD_LIB = False
    EasyClientDastard = None


class MockKeysight33600A:
    """Mock Keysight AWG for dry-run simulation mode."""
    def __init__(self, *args, **kwargs):
        self.offsets = {1: 0.0, 2: 0.0}
        self.channel = self

    def setOutputLoad(self, source, load):
        pass

    def _checkSource(self, source):
        pass

    def clearVolatileMem(self, source):
        pass

    def setByteOrder(self, swap_bytes):
        pass

    def _write_raw(self, data):
        pass

    def setFunction(self, waveform_function, source):
        pass

    def setArbFile(self, source, arb_name):
        pass

    def setArbSampleRate(self, source, rate):
        pass

    def setArbFilter(self, source, filter_state):
        pass

    def setAmplitude(self, source, amp):
        pass

    def setOffset(self, source, offset):
        self.offsets[source] = offset

    def getOffset(self, source):
        return self.offsets.get(source, 0.0)

    def setOutputState(self, source, output_state):
        pass

    def _ask(self, query):
        return "+0, No error"

    def readline(self):
        return b"+0\n"

    def close(self):
        pass


class MockSRS_SIM900:
    """Mock SRS SIM900 mainframe for dry-run simulation mode."""
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class MockSRS_SIM928:
    """Mock SRS SIM928 isolated DC source for dry-run simulation mode."""
    def __init__(self, mainframe, port):
        self.port = port
        self.voltage = 0.0

    def setvolt(self, voltage):
        self.voltage = voltage

    def getvolt(self):
        return self.voltage

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Timing & Mathematical Helpers (Pure time_per_flux_ramp derivation)
# ---------------------------------------------------------------------------
TIME_PER_FLUX_RAMP_POINT_S = 256e-9  # FPGA clock tick (1 / 3.90625 MHz nominal fallback)
DEFAULT_CONFIG_PATH = "/data/experiment.cfg"
SETTINGS_FILE_PATH = "run_settings.json"


def query_time_per_flux_ramp_point(cfg_path=DEFAULT_CONFIG_PATH, default_dt=TIME_PER_FLUX_RAMP_POINT_S):
    """
    Dynamically queries the fine DDC sample rate directly from the FPGA firmware
    via qsdumux / ctrl interface specified in cfg_path.

    The FPGA firmware IP-core (e.g. QSG-NIST:qsg-nist:chpfbfft) defines:
      - ADC sample rate (from TSNUM, TSDENOM, SAMPLESPERCLK)
      - PFB window length (WINDOWLEN)
      - Number of sets (NSETS)
    The fine DDC rate is:
      rate = samplerate / windowlen * nsets
    And the time per flux ramp point is:
      dt = 1.0 / rate (nominally 256 ns = 1 / 3.90625 MHz)

    If the hardware is offline, unconfigured, or qsdumux is unavailable,
    it gracefully falls back to default_dt.
    """
    ctrl_addr = None
    data_addr = None
    fset = "fset0"

    if os.path.exists(cfg_path):
        import configparser
        parser = configparser.ConfigParser()
        try:
            parser.read(cfg_path)
            for sec in ["Global", "global"]:
                if sec in parser:
                    if "ctrl" in parser[sec]:
                        ctrl_addr = parser[sec]["ctrl"].split("#")[0].strip()
                    if "data" in parser[sec]:
                        data_addr = parser[sec]["data"].split("#")[0].strip()
                    if "fset_path" in parser[sec]:
                        raw_fset = parser[sec]["fset_path"].split("#")[0].strip()
                        clean = raw_fset.strip("[]").replace('"', "").replace("'", "").split(",")[0].strip()
                        if clean:
                            fset = clean
                    break
        except Exception as e:
            logger.debug(f"Could not parse ctrl settings from {cfg_path}: {e}")

    if ctrl_addr:
        try:
            import argparse
            import qsdumux.user.abaco_setup as abaco_setup

            cfg = argparse.Namespace()
            cfg.verbose = 0
            cfg.ctrl = ctrl_addr
            cfg.data = data_addr or ctrl_addr
            cfg.rate = False
            cfg.open_data = False
            cfg.quiet = True
            cfg.show_access = False

            ctrlif, dataif, eids, circb = abaco_setup.load_abaco_general(cfg)
            try:
                fullset, frontend, wfplayer, backends = abaco_setup.load_abaco_fullset(cfg, fset, ctrlif, eids)
                if frontend is not None:
                    rate = float(frontend.samplerate) / float(frontend.get_windowlen()) * float(frontend.get_nsets())
                    if rate > 0:
                        dt = 1.0 / rate
                        vlnv = getattr(getattr(frontend, "entityid", None), "vlnv", "unknown")
                        logger.info(f"Retrieved FPGA clock from firmware ({vlnv}): dt={dt*1e9:.3f} ns (f_ddc={rate/1e6:.6f} MHz)")
                        return dt
            finally:
                if hasattr(ctrlif, "close"):
                    ctrlif.close()
                if hasattr(dataif, "close"):
                    dataif.close()
        except Exception as e:
            logger.info(f"Could not query FPGA firmware clock ({e}); falling back to nominal {default_dt*1e9:.1f} ns")

    return default_dt


def get_time_per_flux_ramp_from_cfg(cfg_path=DEFAULT_CONFIG_PATH, query_firmware=True):
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

            dt_point = query_time_per_flux_ramp_point(cfg_path=cfg_path, default_dt=TIME_PER_FLUX_RAMP_POINT_S) if query_firmware else TIME_PER_FLUX_RAMP_POINT_S
            time_per_flux_ramp = fr_points * dt_point
            return time_per_flux_ramp, fr_points

    raise KeyError(f"Could not find 'flux_ramp_points' under [Global] in {cfg_path}")


def derive_timing_from_flux_ramp(time_per_flux_ramp, target_duration_s=0.2,
                                 target_f0_hz=20.0, nominal_awg_rate=5e6):
    """
    Derives all DAQ and AWG parameters strictly from time_per_flux_ramp.
    Harmonizes acquisition duration to an exact integer number of cycles of target_f0_hz
    so every comb tone lands identically on an FFT bin with zero spectral leakage.
    """
    fs_global = 1.0 / time_per_flux_ramp
    k_periods = max(1, int(round(target_duration_s * target_f0_hz)))
    t_nominal = k_periods / target_f0_hz
    N_samples = int(round(t_nominal / time_per_flux_ramp))
    t_duration = N_samples * time_per_flux_ramp

    f_fund = 1.0 / t_duration
    f0 = k_periods * f_fund
    f_low = f0

    n_awg_points = int(round(t_duration * nominal_awg_rate))
    awg_sample_rate = n_awg_points / t_duration

    return {
        "time_per_flux_ramp": time_per_flux_ramp,
        "fs_global": fs_global,
        "N_samples": N_samples,
        "t_duration": t_duration,
        "f_fund": f_fund,
        "f_low": f_low,
        "k0": k_periods,
        "f0": f0,
        "n_awg_points": n_awg_points,
        "awg_sample_rate": awg_sample_rate,
    }


def generate_stimulus_tones(fs_global, f0, f_low, start_exp=0, stop_exp=None, num_high_freq_tones=100,
                            high_f_cutoff=None, skip_line_harmonics=True, line_freq=60.0, line_freq_tol=2.0):
    """
    Generates log-spaced tone multipliers up to the high-frequency cutoff.
    Strictly enforces Nyquist limit: internally rejects any tone >= Nyquist (fs_global / 2).
    Optionally skips power line frequency harmonics (default: 60 Hz multiples).
    Deduplicates frequencies so the fundamental is never doubled.
    """
    nyquist = fs_global / 2.0
    # Strictly enforce f < Nyquist: never accept anything equal to or greater than Nyquist
    max_nyquist_safe = nyquist - (f0 * 0.5)

    if high_f_cutoff is not None and high_f_cutoff > 0:
        effective_max_f = min(float(high_f_cutoff), max_nyquist_safe)
    elif stop_exp is not None:
        effective_max_f = min(f0 * (10 ** stop_exp), max_nyquist_safe)
    else:
        effective_max_f = min(50000.0, max_nyquist_safe)

    effective_max_f = min(effective_max_f, max_nyquist_safe)
    max_allowable_multiplier = int(np.floor(effective_max_f / f0))
    if max_allowable_multiplier < 1:
        max_allowable_multiplier = 1

    oversample = 2.5 if skip_line_harmonics else 1.6
    raw_mults = np.geomspace(1, max_allowable_multiplier, int(num_high_freq_tones * oversample))
    multipliers = np.unique(np.int64(np.round(raw_mults)))
    valid_mults = [int(m) for m in multipliers if (1 <= m <= max_allowable_multiplier) and (m * f0 < nyquist)]

    if skip_line_harmonics and line_freq > 0:
        filtered_mults = []
        for m in valid_mults:
            f = m * f0
            dist_to_harmonic = abs(f - round(f / line_freq) * line_freq)
            if dist_to_harmonic >= line_freq_tol:
                filtered_mults.append(m)
        valid_mults = filtered_mults

    if len(valid_mults) > num_high_freq_tones:
        indices = np.round(np.linspace(0, len(valid_mults) - 1, num_high_freq_tones)).astype(int)
        valid_mults = [valid_mults[i] for i in indices]

    raw_freqs = np.insert(np.array(valid_mults) * f0, 0, f_low)
    # Final strict safety clamp: strictly less than Nyquist
    raw_freqs = raw_freqs[raw_freqs < nyquist]
    _, unique_idx = np.unique(np.round(raw_freqs, 6), return_index=True)
    all_freqs = raw_freqs[np.sort(unique_idx)]
    return valid_mults, all_freqs


def synthesize_comb_waveform(all_freqs, t_duration, awg_sample_rate, multiplier, use_schroeder=True):
    """
    Synthesizes multi-tone cosine waveform with Schroeder crest-factor minimization.

    When use_schroeder is True, phases are set to:
        phi_k = -pi * k * (k - 1) / N
    This drastically reduces the peak-to-average power ratio (PAPR / crest factor)
    by ~sqrt(N), allowing 20 to 100+ tones without clipping the AWG volatile memory.
    """
    npts = int(round(t_duration * awg_sample_rate))
    t = np.arange(npts) / awg_sample_rate
    v_superimposed = np.zeros(npts, dtype=np.float64)
    N = len(all_freqs)

    if use_schroeder and N > 1:
        k = np.arange(N)
        phases = -np.pi * k * (k - 1) / N
    else:
        phases = np.zeros(N)

    for f, phi in zip(all_freqs, phases):
        v_superimposed += np.cos(2 * np.pi * t * f + phi)

    v_out = v_superimposed * multiplier
    peak = float(np.max(np.abs(v_out)))
    scale_factor = 1.0
    if peak > 1.0:
        scale_factor = peak
        v_out = v_out / peak

    return v_out, phases, scale_factor


# ---------------------------------------------------------------------------
# Background Acquisition & Processing Worker
# ---------------------------------------------------------------------------
class CZWorker(QThread):
    """
    Executes CZ data collection on a separate background thread.
    Supports both AWG DC offset and SRS SIM900/SIM928 DC bias sources.
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
        self.mainframe = None
        self.vbox = None

    def abort(self):
        """Signals worker to abort execution immediately."""
        self._is_aborted = True
        self.log_signal.emit("WARN", "ABORT requested by user! Halting acquisition...")
        self.resume()

    def resume(self):
        """Resumes execution after user inspection pause."""
        self._resume_mutex.lock()
        self._is_paused = False
        self._resume_event.wakeAll()
        self._resume_mutex.unlock()

    def sleep_interruptible(self, seconds):
        """Sleeps in small slices, returning False immediately if aborted."""
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._is_aborted:
                return False
            time.sleep(min(0.04, max(0.005, end_time - time.time())))
        return not self._is_aborted

    def run(self):
        self._is_aborted = False
        p = self.params
        dc_source = p.get("dc_source", "AWG")
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_date = datetime.now().strftime("%Y%m%d")

        # Determine save directory
        save_root = Path(p["save_dir"])
        base_dir = save_root / run_date / "Complex_Z" / run_timestamp
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_dir = Path("./data") / run_date / "Complex_Z" / run_timestamp
            base_dir.mkdir(parents=True, exist_ok=True)

        self.log_signal.emit("INFO", f"Run directory: {base_dir}")
        self.log_signal.emit("INFO", f"Active DC Bias Source: {dc_source}")

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
        vpp = p["awg_vpp"]
        load = p["awg_load"]
        v_limit = 10.0 if load.upper() in ("INFINITY", "INF") else 5.0

        if dc_source == "AWG":
            voltage_check_values = bias_values
            if p.get("normal_first", False):
                voltage_check_values = np.append(voltage_check_values, p["vnorm"])
            v_max = np.max(voltage_check_values) + (vpp / 2.0)
            v_min = np.min(voltage_check_values) - (vpp / 2.0)
            headroom = v_limit - max(abs(v_max), abs(v_min))

            if v_max > v_limit or v_min < -v_limit:
                err = f"Voltage limits exceeded! Range [{v_min:+.3f} V, {v_max:+.3f} V] exceeds ±{v_limit:.1f} V."
                self.log_signal.emit("ERROR", err)
                self.finished_signal.emit(False, err)
                return
            self.log_signal.emit("INFO", f"AWG Voltage verification PASS: Headroom = +{headroom:.3f} V")
        else:
            ac_peak = vpp / 2.0
            if ac_peak > v_limit:
                err = f"AWG AC stimulus ({vpp:.4f} Vpp) exceeds ±{v_limit:.1f} V limit!"
                self.log_signal.emit("ERROR", err)
                self.finished_signal.emit(False, err)
                return
            self.log_signal.emit("INFO", f"AWG AC Output verification PASS (Headroom: +{v_limit - ac_peak:.3f} V)")
            self.log_signal.emit("INFO", f"DC Bias will be driven via SRS SIM928 isolated source (Port {p.get('srs_port', 2)})")

        # 3. Connect to AWG and upload waveform
        self.log_signal.emit("INFO", f"Connecting to AWG at {p['awg_address']}...")
        is_dry_run = p.get("dry_run_mode", False) or not HAS_HARDWARE_LIBS

        try:
            if is_dry_run:
                self.fg = MockKeysight33600A()
                self.log_signal.emit("INFO", "Running with Mock Keysight AWG (Dry Run / Simulation mode).")
            elif p["awg_address"].startswith("vxi11://") or "://" in p["awg_address"]:
                self.fg = keysight33600A_usb.Keysight33600A_usb(url=p["awg_address"])
            else:
                self.fg = keysight33600A_usb.Keysight33600A_usb(device=p["awg_address"])
        except Exception as e:
            err = f"Failed to connect to AWG: {e}"
            self.log_signal.emit("ERROR", err)
            self.finished_signal.emit(False, err)
            return

        # 3b. If SRS900 DC source selected, connect to SRS SIM900 mainframe & SIM928
        if dc_source == "SRS900":
            srs_url = p.get("srs_url", "serial:///dev/ttyUSB1?baudrate=9600")
            srs_port = p.get("srs_port", 2)
            self.log_signal.emit("INFO", f"Connecting to SRS SIM900 mainframe at {srs_url} (Port {srs_port})...")
            try:
                if is_dry_run:
                    self.mainframe = MockSRS_SIM900()
                    self.vbox = MockSRS_SIM928(self.mainframe, srs_port)
                    self.log_signal.emit("INFO", "Running with Mock SRS SIM928 (Dry Run / Simulation mode).")
                else:
                    self.mainframe = sim900.SRS_SIM900(url=srs_url)
                    self.vbox = sim928.SRS_SIM928(self.mainframe, srs_port)
                    self.log_signal.emit("INFO", "Connected to SRS SIM900 / SIM928 successfully.")
            except Exception as e:
                err = f"Failed to connect to SRS SIM900/928: {e}"
                self.log_signal.emit("ERROR", err)
                self._safe_cleanup()
                self.finished_signal.emit(False, err)
                return

        # Generate comb waveform
        f_mults, all_freqs = generate_stimulus_tones(
            fs_global, f0, f_low,
            num_high_freq_tones=p.get("num_high_freq_tones", 100),
            high_f_cutoff=p.get("high_f_cutoff", 50000.0),
            skip_line_harmonics=p.get("skip_line_harmonics", True),
            line_freq=p.get("line_freq", 60.0)
        )
        use_schroeder = p.get("use_schroeder_phase", True)
        v_waveform, phases, scale_factor = synthesize_comb_waveform(
            all_freqs, t_duration, awg_sample_rate, p["amp_multiplier"], use_schroeder=use_schroeder
        )
        peak_v = np.max(np.abs(v_waveform))
        self.log_signal.emit("INFO", (
            f"Synthesized {len(all_freqs)} tones ({all_freqs[0]:.1f} Hz to {all_freqs[-1]:.1f} Hz). "
            f"Schroeder phase: {use_schroeder}, Peak: {peak_v:.3f} (scale factor: {scale_factor:.3f})"
        ))

        # Save supplied waveform
        try:
            np.savez(
                base_dir / "supplied_waveform.npz",
                waveform=v_waveform,
                frequencies=all_freqs,
                phases=phases,
                scale_factor=scale_factor,
                use_schroeder=use_schroeder,
                awg_sample_rate=awg_sample_rate,
                amplitude_vpp=vpp,
                multiplier=p["amp_multiplier"],
                fs_global=fs_global,
                time_per_flux_ramp=time_per_flux_ramp,
                dc_source=dc_source,
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

            # In AWG mode, initial offset is first bias; in SRS mode, AWG offset is 0.0 V
            initial_offset = bias_values[0] if dc_source == "AWG" else 0.0
            self.fg.setOffset(source, initial_offset)
            self.fg.setOutputState(source=source, output_state="ON")
            self.sleep_interruptible(0.5)

            if dc_source == "AWG":
                self.log_signal.emit("INFO", f"AWG Ch{source} active: Vpp={vpp} V, Offset={initial_offset:.3f} V")
            else:
                self.log_signal.emit("INFO", f"AWG Ch{source} active (AC stimulus only, Vpp={vpp} V, Offset=0.0 V)")
        except Exception as e:
            err = f"Failed to program AWG: {e}"
            self.log_signal.emit("ERROR", err)
            self._safe_cleanup()
            self.finished_signal.emit(False, err)
            return

        # 4. Connect to DASTARD
        c = None
        if not is_dry_run and HAS_DASTARD_LIB:
            self.log_signal.emit("INFO", "Connecting to DASTARD...")
            for attempt in range(5):
                if self._is_aborted:
                    self._safe_cleanup()
                    self.finished_signal.emit(False, "Aborted before DASTARD connection.")
                    return
                try:
                    c = EasyClientDastard()
                    self.log_signal.emit("INFO", "Connected to DASTARD successfully.")
                    break
                except Exception as e:
                    self.log_signal.emit("WARN", f"DASTARD connection attempt {attempt+1}/5 failed ({e})")
                    if not self.sleep_interruptible(0.5):
                        self._safe_cleanup()
                        self.finished_signal.emit(False, "Aborted during DASTARD connection.")
                        return

            if c is None:
                err = "Failed to connect to DASTARD client."
                self.log_signal.emit("ERROR", err)
                self._safe_cleanup()
                self.finished_signal.emit(False, err)
                return
        else:
            self.log_signal.emit("INFO", "Using simulated detector signal (Dry Run / Simulation mode).")

        # 5. Acquisition Loop
        tau_shift_0V = None
        Z_model_pure = None
        amp0 = None
        f_roll = None
        total_steps = len(bias_values)
        effective_mult = p["amp_multiplier"] / scale_factor
        V_stim = effective_mult * vpp

        try:
            for idx, bias in enumerate(bias_values):
                if self._is_aborted:
                    break

                self.step_started.emit(idx + 1, total_steps, bias)
                self.progress_signal.emit(idx + 1, total_steps)
                self.log_signal.emit("INFO", f"[{idx+1}/{total_steps}] Setting bias = {bias:.3f} V ({dc_source})...")

                # Apply DC bias based on selected DC source
                point_settle_time = p.get("settle_time_s", 0.3)
                if np.isclose(bias, 0.0):
                    point_settle_time = p.get("sc_settle_time_s", 2.0)

                if dc_source == "AWG":
                    if not self._ramp_offset(bias):
                        break
                    if not self.sleep_interruptible(point_settle_time):
                        break
                else:
                    # SRS SIM900/SIM928 bias application
                    if p.get("srs_pulse_vnorm", True) and not np.isclose(bias, 0.0):
                        vnorm_val = p.get("vnorm", 9.5)
                        self.log_signal.emit("INFO", f"Pulsing SRS SIM928 Vnorm = {vnorm_val:.3f} V...")
                        self.vbox.setvolt(vnorm_val)
                        if not self.sleep_interruptible(0.1):
                            break
                    self.vbox.setvolt(bias)
                    if not self.sleep_interruptible(point_settle_time):
                        break

                if self._is_aborted:
                    break

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
                        if self._is_aborted:
                            break
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
                    if self._is_aborted:
                        break
                    selected_index = int(np.argmax(candidate_scores))
                    data = candidate_data[selected_index]
                    data_all = candidate_results[selected_index]

                    # Detect DASTARD SQUID demodulation unlock / flux jump
                    data_check = data_all[ch_key] * 1.0
                    raw_std = float(np.std(data_check))
                    raw_mean = float(np.mean(data_check))
                    raw_min = float(np.min(data_check))
                    raw_max = float(np.max(data_check))
                    if raw_std > 3000.0 or (raw_min < 5000 and raw_max > 55000):
                        self.log_signal.emit(
                            "WARN",
                            f"DASTARD SQUID unlock / flux jump detected at {bias:.3f} V "
                            f"(std={raw_std:.0f}, mean={raw_mean:.0f}). Response will appear as noise!"
                        )
                else:
                    t = np.arange(N_samples) * time_per_flux_ramp
                    data = np.zeros(N_samples)
                    f_r_sim = 1200.0
                    for f, phi_stim in zip(all_freqs, phases):
                        mag = 5000.0 / np.sqrt(1 + (f / f_r_sim)**2)
                        phi_sys = -np.arctan(f / f_r_sim) - 2 * np.pi * f * 15e-6
                        data += mag * np.cos(2 * np.pi * f * t + phi_stim + phi_sys)
                    data += np.random.normal(0, 10, N_samples)
                    data_all = {f"chan{p['channel']}": np.tile(data, p.get("n_averages", 4)), "firstFrameIndex": [0]}

                if self._is_aborted:
                    break

                # Compute FFT
                fft_data_raw = scipy.fft.rfft(data)
                fft_freqs = scipy.fft.rfftfreq(N_samples, d=time_per_flux_ramp)
                f_sch_idx = np.argmin(
                    np.abs(fft_freqs[:, None] - all_freqs[None, :]),
                    axis=0,
                )
                fs_sch = all_freqs
                tone_data_raw = fft_data_raw[f_sch_idx] * np.exp(-1j * phases)
                sc_tone_data_raw = tone_data_raw

                # 0 V Calibration & Superconducting Transfer Function Health Check
                if np.isclose(bias, 0.0):
                    self.log_signal.emit("INFO", "Evaluating Superconducting (0 V) Transfer Function...")

                    def magnitude_model(fs, amp0_val, f_roll_val):
                        return amp0_val / np.sqrt(1 + (fs / f_roll_val)**2)

                    model_mag = Model(magnitude_model)
                    params_mag = Parameters()
                    params_mag.add("amp0_val", value=np.abs(fft_data_raw[0]), min=0, vary=True)
                    params_mag.add("f_roll_val", value=1e3, min=1.0, max=1e6, vary=True)

                    y_data = np.abs(sc_tone_data_raw)
                    res_mag = model_mag.fit(y_data, fs=fs_sch, params=params_mag)

                    amp0 = float(res_mag.params["amp0_val"].value)
                    f_roll = float(res_mag.params["f_roll_val"].value)

                    # Goodness of fit (R^2)
                    fit_best = magnitude_model(fs_sch, amp0, f_roll)
                    ss_res = np.sum((y_data - fit_best)**2)
                    ss_tot = np.sum((y_data - np.mean(y_data))**2)
                    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

                    flip_bias = p.get("flip_bias", False)
                    model_sign = -1.0 if flip_bias else 1.0
                    Z_model_fit = model_sign * amp0 / (1.0 + 1j * fs_sch / f_roll)
                    Z_model_pure = amp0 / (1.0 + 1j * fs_sch / f_roll)

                    # Coherent matched filter for coarse delay estimation across any number of tones & phases
                    H_eq = sc_tone_data_raw / Z_model_fit
                    comb_eq = np.zeros(len(fft_data_raw), dtype=complex)
                    comb_eq[f_sch_idx] = H_eq / np.maximum(np.abs(H_eq), 1e-12)
                    time_impulse = np.fft.irfft(comb_eq)

                    peak_sample = int(np.argmax(time_impulse))
                    if peak_sample > N_samples // 2:
                        tau_coarse = (peak_sample - N_samples) * time_per_flux_ramp
                    else:
                        tau_coarse = peak_sample * time_per_flux_ramp

                    tau_guess = -tau_coarse

                    def real_model(fs, tau_shift):
                        return np.real(sc_tone_data_raw * np.exp(-2j * np.pi * fs_sch * tau_shift))

                    model_real = Model(real_model)
                    params_real = Parameters()
                    params_real.add("tau_shift", value=tau_guess, min=tau_guess - 25e-6, max=tau_guess + 25e-6, vary=True)
                    try:
                        res_real = model_real.fit(np.real(Z_model_fit), fs=fs_sch, params=params_real)
                        tau_shift_0V = float(res_real.params["tau_shift"].value)
                    except Exception as e:
                        self.log_signal.emit("WARN", f"Fine tau fitting note ({e}); using coherent matched filter estimate.")
                        tau_shift_0V = tau_guess

                    sc_stats = {
                        "amp0": amp0,
                        "f_roll": f_roll,
                        "r_squared": r_squared,
                        "tau_shift": tau_shift_0V,
                        "y_data": y_data,
                        "best_fit": fit_best,
                        "all_freqs": all_freqs,
                        "flip_bias": flip_bias,
                    }
                    self.sc_calibrated.emit(sc_stats)
                    self.log_signal.emit("INFO", (
                        f"0 V Fit Results: R² = {r_squared:.4f}, f_roll = {f_roll:.1f} Hz, "
                        f"amp0 = {amp0:.1f}, τ_shift = {tau_shift_0V*1e6:.2f} µs (flip_bias={flip_bias})"
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
                            self._safe_cleanup()
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
                            self._safe_cleanup()
                            self.finished_signal.emit(False, "Aborted by user during 0 V calibration pause.")
                            return
                        self.log_signal.emit("INFO", "User approved 0 V calibration. Resuming sweep...")

                    # Optional normalize TES before sweeping higher voltages
                    if p.get("normal_first", False):
                        vnorm_val = p.get("vnorm", 9.5)
                        self.log_signal.emit("INFO", f"Driving TES normal at Vnorm = {vnorm_val:.3f} V for 1.0 s...")
                        if dc_source == "AWG":
                            self._ramp_offset(vnorm_val)
                        else:
                            self.vbox.setvolt(vnorm_val)
                        if not self.sleep_interruptible(1.0):
                            break
                        post_settle = p.get("post_normal_settle_s", 5.0)
                        if post_settle > 0:
                            self.log_signal.emit("INFO", f"Waiting {post_settle:.1f} s post-normal recovery before sweeping...")
                            first_bias = bias_values[1] if len(bias_values) > 1 else 0.0
                            if dc_source == "AWG":
                                self._ramp_offset(first_bias)
                            else:
                                self.vbox.setvolt(first_bias)
                            if not self.sleep_interruptible(post_settle):
                                break

                applied_tau_shift = tau_shift_0V if tau_shift_0V is not None else 0.0
                raw_unroll = tone_data_raw * np.exp(-2j * np.pi * fs_sch * applied_tau_shift)
                flip_bias = p.get("flip_bias", False)
                model_sign = -1.0 if flip_bias else 1.0
                # Invert unrolled response if flip_bias is True so Re(Z) is positive at DC
                fft_unroll = model_sign * raw_unroll

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
                    time_per_flux_ramp=time_per_flux_ramp,
                    dc_source=dc_source,
                    flip_bias=flip_bias
                )

                plot_data = {
                    "bias": bias,
                    "bias_dir": str(bias_dir),
                    "tau_shift": applied_tau_shift,
                    "fs_sch": fs_sch,
                    "fft_unroll": fft_unroll,
                    "Z_model_pure": Z_model_pure,
                    "amp0": amp0,
                    "f_roll": f_roll,
                    "dc_source": dc_source,
                    "flip_bias": flip_bias,
                }
                self.step_completed.emit(bias, plot_data)

                if p.get("single_0v_test_only", False):
                    self.log_signal.emit("INFO", "Single 0 V SC test completed.")
                    break

        except Exception as e:
            self.log_signal.emit("ERROR", f"Acquisition error: {e}")
            self._safe_cleanup()
            self.finished_signal.emit(False, str(e))
            return

        # Final cleanup
        self._safe_cleanup()
        if self._is_aborted:
            self.finished_signal.emit(False, "Run was aborted by user. All DC sources safely reset to 0.0 V.")
        else:
            self.finished_signal.emit(True, f"Run completed successfully! All data saved to {base_dir}")

    def _ramp_offset(self, target_offset, max_step=0.01, step_delay=0.005):
        """Ramps AWG offset smoothly in small increments to prevent SQUID unlock / flux jumps."""
        if self.fg is None:
            return False
        source = self.params["awg_channel"]
        current_offset = float(self.fg.getOffset(source))
        diff = target_offset - current_offset
        if abs(diff) <= max_step:
            self.fg.setOffset(source, target_offset)
            return True
        n_steps = int(np.ceil(abs(diff) / max_step))
        ramp_vals = np.linspace(current_offset, target_offset, n_steps + 1)[1:]
        for v in ramp_vals:
            if self._is_aborted:
                return False
            self.fg.setOffset(source, float(v))
            time.sleep(step_delay)
        return True

    def _safe_cleanup(self):
        """Safely ramps down the AWG DC offset and/or SRS SIM928 voltage to 0.0 V."""
        if self.fg is not None:
            try:
                self.log_signal.emit("INFO", "Safely resetting AWG Channel offset to 0.0 V and disabling output...")
                self.fg.setOffset(source=self.params["awg_channel"], offset=0.0)
                self.fg.setOutputState(source=self.params["awg_channel"], output_state="OFF")
            except Exception as e:
                self.log_signal.emit("WARN", f"AWG cleanup exception: {e}")
            try:
                self.fg.channel.close()
            except Exception:
                pass
            self.fg = None

        if self.vbox is not None:
            try:
                self.log_signal.emit("INFO", "Safely resetting SRS SIM928 voltage to 0.0 V...")
                self.vbox.setvolt(0.0)
            except Exception as e:
                self.log_signal.emit("WARN", f"SRS SIM928 cleanup exception: {e}")
            self.vbox = None

        if self.mainframe is not None:
            try:
                self.mainframe.close()
            except Exception:
                pass
            self.mainframe = None

    def _save_bias_plot(self, bias_dir, bias, fs_sch, fft_unroll, Z_model_pure, tau_shift, dc_source):
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

        fig.suptitle(f"EZ-CZ Dashboard | Bias = {bias:.3f} V ({dc_source}) | Delay = {tau_shift * 1e6:.2f} µs")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(bias_dir / f"vbias_{bias:.2f}_impedance_plot.png".replace(".", "v", 1))


# ---------------------------------------------------------------------------
# Embedded 4-Panel Matplotlib Canvas Widget
# ---------------------------------------------------------------------------
class CZPlotCanvas(FigureCanvas):
    """Matplotlib canvas containing 4 diagnostic subplots for CZ testing."""
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.dark_mode = False
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor("#ffffff")
        self.axs = self.fig.subplots(2, 2)
        self.fig.tight_layout(pad=3.0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(250, 180)
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
        dc_source = plot_data.get("dc_source", "AWG")

        text_color = "#e0e4ed" if self.dark_mode else "#202124"
        legend_face = "#1e222b" if self.dark_mode else "#ffffff"
        legend_edge = "#444b58" if self.dark_mode else "#9aa0a6"
        self.init_plots()

        # 1. Real and Imaginary parts
        ax1 = self.axs[0, 0]
        c_real = "#00e5ff" if self.dark_mode else "#0072b2"
        c_imag = "#ff4081" if self.dark_mode else "#d55e00"
        ax1.semilogx(fs_sch, np.real(fft_unroll), ".-", color=c_real, label="Real (Data)", lw=1.5)
        ax1.semilogx(fs_sch, np.imag(fft_unroll), ".-", color=c_imag, label="Imag (Data)", lw=1.5)
        if Z_model_pure is not None:
            ax1.semilogx(fs_sch, np.real(Z_model_pure), "--", color=c_real, label="Real (Model)", alpha=0.5)
            ax1.semilogx(fs_sch, np.imag(Z_model_pure), "--", color=c_imag, label="Imag (Model)", alpha=0.5)
        ax1.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        # 2. Nyquist Plot
        ax2 = self.axs[0, 1]
        c_nyq = "#69f0ae" if self.dark_mode else "#009e73"
        ax2.plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", color=c_nyq, label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax2.plot(np.real(Z_model_pure), np.imag(Z_model_pure), "--", color=c_nyq, label="Model", alpha=0.5)
        ax2.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)
        ax2.axis("equal")

        # 3. Magnitude Response
        ax3 = self.axs[1, 0]
        c_mag = "#ffd740" if self.dark_mode else "#cc79a7"
        ax3.loglog(fs_sch, np.abs(fft_unroll), ".-", color=c_mag, label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax3.loglog(fs_sch, np.abs(Z_model_pure), "--", color=c_mag, label="Fit Model", lw=1.2, alpha=0.5)
        ax3.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        # 4. Phase Response
        ax4 = self.axs[1, 1]
        c_phase = "#b388ff" if self.dark_mode else "#56b4e9"
        ax4.semilogx(fs_sch, np.angle(fft_unroll) / np.pi, ".-", color=c_phase, label="Data", lw=1.5)
        if Z_model_pure is not None:
            ax4.semilogx(fs_sch, np.angle(Z_model_pure) / np.pi, "--", color=c_phase, label="Fit Model", lw=1.2, alpha=0.5)
        ax4.set_ylabel("Phase (π rad)", color=text_color, fontsize=9)
        ax4.legend(loc="best", fontsize=7, facecolor=legend_face, edgecolor=legend_edge, labelcolor=text_color)

        tag = " [0 V Superconducting Baseline]" if is_sc_eval or bias == 0 else ""
        self.fig.suptitle(
            f"V_bias = {bias:.3f} V ({dc_source}) | τ_shift = {tau_shift*1e6:.2f} µs{tag}",
            color=text_color, fontsize=12, fontweight="bold", y=0.98
        )
        try:
            self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception:
            pass
        try:
            self.draw()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Analysis Background Worker & Plot Canvas
# ---------------------------------------------------------------------------
ANALYZE_PLOT_TYPES = [
    ("01: Full IV Curve & Processed Bias Points", "01_IVcurve.png"),
    ("02: Rtes vs Vbias Curve", "02_Rtes_vs_Vbias.png"),
    ("03: Shifted Raw Current Traces", "03_ShiftedTraces.png"),
    ("04: Measured Admittance Overview", "04_Admittance_Measured.png"),
    ("05: Admittance Fits vs Hanging Model", "05_Admittance_Fits.png"),
    ("06: Admittance Fit Residuals", "06_Fit_Residuals.png"),
    ("07: Alpha & Beta vs R0/Rn", "07_AlphaBeta_vs_R0Rn.png"),
    ("08: Thermal Parameters vs R0/Rn", "08_OtherParams_vs_R0Rn.png"),
    ("09: Alpha vs Beta Power Law Fit", "09_Alpha_vs_Beta_Fit.png"),
]


class AnalyzeWorker(QThread):
    """
    Executes ez_cz.analyze in a background process, streaming logs and progress to the GUI.
    """
    log_signal = pyqtSignal(str, str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, cmd_args, output_dir, parent=None):
        super().__init__(parent)
        self.cmd_args = cmd_args
        self.output_dir = output_dir
        self.process = None
        self._is_aborted = False

    def run(self):
        try:
            self.status_signal.emit("STATUS: RUNNING ANALYSIS...")
            self.log_signal.emit("INFO", f"Starting analysis: {' '.join(self.cmd_args)}")
            self.process = subprocess.Popen(
                self.cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            step_count = 0
            for line in self.process.stdout:
                if self._is_aborted:
                    break
                line_str = line.strip()
                if not line_str:
                    continue
                if "Processing Vbias =" in line_str:
                    step_count += 1
                    self.status_signal.emit(line_str)
                    prog = min(int(step_count * 2.5), 95)
                    self.progress_signal.emit(prog)
                    self.log_signal.emit("INFO", line_str)
                elif "Fit OK:" in line_str:
                    self.log_signal.emit("INFO", line_str)
                elif "Fit Warning:" in line_str or "Warning:" in line_str:
                    self.log_signal.emit("WARN", line_str)
                elif "Error:" in line_str or "Exception:" in line_str:
                    self.log_signal.emit("ERROR", line_str)
                else:
                    self.log_signal.emit("INFO", line_str)

            self.process.wait()
            ret = self.process.returncode
            if self._is_aborted:
                self.finished_signal.emit(False, "Analysis was aborted by user.")
            elif ret == 0:
                self.progress_signal.emit(100)
                self.status_signal.emit("STATUS: COMPLETE")
                self.finished_signal.emit(True, str(self.output_dir))
            else:
                self.finished_signal.emit(False, f"Analysis exited with code {ret}")
        except Exception as e:
            self.finished_signal.emit(False, f"Analysis failed: {e}")

    def abort(self):
        self._is_aborted = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


class AnalyzePlotCanvas(FigureCanvas):
    """
    Interactive Matplotlib canvas for live Complex Impedance analysis and fitting.
    Supports Nyquist, Bode, Admittance, Residuals, and IV curve views.
    """
    def __init__(self, parent=None, width=9, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#151922")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(250, 180)
        self.show_placeholder("No Complex Z data loaded.\nSelect a raw directory and load data to begin analysis.")

    def show_placeholder(self, text: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#151922")
        ax.text(0.5, 0.5, text, color="#90a4ae", fontsize=12,
                ha="center", va="center", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.8", fc="#1b2028", ec="#2a313d"))
        ax.axis("off")
        self.draw()

    def plot_complex_z(
        self,
        f_data: np.ndarray,
        z_data: np.ndarray,
        f_model: Optional[np.ndarray] = None,
        z_model: Optional[np.ndarray] = None,
        f_fit: Optional[np.ndarray] = None,
        z_fit: Optional[np.ndarray] = None,
        plot_mode: str = "nyquist",
        title: str = "Complex Impedance",
        derived_text: str = "",
    ):
        self.fig.clear()
        valid = np.isfinite(z_data) & np.isfinite(f_data) & (f_data > 0)
        f_clean = f_data[valid]
        z_clean = z_data[valid]

        if len(z_clean) == 0:
            self.show_placeholder("No valid impedance data points to display.")
            return

        max_val = np.max(np.abs(z_clean))
        scale = 1e3 if max_val < 0.1 else 1.0
        unit = "mΩ" if scale == 1e3 else "Ω"

        if plot_mode == "nyquist":
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#1b2028")
            ax.grid(True, color="#2a313d", ls=":", lw=0.8)

            ax.plot(np.real(z_clean) * scale, -np.imag(z_clean) * scale, "o",
                    color="#82b1ff", ms=5, alpha=0.85, label="Measured Data")

            if z_model is not None and len(z_model) > 0:
                ax.plot(np.real(z_model) * scale, -np.imag(z_model) * scale, "--",
                        color="#ffb74d", lw=1.8, label="Predicted Model")

            if z_fit is not None and len(z_fit) > 0:
                ax.plot(np.real(z_fit) * scale, -np.imag(z_fit) * scale, "-",
                        color="#00e676", lw=2.2, label="Fitted Model")

            ax.set_xlabel(f"Re(Z) [{unit}]", color="#cfd8dc", fontsize=10)
            ax.set_ylabel(f"-Im(Z) [{unit}]", color="#cfd8dc", fontsize=10)
            ax.tick_params(colors="#cfd8dc")
            ax.set_title(title, color="#eceff1", fontsize=11, fontweight="bold")
            legend = ax.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc", loc="best")
            if legend:
                legend.get_frame().set_alpha(0.85)

            if derived_text:
                ax.text(0.03, 0.04, derived_text, transform=ax.transAxes, color="#b0bec5", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.4", fc="#151922", ec="#37474f", alpha=0.8))

        elif plot_mode == "bode":
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212, sharex=ax1)
            for ax in (ax1, ax2):
                ax.set_facecolor("#1b2028")
                ax.grid(True, color="#2a313d", ls=":", lw=0.8)
                ax.tick_params(colors="#cfd8dc")

            ax1.semilogx(f_clean, np.real(z_clean) * scale, "o", color="#82b1ff", ms=4, label="Data Re(Z)")
            if z_model is not None:
                fm = f_model if f_model is not None else f_clean
                ax1.semilogx(fm, np.real(z_model) * scale, "--", color="#ffb74d", lw=1.8, label="Model Re(Z)")
            if z_fit is not None:
                ff = f_fit if f_fit is not None else f_clean
                ax1.semilogx(ff, np.real(z_fit) * scale, "-", color="#00e676", lw=2.0, label="Fit Re(Z)")
            ax1.set_ylabel(f"Re(Z) [{unit}]", color="#cfd8dc")
            ax1.set_title(title, color="#eceff1", fontsize=11, fontweight="bold")
            ax1.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc", loc="best")

            ax2.semilogx(f_clean, np.imag(z_clean) * scale, "o", color="#82b1ff", ms=4, label="Data Im(Z)")
            if z_model is not None:
                fm = f_model if f_model is not None else f_clean
                ax2.semilogx(fm, np.imag(z_model) * scale, "--", color="#ffb74d", lw=1.8, label="Model Im(Z)")
            if z_fit is not None:
                ff = f_fit if f_fit is not None else f_clean
                ax2.semilogx(ff, np.imag(z_fit) * scale, "-", color="#00e676", lw=2.0, label="Fit Im(Z)")
            ax2.set_xlabel("Frequency [Hz]", color="#cfd8dc")
            ax2.set_ylabel(f"Im(Z) [{unit}]", color="#cfd8dc")
            ax2.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc", loc="best")

        elif plot_mode == "admittance":
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#1b2028")
            ax.grid(True, color="#2a313d", ls=":", lw=0.8)

            y_clean = 1.0 / z_clean
            ax.plot(np.real(y_clean), np.imag(y_clean), "o", color="#82b1ff", ms=5, label="Measured Y")
            if z_model is not None:
                y_mod = 1.0 / z_model
                ax.plot(np.real(y_mod), np.imag(y_mod), "--", color="#ffb74d", lw=1.8, label="Model Y")
            if z_fit is not None:
                y_fit = 1.0 / z_fit
                ax.plot(np.real(y_fit), np.imag(y_fit), "-", color="#00e676", lw=2.0, label="Fit Y")

            ax.set_xlabel("Re(Y) [S]", color="#cfd8dc")
            ax.set_ylabel("Im(Y) [S]", color="#cfd8dc")
            ax.tick_params(colors="#cfd8dc")
            ax.set_title(f"Admittance Y(f) - {title}", color="#eceff1", fontsize=11, fontweight="bold")
            ax.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc")

        elif plot_mode == "residuals":
            ax1 = self.fig.add_subplot(211)
            ax2 = self.fig.add_subplot(212, sharex=ax1)
            for ax in (ax1, ax2):
                ax.set_facecolor("#1b2028")
                ax.grid(True, color="#2a313d", ls=":", lw=0.8)
                ax.tick_params(colors="#cfd8dc")

            comp_z = z_fit if z_fit is not None else z_model
            if comp_z is not None:
                res_re = (np.real(z_clean) - np.real(comp_z[:len(z_clean)])) * scale
                res_im = (np.imag(z_clean) - np.imag(comp_z[:len(z_clean)])) * scale
                ax1.semilogx(f_clean, res_re, "o-", color="#ff80ab", ms=4, label="Re(Data - Model)")
                ax1.axhline(0, color="#78909c", ls="--", lw=1.0)
                ax1.set_ylabel(f"Residual Re [{unit}]", color="#cfd8dc")
                ax1.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc")
                ax1.set_title(f"Fit Residuals - {title}", color="#eceff1", fontsize=11)

                ax2.semilogx(f_clean, res_im, "o-", color="#80d8ff", ms=4, label="Im(Data - Model)")
                ax2.axhline(0, color="#78909c", ls="--", lw=1.0)
                ax2.set_xlabel("Frequency [Hz]", color="#cfd8dc")
                ax2.set_ylabel(f"Residual Im [{unit}]", color="#cfd8dc")
                ax2.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc")

        self.fig.tight_layout()
        self.draw()

    def plot_iv(self, vbias: np.ndarray, ites: np.ndarray, rtes: np.ndarray, selected_bias: Optional[float] = None, rnormal: Optional[float] = None):
        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)
        for ax in (ax1, ax2):
            ax.set_facecolor("#1b2028")
            ax.grid(True, color="#2a313d", ls=":", lw=0.8)
            ax.tick_params(colors="#cfd8dc")

        ax1.plot(vbias, ites * 1e6, "-", color="#82b1ff", lw=1.8, label="IV Curve")
        ax1.set_ylabel("TES Current [µA]", color="#cfd8dc")
        ax1.set_title("TES Current-Voltage Characteristic", color="#eceff1", fontsize=11, fontweight="bold")

        ax2.plot(vbias, rtes * 1e3, "-", color="#80cbc4", lw=1.8, label="R_tes(Vbias)")
        if rnormal:
            ax2.axhline(rnormal * 1e3, color="#ef5350", ls="--", lw=1.2, label=f"Rn = {rnormal*1e3:.2f} mΩ")
        ax2.set_xlabel("Bias Voltage [V]", color="#cfd8dc")
        ax2.set_ylabel("TES Resistance [mΩ]", color="#cfd8dc")

        if selected_bias is not None:
            idx = np.argmin(np.abs(vbias - selected_bias))
            ax1.plot(vbias[idx], ites[idx] * 1e6, "o", color="#ff1744", ms=8, label=f"Selected: {selected_bias:.3f}V")
            ax2.plot(vbias[idx], rtes[idx] * 1e3, "o", color="#ff1744", ms=8, label=f"Selected: {selected_bias:.3f}V")

        ax1.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc")
        ax2.legend(facecolor="#151922", edgecolor="#37474f", labelcolor="#cfd8dc")
        self.fig.tight_layout()
        self.draw()

    def display_image(self, image_path):
        p = Path(image_path)
        if not p.exists():
            self.show_placeholder(f"Plot file not found:\n{p.name}")
            return
        try:
            img = mpimg.imread(str(p))
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor("#151922")
            ax.imshow(img)
            ax.axis("off")
            self.fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            self.draw()
        except Exception as e:
            self.show_placeholder(f"Failed to load image:\n{e}")


# ---------------------------------------------------------------------------
# Main GUI Window
# ---------------------------------------------------------------------------
class CZMainWindow(QMainWindow):
    """Main Application Window for EZ-CZ Testing."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EZ-CZ Dashboard - Complex Impedance")
        self.setMinimumSize(700, 450)
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1200, max(750, avail.width() - 40))
            h = min(720, max(480, avail.height() - 40))
            self.resize(w, h)
        else:
            self.resize(1100, 640)
        self.worker = None
        self.analyze_worker = None
        self.analyze_plot_files = {}

        # Interactive Analyze State
        self.analyze_sweep_data = None
        self.analyze_iv_data = None
        self.active_model = get_model("Dangling / Hanging (2-body)")
        self.current_model_params = self.active_model.get_parameter_defs()
        self.current_fit_result = None
        self.current_selected_bias = None
        self.param_table_widgets = {}
        self.is_updating_params = False

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
            QTabWidget#main_tabs::pane {
                border: 1px solid #2a313d;
                border-radius: 6px;
                background-color: #12141a;
            }
            QTabBar::tab {
                min-width: 90px;
                padding: 8px 18px;
            }
            QTableWidget {
                background-color: #151922;
                border: 1px solid #2a313d;
                gridline-color: #263238;
                color: #eceff1;
                font-size: 9pt;
            }
            QHeaderView::section {
                background-color: #1e2530;
                color: #82b1ff;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #2a313d;
            }
            QTabWidget#main_tabs > QTabBar::tab {
                background-color: #1a202c;
                color: #b0bec5;
                font-size: 11pt;
                font-weight: bold;
                padding: 8px 18px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid #2d3748;
                border-bottom: none;
                min-width: 90px;
            }
            QTabWidget#main_tabs > QTabBar::tab:selected {
                background-color: #151922;
                color: #00e676;
                border-top: 3px solid #00e676;
                border-bottom: 2px solid #151922;
            }
            QTabWidget#main_tabs > QTabBar::tab:hover:!selected {
                background-color: #232c3d;
                color: #ffffff;
            }
        """

        self.init_ui()
        self.add_bias_range_row(0.5, 1.5, 0.01)
        self.load_settings_from_json()
        self.on_timing_source_changed()
        self.on_dc_source_changed()
        self.update_comb_info()
        self.update_bias_preview()
        self.apply_theme()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("main_tabs")

        # Tab 1: Measure
        self.tab_measure = self._create_measure_tab()
        self.main_tabs.addTab(self.tab_measure, "Measure")

        # Tab 2: Analyze
        self.tab_analyze = self._create_analyze_tab()
        self.main_tabs.addTab(self.tab_analyze, "Analyze")

        main_layout.addWidget(self.main_tabs)

    def _create_measure_tab(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left Pane: Config Tabs & Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Tab Widget for Parameter Sections
        tabs = QTabWidget()
        tabs.setObjectName("measure_subtabs")
        tabs.addTab(self._create_bias_tab(), "Bias / DC")
        tabs.addTab(self._create_awg_tab(), "AWG Comb")
        tabs.addTab(self._create_timing_tab(), "Timing")
        tabs.addTab(self._create_sc_health_tab(), "SC Health")
        left_layout.addWidget(tabs, stretch=0)

        # Safety & Status Group
        status_box = QGroupBox("Acquisition Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setSpacing(6)

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
        self.lbl_sc_health.setWordWrap(True)
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

        self.btn_check_limits = QPushButton("Verify Voltage Limits")
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
            button.setMinimumHeight(36)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_layout.addWidget(self.btn_check_limits, 0, 0)
        btn_layout.addWidget(self.btn_test_0v, 0, 1)
        btn_layout.addWidget(self.btn_start, 1, 0)
        btn_layout.addWidget(self.btn_abort, 1, 1)
        btn_layout.addWidget(self.btn_resume, 2, 0, 1, 2)
        left_layout.addWidget(btn_box)

        # Save settings button
        self.btn_save_config = QPushButton("Save Settings as Default (run_settings.json)")
        self.btn_save_config.setStyleSheet("background-color: #37474f; color: #cfd8dc; font-size: 9pt;")
        self.btn_save_config.clicked.connect(self.save_settings_to_json)
        left_layout.addWidget(self.btn_save_config)

        # Console Log
        log_box = QGroupBox("Activity Console Log")
        log_layout = QVBoxLayout(log_box)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        log_layout.addWidget(self.txt_log)
        left_layout.addWidget(log_box, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        left_scroll.setMinimumWidth(320)
        splitter.addWidget(left_scroll)

        # Right Pane: 4-Panel Live Canvas & Toolbar
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.canvas = CZPlotCanvas(self, width=9, height=8)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #e4e8ed; border: 1px solid #c7cbd1; border-radius: 4px;")

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=1)

        right_widget.setMinimumSize(320, 220)
        splitter.addWidget(right_widget)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([450, 750])
        return splitter

    def _create_analyze_tab(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left Pane: Configuration, Parameters & Fitting Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Complex Z Data & IV Operating Point
        data_box = QGroupBox("Complex Z Data & IV Operating Point")
        data_layout = QGridLayout(data_box)
        data_layout.setSpacing(6)

        data_layout.addWidget(QLabel("Raw Complex Z Dir:"), 0, 0)
        self.edit_analyze_raw_dir = QLineEdit("/data/20261030/Complex_Z/20261030_171549")
        self.btn_browse_analyze_raw = QPushButton("Browse")
        self.btn_browse_analyze_raw.setFixedWidth(75)
        self.btn_browse_analyze_raw.clicked.connect(self.on_browse_analyze_raw_dir)
        self.btn_load_analyze_dir = QPushButton("Load Sweep")
        self.btn_load_analyze_dir.setFixedWidth(95)
        self.btn_load_analyze_dir.setStyleSheet("background-color: #0277bd; color: #ffffff; font-weight: bold;")
        self.btn_load_analyze_dir.clicked.connect(self.on_load_analyze_sweep)
        raw_h = QHBoxLayout()
        raw_h.addWidget(self.edit_analyze_raw_dir)
        raw_h.addWidget(self.btn_browse_analyze_raw)
        raw_h.addWidget(self.btn_load_analyze_dir)
        data_layout.addLayout(raw_h, 0, 1, 1, 3)

        data_layout.addWidget(QLabel("IV Curve File (.npz):"), 1, 0)
        self.edit_analyze_iv_file = QLineEdit("/data/20261023/iv/20261023_181330_iv_IV_19.9mK.npz")
        self.btn_browse_analyze_iv = QPushButton("Browse")
        self.btn_browse_analyze_iv.setFixedWidth(75)
        self.btn_browse_analyze_iv.clicked.connect(self.on_browse_analyze_iv_file)
        iv_h = QHBoxLayout()
        iv_h.addWidget(self.edit_analyze_iv_file)
        iv_h.addWidget(self.btn_browse_analyze_iv)
        data_layout.addLayout(iv_h, 1, 1, 1, 3)

        data_layout.addWidget(QLabel("Channel ID:"), 2, 0)
        self.spin_analyze_channel = QSpinBox()
        self.spin_analyze_channel.setRange(0, 127)
        self.spin_analyze_channel.setValue(63)
        self.spin_analyze_channel.valueChanged.connect(self.on_analyze_channel_changed)
        data_layout.addWidget(self.spin_analyze_channel, 2, 1)

        data_layout.addWidget(QLabel("Bias Point:"), 2, 2)
        self.combo_analyze_bias_points = QComboBox()
        self.combo_analyze_bias_points.addItem("Click 'Load Sweep' to populate")
        self.combo_analyze_bias_points.currentIndexChanged.connect(self.on_analyze_bias_changed)
        data_layout.addWidget(self.combo_analyze_bias_points, 2, 3)

        flags_h = QHBoxLayout()
        self.chk_analyze_flip_bias = QCheckBox("Flip bias (ensure Re(Z) > 0 at DC)")
        self.chk_analyze_flip_bias.setChecked(True)
        self.chk_analyze_flip_bias.toggled.connect(self.on_flip_bias_toggled)
        flags_h.addWidget(self.chk_analyze_flip_bias)

        self.chk_analyze_save_in_raw = QCheckBox("Save in <Complex_Z_Dir>/analysis/")
        self.chk_analyze_save_in_raw.setChecked(True)
        flags_h.addWidget(self.chk_analyze_save_in_raw)
        data_layout.addLayout(flags_h, 3, 0, 1, 4)

        left_layout.addWidget(data_box)

        # 2. Thermal Model & Dynamic Parameters
        model_box = QGroupBox("Thermal Model & Dynamic Parameters")
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(6)

        m_top_h = QHBoxLayout()
        m_top_h.addWidget(QLabel("Thermal Model:"))
        self.combo_analyze_model = QComboBox()
        for m_name in AVAILABLE_MODELS.keys():
            self.combo_analyze_model.addItem(m_name)
        self.combo_analyze_model.setCurrentText("Dangling / Hanging (2-body)")
        self.combo_analyze_model.currentIndexChanged.connect(self.on_analyze_model_changed)
        m_top_h.addWidget(self.combo_analyze_model, stretch=1)
        model_layout.addLayout(m_top_h)

        # Dynamic Parameter Table
        self.table_analyze_params = QTableWidget()
        self.table_analyze_params.setColumnCount(6)
        self.table_analyze_params.setHorizontalHeaderLabels(["Parameter", "Value", "Freeze", "Min", "Max", "± Error"])
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(2, QHeaderView.Interactive)
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(4, QHeaderView.Interactive)
        self.table_analyze_params.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table_analyze_params.setColumnWidth(1, 80)
        self.table_analyze_params.setColumnWidth(2, 55)
        self.table_analyze_params.setColumnWidth(3, 75)
        self.table_analyze_params.setColumnWidth(4, 75)
        self.table_analyze_params.verticalHeader().setVisible(False)
        self.table_analyze_params.setMinimumHeight(170)
        self.table_analyze_params.setMaximumHeight(280)
        model_layout.addWidget(self.table_analyze_params)

        # Operating point & derived parameters badges
        self.lbl_analyze_op_info = QLabel("Operating Point: Load sweep and IV to inspect")
        self.lbl_analyze_op_info.setWordWrap(True)
        self.lbl_analyze_op_info.setStyleSheet("color: #82b1ff; font-weight: bold; font-size: 9pt;")
        model_layout.addWidget(self.lbl_analyze_op_info)

        self.lbl_analyze_derived = QLabel("Derived Parameters: --")
        self.lbl_analyze_derived.setWordWrap(True)
        self.lbl_analyze_derived.setStyleSheet("color: #b0bec5; font-size: 9pt;")
        model_layout.addWidget(self.lbl_analyze_derived)

        left_layout.addWidget(model_box)

        # 3. Model Fitting & Save Actions
        exec_box = QGroupBox("Model Fitting & Save Controls")
        exec_layout = QVBoxLayout(exec_box)
        exec_layout.setSpacing(6)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)

        self.btn_fit_model = QPushButton("FIT MODEL (Selected Bias)")
        self.btn_fit_model.setStyleSheet("background-color: #2e7d32; color: #ffffff; font-size: 10pt; font-weight: bold; padding: 8px;")
        self.btn_fit_model.clicked.connect(self.on_fit_model_clicked)

        self.btn_fit_all_biases = QPushButton("FIT ALL BIAS POINTS (Batch)")
        self.btn_fit_all_biases.setStyleSheet("background-color: #1565c0; color: #ffffff; font-size: 10pt; font-weight: bold; padding: 8px;")
        self.btn_fit_all_biases.clicked.connect(self.on_fit_all_biases_clicked)

        self.btn_save_analysis = QPushButton("Save Fit Results & Plots")
        self.btn_save_analysis.setStyleSheet("background-color: #37474f; color: #cfd8dc; font-weight: bold; padding: 6px;")
        self.btn_save_analysis.clicked.connect(self.on_save_analysis_clicked)

        self.btn_open_analyze_dir = QPushButton("Open analysis/ Folder")
        self.btn_open_analyze_dir.setStyleSheet("background-color: #455a64; color: #cfd8dc; font-weight: bold; padding: 6px;")
        self.btn_open_analyze_dir.clicked.connect(self.on_open_analyze_dir)

        for btn in (self.btn_fit_model, self.btn_fit_all_biases, self.btn_save_analysis, self.btn_open_analyze_dir):
            btn.setFixedHeight(36)

        btn_grid.addWidget(self.btn_fit_model, 0, 0)
        btn_grid.addWidget(self.btn_fit_all_biases, 0, 1)
        btn_grid.addWidget(self.btn_save_analysis, 1, 0)
        btn_grid.addWidget(self.btn_open_analyze_dir, 1, 1)
        exec_layout.addLayout(btn_grid)

        status_h = QHBoxLayout()
        self.lbl_analyze_status = QLabel("STATUS: IDLE")
        self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #b0bec5; font-size: 10pt;")
        status_h.addWidget(self.lbl_analyze_status)
        status_h.addStretch()
        exec_layout.addLayout(status_h)

        self.prog_analyze = QProgressBar()
        self.prog_analyze.setRange(0, 100)
        self.prog_analyze.setValue(0)
        exec_layout.addWidget(self.prog_analyze)

        left_layout.addWidget(exec_box)

        # 4. Activity Log
        log_box = QGroupBox("Analysis Activity & Fit Summary")
        log_layout = QVBoxLayout(log_box)
        self.txt_analyze_log = QTextEdit()
        self.txt_analyze_log.setReadOnly(True)
        log_layout.addWidget(self.txt_analyze_log)
        left_layout.addWidget(log_box, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        left_scroll.setMinimumWidth(320)
        splitter.addWidget(left_scroll)

        # Right Pane: Interactive Live Plot Viewer
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        plot_ctl_box = QGroupBox("Interactive Plot Display")
        plot_ctl_layout = QGridLayout(plot_ctl_box)
        plot_ctl_layout.setSpacing(6)

        plot_ctl_layout.addWidget(QLabel("View:"), 0, 0)
        self.combo_analyze_plot_view = QComboBox()
        self.combo_analyze_plot_view.addItems([
            "Nyquist: -Im(Z) vs Re(Z)",
            "Bode: Re(Z) & Im(Z) vs Frequency",
            "Admittance: Im(Y) vs Re(Y)",
            "Residuals: (Data - Model) vs Frequency",
            "IV Curve & Operating Point",
        ])
        self.combo_analyze_plot_view.currentIndexChanged.connect(self.on_analyze_plot_view_changed)
        plot_ctl_layout.addWidget(self.combo_analyze_plot_view, 0, 1, 1, 3)

        self.btn_prev_bias = QPushButton("< Prev Bias")
        self.btn_prev_bias.clicked.connect(self.on_prev_bias_clicked)
        plot_ctl_layout.addWidget(self.btn_prev_bias, 1, 0)

        self.btn_next_bias = QPushButton("Next Bias >")
        self.btn_next_bias.clicked.connect(self.on_next_bias_clicked)
        plot_ctl_layout.addWidget(self.btn_next_bias, 1, 1)

        self.lbl_analyze_view_info = QLabel("No data loaded")
        self.lbl_analyze_view_info.setWordWrap(True)
        self.lbl_analyze_view_info.setStyleSheet("color: #82b1ff; font-weight: bold; padding: 2px;")
        plot_ctl_layout.addWidget(self.lbl_analyze_view_info, 1, 2, 1, 2)

        right_layout.addWidget(plot_ctl_box)

        self.analyze_canvas = AnalyzePlotCanvas(self, width=9, height=8)
        self.analyze_toolbar = NavigationToolbar(self.analyze_canvas, self)
        self.analyze_toolbar.setStyleSheet("background-color: #e4e8ed; border: 1px solid #c7cbd1; border-radius: 4px;")

        right_layout.addWidget(self.analyze_toolbar)
        right_layout.addWidget(self.analyze_canvas, stretch=1)

        right_widget.setMinimumSize(320, 220)
        splitter.addWidget(right_widget)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([450, 750])

        # Initial parameter table construction
        self.build_parameter_table()

        return splitter

    def _create_bias_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        source_box = QGroupBox("DC Bias Source & Channel Selection")
        source_layout = QGridLayout(source_box)
        source_layout.setSpacing(8)

        # Rolldown Menu for DC Source
        source_layout.addWidget(QLabel("DC Bias Source:"), 0, 0)
        self.combo_dc_source = QComboBox()
        self.combo_dc_source.addItems(["DC source: AWG", "DC source: SRS900"])
        self.combo_dc_source.setCurrentIndex(0)
        self.combo_dc_source.currentIndexChanged.connect(self.on_dc_source_changed)
        source_layout.addWidget(self.combo_dc_source, 0, 1)

        source_layout.addWidget(QLabel("Target Channel (ch_global):"), 0, 2)
        self.spin_channel = QSpinBox()
        self.spin_channel.setRange(0, 2048)
        self.spin_channel.setValue(13)
        source_layout.addWidget(self.spin_channel, 0, 3)

        self.chk_flip_bias = QCheckBox("Flip bias? (Re(Z) > 0 at DC)")
        self.chk_flip_bias.setChecked(False)
        self.chk_flip_bias.setStyleSheet("font-weight: bold; color: #0288d1;")
        self.chk_flip_bias.setToolTip("If enabled, the 0V SC transfer function model is generated with opposite sign so the delay fitter does not crash and Re(Z) is positive at DC.")
        source_layout.addWidget(self.chk_flip_bias, 1, 0, 1, 4)

        self.lbl_dc_source_note = QLabel("AWG Direct Mode: DC offset generated on AWG Ch2 (no external DC box required).")
        self.lbl_dc_source_note.setWordWrap(True)
        self.lbl_dc_source_note.setStyleSheet("color: #00897b; font-size: 8.5pt; font-style: italic;")
        source_layout.addWidget(self.lbl_dc_source_note, 2, 0, 1, 4)

        # SRS900 Configuration Sub-Group
        self.grp_srs = QGroupBox("SRS SIM900 / SIM928 Configuration")
        srs_grid = QGridLayout(self.grp_srs)
        srs_grid.setSpacing(6)

        srs_grid.addWidget(QLabel("Mainframe URL:"), 0, 0)
        self.edit_srs_url = QLineEdit("serial:///dev/ttyUSB1?baudrate=9600")
        srs_grid.addWidget(self.edit_srs_url, 0, 1)

        self.btn_test_srs = QPushButton("Test SRS")
        self.btn_test_srs.setStyleSheet("background-color: #0288d1; color: #ffffff;")
        self.btn_test_srs.clicked.connect(self.on_test_srs_clicked)
        srs_grid.addWidget(self.btn_test_srs, 0, 2)

        srs_grid.addWidget(QLabel("SIM928 Port/Slot:"), 1, 0)
        self.spin_srs_port = QSpinBox()
        self.spin_srs_port.setRange(1, 8)
        self.spin_srs_port.setValue(2)
        srs_grid.addWidget(self.spin_srs_port, 1, 1)

        self.lbl_srs_status = QLabel("Status: Not tested")
        self.lbl_srs_status.setStyleSheet("color: #757575; font-style: italic;")
        srs_grid.addWidget(self.lbl_srs_status, 2, 0, 1, 3)

        self.chk_srs_pulse_vnorm = QCheckBox("Pulse Vnorm (0.1 s) before each bias point")
        self.chk_srs_pulse_vnorm.setChecked(True)
        srs_grid.addWidget(self.chk_srs_pulse_vnorm, 3, 0, 1, 3)

        source_layout.addWidget(self.grp_srs, 3, 0, 1, 4)
        layout.addWidget(source_box)

        # Schedule Box
        sched_box = QGroupBox("Bias Voltage Schedule (V)")
        sched_layout = QVBoxLayout(sched_box)
        sched_layout.setSpacing(6)

        # Baseline & Conditioning Row
        cond_grid = QGridLayout()
        cond_grid.setSpacing(6)

        cond_grid.addWidget(QLabel("Baseline Point:"), 0, 0)
        self.spin_base_bias = QDoubleSpinBox()
        self.spin_base_bias.setRange(0.0, 0.0)
        self.spin_base_bias.setValue(0.0)
        self.spin_base_bias.setDecimals(3)
        self.spin_base_bias.setEnabled(False)
        cond_grid.addWidget(self.spin_base_bias, 0, 1)
        cond_grid.addWidget(QLabel("(Fixed 0.0 V for SC calibration)"), 0, 2)

        self.chk_normal_first = QCheckBox("Pre-normalize TES for 1 s before sweep")
        self.chk_normal_first.setToolTip("SC check, then pulse normal for 1 s and settle before sweep")
        self.chk_normal_first.setChecked(False)
        self.chk_normal_first.toggled.connect(self.update_bias_preview)
        cond_grid.addWidget(self.chk_normal_first, 0, 3)

        cond_grid.addWidget(QLabel("Vnorm (V):"), 1, 0)
        self.spin_vnorm = QDoubleSpinBox()
        self.spin_vnorm.setRange(0.0, 10.0)
        self.spin_vnorm.setValue(9.5)
        self.spin_vnorm.setDecimals(3)
        self.spin_vnorm.setSingleStep(0.5)
        self.spin_vnorm.valueChanged.connect(self.update_bias_preview)
        cond_grid.addWidget(self.spin_vnorm, 1, 1)

        cond_grid.addWidget(QLabel("Post-normal recovery (s):"), 1, 2)
        self.spin_post_normal_settle = QDoubleSpinBox()
        self.spin_post_normal_settle.setRange(0.0, 60.0)
        self.spin_post_normal_settle.setValue(5.0)
        self.spin_post_normal_settle.setDecimals(1)
        self.spin_post_normal_settle.setSingleStep(1.0)
        cond_grid.addWidget(self.spin_post_normal_settle, 1, 3)
        sched_layout.addLayout(cond_grid)

        # Dynamic Bias Ranges Table
        sched_layout.addWidget(QLabel("Sweep Voltage Regions (Start, Stop, Step):"))
        self.tbl_bias_ranges = QTableWidget(0, 5)
        self.tbl_bias_ranges.setHorizontalHeaderLabels(["Region", "Start (V)", "Stop (V)", "Step (V)", "Action"])
        self.tbl_bias_ranges.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_bias_ranges.setMinimumHeight(120)
        self.tbl_bias_ranges.setMaximumHeight(180)
        sched_layout.addWidget(self.tbl_bias_ranges)

        btn_row_layout = QHBoxLayout()
        self.btn_add_bias_row = QPushButton("+ Add Bias Range Row")
        self.btn_add_bias_row.setStyleSheet("background-color: #2e7d32; color: #ffffff; font-weight: bold; padding: 5px 12px;")
        self.btn_add_bias_row.clicked.connect(lambda: self.add_bias_range_row())
        btn_row_layout.addWidget(self.btn_add_bias_row)
        btn_row_layout.addStretch()
        sched_layout.addLayout(btn_row_layout)

        layout.addWidget(sched_box)

        # Bias Points Summary Label
        self.lbl_bias_summary = QLabel("Total Points: -- | Max: -- V")
        self.lbl_bias_summary.setStyleSheet("color: #00897b; font-weight: bold;")
        layout.addWidget(self.lbl_bias_summary)

        # Bias points table preview
        self.tbl_bias = QTableWidget(0, 2)
        self.tbl_bias.setHorizontalHeaderLabels(["Index", "Bias Voltage (V)"])
        self.tbl_bias.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_bias.setMaximumHeight(150)
        layout.addWidget(self.tbl_bias)

        layout.addStretch()
        return tab

    def _create_awg_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        grid.setSpacing(8)

        grid.addWidget(QLabel("AWG Address / URL:"), 0, 0)
        self.edit_awg_addr = QLineEdit("USB::2391::22279::INSTR")
        grid.addWidget(self.edit_awg_addr, 0, 1)

        awg_btn_layout = QHBoxLayout()
        self.btn_test_awg = QPushButton("Test AWG")
        self.btn_test_awg.setStyleSheet("background-color: #0288d1; color: #ffffff; font-weight: bold;")
        self.btn_test_awg.clicked.connect(self.on_test_awg_clicked)
        awg_btn_layout.addWidget(self.btn_test_awg)

        self.btn_detect_awg = QPushButton("Auto-Detect USB")
        self.btn_detect_awg.setStyleSheet("background-color: #455a64; color: #ffffff;")
        self.btn_detect_awg.clicked.connect(self.on_detect_awg_clicked)
        awg_btn_layout.addWidget(self.btn_detect_awg)
        grid.addLayout(awg_btn_layout, 0, 2)

        self.lbl_awg_status = QLabel("Instrument Status: Not tested")
        self.lbl_awg_status.setStyleSheet("color: #757575; font-style: italic;")
        grid.addWidget(self.lbl_awg_status, 1, 1, 1, 2)

        grid.addWidget(QLabel("Output Channel:"), 2, 0)
        self.spin_awg_ch = QSpinBox()
        self.spin_awg_ch.setRange(1, 2)
        self.spin_awg_ch.setValue(2)
        grid.addWidget(self.spin_awg_ch, 2, 1)

        grid.addWidget(QLabel("Termination Load:"), 3, 0)
        self.combo_load = QComboBox()
        self.combo_load.addItems(["INFinity", "50"])
        grid.addWidget(self.combo_load, 3, 1)

        grid.addWidget(QLabel("AC Amplitude Vpp (V):"), 4, 0)
        self.spin_awg_vpp = QDoubleSpinBox()
        self.spin_awg_vpp.setRange(0.001, 1.0)
        self.spin_awg_vpp.setValue(0.02)
        self.spin_awg_vpp.setDecimals(4)
        self.spin_awg_vpp.setSingleStep(0.005)
        grid.addWidget(self.spin_awg_vpp, 4, 1)

        grid.addWidget(QLabel("Multiplier per Sine:"), 5, 0)
        self.spin_sine_mult = QDoubleSpinBox()
        self.spin_sine_mult.setRange(0.001, 0.5)
        self.spin_sine_mult.setValue(0.04)
        self.spin_sine_mult.setDecimals(4)
        self.spin_sine_mult.valueChanged.connect(self.update_comb_info)
        grid.addWidget(self.spin_sine_mult, 5, 1)

        grid.addWidget(QLabel("Nominal AWG Rate (MSa/s):"), 6, 0)
        self.spin_awg_rate = QDoubleSpinBox()
        self.spin_awg_rate.setRange(0.1, 50.0)
        self.spin_awg_rate.setValue(5.0)
        grid.addWidget(self.spin_awg_rate, 6, 1)

        grid.addWidget(QLabel("Settle Time (s):"), 7, 0)
        self.spin_settle = QDoubleSpinBox()
        self.spin_settle.setRange(0.05, 5.0)
        self.spin_settle.setValue(0.3)
        grid.addWidget(self.spin_settle, 7, 1)

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
        self.spin_num_tones.setRange(5, 250)
        self.spin_num_tones.setValue(100)
        self.spin_num_tones.valueChanged.connect(self.update_comb_info)
        grid.addWidget(self.spin_num_tones, 5, 1)

        grid.addWidget(QLabel("High-Freq Cutoff (Hz):"), 6, 0)
        self.spin_high_f_cutoff = QDoubleSpinBox()
        self.spin_high_f_cutoff.setRange(100.0, 500000.0)
        self.spin_high_f_cutoff.setValue(50000.0)
        self.spin_high_f_cutoff.setSingleStep(5000.0)
        self.spin_high_f_cutoff.setDecimals(1)
        self.spin_high_f_cutoff.setToolTip("Upper frequency cutoff for comb tones. Internally rejected if >= Nyquist (fs/2).")
        self.spin_high_f_cutoff.valueChanged.connect(self.update_comb_info)
        grid.addWidget(self.spin_high_f_cutoff, 6, 1)

        self.chk_schroeder = QCheckBox("Schroeder Phase Comb (Prevents AWG clipping)")
        self.chk_schroeder.setToolTip("Disperses tone phases quadratically to minimize crest factor and prevent AWG clipping.")
        self.chk_schroeder.setChecked(True)
        self.chk_schroeder.setStyleSheet("color: #0288d1; font-weight: bold;")
        self.chk_schroeder.toggled.connect(self.update_comb_info)
        grid.addWidget(self.chk_schroeder, 7, 0, 1, 2)

        line_layout = QHBoxLayout()
        self.chk_skip_line = QCheckBox("Skip Line Harmonics")
        self.chk_skip_line.setChecked(True)
        self.chk_skip_line.setStyleSheet("color: #7b1fa2; font-weight: bold;")
        self.chk_skip_line.toggled.connect(self.update_comb_info)
        line_layout.addWidget(self.chk_skip_line)

        line_layout.addWidget(QLabel("Line Freq (Hz):"))
        self.spin_line_freq = QDoubleSpinBox()
        self.spin_line_freq.setRange(10.0, 400.0)
        self.spin_line_freq.setValue(60.0)
        self.spin_line_freq.setDecimals(1)
        self.spin_line_freq.valueChanged.connect(self.update_comb_info)
        line_layout.addWidget(self.spin_line_freq)
        grid.addLayout(line_layout, 8, 0, 1, 2)

        self.lbl_comb_info = QLabel("Comb Status: Initializing...")
        self.lbl_comb_info.setWordWrap(True)
        self.lbl_comb_info.setStyleSheet("color: #00897b; font-style: italic;")
        grid.addWidget(self.lbl_comb_info, 9, 0, 1, 2)

        grid.addWidget(QLabel("Averages per Bias Point:"), 10, 0)
        self.spin_averages = QSpinBox()
        self.spin_averages.setRange(1, 32)
        self.spin_averages.setValue(4)
        grid.addWidget(self.spin_averages, 10, 1)

        grid.addWidget(QLabel("Data Save Root Dir:"), 11, 0)
        self.edit_save_dir = QLineEdit("/data")
        grid.addWidget(self.edit_save_dir, 11, 1)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    def _create_sc_health_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        grid.setSpacing(8)

        self.chk_auto_abort = QCheckBox("Auto-abort sweep if 0V transfer function fails")
        self.chk_auto_abort.setChecked(True)
        self.chk_auto_abort.setStyleSheet("font-weight: bold; color: #d32f2f;")
        grid.addWidget(self.chk_auto_abort, 0, 0, 1, 2)

        self.chk_pause_0v = QCheckBox("Pause after 0V calibration for user visual inspection")
        self.chk_pause_0v.setChecked(False)
        self.chk_pause_0v.setStyleSheet("color: #f57f17;")
        grid.addWidget(self.chk_pause_0v, 1, 0, 1, 2)

        self.chk_flip_bias_sc = QCheckBox("Flip bias? (Re(Z) > 0 at DC)")
        self.chk_flip_bias_sc.setChecked(self.chk_flip_bias.isChecked())
        self.chk_flip_bias_sc.setStyleSheet("font-weight: bold; color: #0288d1;")
        self.chk_flip_bias.toggled.connect(self.chk_flip_bias_sc.setChecked)
        self.chk_flip_bias_sc.toggled.connect(self.chk_flip_bias.setChecked)
        grid.addWidget(self.chk_flip_bias_sc, 2, 0, 1, 2)

        grid.addWidget(QLabel("SC acquisition settle time (s):"), 3, 0)
        self.spin_sc_settle = QDoubleSpinBox()
        self.spin_sc_settle.setRange(0.0, 30.0)
        self.spin_sc_settle.setValue(2.0)
        self.spin_sc_settle.setDecimals(1)
        self.spin_sc_settle.setSingleStep(0.5)
        grid.addWidget(self.spin_sc_settle, 3, 1)

        grid.addWidget(QLabel("Minimum Acceptable R² Fit Quality:"), 4, 0)
        self.spin_min_r2 = QDoubleSpinBox()
        self.spin_min_r2.setRange(0.1, 0.999)
        self.spin_min_r2.setValue(0.85)
        self.spin_min_r2.setDecimals(3)
        self.spin_min_r2.setSingleStep(0.05)
        grid.addWidget(self.spin_min_r2, 4, 1)

        grid.addWidget(QLabel("Allowed f_roll Range (Hz):"), 5, 0)
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
        grid.addLayout(f_h, 5, 1)

        self.chk_dry_run = QCheckBox("Simulation / Dry Run Mode (No hardware)")
        self.chk_dry_run.setChecked(not (HAS_HARDWARE_LIBS and HAS_DASTARD_LIB))
        grid.addWidget(self.chk_dry_run, 6, 0, 1, 2)

        self.chk_dark_mode = QCheckBox("Dark mode")
        self.chk_dark_mode.setChecked(False)
        self.chk_dark_mode.toggled.connect(self.apply_theme)
        grid.addWidget(self.chk_dark_mode, 7, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Event Handlers & Helpers
    # -----------------------------------------------------------------------
    def on_dc_source_changed(self):
        is_srs = (self.combo_dc_source.currentIndex() == 1)
        self.grp_srs.setVisible(is_srs)
        if is_srs:
            self.lbl_dc_source_note.setText("SRS900 Mode: DC bias is generated via SRS SIM928 isolated source.")
            self.lbl_dc_source_note.setStyleSheet("color: #1976d2; font-size: 8.5pt; font-style: italic;")
        else:
            self.lbl_dc_source_note.setText("AWG Direct Mode: DC offset is generated directly on AWG Channel 2 (no external DC box required).")
            self.lbl_dc_source_note.setStyleSheet("color: #00897b; font-size: 8.5pt; font-style: italic;")

    def on_timing_source_changed(self):
        auto = self.chk_auto_cfg.isChecked()
        self.edit_cfg_path.setEnabled(auto)
        self.btn_browse_cfg.setEnabled(auto)
        self.spin_manual_tfr.setEnabled(not auto)

    def on_test_awg_clicked(self):
        addr = self.edit_awg_addr.text().strip()
        self.log("INFO", f"Testing AWG connection at '{addr}'...")
        if not HAS_HARDWARE_LIBS:
            msg = "qsdumux hardware libraries are not available in this Python environment."
            self.lbl_awg_status.setText(f"Status: {msg}")
            self.lbl_awg_status.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.log("ERROR", msg)
            QMessageBox.critical(self, "AWG Connection Error", msg)
            return

        try:
            if addr.startswith("vxi11://") or "://" in addr:
                fg = keysight33600A_usb.Keysight33600A_usb(url=addr)
            else:
                fg = keysight33600A_usb.Keysight33600A_usb(device=addr)
            idn = fg._ask("*IDN?").strip()
            try:
                fg.channel.close()
            except Exception:
                pass
            self.lbl_awg_status.setText(f"Connected: {idn}")
            self.lbl_awg_status.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.log("INFO", f"AWG Identified: {idn}")
            QMessageBox.information(self, "AWG Connected", f"Successfully connected to Keysight AWG!\n\n{idn}")
        except Exception as e:
            err = f"Failed to connect to AWG at '{addr}': {e}"
            self.lbl_awg_status.setText(f"Error: {e}")
            self.lbl_awg_status.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.log("ERROR", err)
            QMessageBox.warning(self, "AWG Connection Failed", err)

    def on_detect_awg_clicked(self):
        self.log("INFO", "Auto-detecting USB AWG...")
        try:
            import usbtmc
            devs = usbtmc.list_devices()
            found = None
            for d in devs:
                if d.idVendor in (0x0957, 0x2A8D):
                    found = f"USB::{d.idVendor}::{d.idProduct}::INSTR"
                    break
            if found:
                self.edit_awg_addr.setText(found)
                self.log("INFO", f"Found USB instrument: {found}")
                self.on_test_awg_clicked()
            else:
                msg = "No Keysight/Agilent USB instrument found."
                self.lbl_awg_status.setText(f"Status: {msg}")
                self.lbl_awg_status.setStyleSheet("color: #f57f17; font-weight: bold;")
                self.log("WARN", msg)
                QMessageBox.information(self, "Auto-Detect AWG", msg)
        except Exception as e:
            self.log("ERROR", f"Auto-detect error: {e}")
            QMessageBox.warning(self, "Auto-Detect Error", str(e))

    def on_test_srs_clicked(self):
        url = self.edit_srs_url.text().strip()
        port = self.spin_srs_port.value()
        self.log("INFO", f"Testing SRS SIM900 mainframe at '{url}' (Port {port})...")
        if not HAS_HARDWARE_LIBS:
            msg = "qsdumux hardware libraries are not available in this Python environment."
            self.log("ERROR", msg)
            QMessageBox.critical(self, "SRS Connection Error", msg)
            return

        try:
            mf = sim900.SRS_SIM900(url=url)
            vb = sim928.SRS_SIM928(mf, port)
            v = vb.getvolt()
            try:
                mf.close()
            except Exception:
                pass
            msg = f"Connected to SRS SIM928 on Port {port}. Current output: {v:.4f} V"
            self.lbl_srs_status.setText(f"Status: OK ({v:.3f} V)")
            self.lbl_srs_status.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.log("INFO", msg)
            QMessageBox.information(self, "SRS Connected", msg)
        except Exception as e:
            err = f"Failed to connect to SRS SIM900/928: {e}"
            self.lbl_srs_status.setText(f"Error: {e}")
            self.lbl_srs_status.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.log("ERROR", err)
            QMessageBox.warning(self, "SRS Connection Failed", err)

    def apply_theme(self):
        dark_mode = self.chk_dark_mode.isChecked()
        if dark_mode:
            self.setStyleSheet(self.dark_stylesheet)
            self.toolbar.setStyleSheet("background-color: #1e2430; border: 1px solid #2a313d; border-radius: 4px;")
            if hasattr(self, "analyze_toolbar"):
                self.analyze_toolbar.setStyleSheet("background-color: #1e2430; border: 1px solid #2a313d; border-radius: 4px;")
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
                    padding: 8px 16px;
                    font-weight: bold;
                }
                QTabBar::tab:selected {
                    background-color: #ffffff;
                    color: #245ea8;
                    border-bottom: 2px solid #245ea8;
                }
                QTableWidget {
                    background-color: #ffffff;
                    border: 1px solid #c7cbd1;
                    gridline-color: #e0e4ed;
                    color: #202124;
                    font-size: 9pt;
                }
                QHeaderView::section {
                    background-color: #e4e8ed;
                    color: #245ea8;
                    font-weight: bold;
                    padding: 4px;
                    border: 1px solid #c7cbd1;
                }
                QTabWidget#main_tabs::pane {
                    border: 1px solid #c7cbd1;
                    border-radius: 6px;
                    background-color: #ffffff;
                }
                QTabWidget#main_tabs > QTabBar::tab {
                    background-color: #e4e8ed;
                    color: #37474f;
                    font-size: 11pt;
                    font-weight: bold;
                    padding: 8px 18px;
                    margin-right: 4px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    border: 1px solid #c7cbd1;
                    border-bottom: none;
                    min-width: 90px;
                }
                QTabWidget#main_tabs > QTabBar::tab:selected {
                    background-color: #ffffff;
                    color: #1976d2;
                    border-top: 3px solid #1976d2;
                    border-bottom: 2px solid #ffffff;
                }
            """)
            self.toolbar.setStyleSheet("background-color: #e4e8ed; border: 1px solid #c7cbd1; border-radius: 4px;")
            if hasattr(self, "analyze_toolbar"):
                self.analyze_toolbar.setStyleSheet("background-color: #e4e8ed; border: 1px solid #c7cbd1; border-radius: 4px;")
        self.canvas.set_theme(dark_mode)

    def on_browse_cfg(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select experiment.cfg", "/data", "Config Files (*.cfg *.ini);;All Files (*)")
        if filename:
            self.edit_cfg_path.setText(filename)

    def add_bias_range_row(self, start=None, stop=None, step=None):
        row = self.tbl_bias_ranges.rowCount()
        self.tbl_bias_ranges.insertRow(row)

        lbl = QLabel(f"Region {row + 1}")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: bold; color: #245ea8;")
        self.tbl_bias_ranges.setCellWidget(row, 0, lbl)

        if start is None or stop is None or step is None:
            if row == 0:
                def_start, def_stop, def_step = 0.5, 1.5, 0.01
            else:
                prev_stop_spin = self.tbl_bias_ranges.cellWidget(row - 1, 2)
                prev_stop = prev_stop_spin.value() if prev_stop_spin else 1.5
                def_start = prev_stop
                def_stop = min(10.0, prev_stop + 3.5)
                def_step = 0.1
        else:
            def_start, def_stop, def_step = float(start), float(stop), float(step)

        spin_start = QDoubleSpinBox()
        spin_start.setRange(0.0, 10.0)
        spin_start.setDecimals(3)
        spin_start.setSingleStep(0.1)
        spin_start.setValue(def_start)
        spin_start.valueChanged.connect(self.update_bias_preview)
        self.tbl_bias_ranges.setCellWidget(row, 1, spin_start)

        spin_stop = QDoubleSpinBox()
        spin_stop.setRange(0.0, 10.0)
        spin_stop.setDecimals(3)
        spin_stop.setSingleStep(0.1)
        spin_stop.setValue(def_stop)
        spin_stop.valueChanged.connect(self.update_bias_preview)
        self.tbl_bias_ranges.setCellWidget(row, 2, spin_stop)

        spin_step = QDoubleSpinBox()
        spin_step.setRange(0.001, 5.0)
        spin_step.setDecimals(3)
        spin_step.setSingleStep(0.01)
        spin_step.setValue(def_step)
        spin_step.valueChanged.connect(self.update_bias_preview)
        self.tbl_bias_ranges.setCellWidget(row, 3, spin_step)

        btn_del = QPushButton("✕ Remove")
        btn_del.setStyleSheet("background-color: #c62828; color: #ffffff; padding: 2px 6px; font-size: 8.5pt;")
        btn_del.clicked.connect(lambda _, r=row: self.remove_bias_range_row(r))
        self.tbl_bias_ranges.setCellWidget(row, 4, btn_del)

        self._refresh_bias_range_row_indices()
        self.update_bias_preview()

    def remove_bias_range_row(self, row):
        if self.tbl_bias_ranges.rowCount() <= 1:
            QMessageBox.information(self, "Minimum Rows", "At least one bias range row is required.")
            return
        self.tbl_bias_ranges.removeRow(row)
        self._refresh_bias_range_row_indices()
        self.update_bias_preview()

    def _refresh_bias_range_row_indices(self):
        for r in range(self.tbl_bias_ranges.rowCount()):
            lbl = self.tbl_bias_ranges.cellWidget(r, 0)
            if lbl:
                lbl.setText(f"Region {r + 1}")
            btn = self.tbl_bias_ranges.cellWidget(r, 4)
            if btn:
                try:
                    btn.clicked.disconnect()
                except Exception:
                    pass
                btn.clicked.connect(lambda _, row_idx=r: self.remove_bias_range_row(row_idx))

    def get_configured_bias_ranges(self):
        ranges = []
        for r in range(self.tbl_bias_ranges.rowCount()):
            s_w = self.tbl_bias_ranges.cellWidget(r, 1)
            e_w = self.tbl_bias_ranges.cellWidget(r, 2)
            st_w = self.tbl_bias_ranges.cellWidget(r, 3)
            if s_w and e_w and st_w:
                ranges.append([s_w.value(), e_w.value(), st_w.value()])
        return ranges

    def get_configured_bias_values(self):
        ranges = self.get_configured_bias_ranges()
        segments = [[0.0]]  # 0.0 V SC baseline is always first
        for (start, stop, step) in ranges:
            step = abs(step)
            if step > 0:
                if stop >= start:
                    segments.append(np.arange(start, stop + 1e-9, step))
                else:
                    segments.append(np.arange(start, stop - 1e-9, -step))
        bias_vals = np.concatenate(segments)
        bias_vals = np.round(bias_vals, 4)
        _, first_indices = np.unique(bias_vals, return_index=True)
        return bias_vals[np.sort(first_indices)]

    def update_bias_preview(self):
        try:
            bias_vals = self.get_configured_bias_values()
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

    def update_comb_info(self):
        try:
            num_tones = self.spin_num_tones.value()
            mult = self.spin_sine_mult.value()
            use_sch = self.chk_schroeder.isChecked()
            skip_line = self.chk_skip_line.isChecked()
            line_f = self.spin_line_freq.value()
            cutoff = self.spin_high_f_cutoff.value()
            line_str = f" ({line_f:.0f} Hz line harmonics skipped)" if skip_line else ""

            if use_sch:
                approx_peak = 1.6 * np.sqrt(num_tones) * mult
                if approx_peak > 1.0:
                    status_text = (
                        f"Comb: {num_tones} tones (cutoff {cutoff*1e-3:.1f} kHz < Nyquist){line_str}\n"
                        f"Peak ~{approx_peak:.2f} V (Schroeder active; auto-scaled to ±1.0 V limit, NO clipping)"
                    )
                    self.lbl_comb_info.setStyleSheet("color: #f57f17; font-weight: bold;")
                else:
                    status_text = (
                        f"Comb: {num_tones} tones (cutoff {cutoff*1e-3:.1f} kHz < Nyquist){line_str}\n"
                        f"Peak ~{approx_peak:.2f} V (Schroeder active, below ±1.0 V limit, NO clipping)"
                    )
                    self.lbl_comb_info.setStyleSheet("color: #2e7d32; font-weight: bold;")
            else:
                peak = (num_tones + 1) * mult
                if peak > 1.0:
                    status_text = (
                        f"Comb: {num_tones} tones (cutoff {cutoff*1e-3:.1f} kHz < Nyquist){line_str}\n"
                        f"In-phase peak = {peak:.2f} V (Exceeds AWG ±1.0 V! Auto-scaled, reduced SNR)"
                    )
                    self.lbl_comb_info.setStyleSheet("color: #d32f2f; font-weight: bold;")
                else:
                    status_text = (
                        f"Comb: {num_tones} tones (cutoff {cutoff*1e-3:.1f} kHz < Nyquist){line_str}\n"
                        f"In-phase peak = {peak:.2f} V"
                    )
                    self.lbl_comb_info.setStyleSheet("color: #00897b;")
            self.lbl_comb_info.setText(status_text)
        except Exception as e:
            self.lbl_comb_info.setText(f"Comb info: {e}")

    def get_time_per_flux_ramp(self):
        if self.chk_auto_cfg.isChecked():
            cfg_file = self.edit_cfg_path.text().strip()
            if os.path.exists(cfg_file):
                tfr, _ = get_time_per_flux_ramp_from_cfg(cfg_file)
                return tfr
            else:
                self.log("WARN", f"Config file '{cfg_file}' not found. Falling back to manual time_per_flux_ramp ({self.spin_manual_tfr.value():.4f} µs).")
                return self.spin_manual_tfr.value() * 1e-6
        else:
            return self.spin_manual_tfr.value() * 1e-6

    def assemble_parameters(self, single_0v_only=False):
        tfr = self.get_time_per_flux_ramp()
        bias_vals = np.array([0.0]) if single_0v_only else self.get_configured_bias_values()
        dc_source = "AWG" if self.combo_dc_source.currentIndex() == 0 else "SRS900"

        return {
            "dc_source": dc_source,
            "srs_url": self.edit_srs_url.text().strip(),
            "srs_port": self.spin_srs_port.value(),
            "srs_pulse_vnorm": self.chk_srs_pulse_vnorm.isChecked(),
            "time_per_flux_ramp": tfr,
            "target_duration_s": self.spin_target_dur.value(),
            "target_f0_hz": self.spin_target_f0.value(),
            "nominal_awg_rate": self.spin_awg_rate.value() * 1e6,
            "num_high_freq_tones": self.spin_num_tones.value(),
            "high_f_cutoff": self.spin_high_f_cutoff.value(),
            "use_schroeder_phase": self.chk_schroeder.isChecked(),
            "skip_line_harmonics": self.chk_skip_line.isChecked(),
            "line_freq": self.spin_line_freq.value(),
            "bias_ranges": self.get_configured_bias_ranges(),
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
            "flip_bias": self.chk_flip_bias.isChecked(),
            "single_0v_test_only": single_0v_only,
            "dry_run_mode": self.chk_dry_run.isChecked(),
        }

    def load_settings_from_json(self, filepath=None):
        """Loads default values from run_settings.json if present."""
        path = Path(filepath or SETTINGS_FILE_PATH)
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            m = data.get("measure", {})
            if "dc_source" in m:
                idx = 0 if m["dc_source"].upper() == "AWG" else 1
                self.combo_dc_source.setCurrentIndex(idx)
            if "channel" in m:
                self.spin_channel.setValue(int(m["channel"]))
            if "v_norm" in m:
                self.spin_vnorm.setValue(float(m["v_norm"]))
            if "vpp" in m:
                self.spin_awg_vpp.setValue(float(m["vpp"]))
            if "amp_mult" in m:
                self.spin_sine_mult.setValue(float(m["amp_mult"]))
            if "awg_rate" in m:
                self.spin_awg_rate.setValue(float(m["awg_rate"]) / 1e6)
            if "f0" in m:
                self.spin_target_f0.setValue(float(m["f0"]))
            if "num_high_freq_tones" in m:
                self.spin_num_tones.setValue(int(m["num_high_freq_tones"]))
            if "high_f_cutoff" in m:
                self.spin_high_f_cutoff.setValue(float(m["high_f_cutoff"]))
            if "use_schroeder" in m:
                self.chk_schroeder.setChecked(bool(m["use_schroeder"]))
            if "skip_line_harmonics" in m:
                self.chk_skip_line.setChecked(bool(m["skip_line_harmonics"]))
            if "line_freq" in m:
                self.spin_line_freq.setValue(float(m["line_freq"]))
            if "flip_bias" in m:
                self.chk_flip_bias.setChecked(bool(m["flip_bias"]))
            if "srs_url" in m:
                self.edit_srs_url.setText(str(m["srs_url"]))
            if "srs_port" in m:
                self.spin_srs_port.setValue(int(m["srs_port"]))
            if "keysight_url" in m:
                self.edit_awg_addr.setText(str(m["keysight_url"]))

            if "bias_ranges" in m and isinstance(m["bias_ranges"], list) and len(m["bias_ranges"]) > 0:
                self.tbl_bias_ranges.setRowCount(0)
                for rng in m["bias_ranges"]:
                    if len(rng) == 3:
                        self.add_bias_range_row(rng[0], rng[1], rng[2])
            elif "bias_dense" in m and len(m["bias_dense"]) == 3:
                self.tbl_bias_ranges.setRowCount(0)
                self.add_bias_range_row(m["bias_dense"][0], m["bias_dense"][1], m["bias_dense"][2])
                if "bias_medium" in m and len(m["bias_medium"]) == 3:
                    self.add_bias_range_row(m["bias_medium"][0], m["bias_medium"][1], m["bias_medium"][2])

            if "output_dir" in m and m["output_dir"]:
                self.edit_save_dir.setText(str(m["output_dir"]))

            # Load analyze settings
            a = data.get("analyze", {})
            if "raw_dir" in a and a["raw_dir"] and hasattr(self, "edit_analyze_raw_dir"):
                self.edit_analyze_raw_dir.setText(str(a["raw_dir"]))
            if "iv_file" in a and a["iv_file"] and hasattr(self, "edit_analyze_iv_file"):
                self.edit_analyze_iv_file.setText(str(a["iv_file"]))
            if "channel_id" in a and hasattr(self, "spin_analyze_channel"):
                self.spin_analyze_channel.setValue(int(a["channel_id"]))
            if "flip_bias" in a and hasattr(self, "chk_analyze_flip_bias"):
                self.chk_analyze_flip_bias.setChecked(bool(a["flip_bias"]))
            if "save_in_raw" in a and hasattr(self, "chk_analyze_save_in_raw"):
                self.chk_analyze_save_in_raw.setChecked(bool(a["save_in_raw"]))
            if "model" in a and hasattr(self, "combo_analyze_model"):
                self.combo_analyze_model.setCurrentText(str(a["model"]))
            self.log("INFO", f"Loaded parameters from {path}")
            if hasattr(self, "edit_analyze_raw_dir") and Path(self.edit_analyze_raw_dir.text().strip()).exists():
                self.on_load_analyze_sweep()
        except Exception as e:
            self.log("WARN", f"Could not load {path}: {e}")

    def save_settings_to_json(self, filepath=None):
        """Saves current GUI configuration back to run_settings.json."""
        path = Path(filepath or SETTINGS_FILE_PATH)
        existing = {}
        if path.exists():
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except Exception:
                pass

        measure_cfg = existing.get("measure", {})
        measure_cfg["dc_source"] = "awg" if self.combo_dc_source.currentIndex() == 0 else "srs900"
        measure_cfg["channel"] = self.spin_channel.value()
        measure_cfg["v_norm"] = self.spin_vnorm.value()
        measure_cfg["vpp"] = self.spin_awg_vpp.value()
        measure_cfg["amp_mult"] = self.spin_sine_mult.value()
        measure_cfg["awg_rate"] = self.spin_awg_rate.value() * 1e6
        measure_cfg["f0"] = self.spin_target_f0.value()
        measure_cfg["num_high_freq_tones"] = self.spin_num_tones.value()
        measure_cfg["high_f_cutoff"] = self.spin_high_f_cutoff.value()
        measure_cfg["use_schroeder"] = self.chk_schroeder.isChecked()
        measure_cfg["skip_line_harmonics"] = self.chk_skip_line.isChecked()
        measure_cfg["line_freq"] = self.spin_line_freq.value()
        measure_cfg["flip_bias"] = self.chk_flip_bias.isChecked()
        measure_cfg["srs_url"] = self.edit_srs_url.text().strip()
        measure_cfg["srs_port"] = self.spin_srs_port.value()
        measure_cfg["keysight_url"] = self.edit_awg_addr.text().strip()

        ranges = self.get_configured_bias_ranges()
        measure_cfg["bias_ranges"] = ranges
        if len(ranges) >= 1:
            measure_cfg["bias_dense"] = ranges[0]
        if len(ranges) >= 2:
            measure_cfg["bias_medium"] = ranges[1]

        measure_cfg["output_dir"] = self.edit_save_dir.text().strip()
        existing["measure"] = measure_cfg

        analyze_cfg = existing.get("analyze", {})
        if hasattr(self, "edit_analyze_raw_dir"):
            analyze_cfg["raw_dir"] = self.edit_analyze_raw_dir.text().strip()
        if hasattr(self, "edit_analyze_iv_file"):
            analyze_cfg["iv_file"] = self.edit_analyze_iv_file.text().strip()
        if hasattr(self, "spin_analyze_channel"):
            analyze_cfg["channel_id"] = self.spin_analyze_channel.value()
        if hasattr(self, "chk_analyze_flip_bias"):
            analyze_cfg["flip_bias"] = self.chk_analyze_flip_bias.isChecked()
        if hasattr(self, "combo_analyze_model"):
            analyze_cfg["model"] = self.combo_analyze_model.currentText()
        if hasattr(self, "chk_analyze_save_in_raw"):
            analyze_cfg["save_in_raw"] = self.chk_analyze_save_in_raw.isChecked()
        existing["analyze"] = analyze_cfg

        try:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            self.log("INFO", f"Configuration successfully saved to {path}")
            QMessageBox.information(self, "Settings Saved", f"Successfully saved defaults to {path}")
        except Exception as e:
            self.log("ERROR", f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Save Error", str(e))

    # -----------------------------------------------------------------------
    # Analyze Tab Handlers (Interactive MDT Model Fitting & Sweeps)
    # -----------------------------------------------------------------------
    def on_browse_analyze_raw_dir(self):
        curr = self.edit_analyze_raw_dir.text().strip() or "/data"
        d = QFileDialog.getExistingDirectory(self, "Select Raw Complex Z Data Directory", curr)
        if d:
            self.edit_analyze_raw_dir.setText(d)
            self.on_load_analyze_sweep()

    def on_browse_analyze_iv_file(self):
        curr = self.edit_analyze_iv_file.text().strip() or "/data"
        f, _ = QFileDialog.getOpenFileName(self, "Select IV Curve File", curr, "NPZ Files (*.npz);;All Files (*)")
        if f:
            self.edit_analyze_iv_file.setText(f)
            self.on_load_analyze_iv()

    def on_analyze_channel_changed(self):
        if hasattr(self, "spin_analyze_pixel"):
            self.spin_analyze_pixel.setValue(self.spin_analyze_channel.value())
        if hasattr(self, "edit_analyze_raw_dir") and Path(self.edit_analyze_raw_dir.text().strip()).exists():
            self.on_load_analyze_sweep()

    def on_flip_bias_toggled(self):
        self.on_load_analyze_sweep()

    def on_load_analyze_iv(self):
        iv_path = Path(self.edit_analyze_iv_file.text().strip())
        if not iv_path.exists():
            return
        try:
            ch = self.spin_analyze_channel.value()
            self.analyze_iv_data = load_iv_curve(iv_path, channel_id=ch)
            self.log_analyze("INFO", f"Loaded IV curve: Ch {ch}, Rn = {self.analyze_iv_data['rnormal']*1e3:.2f} mΩ from {iv_path.name}")
            if self.combo_analyze_plot_view.currentText().startswith("IV"):
                self.on_analyze_plot_view_changed()
            else:
                self.on_analyze_bias_changed()
        except Exception as e:
            self.log_analyze("ERROR", f"Failed loading IV curve: {e}")

    def on_load_analyze_sweep(self):
        raw_dir = Path(self.edit_analyze_raw_dir.text().strip())
        if not raw_dir.exists():
            self.log_analyze("ERROR", f"Complex Z directory does not exist: {raw_dir}")
            return

        ch = self.spin_analyze_channel.value()
        flip = self.chk_analyze_flip_bias.isChecked()

        try:
            self.lbl_analyze_status.setText("STATUS: LOADING DATA...")
            self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #82b1ff; font-size: 10pt;")
            QApplication.processEvents()

            # Ensure IV curve is loaded first
            iv_path = Path(self.edit_analyze_iv_file.text().strip())
            if iv_path.exists() and (self.analyze_iv_data is None or self.analyze_iv_data.get("channel_id") != ch):
                self.analyze_iv_data = load_iv_curve(iv_path, channel_id=ch)

            self.analyze_sweep_data = load_cz_sweep_dataset(raw_dir, channel_id=ch, flip_bias=flip)
            n_biases = self.analyze_sweep_data["num_biases"]
            self.log_analyze("INFO", f"Loaded {n_biases} bias points from {raw_dir.name} (Ch {ch}, tones={len(self.analyze_sweep_data['fs_sch'])})")

            # Update bias points dropdown
            self.combo_analyze_bias_points.blockSignals(True)
            self.combo_analyze_bias_points.clear()

            lookup = self.analyze_iv_data["lookup_op"] if self.analyze_iv_data else None

            preferred_idx = 0
            best_diff = 1e9

            for i, vb in enumerate(self.analyze_sweep_data["bias_voltages"]):
                if lookup:
                    op = lookup(vb)
                    label = f"{vb:.3f} V  ({op['rn_pct']:.1f}% Rn)"
                    if abs(op["rn_pct"] - 35.0) < best_diff:
                        best_diff = abs(op["rn_pct"] - 35.0)
                        preferred_idx = i
                else:
                    label = f"{vb:.3f} V"
                self.combo_analyze_bias_points.addItem(label, userData=vb)

            self.combo_analyze_bias_points.addItem("< All Bias Points (Batch) >", userData=None)
            self.combo_analyze_bias_points.setCurrentIndex(preferred_idx)
            self.combo_analyze_bias_points.blockSignals(False)

            self.lbl_analyze_status.setText(f"STATUS: LOADED ({n_biases} BIASES)")
            self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 10pt;")

            self.on_analyze_bias_changed()

        except Exception as e:
            self.lbl_analyze_status.setText("STATUS: LOAD FAILED")
            self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 10pt;")
            self.log_analyze("ERROR", f"Failed loading sweep: {e}")

    def on_analyze_bias_changed(self):
        if not self.analyze_sweep_data or self.combo_analyze_bias_points.count() == 0:
            return

        selected_vb = self.combo_analyze_bias_points.currentData()
        if selected_vb is None:
            self.lbl_analyze_view_info.setText("Batch mode (< All Bias Points >)")
            return

        self.current_selected_bias = selected_vb
        b_info = self.analyze_sweep_data["bias_points"].get(selected_vb)
        if not b_info:
            return

        ch = self.spin_analyze_channel.value()
        self.lbl_analyze_view_info.setText(f"Ch {ch} | Bias: {selected_vb:.3f} V")

        # Update operating point from IV curve if available
        if self.analyze_iv_data:
            op = self.analyze_iv_data["lookup_op"](selected_vb)
            self.lbl_analyze_op_info.setText(
                f"Operating Point: Vbias = {selected_vb:.3f}V ({op['rn_pct']:.1f}% Rn) | "
                f"R0 = {op['rtes']*1e3:.3f} mΩ | I0 = {op['ites']*1e6:.1f} µA | P0 = {op['p0']*1e12:.2f} pW"
            )
            # Update R0 in parameter table if frozen
            if "R_0" in self.current_model_params and not self.current_model_params["R_0"].vary:
                self.current_model_params["R_0"].value = op["rtes"] * 1e3
                self.sync_param_to_table("R_0", op["rtes"] * 1e3)
        else:
            self.lbl_analyze_op_info.setText(f"Operating Point: Vbias = {selected_vb:.3f}V (No IV file loaded)")

        self.current_fit_result = None
        self.update_live_model_prediction()

    def on_prev_bias_clicked(self):
        idx = self.combo_analyze_bias_points.currentIndex()
        if idx > 0:
            self.combo_analyze_bias_points.setCurrentIndex(idx - 1)

    def on_next_bias_clicked(self):
        idx = self.combo_analyze_bias_points.currentIndex()
        if idx < self.combo_analyze_bias_points.count() - 2:
            self.combo_analyze_bias_points.setCurrentIndex(idx + 1)

    def on_analyze_model_changed(self):
        m_name = self.combo_analyze_model.currentText()
        self.active_model = get_model(m_name)
        old_params = self.current_model_params
        new_params = self.active_model.get_parameter_defs()
        for k, p in new_params.items():
            if k in old_params:
                p.value = old_params[k].value
                p.vary = old_params[k].vary
                p.min_val = old_params[k].min_val
                p.max_val = old_params[k].max_val

        self.current_model_params = new_params
        self.current_fit_result = None
        self.build_parameter_table()
        self.update_live_model_prediction()

    def build_parameter_table(self):
        """Constructs dynamic table rows for the active thermal model."""
        self.is_updating_params = True
        self.table_analyze_params.setRowCount(0)
        self.param_table_widgets = {}

        params = self.current_model_params
        self.table_analyze_params.setRowCount(len(params))

        for r, (k, p) in enumerate(params.items()):
            item_name = QTableWidgetItem(p.label)
            item_name.setToolTip(p.description)
            item_name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table_analyze_params.setItem(r, 0, item_name)

            spin_val = QDoubleSpinBox()
            spin_val.setDecimals(4 if p.value < 10 else 2)
            spin_val.setRange(p.min_val, p.max_val)
            spin_val.setSingleStep(max(p.value * 0.05, 0.001))
            spin_val.setValue(p.value)
            spin_val.setMaximumWidth(85)
            spin_val.valueChanged.connect(lambda v, name=k: self.on_table_param_value_changed(name, v))
            self.table_analyze_params.setCellWidget(r, 1, spin_val)

            chk_freeze = QCheckBox("Freeze")
            chk_freeze.setChecked(not p.vary)
            chk_freeze.toggled.connect(lambda chk, name=k: self.on_table_param_freeze_changed(name, chk))
            self.table_analyze_params.setCellWidget(r, 2, chk_freeze)

            spin_min = QDoubleSpinBox()
            spin_min.setDecimals(4 if p.min_val < 10 else 2)
            spin_min.setRange(0.0, 1e7)
            spin_min.setValue(p.min_val)
            spin_min.setMaximumWidth(75)
            spin_min.valueChanged.connect(lambda v, name=k: self.on_table_param_limit_changed(name, min_v=v))
            self.table_analyze_params.setCellWidget(r, 3, spin_min)

            spin_max = QDoubleSpinBox()
            spin_max.setDecimals(4 if p.max_val < 10 else 2)
            spin_max.setRange(0.0, 1e7)
            spin_max.setValue(p.max_val)
            spin_max.setMaximumWidth(75)
            spin_max.valueChanged.connect(lambda v, name=k: self.on_table_param_limit_changed(name, max_v=v))
            self.table_analyze_params.setCellWidget(r, 4, spin_max)

            lbl_err = QLabel("± --")
            lbl_err.setStyleSheet("color: #90a4ae; font-size: 8pt; padding-left: 4px;")
            self.table_analyze_params.setCellWidget(r, 5, lbl_err)

            self.param_table_widgets[k] = {
                "spin_val": spin_val,
                "chk_freeze": chk_freeze,
                "spin_min": spin_min,
                "spin_max": spin_max,
                "lbl_err": lbl_err,
            }

        self.is_updating_params = False

    def sync_param_to_table(self, name: str, value: float):
        if name in self.param_table_widgets:
            spin = self.param_table_widgets[name]["spin_val"]
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def on_table_param_value_changed(self, name: str, val: float):
        if self.is_updating_params:
            return
        if name in self.current_model_params:
            self.current_model_params[name].value = val
        self.update_live_model_prediction()

    def on_table_param_freeze_changed(self, name: str, is_frozen: bool):
        if name in self.current_model_params:
            self.current_model_params[name].vary = not is_frozen

    def on_table_param_limit_changed(self, name: str, min_v: Optional[float] = None, max_v: Optional[float] = None):
        if name in self.current_model_params:
            if min_v is not None:
                self.current_model_params[name].min_val = min_v
            if max_v is not None:
                self.current_model_params[name].max_val = max_v

    def update_live_model_prediction(self):
        """Instantaneously re-evaluates the theoretical model and updates plot."""
        if not self.analyze_sweep_data or self.current_selected_bias is None:
            return

        b_info = self.analyze_sweep_data["bias_points"].get(self.current_selected_bias)
        if not b_info:
            return

        f_data = b_info["f"]
        z_data = b_info["z_tes"]

        f_dense = np.geomspace(max(f_data[0] * 0.5, 1.0), max(f_data[-1] * 2.0, 1e5), 300)
        p_vals = {k: p.value for k, p in self.current_model_params.items()}

        try:
            z_pred_dense = self.active_model.calc_z_tes(f_dense, p_vals)
            derived = self.active_model.calc_derived(p_vals)

            d_parts = []
            if "P0_pW" in derived:
                d_parts.append(f"P0={derived['P0_pW']:.2f} pW")
            if "L_I" in derived:
                d_parts.append(f"LI={derived['L_I']:.2f}")
            if "tau_ms" in derived:
                d_parts.append(f"τ={derived['tau_ms']:.2f} ms")
            if "tau_I_ms" in derived and np.isfinite(derived["tau_I_ms"]):
                d_parts.append(f"τ_I={derived['tau_I_ms']:.2f} ms")
            if "tau_abs_ms" in derived:
                d_parts.append(f"τ_abs={derived['tau_abs_ms']:.2f} ms")
            self.lbl_analyze_derived.setText("Derived: " + " | ".join(d_parts) if d_parts else "Derived: --")

            self.draw_analyze_plot(f_data, z_data, f_dense, z_pred_dense)
        except Exception:
            pass

    def draw_analyze_plot(self, f_data, z_data, f_model=None, z_model=None):
        mode_text = self.combo_analyze_plot_view.currentText()
        vb = self.current_selected_bias or 0.0

        if mode_text.startswith("Nyquist"):
            z_fit = self.current_fit_result.z_dense if self.current_fit_result else None
            f_fit = self.current_fit_result.f_dense if self.current_fit_result else None
            self.analyze_canvas.plot_complex_z(
                f_data, z_data, f_model=f_model, z_model=z_model,
                f_fit=f_fit, z_fit=z_fit,
                plot_mode="nyquist",
                title=f"Nyquist Plot: Vbias = {vb:.3f} V ({self.active_model.name})",
            )
        elif mode_text.startswith("Bode"):
            z_fit = self.current_fit_result.z_dense if self.current_fit_result else None
            f_fit = self.current_fit_result.f_dense if self.current_fit_result else None
            self.analyze_canvas.plot_complex_z(
                f_data, z_data, f_model=f_model, z_model=z_model,
                f_fit=f_fit, z_fit=z_fit,
                plot_mode="bode",
                title=f"Frequency Response: Vbias = {vb:.3f} V",
            )
        elif mode_text.startswith("Admittance"):
            z_fit = self.current_fit_result.z_dense if self.current_fit_result else None
            f_fit = self.current_fit_result.f_dense if self.current_fit_result else None
            self.analyze_canvas.plot_complex_z(
                f_data, z_data, f_model=f_model, z_model=z_model,
                f_fit=f_fit, z_fit=z_fit,
                plot_mode="admittance",
                title=f"Admittance Y(f): Vbias = {vb:.3f} V",
            )
        elif mode_text.startswith("Residuals"):
            z_fit = self.current_fit_result.z_dense if self.current_fit_result else z_model
            self.analyze_canvas.plot_complex_z(
                f_data, z_data, f_model=f_model, z_model=z_model,
                f_fit=f_model, z_fit=z_fit,
                plot_mode="residuals",
                title=f"Fit Residuals: Vbias = {vb:.3f} V",
            )
        elif mode_text.startswith("IV"):
            if self.analyze_iv_data:
                self.analyze_canvas.plot_iv(
                    self.analyze_iv_data["vbias"],
                    self.analyze_iv_data["ites"],
                    self.analyze_iv_data["rtes"],
                    selected_bias=vb,
                    rnormal=self.analyze_iv_data.get("rnormal"),
                )
            else:
                self.analyze_canvas.show_placeholder("No IV curve loaded. Browse to an IV file to view.")

    def on_analyze_plot_view_changed(self):
        if self.analyze_sweep_data and self.current_selected_bias is not None:
            self.update_live_model_prediction()
        elif self.combo_analyze_plot_view.currentText().startswith("IV"):
            if self.analyze_iv_data:
                self.analyze_canvas.plot_iv(
                    self.analyze_iv_data["vbias"],
                    self.analyze_iv_data["ites"],
                    self.analyze_iv_data["rtes"],
                    selected_bias=self.current_selected_bias,
                    rnormal=self.analyze_iv_data.get("rnormal"),
                )

    def on_fit_model_clicked(self):
        """Runs fast, stable bounded least-squares fit on selected bias point."""
        if not self.analyze_sweep_data or self.current_selected_bias is None:
            QMessageBox.warning(self, "No Data", "Please load a Complex Z sweep first.")
            return

        b_info = self.analyze_sweep_data["bias_points"].get(self.current_selected_bias)
        if not b_info:
            return

        f_data = b_info["f"]
        z_data = b_info["z_tes"]

        t0 = time.time()
        self.lbl_analyze_status.setText("STATUS: FITTING...")
        self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #ffb74d; font-size: 10pt;")
        QApplication.processEvents()

        res = self.active_model.fit(f_data, z_data, self.current_model_params)
        elapsed_ms = (time.time() - t0) * 1000.0

        if res.success:
            self.current_fit_result = res
            self.lbl_analyze_status.setText(f"STATUS: FIT CONVERGED ({elapsed_ms:.1f} ms, χ²={res.chi2:.2e})")
            self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 10pt;")

            self.is_updating_params = True
            for k, p in res.params.items():
                if k in self.param_table_widgets:
                    spin = self.param_table_widgets[k]["spin_val"]
                    spin.blockSignals(True)
                    spin.setValue(p.value)
                    spin.blockSignals(False)

                    err = res.errors.get(k, 0.0)
                    lbl_e = self.param_table_widgets[k]["lbl_err"]
                    if p.vary and err > 0:
                        lbl_e.setText(f"± {err:.3f}")
                    else:
                        lbl_e.setText("± --")
            self.is_updating_params = False

            vb = self.current_selected_bias
            self.log_analyze("INFO", f"=== Fit OK: Vbias={vb:.3f}V ({self.active_model.name}) ===")
            for k, p in res.params.items():
                err = res.errors.get(k, 0.0)
                status = f"± {err:.3f}" if p.vary else "[FROZEN]"
                self.log_analyze("INFO", f"  {k}: {p.value:.4f} {p.unit} {status}")
            self.log_analyze("INFO", f"  Convergence: {res.nfev} evals in {elapsed_ms:.1f} ms | redchi={res.redchi:.3e}")

            self.update_live_model_prediction()
        else:
            self.lbl_analyze_status.setText("STATUS: FIT FAILED")
            self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 10pt;")
            self.log_analyze("ERROR", f"Fit failed: {res.message}")

    def on_fit_all_biases_clicked(self):
        """Batch fits all bias points across the transition and exports results."""
        if not self.analyze_sweep_data:
            QMessageBox.warning(self, "No Data", "Please load a Complex Z sweep first.")
            return

        voltages = self.analyze_sweep_data["bias_voltages"]
        if not voltages:
            return

        raw_dir = Path(self.edit_analyze_raw_dir.text().strip())
        out_dir = raw_dir / "analysis" if self.chk_analyze_save_in_raw.isChecked() else Path("/data/Results/Complex_Z")
        out_dir.mkdir(parents=True, exist_ok=True)

        ch = self.spin_analyze_channel.value()
        self.btn_fit_all_biases.setEnabled(False)
        self.btn_fit_model.setEnabled(False)
        self.prog_analyze.setValue(0)

        self.log_analyze("INFO", f"Starting batch fit across {len(voltages)} bias points using {self.active_model.name}...")
        results_rows = []

        try:
            import pandas as pd
            for idx, vb in enumerate(voltages):
                b_info = self.analyze_sweep_data["bias_points"].get(vb)
                if not b_info:
                    continue

                f_d = b_info["f"]
                z_d = b_info["z_tes"]

                if self.analyze_iv_data:
                    op = self.analyze_iv_data["lookup_op"](vb)
                    if "R_0" in self.current_model_params:
                        self.current_model_params["R_0"].value = op["rtes"] * 1e3

                res = self.active_model.fit(f_d, z_d, self.current_model_params)
                if res.success:
                    row = {"Vbias_V": vb, "success": True, "chi2": res.chi2, "redchi": res.redchi}
                    for k, p in res.params.items():
                        row[k] = p.value
                        row[f"{k}_err"] = res.errors.get(k, 0.0)
                    for dk, dv in res.derived.items():
                        row[dk] = dv
                    results_rows.append(row)

                prog = int((idx + 1) / len(voltages) * 100)
                self.prog_analyze.setValue(prog)
                self.lbl_analyze_status.setText(f"BATCH FIT: {idx+1}/{len(voltages)} ({vb:.3f}V)")
                QApplication.processEvents()

            if results_rows:
                df = pd.DataFrame(results_rows)
                csv_path = out_dir / f"fit_summary_Ch{ch}_{self.active_model.name.split()[0]}.csv"
                df.to_csv(csv_path, index=False)
                self.log_analyze("INFO", f"Batch fit complete! {len(results_rows)} points fitted.")
                self.log_analyze("INFO", f"Saved summary CSV to: {csv_path}")
                self.lbl_analyze_status.setText("STATUS: BATCH FIT COMPLETE")
                self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #00e676; font-size: 10pt;")
            else:
                self.lbl_analyze_status.setText("STATUS: NO FITS CONVERGED")
                self.lbl_analyze_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 10pt;")

        except Exception as e:
            self.log_analyze("ERROR", f"Batch fit failed: {e}")
        finally:
            self.btn_fit_all_biases.setEnabled(True)
            self.btn_fit_model.setEnabled(True)

    def on_save_analysis_clicked(self):
        """Saves current plot and parameter JSON into <Complex_Z_Dir>/analysis/."""
        raw_dir = Path(self.edit_analyze_raw_dir.text().strip())
        out_dir = raw_dir / "analysis" if self.chk_analyze_save_in_raw.isChecked() else Path("/data/Results/Complex_Z")
        out_dir.mkdir(parents=True, exist_ok=True)

        ch = self.spin_analyze_channel.value()
        vb = self.current_selected_bias or 0.0

        fig_path = out_dir / f"Ch{ch}_Vbias_{vb:.3f}V_ComplexZ_fit.png"
        self.analyze_canvas.fig.savefig(fig_path, dpi=200, bbox_inches="tight")

        param_dict = {
            "vbias": vb,
            "channel": ch,
            "model": self.active_model.name,
            "params": {k: {"value": p.value, "vary": p.vary, "unit": p.unit, "error": self.current_fit_result.errors.get(k, 0.0) if self.current_fit_result else 0.0}
                       for k, p in self.current_model_params.items()},
        }
        json_path = out_dir / f"Ch{ch}_Vbias_{vb:.3f}V_parameters.json"
        with open(json_path, "w") as f:
            json.dump(param_dict, f, indent=2)

        self.log_analyze("INFO", f"Saved plot to: {fig_path.name}")
        self.log_analyze("INFO", f"Saved parameters to: {json_path.name}")
        QMessageBox.information(self, "Saved", f"Analysis results saved to:\n{out_dir}")

    def on_open_analyze_dir(self):
        raw_dir = Path(self.edit_analyze_raw_dir.text().strip())
        target = raw_dir / "analysis" if self.chk_analyze_save_in_raw.isChecked() else raw_dir
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(target))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as e:
            self.log_analyze("ERROR", f"Could not open directory: {e}")

    def log_analyze(self, level, message):
        color_map = {
            "INFO": "#90caf9",
            "WARN": "#ffb74d",
            "ERROR": "#ef5350",
            "DEBUG": "#b0bec5",
        }
        color = color_map.get(level.upper(), "#ffffff")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_analyze_log.append(
            f'<span style="color: #78909c;">[{timestamp}]</span> '
            f'<b style="color: {color};">[{level.upper()}]</b> '
            f'<span style="color: #eceff1;">{message}</span>'
        )
        self.txt_analyze_log.verticalScrollBar().setValue(
            self.txt_analyze_log.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------------------------
    # Action Button Handlers
    # -----------------------------------------------------------------------
    def on_verify_limits_clicked(self):
        try:
            dc_source = "AWG" if self.combo_dc_source.currentIndex() == 0 else "SRS900"
            bias_vals = self.get_configured_bias_values()
            vpp = self.spin_awg_vpp.value()
            load = self.combo_load.currentText()
            v_limit = 10.0 if load.upper() in ("INFINITY", "INF") else 5.0

            self.log("INFO", "=" * 55)
            self.log("INFO", f"Voltage Verification Check ({dc_source}):")

            if dc_source == "AWG":
                voltage_check_values = bias_vals
                if self.chk_normal_first.isChecked():
                    voltage_check_values = np.append(voltage_check_values, self.spin_vnorm.value())
                v_max = np.max(voltage_check_values) + (vpp / 2.0)
                v_min = np.min(voltage_check_values) - (vpp / 2.0)
                headroom = v_limit - max(abs(v_max), abs(v_min))

                self.log("INFO", f"  AWG Load: {load} | Max allowable: ±{v_limit:.2f} V")
                self.log("INFO", f"  Peak max: {v_max:+.4f} V | Peak min: {v_min:+.4f} V")

                if v_max > v_limit or v_min < -v_limit:
                    msg = f"LIMIT EXCEEDED! Range [{v_min:+.3f} V, {v_max:+.3f} V] exceeds ±{v_limit:.1f} V."
                    self.log("ERROR", msg)
                    QMessageBox.critical(self, "Limit Error", msg)
                else:
                    msg = f"AWG Voltage verification PASSED! Headroom: +{headroom:.3f} V"
                    self.log("INFO", msg)
                    QMessageBox.information(self, "Limit Verification", msg)
            else:
                ac_peak = vpp / 2.0
                headroom = v_limit - ac_peak
                self.log("INFO", f"  AWG AC Stimulus: {vpp:.4f} Vpp (Peak ±{ac_peak:.4f} V)")
                self.log("INFO", f"  SRS SIM928 DC Bias Range: [{np.min(bias_vals):.3f} V, {np.max(bias_vals):.3f} V] (Rated to ±20.0 V)")
                msg = f"SRS900 & AWG Verification PASSED! AWG Headroom: +{headroom:.3f} V"
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

        state_tag = "0V TEST ONLY" if single_0v_only else f"RUNNING ({params['dc_source']})"
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

    def _emergency_zero_hardware(self):
        """Directly zeroes voltages from the GUI main thread in case the worker was terminated."""
        self.log("WARN", "Safely zeroing hardware voltages from main thread...")
        try:
            awg_addr = self.edit_awg_addr.text().strip()
            awg_ch = self.spin_awg_ch.value()
            if HAS_HARDWARE_LIBS and keysight33600A_usb is not None:
                if awg_addr.startswith("vxi11://") or "://" in awg_addr:
                    fg = keysight33600A_usb.Keysight33600A_usb(url=awg_addr)
                else:
                    fg = keysight33600A_usb.Keysight33600A_usb(device=awg_addr)
                fg.setOffset(source=awg_ch, offset=0.0)
                fg.setOutputState(source=awg_ch, output_state="OFF")
                try:
                    fg.channel.close()
                except Exception:
                    pass
                self.log("INFO", f"Emergency: AWG Ch{awg_ch} DC offset zeroed and output disabled.")
        except Exception as e:
            self.log("WARN", f"Emergency AWG zeroing note: {e}")

        if self.combo_dc_source.currentIndex() == 1 and HAS_HARDWARE_LIBS and sim900 is not None:
            try:
                srs_url = self.edit_srs_url.text().strip()
                srs_port = self.spin_srs_port.value()
                mf = sim900.SRS_SIM900(url=srs_url)
                vb = sim928.SRS_SIM928(mf, srs_port)
                vb.setvolt(0.0)
                try:
                    mf.close()
                except Exception:
                    pass
                self.log("INFO", f"Emergency: SRS SIM928 Port {srs_port} voltage zeroed.")
            except Exception as e:
                self.log("WARN", f"Emergency SRS zeroing note: {e}")

    def on_abort_clicked(self):
        self.lbl_status.setText("STATE: ABORTING...")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #ff1744; font-size: 11pt;")
        self.btn_abort.setEnabled(False)
        self.log("WARN", "Abort button clicked by user! Halting worker and zeroing voltages...")

        if self.worker is not None and self.worker.isRunning():
            self.worker.abort()
            if not self.worker.wait(1000):
                self.log("WARN", "Worker thread blocked on external I/O. Forcing thread termination...")
                self.worker.terminate()
                self.worker.wait(500)

        self._emergency_zero_hardware()
        self.on_worker_finished(False, "Measurement sweep aborted. Hardware voltages safely zeroed.")

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
        if "bias_dir" in plot_data:
            try:
                bias_str = f"{bias_v:.2f}".replace(".", "v")
                plot_file = Path(plot_data["bias_dir"]) / f"vbias_{bias_str}_impedance_plot.png"
                self.canvas.fig.savefig(plot_file, dpi=100)
            except Exception as e:
                self.log("WARN", f"Could not save bias plot image: {e}")

    @pyqtSlot(dict)
    def on_sc_calibrated(self, sc_stats):
        r2 = sc_stats["r_squared"]
        f_roll = sc_stats["f_roll"]
        self.lbl_sc_health.setText(f"SC 0V Health: PASS (R² = {r2:.4f}, f_roll = {f_roll:.1f} Hz)")
        self.lbl_sc_health.setStyleSheet("padding: 4px; border-radius: 4px; background-color: #1b5e20; color: #b9f6ca; font-weight: bold;")

    @pyqtSlot(str, dict)
    def on_sc_check_failed(self, reason, sc_stats):
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
            save_dir = Path(self.edit_save_dir.text().strip())
            if save_dir.exists():
                candidates = sorted(save_dir.glob("*/Complex_Z/*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    self.edit_analyze_raw_dir.setText(str(candidates[0]))
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
            "INFO": "#00897b" if not self.chk_dark_mode.isChecked() else "#80cbc4",
            "WARN": "#f57f17" if not self.chk_dark_mode.isChecked() else "#ffd54f",
            "ERROR": "#d32f2f" if not self.chk_dark_mode.isChecked() else "#ef5350"
        }
        color = color_map.get(level.upper(), "#333333" if not self.chk_dark_mode.isChecked() else "#e0e4ed")
        html = f"<span style='color:#78909c;'>[{timestamp}]</span> <b style='color:{color};'>[{level.upper()}]</b> {msg}"
        self.txt_log.append(html)
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        """Safe shutdown on window close."""
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "An acquisition run is currently active! Abort and safely zero all voltages before exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.worker.abort()
                if not self.worker.wait(1000):
                    self.worker.terminate()
                    self.worker.wait(500)
                self._emergency_zero_hardware()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ===========================================================================
# Application Entrypoint
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="EZ-CZ Dashboard GUI")
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
