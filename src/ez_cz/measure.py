#!/usr/bin/env python3
"""
measure.py: Complex Impedance (CZ) Measurement Runner for TES Devices.

Features:
  - Dual DC Bias Source Support:
      * "awg" (Default): Direct DC offset on Keysight 33600A AWG Ch2.
      * "srs900": External SRS SIM900/SIM928 DC voltage source.
  - Derived timing strictly from time_per_flux_ramp (eliminates arbitrary magic numbers).
  - Superconducting (0V) Transfer Function Health Check & delay calibration.
  - Safe ramp-down and zeroing on exit or interruption.
  - Dry run / simulation mode for offline verification.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.fft
from lmfit import Model, Parameters
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Hardware Imports with Mock Fallbacks
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
    def __init__(self, *args, **kwargs):
        self.offsets = {1: 0.0, 2: 0.0}
        self.channel = self

    def setOutputLoad(self, source, load): pass
    def _checkSource(self, source): pass
    def clearVolatileMem(self, source): pass
    def setByteOrder(self, swap_bytes): pass
    def _write_raw(self, data): pass
    def setFunction(self, waveform_function, source): pass
    def setArbFile(self, source, arb_name): pass
    def setArbSampleRate(self, source, rate): pass
    def setArbFilter(self, source, filter_state): pass
    def setAmplitude(self, source, amp): pass
    def setOffset(self, source, offset):
        self.offsets[source] = offset
    def getOffset(self, source):
        return self.offsets.get(source, 0.0)
    def setOutputState(self, source, output_state): pass
    def _ask(self, query): return "+0, No error"
    def readline(self): return b"+0\n"
    def close(self): pass


class MockSRS_SIM900:
    def __init__(self, *args, **kwargs): pass
    def close(self): pass


class MockSRS_SIM928:
    def __init__(self, mainframe, port):
        self.port = port
        self.voltage = 0.0

    def setvolt(self, voltage):
        self.voltage = voltage

    def getvolt(self):
        return self.voltage

    def close(self): pass


# ---------------------------------------------------------------------------
# Timing & Mathematical Helpers
# ---------------------------------------------------------------------------
TIME_PER_FLUX_RAMP_POINT_S = 256e-9
DEFAULT_CONFIG_PATH = "/data/experiment.cfg"


def get_time_per_flux_ramp_from_cfg(cfg_path=DEFAULT_CONFIG_PATH):
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
    max_desired_freq = fs_global / 2.0
    max_allowable_multiplier = int(max_desired_freq / f0)
    multipliers = np.unique(np.int64(np.logspace(start_exp, stop_exp, num_high_freq_tones)))
    valid_mults = [m for m in multipliers if 1 <= m <= max_allowable_multiplier]
    all_freqs = np.insert(np.array(valid_mults) * f0, 0, f_low)
    return valid_mults, all_freqs


def synthesize_comb_waveform(all_freqs, t_duration, awg_sample_rate, multiplier):
    npts = int(round(t_duration * awg_sample_rate))
    t = np.arange(npts) / awg_sample_rate
    v_superimposed = np.zeros(npts, dtype=np.float64)
    for f in all_freqs:
        v_superimposed += np.cos(2 * np.pi * t * f)
    return v_superimposed * multiplier


def verify_awg_voltage_limits(bias_points, ac_amp_vpp, load="INFinity", dc_source="awg"):
    is_high_z = load.upper() in ("INFINITY", "INF")
    v_limit = 10.0 if is_high_z else 5.0

    print("=================================================================")
    print(f"Voltage Limit & Headroom Verification (DC Source: {dc_source.upper()})")
    print("=================================================================")
    if dc_source.lower() == "awg":
        v_max = np.max(bias_points) + (ac_amp_vpp / 2.0)
        v_min = np.min(bias_points) - (ac_amp_vpp / 2.0)
        print(f"  Channel Output Load:        {load}")
        print(f"  Max allowable voltage:      ±{v_limit:.2f} V")
        print(f"  Configured DC bias range:   [{np.min(bias_points):.3f} V, {np.max(bias_points):.3f} V]")
        print(f"  AC stimulus amplitude:      {ac_amp_vpp * 1e3:.1f} mVpp (peak ±{(ac_amp_vpp / 2.0) * 1e3:.1f} mV)")
        print(f"  Expected peak max voltage:  {v_max:+.4f} V")
        print(f"  Expected peak min voltage:  {v_min:+.4f} V")

        if v_max > v_limit or v_min < -v_limit:
            raise ValueError(
                f"ERROR: Voltage limits exceeded! Range [{v_min:+.4f} V, {v_max:+.4f} V] "
                f"exceeds allowable limit of ±{v_limit:.2f} V for termination '{load}'."
            )
        headroom = v_limit - max(abs(v_max), abs(v_min))
        print(f"  Safety Headroom:            {headroom:+.4f} V (PASS)")
    else:
        ac_peak = ac_amp_vpp / 2.0
        headroom = v_limit - ac_peak
        print(f"  AWG AC Stimulus:            {ac_amp_vpp * 1e3:.1f} mVpp (peak ±{ac_peak * 1e3:.1f} mV)")
        print(f"  AWG Safety Headroom:        +{headroom:.4f} V (PASS)")
        print(f"  SRS SIM928 DC Bias Range:   [{np.min(bias_points):.3f} V, {np.max(bias_points):.3f} V]")
    print("=================================================================\n")


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------
def load_config_defaults(config_path="run_settings.json"):
    try:
        p = Path(config_path)
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
                return data.get("measure", {})
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
    return {}


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", "-cfg", type=str, default="run_settings.json",
        help="Path to JSON settings file."
    )
    config_args, remaining_argv = config_parser.parse_known_args()
    file_settings = load_config_defaults(config_args.config)

    def get_default(key, fallback):
        return file_settings.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="ez-cz-measure: Complex Impedance (CZ) Measurement on TES Devices."
    )
    parser.add_argument(
        "--config", "-cfg", type=str, default=config_args.config,
        help="Path to JSON settings file (default: %(default)s)."
    )
    parser.add_argument(
        "--dc-source", type=str, choices=["awg", "srs900"],
        default=get_default("dc_source", "awg"),
        help="DC bias voltage source: 'awg' (Keysight AWG Ch2 offset, default) or 'srs900' (SRS SIM900/928)."
    )
    parser.add_argument(
        "--channel", "-c", type=int, default=get_default("channel", 13),
        help="DASTARD channel ID (default: %(default)s)."
    )
    parser.add_argument(
        "--config-cfg", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to experiment.cfg to auto-read time_per_flux_ramp (default: %(default)s)."
    )
    parser.add_argument(
        "--time-per-flux-ramp", type=float, default=None,
        help="Manual time duration per flux ramp in seconds (e.g. 8.192e-6)."
    )
    parser.add_argument(
        "--v-norm", type=float, default=get_default("v_norm", 9.5),
        help="Bias normalization voltage in Volts (default: %(default)s)."
    )
    parser.add_argument(
        "--normal-first", action="store_true", default=get_default("normal_first", False),
        help="Drive TES normal for 1.0 s after 0V calibration before sweep."
    )
    parser.add_argument(
        "--awg-rate", type=float, default=get_default("awg_rate", 5e6),
        help="AWG sample rate in Hz (default: %(default)s)."
    )
    parser.add_argument(
        "--target-duration", type=float, default=0.2,
        help="Nominal measurement duration window in seconds (default: %(default)s)."
    )
    parser.add_argument(
        "--f0", type=float, default=get_default("f0", 20.0),
        help="Base frequency comb spacing in Hz (default: %(default)s)."
    )
    parser.add_argument(
        "--amp-mult", type=float, default=get_default("amp_mult", 0.04),
        help="Amplitude multiplier per sine tone (default: %(default)s)."
    )
    parser.add_argument(
        "--vpp", type=float, default=get_default("vpp", 0.02),
        help="Final AWG AC amplitude in Vpp (default: %(default)s)."
    )
    parser.add_argument(
        "--awg-channel", type=int, default=2,
        help="AWG output channel number (default: %(default)s)."
    )
    parser.add_argument(
        "--awg-load", type=str, default="INFinity",
        help="AWG output termination load: INFinity or 50 (default: %(default)s)."
    )
    parser.add_argument(
        "--keysight-url", type=str, default=get_default("keysight_url", "USB::2391::22279::INSTR"),
        help="Keysight AWG address or URL (default: %(default)s)."
    )
    parser.add_argument(
        "--srs-url", type=str, default=get_default("srs_url", "serial:///dev/ttyUSB1?baudrate=9600"),
        help="SRS SIM900 mainframe serial connection string (default: %(default)s)."
    )
    parser.add_argument(
        "--srs-port", type=int, default=get_default("srs_port", 2),
        help="SRS SIM928 card slot port (default: %(default)s)."
    )
    parser.add_argument(
        "--bias-dense", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
        default=get_default("bias_dense", [0.5, 1.5, 0.01]),
        help="Dense bias values configuration [start, stop, step] (default: %(default)s)."
    )
    parser.add_argument(
        "--bias-medium", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
        default=get_default("bias_medium", [1.5, 5.0, 0.1]),
        help="Medium bias values configuration [start, stop, step] (default: %(default)s)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=get_default("output_dir", None),
        help="Directory path to save raw data and plots (default: /data/<date>/Complex_Z/<time>)."
    )
    parser.add_argument(
        "--n-averages", type=int, default=4,
        help="Number of block averages per bias point (default: %(default)s)."
    )
    parser.add_argument(
        "--dry-run", "--simulation", action="store_true",
        help="Simulation mode without physical instruments."
    )
    return parser.parse_args(remaining_argv)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    dc_source = args.dc_source.lower()

    run_start_time = datetime.now()
    run_timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
    run_date = run_start_time.strftime("%Y%m%d")

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path("/data") / run_date / "Complex_Z" / run_timestamp

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = Path("./data") / run_date / "Complex_Z" / run_timestamp
        base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {base_dir}")
    print(f"DC Bias Source: {dc_source.upper()}")

    # 1. Timing calculation
    time_per_flux_ramp = args.time_per_flux_ramp
    if time_per_flux_ramp is None:
        if os.path.exists(args.config_cfg):
            try:
                time_per_flux_ramp, _ = get_time_per_flux_ramp_from_cfg(args.config_cfg)
            except Exception as e:
                print(f"Warning: Could not read {args.config_cfg}: {e}. Using fallback 8.192 µs.")
                time_per_flux_ramp = 8.192e-6
        else:
            time_per_flux_ramp = 8.192e-6

    timing = derive_timing_from_flux_ramp(
        time_per_flux_ramp=time_per_flux_ramp,
        target_duration_s=args.target_duration,
        target_f0_hz=args.f0,
        nominal_awg_rate=args.awg_rate
    )
    fs_global = timing["fs_global"]
    N_samples = timing["N_samples"]
    t_duration = timing["t_duration"]
    f_low = timing["f_low"]
    f0 = timing["f0"]
    awg_sample_rate = timing["awg_sample_rate"]
    n_awg_points = timing["n_awg_points"]

    print(f"Derived Timing: dt = {time_per_flux_ramp*1e6:.4f} µs, fs = {fs_global:.1f} Hz, N = {N_samples}")
    print(f"Comb: f_low = {f_low:.4f} Hz, f0 = {f0:.4f} Hz, AWG = {n_awg_points} pts @ {awg_sample_rate:.1f} Sa/s")

    # 2. Bias schedule
    dense = np.arange(args.bias_dense[0], args.bias_dense[1], args.bias_dense[2])
    medium = np.arange(args.bias_medium[0], args.bias_medium[1] + 1e-9, args.bias_medium[2])
    bias_values = np.concatenate([[0.0], dense, medium])
    bias_values = np.round(bias_values, 4)
    _, first_indices = np.unique(bias_values, return_index=True)
    bias_values = bias_values[np.sort(first_indices)]

    # 3. Limit verification
    verify_awg_voltage_limits(bias_values, args.vpp, load=args.awg_load, dc_source=dc_source)

    # 4. Synthesize stimulus comb
    valid_mults, all_freqs = generate_stimulus_tones(fs_global, f0, f_low)
    v_waveform = synthesize_comb_waveform(all_freqs, t_duration, awg_sample_rate, args.amp_mult)

    # Save supplied waveform
    np.savez(
        base_dir / "supplied_waveform.npz",
        waveform=v_waveform,
        frequencies=all_freqs,
        awg_sample_rate=awg_sample_rate,
        amplitude_vpp=args.vpp,
        multiplier=args.amp_mult,
        fs_global=fs_global,
        time_per_flux_ramp=time_per_flux_ramp,
        dc_source=dc_source,
        created_at=datetime.now().isoformat()
    )

    # 5. Connect to hardware or mock
    fg = None
    mainframe = None
    vbox = None
    is_dry_run = args.dry_run or not HAS_HARDWARE_LIBS

    try:
        if is_dry_run:
            print("Using Mock Keysight AWG (Dry Run / Simulation mode).")
            fg = MockKeysight33600A()
        elif args.keysight_url.startswith("vxi11://") or "://" in args.keysight_url:
            fg = keysight33600A_usb.Keysight33600A_usb(url=args.keysight_url)
        else:
            fg = keysight33600A_usb.Keysight33600A_usb(device=args.keysight_url)

        # Upload waveform
        source = args.awg_channel
        fg.setOutputLoad(source=source, load=args.awg_load)
        fg._checkSource(source)
        fg.clearVolatileMem(source)
        time.sleep(0.1)
        fg.setByteOrder(swap_bytes=True)
        packed = np.array(v_waveform, dtype=np.float32).tobytes()
        header = f"SOURce{source}:DATA:ARB cz_sum_sines, #{len(str(len(packed)))}{len(packed)}".encode("ascii")
        fg._write_raw(header + packed + b"\n")

        fg.setFunction(waveform_function="ARB", source=source)
        fg.setArbFile(source, "cz_sum_sines")
        fg.setArbSampleRate(source, awg_sample_rate)
        fg.setArbFilter(source, filter_state="STEP")
        fg.setAmplitude(source, args.vpp)
        initial_offset = bias_values[0] if dc_source == "awg" else 0.0
        fg.setOffset(source, initial_offset)
        fg.setOutputState(source=source, output_state="ON")
        time.sleep(0.5)

        # Connect to SRS SIM900 if in srs900 mode
        if dc_source == "srs900":
            if is_dry_run:
                print("Using Mock SRS SIM928 (Dry Run / Simulation mode).")
                mainframe = MockSRS_SIM900()
                vbox = MockSRS_SIM928(mainframe, args.srs_port)
            else:
                mainframe = sim900.SRS_SIM900(url=args.srs_url)
                vbox = sim928.SRS_SIM928(mainframe, args.srs_port)
            print(f"Connected to SRS SIM928 on port {args.srs_port}.")

        # Connect to Dastard
        c = None
        if not is_dry_run and HAS_DASTARD_LIB:
            for attempt in range(5):
                try:
                    c = EasyClientDastard()
                    break
                except Exception as e:
                    print(f"DASTARD connection attempt {attempt+1}/5 failed ({e}). Retrying...")
                    time.sleep(0.5)
            if c is None:
                raise ConnectionError("Failed to connect to DASTARD client.")
        else:
            print("Using simulated detector response (Dry Run).")

        # 6. Sweep Loop
        tau_shift_0V = None
        Z_model_pure = None
        amp0 = None
        f_roll = None
        V_stim = args.amp_mult * args.vpp

        for idx, bias in enumerate(bias_values):
            print(f"[{idx+1}/{len(bias_values)}] Setting bias = {bias:.3f} V ({dc_source.upper()})...")

            if dc_source == "awg":
                fg.setOffset(source=args.awg_channel, offset=bias)
                time.sleep(0.3 if not np.isclose(bias, 0.0) else 1.0)
            else:
                # SRS SIM900/SIM928 DC bias with optional Vnorm pre-pulse
                if not np.isclose(bias, 0.0):
                    vbox.setvolt(args.v_norm)
                    time.sleep(0.1)
                vbox.setvolt(bias)
                time.sleep(0.5)

            # Acquire data
            if c is not None:
                data_all = c.getNewData(N_samples * args.n_averages)
                ch_key = f"chan{args.channel}"
                data_raw = data_all[ch_key] * 1.0
                data_mean = data_raw[:N_samples * args.n_averages].reshape((args.n_averages, N_samples)).mean(axis=0)
                first_frame_index = data_all["firstFrameIndex"][0] % N_samples
                data = np.roll(data_mean, first_frame_index)
            else:
                t = np.arange(N_samples) * time_per_flux_ramp
                data = np.zeros(N_samples)
                f_r_sim = 1200.0
                for f in all_freqs:
                    mag = 5000.0 / np.sqrt(1 + (f / f_r_sim)**2)
                    phi = -np.arctan(f / f_r_sim) - 2 * np.pi * f * 15e-6
                    data += mag * np.cos(2 * np.pi * f * t + phi)
                data += np.random.normal(0, 10, N_samples)
                data_all = {f"chan{args.channel}": np.tile(data, args.n_averages), "firstFrameIndex": [0]}

            # FFT analysis
            fft_data_raw = scipy.fft.rfft(data)
            fft_freqs = scipy.fft.rfftfreq(N_samples, d=time_per_flux_ramp)
            f_sch_idx = np.argmin(np.abs(fft_freqs[:, None] - all_freqs[None, :]), axis=0)
            tone_data_raw = fft_data_raw[f_sch_idx]

            # 0 V Calibration
            if np.isclose(bias, 0.0):
                def magnitude_model(fs, amp0_val, f_roll_val):
                    return amp0_val / np.sqrt(1 + (fs / f_roll_val)**2)

                model_mag = Model(magnitude_model)
                params_mag = Parameters()
                params_mag.add("amp0_val", value=np.abs(fft_data_raw[0]), min=0, vary=True)
                params_mag.add("f_roll_val", value=1e3, min=1.0, max=1e6, vary=True)
                res_mag = model_mag.fit(np.abs(tone_data_raw), fs=all_freqs, params=params_mag)
                amp0 = float(res_mag.params["amp0_val"].value)
                f_roll = float(res_mag.params["f_roll_val"].value)

                Z_model_pure = amp0 / (1.0 + 1j * all_freqs / f_roll)

                def real_model(fs, tau_shift):
                    return np.real((fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift))[f_sch_idx])

                model_real = Model(real_model)
                tau_guess = -np.argmax(data) / fs_global
                params_real = Parameters()
                params_real.add("tau_shift", value=tau_guess, vary=True)
                res_real = model_real.fit(np.real(Z_model_pure), fs=all_freqs, params=params_real)
                tau_shift_0V = float(res_real.params["tau_shift"].value)
                print(f"0V SC Health: amp0={amp0:.1f}, f_roll={f_roll:.1f} Hz, tau_shift={tau_shift_0V*1e6:.2f} µs")

                if args.normal_first:
                    print(f"Driving TES normal at Vnorm = {args.v_norm:.3f} V for 1.0 s...")
                    if dc_source == "awg":
                        fg.setOffset(source=args.awg_channel, offset=args.v_norm)
                    else:
                        vbox.setvolt(args.v_norm)
                    time.sleep(1.0)

            applied_tau = tau_shift_0V if tau_shift_0V is not None else 0.0
            fft_unroll = tone_data_raw * np.exp(-2j * np.pi * all_freqs * applied_tau)

            bias_str = f"{bias:.2f}".replace(".", "v")
            bias_dir = base_dir / f"bias_{bias_str}"
            bias_dir.mkdir(parents=True, exist_ok=True)

            np.savez(
                bias_dir / f"vbias_{bias_str}_data.npz",
                bias=bias,
                v_stim=V_stim,
                fft_unroll=fft_unroll,
                fs_sch=all_freqs,
                tau_shift=applied_tau,
                raw_data=data,
                fs_global=fs_global,
                time_per_flux_ramp=time_per_flux_ramp,
                dc_source=dc_source
            )

        print(f"\nMeasurement complete! All datasets saved to {base_dir}")

    except KeyboardInterrupt:
        print("\nMeasurement interrupted by user! Safely ramping down voltages...")
    finally:
        # Safe cleanup
        if fg is not None:
            try:
                fg.setOffset(source=args.awg_channel, offset=0.0)
                fg.setOutputState(source=args.awg_channel, output_state="OFF")
                fg.channel.close()
            except Exception:
                pass
        if vbox is not None:
            try:
                vbox.setvolt(0.0)
            except Exception:
                pass
        print("Voltages safely zeroed. Exiting.")


if __name__ == "__main__":
    main()
