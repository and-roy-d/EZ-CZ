#!/usr/bin/env python3
"""
cz_test4.py: Complex Impedance (CZ) Data Taker with Built-In Software Sum.

Successor to cz_test3.py.
This script uses Channel 2 of the Keysight 33600A AWG to simultaneously output
the multi-frequency sum-of-sines stimulus and the DC bias voltage via the
hardware DC offset parameter (software-summed Option B1). No external DC voltage
source (SIM928 / QDAC) or external hardware summing amplifier is required.

Compliance & Headroom:
- Termination is set to High-Z (`INFinity`), giving a ±10.0 V output range.
- For 5.0 V DC bias + 20 mVpp AC excitation, the peak voltage is 5.01 V,
  leaving ~4.99 V of safety headroom below the instrument's 10 V limit.
"""

import numpy as np
import time
import pylab as plt
import scipy
from scipy.fft import rfft, rfftfreq
from lmfit import Model, Parameters
import os
import argparse
from datetime import datetime
from pathlib import Path

import qsdumux
import qsdumux.instruments.keysight33600A_usb as keysight33600A_usb
from dastardcommander import EasyClientDastard


# ---------------------------------------------------------------------------
# Target directory setup
# ---------------------------------------------------------------------------
run_start_time = datetime.now()
run_timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
run_date = run_start_time.strftime("%Y%m%d")

base_dir = Path("/data") / run_date / "Complex_Z" / run_timestamp
base_dir.mkdir(parents=True, exist_ok=True)

# Plot setup
plt.ion()
plt.close("all")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
TIME_PER_FLUX_RAMP_POINT_S = 256e-9  # Base clock tick per flux ramp point (1 / 3.90625 MHz)
CONFIG_FILE_PATH = "/data/experiment.cfg"

AWG_ADDRESS = "USB::2391::22279::INSTR"  # Default USB address (or vxi11://192.168.0.112:2021)
AWG_OUTPUT_NUM = 2
AWG_LOAD = "INFinity"  # Required for > 5 V total swing (allows up to ±10 V)

# User experiment targets
TARGET_DURATION_S = 0.2               # Nominal CZ period window (~5 Hz lowest tone)
TARGET_F0_HZ = 20.0                   # Nominal base tone comb spacing (~20 Hz)
NOMINAL_AWG_RATE_HZ = 5e6             # Target AWG sample rate (Hz)
amplitude_multiplier_per_sine = 0.04
final_awg_amplitude_Vpp = 0.02

ch_global = 13

# Bias schedule (0 V baseline, dense grid 0.5-1.5 V, medium grid 1.5-5.0 V)
dense = np.arange(0.5, 1.5, 0.01)
medium = np.arange(1.5, 5.0 + 1e-9, 0.1)
bias_values = np.concatenate([[0.0], dense, medium])

assert bias_values[0] == 0.0, "First bias point must be 0.0 V for baseline/tau calibration."


def get_time_per_flux_ramp_from_cfg(cfg_path=CONFIG_FILE_PATH):
    """
    Auto-reads flux_ramp_points directly from experiment.cfg and calculates:
      time_per_flux_ramp = flux_ramp_points * 256 ns
    """
    import configparser

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

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
                raise ValueError(f"Invalid flux_ramp_points ({fr_points}) in {cfg_path}")

            time_per_flux_ramp = fr_points * TIME_PER_FLUX_RAMP_POINT_S
            print(f"Auto-read {cfg_path} [{section}]: flux_ramp_points = {fr_points}")
            print(f"Time per flux ramp: {time_per_flux_ramp * 1e6:.4f} µs (DAQ fs = {1.0 / time_per_flux_ramp:.2f} Hz)")
            return time_per_flux_ramp, fr_points

    raise KeyError(f"Could not find 'flux_ramp_points' under [Global] in {cfg_path}")


def derive_timing_from_flux_ramp(time_per_flux_ramp, target_duration_s=TARGET_DURATION_S,
                                 target_f0_hz=TARGET_F0_HZ, nominal_awg_rate=NOMINAL_AWG_RATE_HZ):
    """
    Derives all DAQ and AWG parameters strictly from time_per_flux_ramp.
    Eliminates all arbitrary magic numbers:
      - 1 DAQ sample = 1 flux ramp period (time_per_flux_ramp)
      - DAQ sample rate: fs = 1 / time_per_flux_ramp
      - Block size: N_samples = round(target_duration_s / time_per_flux_ramp)
      - Synchronized duration: t_duration = N_samples * time_per_flux_ramp
      - FFT bin spacing: f_fund = 1 / t_duration
      - Stimulus harmonics: f_low = f_fund, f0 = round(target_f0 / f_fund) * f_fund
      - AWG sample count & rate: N_awg = round(t_duration * nominal_awg_rate),
                                 awg_rate = N_awg / t_duration
    """
    fs_global = 1.0 / time_per_flux_ramp

    # Number of flux ramps in the nominal period
    N_samples = int(round(target_duration_s / time_per_flux_ramp))
    t_duration = N_samples * time_per_flux_ramp

    # Fundamental harmonic = FFT bin spacing (zero spectral leakage)
    f_fund = 1.0 / t_duration
    f_low = f_fund

    # Base tone spacing harmonically locked to f_fund
    k0 = max(1, int(round(target_f0_hz / f_fund)))
    f0 = k0 * f_fund

    # AWG waveform sizing and exact matching rate
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


# ---------------------------------------------------------------------------
# AWG Limit Verification Helper
# ---------------------------------------------------------------------------
def verify_awg_voltage_limits(bias_points, ac_amp_vpp, load="INFinity"):
    """
    Verifies that the combined DC bias and AC peak-to-peak amplitude do not
    exceed the allowable voltage compliance window of the Keysight 33600A.

    Keysight 33600A limits:
      - 50 Ohm load:       |V_offset| + V_pp / 2 <= 5.0 V
      - High-Z (INFinity): |V_offset| + V_pp / 2 <= 10.0 V
    """
    v_max = np.max(bias_points) + (ac_amp_vpp / 2.0)
    v_min = np.min(bias_points) - (ac_amp_vpp / 2.0)

    is_high_z = load.upper() in ("INFINITY", "INF")
    v_limit = 10.0 if is_high_z else 5.0

    print("=================================================================")
    print("AWG Voltage Limit & Headroom Verification (Keysight 33600A)")
    print("=================================================================")
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
    print("=================================================================\n")


# ---------------------------------------------------------------------------
# Waveform Generation & Instrument Communication Helpers
# ---------------------------------------------------------------------------
def connect_keysight_fg(address=AWG_ADDRESS):
    """Connects to Keysight 33600A/33622A via USB address or VXI-11 URL."""
    if address.startswith("vxi11://") or "://" in address:
        return keysight33600A_usb.Keysight33600A_usb(url=address)
    return keysight33600A_usb.Keysight33600A_usb(device=address)


def generate_freq_mutiples_logspaced(fs_global, f0=20, start_exp=0, stop_exp=3, num_high_freq_tones=20):
    max_desired_freq = fs_global / 2
    max_allowable_multiplier = int(max_desired_freq / f0)
    multipliers = np.unique(np.int64(np.logspace(start_exp, stop_exp, num_high_freq_tones)))
    return [m for m in multipliers if 1 <= m <= max_allowable_multiplier]


def cos_superimposed(fmults, f0, f_low, t_duration, output_rate_hz, multiplier):
    npts = int(round(t_duration * output_rate_hz))
    t = np.arange(npts) / output_rate_hz
    v_superimposed = np.zeros(npts)
    all_freqs = np.insert(np.array(fmults) * f0, 0, f_low)
    for f in all_freqs:
        v_superimposed += np.cos(2 * np.pi * t * f)
    return all_freqs, v_superimposed * multiplier


def sendArbFloatBinaryBlock(instr, data, source, arb_name="cz_sum_sines"):
    instr._checkSource(source)
    instr.clearVolatileMem(source)
    time.sleep(0.1)
    instr.setByteOrder(swap_bytes=True)
    packed_bytes = np.array(data, dtype=np.float32).tobytes()
    data_length = len(packed_bytes)
    header = f"SOURce{source}:DATA:ARB {arb_name}, #{len(str(data_length))}{data_length}".encode("ascii")
    instr._write_raw(header + packed_bytes + b"\n")


def check_fg_errors(fg):
    errors = []
    for _ in range(100):
        fg._write_raw(b"SYST:ERR?")
        line = fg.readline()
        if line.startswith("+0"):
            break
        errors.append(line)
    if errors:
        print(f"AWG System Warnings/Errors: {errors}")


def setup_awg_with_waveform(fg, vals, output_num=2, load="INFinity", sample_rate=5e6, vpp=0.02, initial_offset=0.0):
    """
    Configures output termination load, uploads the arbitrary waveform once,
    configures sample rate, amplitude, initial offset, and enables output.
    """
    print(f"Configuring Keysight 33600A Channel {output_num}...")
    fg.setOutputLoad(source=output_num, load=load)
    sendArbFloatBinaryBlock(fg, vals, output_num, "cz_sum_sines")
    fg.setFunction(waveform_function="ARB", source=output_num)
    fg.setArbFile(output_num, "cz_sum_sines")
    fg.setArbSampleRate(output_num, sample_rate)
    fg.setArbFilter(output_num, filter_state="STEP")
    fg.setAmplitude(output_num, vpp)
    fg.setOffset(output_num, initial_offset)
    fg.setOutputState(source=output_num, output_state="ON")
    time.sleep(1.0)
    check_fg_errors(fg)
    print(f"Keysight 33600A Channel {output_num} active (Load={load}, Vpp={vpp} V, Offset={initial_offset} V).")


def apply_awg_bias(fg, bias, output_num=2, settle_time=0.3):
    """Sets DC bias voltage via the AWG's hardware offset parameter."""
    print(f"Applying AWG Ch{output_num} DC bias offset: {bias:.3f} V")
    fg.setOffset(source=output_num, offset=bias)
    time.sleep(settle_time)


# ---------------------------------------------------------------------------
# Data Storage Helper
# ---------------------------------------------------------------------------
def save_all_data_npz(save_dir, bias_val, v_stim, data_all, N_samples, N_averages,
                      amplitude_vpp, fs_sch, fft_freqs, tau_shift, fs_global, multiplier):
    os.makedirs(save_dir, exist_ok=True)

    bias_str = f"{bias_val:.2f}".replace(".", "v")
    amp_str = f"{amplitude_vpp:.3f}".replace(".", "v")
    filename = f"vbias_{bias_str}_vac_{amp_str}vpp.npz"
    full_path = save_dir / filename

    data_dict = {}
    fft_real_dict = {}
    fft_imag_dict = {}

    true_voltage_at_frequencies = np.array([multiplier * amplitude_vpp for _ in fft_freqs])
    f_sch_idx = np.searchsorted(fft_freqs, fs_sch)

    for k, v in data_all.items():
        if k.startswith("chan"):
            avg = v[:N_samples * N_averages].reshape((N_averages, N_samples)).mean(axis=0)

            firstFrameIndexMod = data_all["firstFrameIndex"][0] % N_samples
            data = np.roll(avg, firstFrameIndexMod)
            data_dict[k] = data

            # Unroll FFT phase
            fft_data = rfft(data)
            fft_unrolled = fft_data * np.exp(-2j * np.pi * fft_freqs * tau_shift)
            fft_unrolled_selected = fft_unrolled[f_sch_idx]

            fft_real_dict[k] = np.real(fft_unrolled_selected)
            fft_imag_dict[k] = np.imag(fft_unrolled_selected)

    np.savez(
        full_path,
        data=data_dict,
        V_stim=v_stim,
        fft_real=fft_real_dict,
        fft_imag=fft_imag_dict,
        fs_sch=fs_sch,
        true_voltage_at_frequencies=true_voltage_at_frequencies,
        fs_global=fs_global,
        dt=1.0 / fs_global
    )
    print(f"Saved dataset to: {full_path}")


def save_supplied_waveform(save_dir, v_waveform, freqs, output_rate_hz, amplitude_vpp, multiplier, fs_global, fr_points):
    """
    Saves the exact stimulus waveform supplied to the AWG into the data run directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_dir / "supplied_waveform.npz"

    t = np.arange(len(v_waveform)) / output_rate_hz
    np.savez(
        npz_path,
        waveform=v_waveform,
        frequencies=freqs,
        time=t,
        output_rate_hz=output_rate_hz,
        amplitude_vpp=amplitude_vpp,
        multiplier=multiplier,
        fs_global=fs_global,
        flux_ramp_points=fr_points,
        n_samples_waveform=len(v_waveform),
        f_low=freqs[0] if len(freqs) > 0 else None,
        f_high=freqs[-1] if len(freqs) > 0 else None,
        created_at=datetime.now().isoformat()
    )
    print(f"Saved supplied AWG waveform to: {npz_path}")


# ===========================================================================
# Main Execution
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cz_test4: Complex Impedance (CZ) with Software-Summed Bias")
    parser.add_argument("--address", default=AWG_ADDRESS, help=f"AWG address or URL (default: {AWG_ADDRESS})")
    parser.add_argument("--config", default=CONFIG_FILE_PATH, help=f"Path to experiment.cfg (default: {CONFIG_FILE_PATH})")
    parser.add_argument("--time-per-flux-ramp", type=float, default=None,
                        help="Time duration per flux ramp in seconds (e.g. 8.192e-6). If omitted, read from experiment.cfg")
    parser.add_argument("--target-duration", type=float, default=TARGET_DURATION_S,
                        help=f"Nominal measurement duration in seconds (default: {TARGET_DURATION_S} s)")
    parser.add_argument("--target-f0", type=float, default=TARGET_F0_HZ,
                        help=f"Nominal tone spacing in Hz (default: {TARGET_F0_HZ} Hz)")
    parser.add_argument("--test-awg", action="store_true", help="Test AWG upload and offset stepping up to 5V without DASTARD")
    args = parser.parse_args()

    # 1. Determine time_per_flux_ramp (the single fundamental physical clock of the measurement)
    if args.time_per_flux_ramp is not None:
        time_per_flux_ramp = args.time_per_flux_ramp
        fr_points = int(round(time_per_flux_ramp / TIME_PER_FLUX_RAMP_POINT_S))
        print(f"Using CLI time_per_flux_ramp = {time_per_flux_ramp * 1e6:.4f} µs (~{fr_points} points)")
    else:
        time_per_flux_ramp, fr_points = get_time_per_flux_ramp_from_cfg(cfg_path=args.config)

    # 2. Derive all timing, sample counts, frequencies, and AWG rates strictly from time_per_flux_ramp
    timing = derive_timing_from_flux_ramp(
        time_per_flux_ramp=time_per_flux_ramp,
        target_duration_s=args.target_duration,
        target_f0_hz=args.target_f0,
        nominal_awg_rate=NOMINAL_AWG_RATE_HZ
    )
    fs_global = timing["fs_global"]
    N_samples = timing["N_samples"]
    t_duration = timing["t_duration"]
    f_low = timing["f_low"]
    f0 = timing["f0"]
    awg_sample_rate = timing["awg_sample_rate"]
    n_awg_points = timing["n_awg_points"]

    print("=================================================================")
    print("Flux-Ramp Synchronized Timing")
    print("=================================================================")
    print(f"  Time per flux ramp (dt):    {time_per_flux_ramp * 1e6:.4f} µs")
    print(f"  DAQ Sample Rate (fs):       {fs_global:.2f} Hz")
    print(f"  Samples per cycle:          {N_samples} flux ramps (exact integer)")
    print(f"  Synchronized duration (T):  {t_duration:.6f} s")
    print(f"  Fundamental FFT bin (df):   {f_low:.6f} Hz")
    print(f"  Base comb spacing (f0):     {f0:.6f} Hz ({timing['k0']} x df)")
    print(f"  AWG buffer:                 {n_awg_points} points @ {awg_sample_rate:.2f} Sa/s")
    print(f"  AWG buffer playback time:   {n_awg_points / awg_sample_rate:.6f} s (identically equal to T)")
    print("=================================================================\n")

    # 3. Pre-flight check AWG limits
    verify_awg_voltage_limits(bias_values, final_awg_amplitude_Vpp, load=AWG_LOAD)

    # 4. Connect to Keysight 33600A and upload multi-tone waveform
    print(f"Connecting to Keysight 33600A at {args.address}...")
    fg = connect_keysight_fg(args.address)

    f_mults_list = generate_freq_mutiples_logspaced(fs_global, f0=f0)
    fs_sch, v_sch = cos_superimposed(
        f_mults_list, f0, f_low, t_duration, awg_sample_rate, amplitude_multiplier_per_sine
    )
    print(f"Generated {len(fs_sch)} stimulus tones from {fs_sch[0]:.2f} Hz to {fs_sch[-1]:.2f} Hz (Nyquist limit: {fs_global/2:.1f} Hz)")

    # Save the supplied waveform into the run's data directory
    save_supplied_waveform(
        save_dir=base_dir,
        v_waveform=v_sch,
        freqs=fs_sch,
        output_rate_hz=awg_sample_rate,
        amplitude_vpp=final_awg_amplitude_Vpp,
        multiplier=amplitude_multiplier_per_sine,
        fs_global=fs_global,
        fr_points=fr_points
    )

    setup_awg_with_waveform(
        fg,
        vals=v_sch,
        output_num=AWG_OUTPUT_NUM,
        load=AWG_LOAD,
        sample_rate=awg_sample_rate,
        vpp=final_awg_amplitude_Vpp,
        initial_offset=bias_values[0]
    )

    # Optional dry run mode to verify hardware stepping without DASTARD
    if args.test_awg:
        print("\n--- Running AWG Hardware Test Mode (--test-awg) ---")
        test_points = [0.0, 0.5, 1.0, 2.5, 5.0]
        try:
            for v_test in test_points:
                apply_awg_bias(fg, bias=v_test, output_num=AWG_OUTPUT_NUM, settle_time=0.2)
                readback = fg.getOffset(AWG_OUTPUT_NUM)
                err = fg._ask("SYST:ERR?")
                print(f"  Target: {v_test:.3f} V | Readback: {readback:.4f} V | SCPI Error: {err}")
                assert abs(readback - v_test) < 0.05, f"Offset readback mismatch ({readback} != {v_test})"
                assert err.startswith("+0"), f"SCPI Error: {err}"
            print("\n*** AWG Hardware Test PASSED: Successfully stepped to 5.0 V with zero SCPI errors! ***\n")
        finally:
            print("Resetting AWG Ch2 offset to 0.0 V and disabling output...")
            fg.setOffset(source=AWG_OUTPUT_NUM, offset=0.0)
            fg.setOutputState(source=AWG_OUTPUT_NUM, output_state="OFF")
            try:
                fg.channel.close()
            except Exception:
                pass
        exit(0)

    # 4. Connect to DASTARD
    print("Connecting to DASTARD...")
    c = None
    for attempt in range(10):
        try:
            c = EasyClientDastard()
            # Double check live sample rate reported by DASTARD
            live_rate = getattr(c, "sample_rate", None)
            if live_rate and abs(live_rate - fs_global) > 1.0:
                print(f"Updating timing to match live DASTARD rate: {live_rate:.4f} Hz")
                time_per_flux_ramp = 1.0 / float(live_rate)
                timing = derive_timing_from_flux_ramp(time_per_flux_ramp)
                fs_global = timing["fs_global"]
                N_samples = timing["N_samples"]
                t_duration = timing["t_duration"]
                f_low = timing["f_low"]
                f0 = timing["f0"]
            break
        except Exception as e:
            print(f"DASTARD connection attempt {attempt+1}/10 failed ({e}). Retrying...")
            time.sleep(0.5)

    if c is None:
        raise ConnectionError("Failed to connect to DASTARD.")

    tau_shift_0V = None
    fft_unrolls_by_bias = {}
    raw_data_by_bias = {}
    V_stim = amplitude_multiplier_per_sine * final_awg_amplitude_Vpp

    try:
        for idx, bias in enumerate(bias_values):
            # Step the DC offset on Keysight Ch2
            apply_awg_bias(fg, bias=bias, output_num=AWG_OUTPUT_NUM, settle_time=0.3)
            print(f"Now starting {bias:.3f} V bias point. Step {idx+1} of {len(bias_values)}....")

            print(f"[{idx+1}/{len(bias_values)}] Requesting {N_samples} samples from DASTARD at bias = {bias:.3f} V")
            data_all = c.getNewData(N_samples * 4)
            data_raw = data_all[f"chan{ch_global}"] * 1.0
            data_mean = data_raw[:N_samples * 4].reshape((4, N_samples)).mean(axis=0)
            firstFrameIndexMod = data_all["firstFrameIndex"][0] % N_samples
            data = np.roll(data_mean, firstFrameIndexMod)
            raw_data_by_bias[bias] = data

            fft_data_raw = scipy.fft.rfft(data)
            fft_freqs = scipy.fft.rfftfreq(N_samples, d=1/fs_global)
            f_sch_idx = np.searchsorted(fft_freqs, fs_sch)

            # Circuit model calibration at 0 V
            Rs = 250e-6
            L = 1000e-9
            omega_fft_model = 2 * np.pi * fs_sch
            Z_L = 1j * omega_fft_model * L
            Z_model = Rs / (Rs + Z_L) * 9.6e6

            if idx == 0:
                def real_model(fs, tau_shift):
                    return np.real((fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift))[f_sch_idx])

                def magnitude_model(fs, amp0, f_roll):
                    return amp0 / np.sqrt(1 + (fs / f_roll)**2)

                model_mag = Model(magnitude_model)
                params_mag = Parameters()
                params_mag.add("amp0", value=np.abs(fft_data_raw[0]), vary=True)
                params_mag.add("f_roll", value=1e3, vary=True)
                result_mag = model_mag.fit(np.abs(fft_data_raw[f_sch_idx]), fs=fs_sch, params=params_mag)
                amp0 = result_mag.params["amp0"].value
                f_roll = result_mag.params["f_roll"].value

                Z_model_pure = amp0 / (1 - 1j * fs_sch / f_roll)

                model_real = Model(real_model)
                tau_guess = -np.argmax(data) / fs_global
                params = Parameters()
                params.add("tau_shift", value=tau_guess, vary=True)
                result_real = model_real.fit(np.real(Z_model_pure), fs=fs_sch, params=params)
                print(result_real.fit_report())
                tau_shift_0V = result_real.params["tau_shift"].value
            else:
                Z_model_pure = amp0 / (1 - 1j * fs_sch / f_roll)

            fft_unroll = (fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift_0V))[f_sch_idx]
            fft_unrolls_by_bias[bias] = fft_unroll
            bias_str = f"{bias:.2f}".replace(".", "v")

            # Save diagnostic plots
            bias_dir = base_dir / f"bias_{bias_str}"
            bias_dir.mkdir(parents=True, exist_ok=True)

            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Bias = {bias:.2f} V (AWG Offset), τ_shift = {tau_shift_0V:.6e} s", fontsize=14)
            axs = axs.flatten()

            axs[0].semilogx(fs_sch, np.real(fft_unroll), ".-", label="real")
            axs[0].semilogx(fs_sch, np.imag(fft_unroll), ".-", label="imag")
            axs[0].semilogx(fs_sch, np.real(Z_model_pure), ".-", label="real model")
            axs[0].semilogx(fs_sch, np.imag(Z_model_pure), ".-", label="imag model")
            axs[0].set_title("Real and Imaginary Parts")
            axs[0].legend()
            axs[0].grid(True, which="both")

            axs[1].plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", label="data")
            axs[1].plot(np.real(Z_model_pure), np.imag(Z_model_pure), ".-", label="model")
            axs[1].set_title("Nyquist Plot")
            axs[1].grid(True)
            axs[1].axis("equal")

            axs[2].loglog(fs_sch, np.abs(fft_unroll), ".-", label="data")
            axs[2].loglog(fs_sch, np.abs(Z_model_pure), ".-", label="model")
            axs[2].set_title("Magnitude Response")
            axs[2].grid(True, which="both")

            axs[3].semilogx(fs_sch, np.angle(fft_unroll) / np.pi, ".-", label="data")
            axs[3].semilogx(fs_sch, np.angle(Z_model_pure) / np.pi, ".-", label="model")
            axs[3].set_title("Phase Response")
            axs[3].grid(True, which="both")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            plot_path = bias_dir / f"vbias_{bias_str}_impedance_plot.png"
            fig.savefig(plot_path)
            plt.close("all")

            save_all_data_npz(
                save_dir=bias_dir,
                bias_val=bias,
                v_stim=V_stim,
                data_all=data_all,
                N_samples=N_samples,
                N_averages=4,
                amplitude_vpp=final_awg_amplitude_Vpp,
                fs_sch=fs_sch,
                fft_freqs=fft_freqs,
                tau_shift=tau_shift_0V,
                fs_global=fs_global,
                multiplier=amplitude_multiplier_per_sine,
            )

            print(f"Bias {bias:.3f} V done. Saved to {bias_dir}")
            print("-----------------------------------------------------------------------\n")

    finally:
        # Safe ramp down to 0 V
        print("Cleaning up: Safely resetting AWG Channel 2 DC offset to 0.0 V...")
        try:
            fg.setOffset(source=AWG_OUTPUT_NUM, offset=0.0)
            print("AWG offset reset to 0.0 V.")
        except Exception as e:
            print(f"Warning: Failed to reset AWG offset to 0.0 V: {e}")
        try:
            fg.channel.close()
        except Exception:
            pass

