#!/usr/bin/env python3
"""
analyze.py: Complex Impedance (CZ) Data Analysis & Thermal Model Fitting.

Processes measured complex impedance sweeps across TES bias points:
  - Computes unrolled frequency response and calibrated impedance Z(f).
  - Fits single-pole superconducting (0V) transfer function.
  - Fits hanging-body two-fluid thermal models (C_tes, C_1, G_tes, G_1, alpha, beta).
  - Generates Nyquist, Cole-Cole, Alpha vs Beta, and parameter evolution plots.
  - Exports fitted thermal parameters to summary CSV and figure sets.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import warnings
from scipy.fft import rfft, irfft, rfftfreq
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import coolwarm
import re
import sys
import json
import argparse
import pandas as pd
from lmfit import Model, Parameters
from scipy.optimize import curve_fit as scipy_curve_fit

PHI0 = scipy.constants.value(u"mag. flux quantum")

warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', message='divide by zero encountered in true_divide')
warnings.filterwarnings('ignore', message='invalid value encountered in reciprocal')

plt.rcParams['font.size'] = 10


def convert_ang2_to_ites(ang2_data):
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / PHI0
    arbs_per_phi0 = 1
    amp_per_arb = 1 / min_phi0_per_amp / arbs_per_phi0
    return ang2_data * amp_per_arb


def convert_vbias_to_ibias(vbias, rbias):
    return vbias / rbias


def get_ites_from_iv_curve(iv_file_path, rbias, channel_id, correct_shift=False):
    try:
        data = np.load(iv_file_path, allow_pickle=True)
        vbias_full = data['vb']
        ang2_full_all_ch = data['ang2']
        if channel_id >= ang2_full_all_ch.shape[1]:
            print(f"Error: IV channel_id {channel_id} out of bounds for ang2 shape {ang2_full_all_ch.shape}")
            return None, None
        ang2_channel = ang2_full_all_ch[:, channel_id]
        ites_full = convert_ang2_to_ites(ang2_channel)
        if correct_shift and len(vbias_full) > 0:
            zero_bias_idx = np.argmin(np.abs(vbias_full))
            ites_offset = ites_full[zero_bias_idx]
            ites_full -= ites_offset
        return vbias_full, ites_full
    except FileNotFoundError:
        print(f"Error: IV file not found at '{iv_file_path}'.")
        return None, None
    except Exception as e:
        print(f"Error in get_ites_from_iv_curve: {e}")
        return None, None


def arb_to_amp(in_val):
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / PHI0
    arbs_per_phi0 = 4096
    amp_per_arb = 1 / min_phi0_per_amp / arbs_per_phi0
    return in_val * amp_per_arb


def Rtes(ibias, ites, Rshunt=250e-6):
    ites_safe = np.where(ites == 0, np.nan, ites)
    rtes_values = Rshunt * (ibias - ites_safe) / ites_safe
    return rtes_values


def get_g_tes_b(channel_id_func, g_df_func, channel_map_func, Tc_func, n_thermal_exponent_func):
    if channel_id_func not in channel_map_func:
        print(f"Error: Channel ID {channel_id_func} not found in channel_to_pixel_map.")
        return None
    pixel_number_func = channel_map_func[channel_id_func]
    try:
        if g_df_func is None:
            return None
        g_at_30mk = g_df_func.loc[pixel_number_func]['G_at_30mK']
        g_tes_b_final = g_at_30mk * 1e-12 * (Tc_func / 30e-3) ** (n_thermal_exponent_func - 1)
        print(f"\n--- Thermal Parameter g_tes_b (Ch {channel_id_func}, Px {pixel_number_func}) ---")
        print(f"G@30mK (file): {g_at_30mk:.2f} pW/K")
        print(f"Scaled g_tes_b: {g_tes_b_final * 1e12:.2f} pW/K (using (n-1) scaling from Tbath=30mK to Tc)")
        print("--------------------------------------------------\n")
        return g_tes_b_final
    except KeyError:
        print(f"Error: Pixel_Number {pixel_number_func} (for Ch {channel_id_func}) not found in G data file's index.")
        return None
    except Exception as e:
        print(f"Error in get_g_tes_b for Px {pixel_number_func}: {e}")
        return None


def calculate_z_hanging_model_complex(f, alpha, beta, C_1, C_tes, g_tes_1, R0, P0, T0_model, g_tes_b):
    omega = 2 * np.pi * f
    if g_tes_1 <= 1e-12 or (g_tes_1 + g_tes_b) <= 1e-12 or T0_model <= 1e-6 or C_1 < 1e-18 or C_tes < 1e-18:
        return np.full_like(f, np.nan + 1j * np.nan, dtype=complex)
    L_H = (P0 * alpha) / ((g_tes_1 + g_tes_b) * T0_model)
    if np.abs(1 - L_H) < 1e-9:
        return np.full_like(f, np.nan + 1j * np.nan, dtype=complex)
    tau_1 = C_1 / g_tes_1
    tau_I = C_tes / ((g_tes_1 + g_tes_b) * (1 - L_H))
    sub_term_denom_complex = (g_tes_1 + g_tes_b) * (1 - L_H) * (1 + 1j * omega * tau_1)
    sub_term_complex = np.full_like(f, np.nan + 1j * np.nan, dtype=complex)
    safe_sub_denom_indices = np.abs(sub_term_denom_complex) >= 1e-12
    if np.any(safe_sub_denom_indices):
        sub_term_complex[safe_sub_denom_indices] = g_tes_1 / sub_term_denom_complex[safe_sub_denom_indices]
    main_denom_complex = 1 + 1j * omega * tau_I - sub_term_complex
    second_term_complex = np.full_like(f, np.nan + 1j * np.nan, dtype=complex)
    safe_main_denom_indices = np.abs(main_denom_complex) >= 1e-12
    if np.any(safe_main_denom_indices):
        second_term_complex[safe_main_denom_indices] = (L_H / (1 - L_H)) * (R0 * (2 + beta)) / main_denom_complex[
            safe_main_denom_indices]
    z_model_complex = R0 * (1 + beta) + second_term_complex
    z_model_complex[~np.isfinite(z_model_complex)] = np.nan + 1j * np.nan
    return z_model_complex


def z_tes_hanging_model_for_lmfit(f, alpha, beta, C_1, C_tes, g_tes_1, R0, P0, T0_model, g_tes_b):
    z_model_complex = calculate_z_hanging_model_complex(f, alpha, beta, C_1, C_tes, g_tes_1, R0, P0, T0_model, g_tes_b)
    if np.any(np.isnan(z_model_complex)):
        return np.full(2 * len(f), np.inf)
    return np.concatenate([np.real(z_model_complex), np.imag(z_model_complex)])


def power_law_model(beta_data, A, n_power):
    return A * beta_data ** n_power


def parse_pixel_map(s):
    try:
        parts = s.split(':')
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError("Pixel map item must be in format channel_id:pixel_number (e.g. 13:13)")


def load_config_defaults(config_path="run_settings.json"):
    try:
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                data = json.load(f)
                return data.get("analyze", {})
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
    return {}


def main():
    # Pass 1: Parse only the --config / -cfg argument
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", "-cfg", type=str, default="run_settings.json",
        help="Path to the JSON settings file."
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    # Load settings from file
    file_settings = load_config_defaults(config_args.config)

    # Pass 2: Define and parse all other arguments with defaults merged from file
    parser = argparse.ArgumentParser(
        description="Analyze complex impedance data and fit thermal parameters."
    )
    # Re-add --config so it shows in help
    parser.add_argument(
        "--config", "-cfg", type=str, default=config_args.config,
        help="Path to the JSON settings file (default: run_settings.json)."
    )

    def get_default(key, fallback):
        return file_settings.get(key, fallback)

    # Process pixel_map format from json (list of lists to list of tuples)
    json_pixel_map = get_default("pixel_map", [(13, 13)])
    default_pixel_map = [tuple(item) if isinstance(item, list) else item for item in json_pixel_map]

    parser.add_argument(
        "--raw-dir", "-d", type=str, default=get_default("raw_dir", "/data/20250807/Complex_Z/20250807_141314"),
        help="Raw data directory containing bias_* folders (default: %(default)s)."
    )
    parser.add_argument(
        "--iv-file", "-i", type=str, default=get_default("iv_file", "20250807_150622_iv.npz"),
        help="IV curve npz filename (default: %(default)s)."
    )
    parser.add_argument(
        "--channel-id", "-c", type=int, default=get_default("channel_id", 13),
        help="Dastard channel ID to analyze (default: %(default)s)."
    )
    parser.add_argument(
        "--zero-bias-time-shift", "-t", type=float, default=get_default("zero_bias_time_shift", -2.70625e-2),
        help="Superconducting time shift value in seconds (default: %(default)s)."
    )
    parser.add_argument(
        "--tc", type=float, default=get_default("tc", 120e-3),
        help="Device Tc in Kelvin (default: %(default)s)."
    )
    parser.add_argument(
        "--tbath", type=float, default=get_default("tbath", 20e-3),
        help="Device Tbath in Kelvin (default: %(default)s)."
    )
    parser.add_argument(
        "--n-thermal", type=float, default=get_default("n_thermal", 4.0),
        help="Thermal exponent n (default: %(default)s)."
    )
    parser.add_argument(
        "--dummy-g-tes-b", type=float, default=get_default("dummy_g_tes_b", 200e-9),
        help="Dummy g_tes_b value in W/K if CSV lookup is not available (default: %(default)s)."
    )
    parser.add_argument(
        "--g-data-file", type=str, default=get_default("g_data_file", "/home/pcuser/Runs/Cooldown_A18/Results/G_parameter_summary.csv"),
        help="CSV file path containing precomputed G data (default: %(default)s)."
    )
    parser.add_argument(
        "--pixel-map", nargs="+", type=parse_pixel_map, default=default_pixel_map,
        help="List of channel_id:pixel_number mapping items (default: %(default)s)."
    )
    parser.add_argument(
        "--rbias-iv", type=float, default=get_default("rbias_iv", 2000.0),
        help="Rbias used during IV curve in Ohms (default: %(default)s)."
    )
    parser.add_argument(
        "--rshunt-uohm", type=float, default=get_default("rshunt_uOhm", 250.0),
        help="Rshunt in micro-Ohms (default: %(default)s)."
    )
    parser.add_argument(
        "--l-nyq-nh", type=float, default=get_default("l_nyq_nH", 74.8),
        help="Nyquist inductor L in nano-Henries (default: %(default)s)."
    )
    parser.add_argument(
        "--output-base-dir", "-o", type=str, default=get_default("output_base_dir", "/home/pcuser/Runs/Complex_Z/"),
        help="Base directory to save output results and plots (default: %(default)s)."
    )
    parser.add_argument(
        "--save-plots", action="store_true", default=get_default("save_plots", True),
        help="Save output plots and fit summary CSV files (default: %(default)s)."
    )
    parser.add_argument(
        "--no-save-plots", action="store_false", dest="save_plots",
        help="Do not save plots, only display them in GUI."
    )

    args = parser.parse_args(remaining_argv)

    # Paths setup and Windows fallbacks
    base_dir_raw = Path(args.raw_dir)
    if not base_dir_raw.exists():
        print(f"Warning: Raw data directory {base_dir_raw} does not exist. Checking local path `./data`...")
        # Try local path conversion
        local_raw = Path(".") / "data" / "20250807" / "Complex_Z" / "20250807_141314"
        if local_raw.exists():
            base_dir_raw = local_raw
            print(f"Using local raw data directory: {base_dir_raw}")

    output_base = Path(args.output_base_dir)
    try:
        output_base.mkdir(parents=True, exist_ok=True)
    except Exception:
        output_base = Path("./results/Complex_Z")
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"Using local output directory: {output_base}")

    FS_SAMPLE_RATE = 250000

    channel_to_pixel_map = dict(args.pixel_map)
    channels_to_process = [args.channel_id] if args.channel_id in channel_to_pixel_map else list(channel_to_pixel_map.keys())

    if not channels_to_process:
        print(f"Error: Target channel {args.channel_id} is not mapped in pixel map {channel_to_pixel_map}.")
        sys.exit(1)

    # Load G summary data if available
    g_dataframe_global = None
    g_file = Path(args.g-data-file) if hasattr(args, 'g-data-file') else Path(args.g_data_file)
    if g_file.exists():
        try:
            g_dataframe_global = pd.read_csv(g_file, index_col=None)
            g_dataframe_global.set_index('Pixel_Number', inplace=True)
            print(f"Loaded G parameters from: {g_file}")
        except Exception as e:
            print(f"Warning: Could not read G data file '{g_file}': {e}. Will fall back to dummy g_tes_b.")
    else:
        print(f"Warning: G data file '{g_file}' not found. Will fall back to dummy g_tes_b.")

    for channel in channels_to_process:
        CHANNEL_ID = channel
        channel_id_str = f"chan{CHANNEL_ID}"
        iv_channel_id_for_offset = CHANNEL_ID  # Using the channel directly for IV index (or offset if needed)

        print(f"\n" + "=" * 70)
        print(f"========== Processing Channel ID: {CHANNEL_ID} ==========")
        print("=" * 70)

        pixel_number_for_dir = channel_to_pixel_map.get(CHANNEL_ID, CHANNEL_ID)
        pixel_output_dir = output_base / f"Pixel_{pixel_number_for_dir}"
        if args.save_plots:
            pixel_output_dir.mkdir(parents=True, exist_ok=True)

        # Get g_tes_b
        g_tes_b_device = get_g_tes_b(CHANNEL_ID, g_dataframe_global, channel_to_pixel_map, args.tc, args.n_thermal)
        if g_tes_b_device is None:
            g_tes_b_device = args.dummy_g_tes_b
            print(f"Using dummy/fallback g_tes_b: {g_tes_b_device * 1e12:.2f} pW/K")

        # Load IV curves
        date_str_iv = args.iv_file.split('_')[0]
        iv_path = Path(f"/data/{date_str_iv}/iv/{args.iv_file}")
        if not iv_path.exists():
            iv_path = Path(".") / args.iv_file
        if not iv_path.exists():
            # Search raw directory or parent
            iv_path = base_dir_raw.parent.parent.parent / "iv" / args.iv_file
        if not iv_path.exists():
            # Check current workspace directory
            iv_path = Path(args.iv_file)

        if not iv_path.exists():
            print(f"Error: IV file not found for Ch {CHANNEL_ID}. Checked paths including '{iv_path}'. Skipping channel.")
            continue

        vbias_iv_full, ites_iv_full = get_ites_from_iv_curve(
            iv_path, rbias=args.rbias_iv, channel_id=iv_channel_id_for_offset, correct_shift=True
        )
        if vbias_iv_full is None:
            print(f"Error: Failed to load IV data for Ch {CHANNEL_ID} from {iv_path}. Skipping channel.")
            continue
        print(f"IV-Curve Loaded for Ch {CHANNEL_ID} from: {iv_path}")

        ibias_iv_full = convert_vbias_to_ibias(vbias_iv_full, args.rbias_iv)
        rtes_iv_full = Rtes(ibias_iv_full, ites_iv_full, Rshunt=args.rshunt_uohm * 1e-6)

        Rnormal, stop_bias_val = None, None
        positive_rtes_indices = np.where((rtes_iv_full > 50e-6) & np.isfinite(rtes_iv_full))[0]
        if len(positive_rtes_indices) > 0:
            vbias_pos_rtes = vbias_iv_full[positive_rtes_indices]
            ites_pos_rtes = ites_iv_full[positive_rtes_indices]
            rtes_pos_rtes = rtes_iv_full[positive_rtes_indices]
            min_ites_idx_subset = np.argmin(np.abs(ites_pos_rtes))
            stop_bias_val = vbias_pos_rtes[min_ites_idx_subset]
            Rnormal = rtes_pos_rtes[min_ites_idx_subset]
            print(f"Rnormal for Ch {CHANNEL_ID}: {Rnormal:.3e} Ohms at Vbias={stop_bias_val:.4f}V")
            start_bias_val_filter = vbias_pos_rtes[-4] if len(vbias_pos_rtes) >= 4 else vbias_pos_rtes[0]
        else:
            print(f"Warning: Rnormal could not be determined for Ch {CHANNEL_ID}. Using default bias filter.")
            start_bias_val_filter = 0.0
        end_bias_val_filter = stop_bias_val if stop_bias_val is not None else 0.3
        print(f"Bias filter for Ch {CHANNEL_ID}: {start_bias_val_filter:.4f}V to {end_bias_val_filter:.4f}V")

        # Process zero bias file
        zero_bias_freq_currents = {}
        processed_zero_bias = False
        for bias_dir_path_loop in sorted(base_dir_raw.glob('bias_*')):
            bias_val_temp = float(bias_dir_path_loop.name.split('_', 1)[-1].replace('v', '.'))
            if np.isclose(bias_val_temp, 0.0):
                npz_files_in_dir = list(bias_dir_path_loop.glob("*.npz"))
                if not npz_files_in_dir:
                    continue
                npz_file_raw_data_temp = npz_files_in_dir[0]
                try:
                    data_raw_temp = np.load(npz_file_raw_data_temp, allow_pickle=True)
                    raw_data_dict = data_raw_temp["data"].item()
                    if channel_id_str not in raw_data_dict:
                        print(f"Warning: {channel_id_str} not in zero-bias file {npz_file_raw_data_temp}. Skipping.")
                        break
                    raw_current_data_arb_temp = raw_data_dict[channel_id_str]
                    raw_current_data_amps_temp = arb_to_amp(raw_current_data_arb_temp)
                    fs_sch_array_temp = data_raw_temp['fs_sch']
                    if len(fs_sch_array_temp) == 0:
                        continue
                    n_points_temp = len(raw_current_data_amps_temp)
                    full_fft_frequencies_temp = rfftfreq(n_points_temp, 1 / FS_SAMPLE_RATE)
                    idx_0_iv = np.argmin(np.abs(vbias_iv_full - 0.0))
                    target_dc_level_0 = ites_iv_full[idx_0_iv]
                    current_mean_0 = np.mean(raw_current_data_amps_temp)
                    raw_current_data_dc_replaced_amps_0 = (raw_current_data_amps_temp - current_mean_0) + target_dc_level_0
                    raw_current_fft_dc_replaced_0 = rfft(raw_current_data_dc_replaced_amps_0)
                    phase_factor_0 = np.exp(-2j * np.pi * full_fft_frequencies_temp * args.zero_bias_time_shift)
                    raw_current_fft_shifted_0 = raw_current_fft_dc_replaced_0 * phase_factor_0
                    for stim_freq_0 in fs_sch_array_temp:
                        f_idx_0 = np.searchsorted(full_fft_frequencies_temp, stim_freq_0)
                        if f_idx_0 < len(raw_current_fft_shifted_0):
                            zero_bias_freq_currents[stim_freq_0] = raw_current_fft_shifted_0[f_idx_0]
                    print(f"Processed zero-bias data from {npz_file_raw_data_temp} for Ch {CHANNEL_ID}.")
                    processed_zero_bias = True
                    break
                except Exception as e:
                    print(f"Warning: Failed to load zero-bias file {npz_file_raw_data_temp}: {e}")
                    continue

        if not processed_zero_bias:
            print(f"Error: Zero-bias data not found/processed for Ch {CHANNEL_ID}. Skipping channel.")
            continue

        # Load raw complex Z data
        all_raw_data_info = []
        processed_bias_vals_plot = []

        for bias_dir_path in sorted(base_dir_raw.glob('bias_*')):
            bias_val = float(bias_dir_path.name.split('_', 1)[-1].replace('v', '.'))
            if not (start_bias_val_filter <= bias_val <= end_bias_val_filter) or np.isclose(bias_val, 0.0):
                continue
            npz_files_in_dir = list(bias_dir_path.glob("*.npz"))
            if not npz_files_in_dir:
                continue
            npz_file_raw_data = npz_files_in_dir[0]
            try:
                data_raw = np.load(npz_file_raw_data, allow_pickle=True)
                raw_data_dict = data_raw["data"].item()
                if channel_id_str not in raw_data_dict:
                    print(f"Warning: {channel_id_str} not in bias file {npz_file_raw_data}. Skipping bias {bias_val} V.")
                    continue
                raw_current_data_arb = raw_data_dict[channel_id_str]
                raw_current_data_amps = arb_to_amp(raw_current_data_arb)
                fs_sch_array = data_raw['fs_sch']
                if len(fs_sch_array) == 0:
                    continue
                processed_bias_vals_plot.append(bias_val)
                n_points = len(raw_current_data_amps)
                full_fft_frequencies = rfftfreq(n_points, 1 / FS_SAMPLE_RATE)
                all_raw_data_info.append((bias_val, raw_current_data_amps, fs_sch_array, full_fft_frequencies, n_points))
            except Exception as e:
                print(f"Warning: Error reading file {npz_file_raw_data}: {e}")
                continue

        if not all_raw_data_info:
            print(f"No Complex Z data in filter range for Ch {CHANNEL_ID}. Skipping channel.")
            continue

        # Calculate Admittance and Fit Parameters
        shifted_raw_data_by_bias = []
        admittance_data_shifted = {}
        fit_results = {}

        gmodel_ch = Model(z_tes_hanging_model_for_lmfit, independent_vars=['f'])

        for bias_val, raw_current_data_amps, fs_sch_array, full_fft_frequencies, n_points in all_raw_data_info:
            print(f"  Processing Vbias = {bias_val:.4f}V for Ch {CHANNEL_ID}...")
            idx_iv_op = np.argmin(np.abs(vbias_iv_full - bias_val))
            target_dc_level = ites_iv_full[idx_iv_op]
            current_mean_raw = np.mean(raw_current_data_amps)
            raw_current_dc_replaced = (raw_current_data_amps - current_mean_raw) + target_dc_level
            current_time_shift = args.zero_bias_time_shift
            raw_fft_dc_replaced = rfft(raw_current_dc_replaced)
            phase_factor = np.exp(-2j * np.pi * full_fft_frequencies * current_time_shift)
            raw_fft_shifted = raw_fft_dc_replaced * phase_factor
            shifted_raw_data_by_bias.append((bias_val, irfft(raw_fft_shifted)))

            adm_re_measured, adm_im_measured = [], []
            for stim_freq in fs_sch_array:
                f_idx = np.searchsorted(full_fft_frequencies, stim_freq)
                if f_idx >= len(raw_fft_shifted):
                    Y_tes_calculated = np.nan + 1j * np.nan
                else:
                    I_bias_fft_component = raw_fft_shifted[f_idx]
                    I_0_ref_fft_component = zero_bias_freq_currents.get(stim_freq)
                    Z_shunt_L_circuit = 1j * 2 * np.pi * stim_freq * (args.l_nyq_nh * 1e-9) + (args.rshunt_uohm * 1e-6)
                    if I_0_ref_fft_component is None or np.abs(I_0_ref_fft_component) < 1e-15:
                        Z_tes_calculated = np.nan + 1j * np.nan
                    else:
                        Ratio_I = I_bias_fft_component / I_0_ref_fft_component
                        Z_tes_calculated = Z_shunt_L_circuit * (1 / Ratio_I - 1) if np.abs(
                            Ratio_I) > 1e-12 else np.nan + 1j * np.nan
                    Y_tes_calculated = 1.0 / Z_tes_calculated
                adm_re_measured.append(np.real(Y_tes_calculated))
                adm_im_measured.append(np.imag(Y_tes_calculated))
            admittance_data_shifted[bias_val] = (fs_sch_array, np.array(adm_re_measured), np.array(adm_im_measured))

            R0_op = rtes_iv_full[idx_iv_op]
            I0_op = ites_iv_full[idx_iv_op]
            P0_op = I0_op ** 2 * R0_op
            T0_for_model_fit = args.tc

            Y_measured_complex = np.array(adm_re_measured) + 1j * np.array(adm_im_measured)
            Z_measured_complex = 1.0 / Y_measured_complex
            valid_Z_mask = np.isfinite(Z_measured_complex)
            fs_sch_for_fitting = fs_sch_array[valid_Z_mask]

            if len(fs_sch_for_fitting) < 5:
                print(f"    Skipping fit: Not enough valid data points ({len(fs_sch_for_fitting)}).")
                continue

            Z_measured_clean_real = np.real(Z_measured_complex)[valid_Z_mask]
            Z_measured_clean_imag = np.imag(Z_measured_complex)[valid_Z_mask]
            y_data_to_fit_clean = np.concatenate([Z_measured_clean_real, Z_measured_clean_imag])

            params_lmfit = Parameters()
            params_lmfit.add('alpha', value=400, min=0, max=4000)
            params_lmfit.add('beta', value=5, min=0, max=100)
            params_lmfit.add('C_1', value=100e-15, min=1e-16, max=1e-11)
            params_lmfit.add('C_tes', value=500e-15, min=1e-15, max=1e-11)
            params_lmfit.add('g_tes_1', value=g_tes_b_device / 5 if g_tes_b_device else 1e-9, min=1e-12,
                             max=g_tes_b_device * 50 if g_tes_b_device else 1e-7)
            params_lmfit.add('R0', value=R0_op, vary=False)
            params_lmfit.add('P0', value=P0_op, vary=False)
            params_lmfit.add('T0_model', value=T0_for_model_fit, vary=False)
            params_lmfit.add('g_tes_b', value=g_tes_b_device, vary=False)

            try:
                result = gmodel_ch.fit(y_data_to_fit_clean, params_lmfit, f=fs_sch_for_fitting,
                                       method='leastsq', fit_kws={'ftol': 1e-9, 'xtol': 1e-9, 'maxfev': 30000})
                if result.success:
                    popt_lmfit = result.params
                    fit_results[bias_val] = {
                        'alpha': popt_lmfit['alpha'].value,
                        'alpha_err': popt_lmfit['alpha'].stderr if popt_lmfit['alpha'].stderr is not None else 0,
                        'beta': popt_lmfit['beta'].value,
                        'beta_err': popt_lmfit['beta'].stderr if popt_lmfit['beta'].stderr is not None else 0,
                        'C_1': popt_lmfit['C_1'].value,
                        'C_1_err': popt_lmfit['C_1'].stderr if popt_lmfit['C_1'].stderr is not None else 0,
                        'C_tes': popt_lmfit['C_tes'].value,
                        'C_tes_err': popt_lmfit['C_tes'].stderr if popt_lmfit['C_tes'].stderr is not None else 0,
                        'g_tes_1': popt_lmfit['g_tes_1'].value,
                        'g_tes_1_err': popt_lmfit['g_tes_1'].stderr if popt_lmfit['g_tes_1'].stderr is not None else 0,
                        'R0': R0_op, 'P0': P0_op, 'T0_model': T0_for_model_fit, 'g_tes_b': g_tes_b_device,
                        'fs_fit': fs_sch_for_fitting, 'lmfit_result_obj': result
                    }
                    print(
                        f"    Fit OK: α={popt_lmfit['alpha'].value:.1f}, β={popt_lmfit['beta'].value:.2f}, C1={popt_lmfit['C_1'].value:.1e}, Ctes={popt_lmfit['C_tes'].value:.1e}, g1={popt_lmfit['g_tes_1'].value:.1e}")
                else:
                    print(f"    Fit Warning: lmfit did not report success. Message: {result.message}")
            except Exception as e:
                print(f"    Fit Exception: {e}")

        # Plotting & Saving (Per Channel)
        if not fit_results:
            print(f"No successful fits for Ch {CHANNEL_ID} to plot or save.")
            continue

        # Plot 1: IV Curve
        fig_iv = plt.figure(figsize=(10, 7))
        plt.plot(vbias_iv_full * 1e3, ites_iv_full * 1e6, 'b-', label='Full IV Curve')
        if processed_bias_vals_plot:
            ites_at_proc = [ites_iv_full[np.argmin(np.abs(pb - vbias_iv_full))] for pb in processed_bias_vals_plot]
            plt.plot(np.array(processed_bias_vals_plot) * 1e3, np.array(ites_at_proc) * 1e6, 'ro', ms=5,
                     label='Processed Bias Pts')
        plt.xlabel("Bias Voltage (mV)")
        plt.ylabel("TES Current (μA)")
        plt.title(f"IV Curve: IV Ch {iv_channel_id_for_offset} (Z Ch {CHANNEL_ID}, Px {pixel_number_for_dir})")
        plt.grid(True, ls=':')
        plt.legend()
        plt.tight_layout()
        if args.save_plots:
            fig_iv.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_01_IVcurve.png")
        plt.close(fig_iv)

        # Plot 2: Rtes vs Vbias
        if Rnormal is not None:
            fig_rtes = plt.figure(figsize=(10, 7))
            plt.plot(vbias_iv_full * 1e3, rtes_iv_full * 1e3, 'g-', label='Full Rtes Curve')
            plt.axhline(y=Rnormal * 1e3, color='dimgray', ls='--', lw=1.5, label=f'$R_n$={Rnormal * 1e3:.2f} mΩ')
            if processed_bias_vals_plot:
                rtes_at_proc_for_plot = [rtes_iv_full[np.argmin(np.abs(pb - vbias_iv_full))] for pb in
                                         processed_bias_vals_plot]

                plt.plot(np.array(processed_bias_vals_plot) * 1e3, np.array(rtes_at_proc_for_plot) * 1e3, 'ro', ms=7,
                         label='Processed Bias Pts')
                for v_bp, r_tp in zip(processed_bias_vals_plot, rtes_at_proc_for_plot):
                    plt.text(v_bp * 1e3, r_tp * 1e3 + 0.05 * Rnormal * 1e3, f'{(r_tp / Rnormal) * 100:.1f}%',
                             fontsize=8, ha='center', va='bottom',
                             bbox=dict(fc='w', alpha=0.6, ec='none', pad=0.5))
            plt.xlabel("Bias Voltage (mV)")
            plt.ylabel("TES Resistance (mΩ)")
            plt.title(f"Rtes vs. Vbias: IV Ch {iv_channel_id_for_offset} (Px {pixel_number_for_dir})")
            plt.grid(True, ls=':')
            plt.legend()
            plt.tight_layout()
            if args.save_plots:
                fig_rtes.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_02_Rtes_vs_Vbias.png")
            plt.close(fig_rtes)

        # Plot 3: Shifted Raw traces
        if shifted_raw_data_by_bias:
            fig_raw = plt.figure(figsize=(12, 7))
            ax_raw = fig_raw.gca()
            sorted_shifted = sorted(shifted_raw_data_by_bias, key=lambda x: x[0])
            bias_cmap_raw = [item[0] for item in sorted_shifted]
            if bias_cmap_raw:
                norm_raw = Normalize(vmin=min(bias_cmap_raw), vmax=max(bias_cmap_raw))
                cmap_raw = coolwarm
                if sorted_shifted and len(sorted_shifted[0]) > 1 and len(sorted_shifted[0][1]) > 0:
                    time_ms = np.arange(len(sorted_shifted[0][1])) / FS_SAMPLE_RATE * 1e3
                    for bv_plt, data_plt in sorted_shifted:
                        ax_raw.plot(time_ms, data_plt, color=cmap_raw(norm_raw(bv_plt)), alpha=0.8, lw=1)
                    cb = fig_raw.colorbar(plt.cm.ScalarMappable(norm=norm_raw, cmap=cmap_raw), ax=ax_raw, pad=0.02)
                    cb.set_label("Bias Voltage (V)")
            ax_raw.set_xlabel("Time (ms)")
            ax_raw.set_ylabel(f"Shifted Current (Ch {CHANNEL_ID}) [A]")
            ax_raw.set_title(
                f"Shifted Raw Current (Time Shift = {args.zero_bias_time_shift:.3e} s, Px {pixel_number_for_dir})")
            ax_raw.grid(True, ls=':')
            fig_raw.tight_layout()
            if args.save_plots:
                fig_raw.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_03_ShiftedTraces.png")
            plt.close(fig_raw)

        # Plot 4: Admittance Overview
        if admittance_data_shifted:
            fig_adm_m, axs_adm_m = plt.subplots(2, 2, figsize=(16, 10))
            fig_adm_m.suptitle(
                f"Measured Admittance: Ch {CHANNEL_ID}, Px {pixel_number_for_dir} (Shift={args.zero_bias_time_shift:.3e}s)",
                fontsize=16)
            bias_adm_m = sorted(admittance_data_shifted.keys())
            if bias_adm_m:
                norm_adm_m = Normalize(vmin=min(bias_adm_m), vmax=max(bias_adm_m))
                cmap_adm_m = coolwarm
                for bv_plt in bias_adm_m:
                    fs_plt, re_plt, im_plt = admittance_data_shifted[bv_plt]
                    color = cmap_adm_m(norm_adm_m(bv_plt))
                    axs_adm_m[0, 0].semilogx(fs_plt, re_plt, 'o-', c=color, ms=4, lw=1)
                    axs_adm_m[0, 1].semilogx(fs_plt, im_plt, 'o-', c=color, ms=4, lw=1)
                    axs_adm_m[1, 0].semilogx(fs_plt, np.sqrt(re_plt ** 2 + im_plt ** 2), 'o-', c=color, ms=4, lw=1)
                    axs_adm_m[1, 1].plot(re_plt, im_plt, 'o-', c=color, ms=4, lw=1)
                ylabels = [r"Re(Y_m)", "Im(Y_m)", "|Y_m|", "Nyquist Y_m"]
                for i_ax, ax in enumerate(axs_adm_m.flat):
                    ax.set_xlabel("Freq (Hz)" if i_ax < 3 else r"Re(Y) ($\Omega^{-1}$)")
                    ax.set_ylabel(r"Y ($\Omega^{-1}$)" if i_ax < 3 else r"Im(Y) ($\Omega^{-1}$)")
                    ax.grid(True, which="both", ls=':')
                axs_adm_m[1, 1].axis('equal')
                fig_adm_m.tight_layout(rect=[0, 0, 0.9, 0.95])
                cbar_ax = fig_adm_m.add_axes([0.92, 0.15, 0.02, 0.7])
                fig_adm_m.colorbar(plt.cm.ScalarMappable(norm=norm_adm_m, cmap=cmap_adm_m), cax=cbar_ax,
                                   label='Vbias (V)')
            if args.save_plots:
                fig_adm_m.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_04_Admittance_Measured.png")
            plt.close(fig_adm_m)

        # Plot 5: Admittance Fits
        if fit_results:
            fig_fits, axs_fits = plt.subplots(2, 2, figsize=(8.5, 6.5), dpi=200)
            bias_fitted_plot = sorted(fit_results.keys())
            if bias_fitted_plot:
                norm_fit_plot = Normalize(vmin=min(bias_fitted_plot), vmax=max(bias_fitted_plot))
                cmap_fit_plot = coolwarm
                for i, bv_plot in enumerate(bias_fitted_plot):
                    params_dict = fit_results[bv_plot]
                    color = cmap_fit_plot(norm_fit_plot(bv_plot))
                    fs_m, adm_re_m, adm_im_m = admittance_data_shifted[bv_plot]
                    fs_f = params_dict.get('fs_fit', fs_m)
                    if len(fs_f) == 0:
                        continue
                    Z_model = calculate_z_hanging_model_complex(fs_f, params_dict['alpha'], params_dict['beta'],
                                                                params_dict['C_1'],
                                                                params_dict['C_tes'], params_dict['g_tes_1'],
                                                                params_dict['R0'],
                                                                params_dict['P0'], params_dict['T0_model'],
                                                                params_dict['g_tes_b'])
                    Y_model = 1.0 / Z_model
                    adm_re_f, adm_im_f, adm_mag_f = np.real(Y_model), np.imag(Y_model), np.abs(Y_model)
                    lbl_d = 'Data' if i < 1 else None
                    lbl_f = 'Fit' if i < 1 else None
                    axs_fits[0, 0].semilogx(fs_m, adm_re_m, 'o', c=color, ms=2, alpha=0.6, label=lbl_d)
                    axs_fits[0, 0].semilogx(fs_f, adm_re_f, '-', c=color, lw=1, label=lbl_f)
                    axs_fits[0, 1].semilogx(fs_m, adm_im_m, 'o', c=color, ms=2, alpha=0.6)
                    axs_fits[0, 1].semilogx(fs_f, adm_im_f, '-', c=color, lw=1)
                    axs_fits[1, 0].semilogx(fs_m, np.sqrt(adm_re_m ** 2 + adm_im_m ** 2), 'o', c=color, ms=2, alpha=0.6)
                    axs_fits[1, 0].semilogx(fs_f, adm_mag_f, '-', c=color, lw=1)
                    axs_fits[1, 1].plot(adm_re_m, adm_im_m, 'o', c=color, ms=2, alpha=0.6)
                    axs_fits[1, 1].plot(adm_re_f, adm_im_f, '-', c=color, lw=1)
                ylabels_fits = [r"Re($Y_{TES}$) ($\Omega^{-1}$)", r"Im($Y_{TES}$) ($\Omega^{-1}$)", r"|$Y_{TES}$| ($\Omega^{-1}$)", r"Im($Y_{TES}$) ($\Omega^{-1}$)"]
                for i_ax, ax_fit in enumerate(axs_fits.flat):
                    ax_fit.set_xlabel("Freq (Hz)" if i_ax < 3 else r"Re(Y) ($\Omega^{-1}$)")
                    ax_fit.set_ylabel(ylabels_fits[i_ax])
                    ax_fit.grid(True, which="both", ls=':', lw=0.5)
                axs_fits[1, 1].axis('equal')
                if bias_fitted_plot:
                    axs_fits[0, 0].legend(fontsize='x-small', loc='best')
                fig_fits.tight_layout(rect=[0, 0, 0.9, 0.95])
            if args.save_plots:
                fig_fits.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_05_Admittance_Fits.png")
            plt.close(fig_fits)

        # Plot 6: Residuals
        if fit_results:
            fig_res, axs_res_plot = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True, sharex=True)
            fig_res.suptitle(
                f"Hanging Body Fit Residuals: Ch {CHANNEL_ID}, Px {pixel_number_for_dir} (n_exp={args.n_thermal})",
                fontsize=16)
            bias_fitted_plot_res = sorted(fit_results.keys())
            if bias_fitted_plot_res:
                norm_res_plot_loc = Normalize(vmin=min(bias_fitted_plot_res), vmax=max(bias_fitted_plot_res))
                cmap_res_plot_loc = coolwarm
                for i, bv_plot in enumerate(bias_fitted_plot_res):
                    params_dict = fit_results[bv_plot]
                    color = cmap_res_plot_loc(norm_res_plot_loc(bv_plot))
                    fs_m, adm_re_m, adm_im_m = admittance_data_shifted[bv_plot]
                    adm_mag_m = np.sqrt(adm_re_m ** 2 + adm_im_m ** 2)
                    Z_model_fs_m = calculate_z_hanging_model_complex(fs_m, params_dict['alpha'], params_dict['beta'],
                                                                     params_dict['C_1'],
                                                                     params_dict['C_tes'], params_dict['g_tes_1'],
                                                                     params_dict['R0'],
                                                                     params_dict['P0'], params_dict['T0_model'],
                                                                     params_dict['g_tes_b'])
                    Y_model_fs_m = 1.0 / Z_model_fs_m
                    valid_Y = np.isfinite(Y_model_fs_m)
                    res_re, res_im, res_mag = np.full_like(fs_m, np.nan), np.full_like(fs_m, np.nan), np.full_like(fs_m, np.nan)
                    res_re[valid_Y] = adm_re_m[valid_Y] - np.real(Y_model_fs_m[valid_Y])
                    res_im[valid_Y] = adm_im_m[valid_Y] - np.imag(Y_model_fs_m[valid_Y])
                    res_mag[valid_Y] = adm_mag_m[valid_Y] - np.abs(Y_model_fs_m[valid_Y])
                    lbl_res = f'Vb={bv_plot:.3f}V' if i < 2 else None
                    axs_res_plot[0].semilogx(fs_m, res_re, 'o-', c=color, ms=3, alpha=0.7, lw=1, label=lbl_res)
                    axs_res_plot[1].semilogx(fs_m, res_im, 'o-', c=color, ms=3, alpha=0.7, lw=1)
                    axs_res_plot[2].semilogx(fs_m, res_mag, 'o-', c=color, ms=3, alpha=0.7, lw=1)
                titles_res = ["Residual Re(Y)", "Residual Im(Y)", "Residual |Y|"]
                ylabels_res = ["ΔRe(Y) (S)", "ΔIm(Y) (S)", "Δ|Y| (S)"]
                for i_ax, ax_res_loc in enumerate(axs_res_plot):
                    ax_res_loc.set_title(titles_res[i_ax])
                    ax_res_loc.set_ylabel(ylabels_res[i_ax])
                    ax_res_loc.grid(True, which="both", ls=':')
                axs_res_plot[2].set_xlabel("Freq (Hz)")
                if bias_fitted_plot_res:
                    axs_res_plot[0].legend(fontsize='x-small', loc='best')
            if args.save_plots:
                fig_res.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_06_Fit_Residuals.png")
            plt.close(fig_res)

        # Plot 7: Alpha and Beta vs R0/Rn
        if fit_results and Rnormal is not None:
            bias_p_ab = sorted(fit_results.keys())
            R0_norm_ab = [fit_results[b]['R0'] / Rnormal for b in bias_p_ab]
            alpha_values_plot = np.array([fit_results[b]['alpha'] for b in bias_p_ab])
            alpha_errors_plot = np.array([fit_results[b].get('alpha_err', 0) for b in bias_p_ab])
            alpha_errors_plot = np.array([0 if e is None or e < 0 or not np.isfinite(e) else e for e in alpha_errors_plot])
            beta_values_plot = np.array([fit_results[b]['beta'] for b in bias_p_ab])
            beta_errors_plot = np.array([fit_results[b].get('beta_err', 0) for b in bias_p_ab])
            beta_errors_plot = np.array([0 if e is None or e < 0 or not np.isfinite(e) else e for e in beta_errors_plot])

            fig_ab, axs_ab_plot = plt.subplots(2, 1, figsize=(10, 8), sharex=True, tight_layout=True)
            fig_ab.suptitle(
                f"Fitted Alpha & Beta vs R0/Rn: Ch {CHANNEL_ID}, Px {pixel_number_for_dir} (n_exp={args.n_thermal})",
                fontsize=14)
            axs_ab_plot[0].errorbar(R0_norm_ab, alpha_values_plot, yerr=alpha_errors_plot, fmt='o', capsize=4,
                                    elinewidth=1.5, markersize=5, ecolor='gray', mec='blue', mfc='lightblue')
            axs_ab_plot[0].set_ylabel("Alpha (α)")
            axs_ab_plot[0].grid(True, ls=':')
            axs_ab_plot[1].errorbar(R0_norm_ab, beta_values_plot, yerr=beta_errors_plot, fmt='o', capsize=4,
                                    elinewidth=1.5, markersize=5, ecolor='gray', mec='green', mfc='lightgreen')
            axs_ab_plot[1].set_ylabel("Beta (β)")
            axs_ab_plot[1].set_xlabel("TES Resistance (R0/Rn)")
            axs_ab_plot[1].grid(True, ls=':')
            if args.save_plots:
                fig_ab.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_07_AlphaBeta_vs_R0Rn.png")
            plt.close(fig_ab)

        # Plot 8: Other parameters vs R0/Rn
        if fit_results and Rnormal is not None:
            bias_p_other_plot = sorted(fit_results.keys())
            R0_norm_other_plot = [fit_results[b]['R0'] / Rnormal for b in bias_p_other_plot]
            params_other_plot = ['C_1', 'C_tes', 'g_tes_1']
            labels_other_plot = ['C_1 (J/K)', 'C_tes (J/K)', 'g_tes_1 (W/K)']
            colors_other_plot = ['purple', 'orange', 'brown']
            fig_other, axs_other_plot = plt.subplots(len(params_other_plot), 1,
                                                     figsize=(10, 2.7 * len(params_other_plot)), sharex=True,
                                                     tight_layout=True)
            if len(params_other_plot) == 1:
                axs_other_plot = [axs_other_plot]
            fig_other.suptitle(
                f"Other Fitted Parameters vs R0/Rn: Ch {CHANNEL_ID}, Px {pixel_number_for_dir} (n_exp={args.n_thermal})",
                fontsize=14)
            for i, param_name in enumerate(params_other_plot):
                values = [fit_results[b][param_name] for b in bias_p_other_plot]
                errors = [fit_results[b].get(f"{param_name}_err", 0) for b in bias_p_other_plot]
                errors = [0 if e is None or e < 0 or not np.isfinite(e) else e for e in errors]
                axs_other_plot[i].errorbar(R0_norm_other_plot, values, yerr=errors, fmt='o', capsize=4, elinewidth=1.5,
                                           markersize=5, ecolor='gray', mec=colors_other_plot[i], mfc=plt.cm.Pastel1(i))
                axs_other_plot[i].set_ylabel(labels_other_plot[i])
                axs_other_plot[i].grid(True, ls=':')
            axs_other_plot[-1].set_xlabel("TES Resistance (R0/Rn)")
            if args.save_plots:
                fig_other.savefig(pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_08_OtherParams_vs_R0Rn.png")
            plt.close(fig_other)

        A_fit_powerlaw, n_fit_powerlaw = np.nan, np.nan
        if fit_results:
            alphas_all_plot = np.array([fit_results[b]['alpha'] for b in fit_results.keys()])
            alpha_errors_all_plot = np.array([fit_results[b].get('alpha_err', 0) for b in fit_results.keys()])
            alpha_errors_all_plot = np.array([0 if e is None or e < 0 or not np.isfinite(e) else e for e in alpha_errors_all_plot])
            betas_all_plot = np.array([fit_results[b]['beta'] for b in fit_results.keys()])
            beta_errors_all_plot = np.array([fit_results[b].get('beta_err', 0) for b in fit_results.keys()])
            beta_errors_all_plot = np.array([0 if e is None or e < 0 or not np.isfinite(e) else e for e in beta_errors_all_plot])
            valid_ab_mask_plot = np.isfinite(alphas_all_plot) & np.isfinite(betas_all_plot) & (betas_all_plot > 0) & (
                        alphas_all_plot > 20) & np.isfinite(alpha_errors_all_plot)
            alphas_clean_plot = alphas_all_plot[valid_ab_mask_plot]
            betas_clean_plot = betas_all_plot[valid_ab_mask_plot]
            alpha_errors_clean_plot = alpha_errors_all_plot[valid_ab_mask_plot]
            beta_errors_clean_plot = beta_errors_all_plot[valid_ab_mask_plot]
            sigma_for_fit_plot = alpha_errors_clean_plot.copy()
            sigma_for_fit_plot[sigma_for_fit_plot <= 1e-9] = 1.0

            if len(alphas_clean_plot) > 1 and len(betas_clean_plot) > 1:
                fig_alpha_beta = plt.figure(figsize=(8, 7))
                plt.errorbar(betas_clean_plot, alphas_clean_plot, yerr=alpha_errors_clean_plot,
                             xerr=beta_errors_clean_plot,
                             fmt='o', label='Data', color='teal', markersize=6, capsize=3, elinewidth=1, ecolor='gray')
                try:
                    popt_ab, pcov_ab = scipy_curve_fit(power_law_model, betas_clean_plot, alphas_clean_plot,
                                                       p0=[np.nanmax(alphas_clean_plot) if len(alphas_clean_plot) > 0 else 100, 1.0],
                                                       sigma=sigma_for_fit_plot, absolute_sigma=True)
                    A_fit_powerlaw, n_fit_powerlaw = popt_ab[0], popt_ab[1]
                    if len(betas_clean_plot) > 0 and min(betas_clean_plot) > 0:
                        beta_fit_line = np.logspace(np.log10(min(betas_clean_plot) * 0.8),
                                                    np.log10(max(betas_clean_plot) * 1.2), 100)
                        alpha_fit_line = power_law_model(beta_fit_line, A_fit_powerlaw, n_fit_powerlaw)
                        plt.loglog(beta_fit_line, alpha_fit_line, '-', color='crimson', label='Fit')
                        fit_text = f'α = {round(A_fit_powerlaw)} $\\cdot$ β$^{{{n_fit_powerlaw:.2f}}}$'
                        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, fontsize=12,
                                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
                except Exception as e:
                    print(f"Could not fit or plot Alpha vs Beta power law for Ch {CHANNEL_ID}: {e}")
                plt.xlabel(r"$\beta$")
                plt.ylabel(r"$\alpha$")
                plt.title(fr"$\alpha$ vs. $\beta$ for Ch {CHANNEL_ID}, Px {pixel_number_for_dir}")
                plt.loglog()
                plt.legend()
                plt.grid(True, which="both", ls=':')
                plt.tight_layout()
                if args.save_plots:
                    fig_alpha_beta.savefig(
                        pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_09_Alpha_vs_Beta_Fit.png")
                plt.close(fig_alpha_beta)
            else:
                print(f"Not enough valid (alpha, beta) points to create Alpha vs Beta plot or fit for Ch {CHANNEL_ID}.")

        if fit_results and args.save_plots:
            fit_data_list = []
            for bias_v, params in fit_results.items():
                row = {'Bias_Voltage (V)': bias_v,
                       'R0 (Ohm)': params['R0'],
                       'R0/Rn': params['R0'] / Rnormal if Rnormal is not None and Rnormal > 0 else np.nan,
                       'P0 (W)': params['P0'],
                       'T0_model (K)': params['T0_model'],
                       'g_tes_b (W/K)': params['g_tes_b'],
                       'alpha': params['alpha'], 'alpha_err': params.get('alpha_err', np.nan),
                       'beta': params['beta'], 'beta_err': params.get('beta_err', np.nan),
                       'C_1 (J/K)': params['C_1'], 'C_1_err': params.get('C_1_err', np.nan),
                       'C_tes (J/K)': params['C_tes'], 'C_tes_err': params.get('C_tes_err', np.nan),
                       'g_tes_1 (W/K)': params['g_tes_1'], 'g_tes_1_err': params.get('g_tes_1_err', np.nan),
                       'Alpha_Beta_PowerLaw_A': A_fit_powerlaw,
                       'Alpha_Beta_PowerLaw_n_exp': n_fit_powerlaw
                       }
                fit_data_list.append(row)

            fit_df_channel = pd.DataFrame(fit_data_list)
            csv_filepath = pixel_output_dir / f"Ch{CHANNEL_ID}_Px{pixel_number_for_dir}_FitParameters.csv"
            try:
                fit_df_channel.to_csv(csv_filepath, index=False, float_format='%.5e')
                print(f"Fit parameters for Ch {CHANNEL_ID} saved to: {csv_filepath}")
            except Exception as e:
                print(f"Error saving fit parameters to CSV for Ch {CHANNEL_ID}: {e}")
        else:
            print(f"No fit results to save to CSV for Ch {CHANNEL_ID}.")

    # Show plots if not saving
    if not args.save_plots:
        plt.show()
    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    main()
