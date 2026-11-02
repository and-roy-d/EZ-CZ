"""
data_io.py: Robust I/O and processing utilities for Complex Impedance (CZ) sweeps and IV curves.
"""

import os
import re
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import scipy.constants
from scipy.fft import rfft, irfft, rfftfreq

PHI0 = scipy.constants.value("mag. flux quantum")


def convert_ang2_to_ites(ang2_data: np.ndarray) -> np.ndarray:
    """Converts DASTARD angular demodulation units to TES current in Amperes."""
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / PHI0
    arbs_per_phi0 = 1
    amp_per_arb = 1.0 / (min_phi0_per_amp * arbs_per_phi0)
    return ang2_data * amp_per_arb


def arb_to_amp(in_val: np.ndarray) -> np.ndarray:
    """Converts 12-bit raw DAQ counts to TES current in Amperes."""
    min_SI = 248e-12
    min_phi0_per_amp = min_SI / PHI0
    arbs_per_phi0 = 4096
    amp_per_arb = 1.0 / (min_phi0_per_amp * arbs_per_phi0)
    return in_val * amp_per_arb


def convert_vbias_to_ibias(vbias: np.ndarray, rbias: float) -> np.ndarray:
    """Converts room-temperature DAC voltage to bias current in Amperes."""
    return vbias / rbias


def Rtes(ibias: np.ndarray, ites: np.ndarray, Rshunt: float = 250e-6) -> np.ndarray:
    """Calculates TES resistance from bias and TES currents in Ohms."""
    ites_safe = np.where(np.abs(ites) < 1e-18, np.nan, ites)
    rtes = Rshunt * (ibias - ites_safe) / ites_safe
    return rtes


def load_iv_curve(
    iv_file_path: Path,
    rbias: float = 1000.0,
    channel_id: int = 63,
    rshunt: float = 250e-6,
    correct_shift: bool = True,
    manual_rn: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Loads an IV curve (.npz) and extracts Vbias, Ites, Rtes, and normal resistance.
    Normal resistance Rn is calculated from the superconducting and normal slope ratio:
        Rn = Rshunt * (slope_sc / slope_n - 1)
    or taken from manual_rn if provided.
    """
    p = Path(iv_file_path)
    if not p.exists():
        raise FileNotFoundError(f"IV file not found at: {p}")

    data = np.load(p, allow_pickle=True)
    if "vb" not in data or "ang2" not in data:
        raise ValueError(f"Missing 'vb' or 'ang2' in IV file: {p}")

    vbias_full = np.asarray(data["vb"], dtype=float)
    ang2_full = np.asarray(data["ang2"], dtype=float)

    if ang2_full.ndim == 2:
        if channel_id >= ang2_full.shape[1]:
            ch_idx = 0
        else:
            ch_idx = channel_id
        ang2_ch = ang2_full[:, ch_idx]
    else:
        ang2_ch = ang2_full

    ites_full = convert_ang2_to_ites(ang2_ch)
    if correct_shift and len(vbias_full) > 0:
        zero_idx = np.argmin(np.abs(vbias_full))
        ites_offset = ites_full[zero_idx]
        ites_full = ites_full - ites_offset

    ibias_full = convert_vbias_to_ibias(vbias_full, rbias)
    rtes_full = Rtes(ibias_full, ites_full, Rshunt=rshunt)

    # Determine Normal Resistance from slope ratio:
    # Rn = Rshunt * (slope_sc / slope_n - 1)
    slope_sc = None
    slope_n = None
    rn_calc = None

    if len(vbias_full) >= 10:
        sort_order = np.argsort(vbias_full)
        vb_s = vbias_full[sort_order]
        it_s = ites_full[sort_order]

        # 1. Superconducting slope (low positive Vbias up to peak current)
        max_idx = int(np.argmax(it_s))
        vb_peak = float(vb_s[max_idx]) if max_idx > 0 else 0.2
        sc_mask = (vb_s > 0.002) & (vb_s < max(vb_peak * 0.7, 0.05))
        if np.sum(sc_mask) >= 3:
            p_sc = np.polyfit(vb_s[sc_mask], it_s[sc_mask], 1)
            slope_sc = float(p_sc[0])
        elif vb_peak > 0:
            slope_sc = float(it_s[max_idx] / vb_peak)

        # 2. Normal slope (high Vbias ohmic branch)
        v_max = float(vb_s.max())
        v_norm_min = max(v_max * 0.65, 2.0)
        norm_mask = vb_s >= v_norm_min
        if np.sum(norm_mask) >= 3:
            p_norm = np.polyfit(vb_s[norm_mask], it_s[norm_mask], 1)
            slope_n = float(p_norm[0])
        else:
            norm_mask = vb_s >= (v_max * 0.5)
            if np.sum(norm_mask) >= 3:
                p_norm = np.polyfit(vb_s[norm_mask], it_s[norm_mask], 1)
                slope_n = float(p_norm[0])

        if slope_sc is not None and slope_n is not None and slope_n > 0 and slope_sc > slope_n:
            ratio = slope_sc / slope_n
            rn_calc = float(rshunt * (ratio - 1.0))

    if manual_rn is not None and manual_rn > 0:
        rnormal = float(manual_rn)
    elif rn_calc is not None and rn_calc > 0:
        rnormal = float(rn_calc)
    else:
        # Fallback if IV curve slope calculation fails to find Rn
        valid_r = rtes_full[np.isfinite(rtes_full) & (rtes_full > 0)]
        rnormal = float(np.max(valid_r)) if len(valid_r) > 0 else 0.025

    out_dict: Dict[str, Any] = {
        "vbias": vbias_full,
        "ites": ites_full,
        "ibias": ibias_full,
        "rtes": rtes_full,
        "rnormal": rnormal,
        "channel_id": channel_id,
        "rbias": rbias,
        "slope_sc": slope_sc,
        "slope_n": slope_n,
        "rn_calculated": rn_calc,
    }

    def lookup_op(vbias_target: float) -> Dict[str, float]:
        idx = np.argmin(np.abs(vbias_full - vbias_target))
        vb_match = float(vbias_full[idx])
        ites_match = float(ites_full[idx])
        rtes_match = float(rtes_full[idx])
        rn_curr = float(out_dict.get("rnormal", rnormal))
        if not np.isfinite(rtes_match) or rtes_match <= 0:
            rtes_match = max(float(rn_curr * (vb_match / max(vbias_full[0], 1e-3))), 1e-5)
        p0_match = float(ites_match**2 * rtes_match)
        rn_pct = float((rtes_match / rn_curr) * 100.0) if (rn_curr and rn_curr > 0) else 0.0
        return {
            "vbias": vb_match,
            "ites": ites_match,
            "rtes": rtes_match,
            "p0": p0_match,
            "rn_pct": rn_pct,
            "rnormal": rn_curr,
        }

    out_dict["lookup_op"] = lookup_op
    return out_dict


def _extract_channel_fft(
    data_dict: Any,
    channel_id: int,
    fs_sch: np.ndarray,
    tau_shift: float = 0.0,
    fs_global: float = 250000.0,
) -> Tuple[Optional[np.ndarray], bool, Optional[int]]:
    """
    Extracts complex tone FFT array for the requested channel from a CZ dataset npz.
    Returns: (u_fft, is_multichannel, detected_channel_id)
    """
    candidate_keys = [
        f"chan{channel_id}",
        f"ch{channel_id}",
        f"channel_{channel_id}",
        f"channel{channel_id}",
        str(channel_id),
        channel_id,
    ]

    # 1. Multi-channel dictionary: fft_real and fft_imag
    if "fft_real" in data_dict and "fft_imag" in data_dict:
        re_obj = data_dict["fft_real"].item() if (hasattr(data_dict["fft_real"], "ndim") and data_dict["fft_real"].ndim == 0) else data_dict["fft_real"]
        im_obj = data_dict["fft_imag"].item() if (hasattr(data_dict["fft_imag"], "ndim") and data_dict["fft_imag"].ndim == 0) else data_dict["fft_imag"]
        if isinstance(re_obj, dict):
            matched_key = None
            for k in candidate_keys:
                if k in re_obj and (im_obj is None or k in im_obj):
                    matched_key = k
                    break
            if matched_key is not None:
                return np.asarray(re_obj[matched_key], dtype=float) + 1j * np.asarray(im_obj[matched_key], dtype=float), True, channel_id
            return None, True, None

    # 2. fft_unroll (dict, 2D array, or 1D array)
    if "fft_unroll" in data_dict:
        fft_val = data_dict["fft_unroll"]
        if hasattr(fft_val, "ndim") and fft_val.ndim == 0 and isinstance(fft_val.item(), dict):
            d = fft_val.item()
            matched_key = None
            for k in candidate_keys:
                if k in d:
                    matched_key = k
                    break
            if matched_key is not None:
                return np.asarray(d[matched_key], dtype=complex), True, channel_id
            return None, True, None
        elif hasattr(fft_val, "ndim") and fft_val.ndim == 2:
            sh = fft_val.shape
            if sh[0] > sh[1]:
                # (freqs, channels)
                if channel_id < sh[1]:
                    return np.asarray(fft_val[:, channel_id], dtype=complex), True, channel_id
                return None, True, None
            else:
                # (channels, freqs)
                if channel_id < sh[0]:
                    return np.asarray(fft_val[channel_id, :], dtype=complex), True, channel_id
                return None, True, None
        elif hasattr(fft_val, "ndim") and fft_val.ndim == 1:
            det_ch = None
            for k in ("channel", "channel_id", "chan"):
                if k in data_dict:
                    try:
                        det_ch = int(data_dict[k])
                        break
                    except Exception:
                        pass
            return np.asarray(fft_val, dtype=complex), False, det_ch

    # 3. Multi-channel raw demodulation: 'data' dict
    if "data" in data_dict:
        data_obj = data_dict["data"].item() if (hasattr(data_dict["data"], "ndim") and data_dict["data"].ndim == 0) else data_dict["data"]
        if isinstance(data_obj, dict):
            matched_key = None
            for k in candidate_keys:
                if k in data_obj:
                    matched_key = k
                    break
            if matched_key is not None:
                raw_ch = np.asarray(data_obj[matched_key], dtype=float)
                raw_amps = arb_to_amp(raw_ch)
                n_pts = len(raw_amps)
                freqs_fft = rfftfreq(n_pts, 1.0 / fs_global)
                fft_dc = rfft(raw_amps - np.mean(raw_amps)) * np.exp(-2j * np.pi * freqs_fft * tau_shift)
                u_fft = np.array([fft_dc[np.argmin(np.abs(freqs_fft - f))] for f in fs_sch])
                return u_fft, True, channel_id
            return None, True, None

    # 4. Single-channel raw_data demodulation
    if "raw_data" in data_dict:
        fs_sample = float(data_dict.get("fs_global", fs_global))
        raw_amps = arb_to_amp(np.asarray(data_dict["raw_data"], dtype=float))
        n_pts = len(raw_amps)
        freqs_fft = rfftfreq(n_pts, 1.0 / fs_sample)
        tau = float(data_dict.get("tau_shift", tau_shift))
        fft_dc = rfft(raw_amps - np.mean(raw_amps)) * np.exp(-2j * np.pi * freqs_fft * tau)
        u_fft = np.array([fft_dc[np.argmin(np.abs(freqs_fft - f))] for f in fs_sch])
        det_ch = None
        for k in ("channel", "channel_id", "chan"):
            if k in data_dict:
                try:
                    det_ch = int(data_dict[k])
                    break
                except Exception:
                    pass
        return u_fft, False, det_ch

    return None, False, None


def load_cz_sweep_dataset(
    raw_dir: Path,
    channel_id: int = 63,
    rshunt_uohm: float = 250.0,
    lnyq_nh: Optional[float] = None,
    flip_bias: bool = True,
    sc_ref_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Scans a raw Complex Z measurement directory, locates the zero-bias superconducting
    reference (or uses an external SC reference if provided), and computes complex impedance
    Z_tes(f) for every bias point for the specified channel.
    """
    base_dir = Path(raw_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Raw Complex Z directory not found: {base_dir}")

    bias_folders = sorted(base_dir.glob("bias_*"))
    if not bias_folders:
        raise ValueError(f"No bias_* subfolders found in {base_dir}")

    # 1. Locate 0V superconducting reference file
    ref_0v_npz: Optional[Path] = None
    ref_0v_folder: Optional[Path] = None

    if sc_ref_path:
        sc_p = Path(sc_ref_path)
        if sc_p.is_file() and sc_p.suffix == ".npz":
            ref_0v_npz = sc_p
            ref_0v_folder = sc_p.parent
        elif sc_p.is_dir():
            ref_0v_folder = sc_p
            ch_candidates = [
                f"*chan{channel_id}*.npz",
                f"*ch{channel_id}*.npz",
                f"*channel_{channel_id}*.npz",
                f"*{channel_id}*.npz",
            ]
            for pat in ch_candidates:
                matched = sorted(sc_p.glob(pat))
                if matched:
                    ref_0v_npz = matched[0]
                    break
            if ref_0v_npz is None:
                all_npzs = sorted(sc_p.glob("*.npz"))
                if all_npzs:
                    ref_0v_npz = all_npzs[0]

    # If no external sc_ref specified or not found, search in base_dir
    if ref_0v_npz is None:
        for bf in bias_folders:
            try:
                vb_val = float(bf.name.split("_", 1)[-1].replace("v", "."))
                if np.isclose(vb_val, 0.0, atol=1e-3):
                    ref_0v_folder = bf
                    break
            except Exception:
                continue

        if ref_0v_folder is not None:
            ch_candidates = [
                f"*chan{channel_id}*.npz",
                f"*ch{channel_id}*.npz",
                f"*channel_{channel_id}*.npz",
                f"*{channel_id}*.npz",
            ]
            for pat in ch_candidates:
                matched = sorted(ref_0v_folder.glob(pat))
                if matched:
                    ref_0v_npz = matched[0]
                    break
            if ref_0v_npz is None:
                all_npzs = sorted(ref_0v_folder.glob("*.npz"))
                if all_npzs:
                    ref_0v_npz = all_npzs[0]

    if ref_0v_npz is None or not ref_0v_npz.exists():
        raise ValueError(f"Could not locate 0V superconducting reference file in {base_dir}")

    # Load 0V reference tones and FFT
    data_0v = np.load(ref_0v_npz, allow_pickle=True)
    if "fs_sch" not in data_0v:
        raise ValueError(f"Missing 'fs_sch' frequencies in {ref_0v_npz}")

    fs_sch_ref = np.asarray(data_0v["fs_sch"], dtype=float)
    tau_shift_0v = float(data_0v.get("tau_shift", 0.0))
    fs_global_0v = float(data_0v.get("fs_global", 250000.0))

    u0_fft, is_multichannel, detected_channel = _extract_channel_fft(
        data_0v, channel_id, fs_sch_ref, tau_shift=tau_shift_0v, fs_global=fs_global_0v
    )
    is_channel_mismatch = (u0_fft is None)

    # Extract or predict f_roll and L_nyq from the SC transfer function (hardware-agnostic)
    f_roll_0v = None
    if u0_fft is not None:
        if "f_roll" in data_0v and not is_multichannel:
            try:
                f_roll_0v = float(data_0v["f_roll"])
            except Exception:
                f_roll_0v = None

        if f_roll_0v is None:
            try:
                from scipy.optimize import curve_fit
                y_mag = np.abs(u0_fft)
                valid = np.isfinite(fs_sch_ref) & np.isfinite(y_mag) & (fs_sch_ref > 0) & (y_mag > 0)
                if np.sum(valid) > 4:
                    def _mag_lorentz(f, a0, fr):
                        return a0 / np.sqrt(1.0 + (f / fr)**2)
                    popt, _ = curve_fit(
                        _mag_lorentz,
                        fs_sch_ref[valid],
                        y_mag[valid],
                        p0=[float(np.max(y_mag[valid])), 1000.0],
                        bounds=([0.0, 1.0], [np.inf, 1e6])
                    )
                    f_roll_0v = float(popt[1])
            except Exception:
                pass

    predicted_lnyq_nh = None
    if f_roll_0v is not None and f_roll_0v > 0:
        # L = R_shunt / (2 * pi * f_roll)
        predicted_lnyq_nh = (rshunt_uohm * 1e3) / (2.0 * np.pi * f_roll_0v)

    if lnyq_nh is not None:
        lnyq_val_nh = float(lnyq_nh)
    elif predicted_lnyq_nh is not None:
        lnyq_val_nh = float(predicted_lnyq_nh)
    else:
        lnyq_val_nh = 100.0

    R_shunt = rshunt_uohm * 1e-6
    L_nyq = lnyq_val_nh * 1e-9

    bias_points_dict = {}
    bias_voltages = []

    for bf in bias_folders:
        try:
            vb_val = float(bf.name.split("_", 1)[-1].replace("v", "."))
        except Exception:
            continue

        if np.isclose(vb_val, 0.0, atol=1e-3):
            continue

        npz_files = list(bf.glob("*.npz"))
        if not npz_files:
            continue

        try:
            data_bias = np.load(npz_files[0], allow_pickle=True)
            fs_bias = np.asarray(data_bias.get("fs_sch", fs_sch_ref), dtype=float)
            tau_shift_bias = float(data_bias.get("tau_shift", tau_shift_0v))
            fs_global_bias = float(data_bias.get("fs_global", fs_global_0v))
            u_bias, _, _ = _extract_channel_fft(
                data_bias, channel_id, fs_bias, tau_shift=tau_shift_bias, fs_global=fs_global_bias
            )

            if u_bias is None or len(u_bias) != len(fs_bias):
                continue

            if u0_fft is not None:
                if len(u0_fft) != len(fs_bias) or not np.allclose(fs_sch_ref, fs_bias):
                    u0_re = np.interp(fs_bias, fs_sch_ref, np.real(u0_fft))
                    u0_im = np.interp(fs_bias, fs_sch_ref, np.imag(u0_fft))
                    u0_eff = u0_re + 1j * u0_im
                else:
                    u0_eff = u0_fft
            else:
                # Synthesize ideal single-pole SC response if u0_fft was missing
                u0_eff = 1.0 / (R_shunt + 1j * 2.0 * np.pi * fs_bias * L_nyq)

            # Calculate complex impedance:
            # Transfer function ratio: I_bias / I_0 = Z_circuit / Z_total
            # where Z_circuit = R_shunt + 1j * omega * L_nyq
            # and Z_total = Z_tes + R_shunt + 1j * omega * L_nyq = Z_tes + Z_circuit
            Z_circuit = 1j * 2.0 * np.pi * fs_bias * L_nyq + R_shunt
            ratio_I = u_bias / u0_eff
            safe_ratio = np.where(np.abs(ratio_I) > 1e-12, ratio_I, np.nan + 1j * np.nan)
            Z_total = Z_circuit / safe_ratio
            Z_tes = Z_total - Z_circuit

            valid_mask = np.isfinite(Z_tes)

            # Flip polarity if requested: ensure Re(Z) > 0 at DC
            is_flip = flip_bias or bool(data_bias.get("flip_bias", False))
            if is_flip and (np.sum(valid_mask) > 0 and np.real(Z_tes[valid_mask][0]) < 0):
                Z_tes = -Z_tes
                Z_total = Z_tes + Z_circuit

            Y_tes = 1.0 / np.where(np.abs(Z_tes) > 1e-15, Z_tes, np.nan + 1j * np.nan)
            Y_total = 1.0 / np.where(np.abs(Z_total) > 1e-15, Z_total, np.nan + 1j * np.nan)

            r0_dc = float(np.abs(np.real(Z_tes[valid_mask][0]))) if np.sum(valid_mask) > 0 else 0.001

            bias_points_dict[vb_val] = {
                "vbias": vb_val,
                "folder": bf,
                "file": npz_files[0],
                "f": fs_bias,
                "z_tes": Z_tes,
                "z_total": Z_total,
                "y_tes": Y_tes,
                "y_total": Y_total,
                "r0_dc": r0_dc,
                "u_bias": u_bias,
                "u0_fft": u0_eff,
                "safe_ratio": safe_ratio,
                "valid_mask": np.isfinite(Z_tes),
            }
            bias_voltages.append(vb_val)
        except Exception:
            continue

    bias_voltages.sort()

    return {
        "raw_dir": base_dir,
        "ref_0v_folder": ref_0v_folder,
        "ref_0v_npz": ref_0v_npz,
        "fs_sch": fs_sch_ref,
        "u0_fft": u0_fft,
        "f_roll_0v": f_roll_0v,
        "predicted_lnyq_nh": predicted_lnyq_nh,
        "is_multichannel": is_multichannel,
        "is_channel_mismatch": is_channel_mismatch,
        "detected_channel": detected_channel,
        "channel_id": channel_id,
        "rshunt_uohm": rshunt_uohm,
        "lnyq_nh": lnyq_val_nh,
        "bias_points": bias_points_dict,
        "bias_voltages": bias_voltages,
        "num_biases": len(bias_voltages),
    }


def recompute_dataset_circuit_impedance(
    dataset: dict,
    rshunt_uohm: float,
    lnyq_nh: float,
    flip_bias: bool = True
):
    """
    Dynamically recomputes Z_circuit, Z_total, Z_tes, Y_tes, Y_total across all bias points
    using updated R_shunt and L_nyq values:
        Z_circuit = R_shunt + 1j * 2*pi*f * L_nyq
        Z_total = Z_circuit / (u_bias / u0_fft)
        Z_tes = Z_total - Z_circuit
        Y_tes = 1 / Z_tes
        Y_total = 1 / Z_total
    """
    R_sh = rshunt_uohm * 1e-6
    L_val = lnyq_nh * 1e-9
    dataset["rshunt_uohm"] = rshunt_uohm
    dataset["lnyq_nh"] = lnyq_nh

    for vb, b_info in dataset.get("bias_points", {}).items():
        fs = b_info["f"]
        safe_ratio = b_info.get("safe_ratio")
        if safe_ratio is None:
            continue
        Z_circ = 1j * 2.0 * np.pi * fs * L_val + R_sh
        Z_tot = Z_circ / safe_ratio
        Z_t = Z_tot - Z_circ
        valid = np.isfinite(Z_t)
        if flip_bias and (np.sum(valid) > 0 and np.real(Z_t[valid][0]) < 0):
            Z_t = -Z_t
            Z_tot = Z_t + Z_circ

        b_info["z_tes"] = Z_t
        b_info["z_total"] = Z_tot
        b_info["y_tes"] = 1.0 / np.where(np.abs(Z_t) > 1e-15, Z_t, np.nan + 1j * np.nan)
        b_info["y_total"] = 1.0 / np.where(np.abs(Z_tot) > 1e-15, Z_tot, np.nan + 1j * np.nan)
        b_info["valid_mask"] = np.isfinite(Z_t)


def compute_lindeman_two_state_impedance(
    u_bias: np.ndarray,
    u_sc: np.ndarray,
    u_normal: np.ndarray,
    r_normal: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes TES complex impedance Z_tes(f) and equivalent circuit impedance Z_eq(f)
    using the empirical 2-state calibration method from Lindeman et al. (2007) [arXiv:0704.3066],
    Eq. (8) & Section III.

    Given a superconducting transfer function u_sc(f) and a normal-state transfer function
    u_normal(f) with normal resistance r_normal:
        Z_eq(f) = r_normal / ( (u_sc(f) / u_normal(f)) - 1 )
        Z_tes(f) = Z_eq(f) * ( (u_sc(f) / u_bias(f)) - 1 )

    Returns:
        (Z_tes, Z_eq)
    """
    ratio_norm = np.where(np.abs(u_normal) > 1e-15, u_sc / u_normal, np.nan + 1j * np.nan)
    denom_eq = ratio_norm - 1.0
    safe_denom_eq = np.where(np.abs(denom_eq) > 1e-12, denom_eq, np.nan + 1j * np.nan)
    Z_eq = r_normal / safe_denom_eq

    ratio_bias = np.where(np.abs(u_bias) > 1e-15, u_sc / u_bias, np.nan + 1j * np.nan)
    Z_tes = Z_eq * (ratio_bias - 1.0)
    return Z_tes, Z_eq
