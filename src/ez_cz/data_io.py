"""
data_io.py: Robust I/O and processing utilities for Complex Impedance (CZ) sweeps and IV curves.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
        # Fallback if IV curve fails to find Rn (~25 mOhm for channels >= 32)
        rnormal = 0.025 if channel_id >= 32 else 0.003

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


def load_cz_sweep_dataset(
    raw_dir: Path,
    channel_id: int = 63,
    rshunt_uohm: float = 250.0,
    lnyq_nh: float = 74.8,
    flip_bias: bool = True,
) -> Dict[str, Any]:
    """
    Scans a raw Complex Z measurement directory, locates the zero-bias superconducting
    reference, and computes complex impedance Z_tes(f) for every bias point.
    """
    base_dir = Path(raw_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Raw Complex Z directory not found: {base_dir}")

    bias_folders = sorted(base_dir.glob("bias_*"))
    if not bias_folders:
        raise ValueError(f"No bias_* subfolders found in {base_dir}")

    # 1. Identify 0V superconducting reference folder
    ref_0v_folder = None
    for bf in bias_folders:
        try:
            vb_val = float(bf.name.split("_", 1)[-1].replace("v", "."))
            if np.isclose(vb_val, 0.0, atol=1e-3):
                ref_0v_folder = bf
                break
        except Exception:
            continue

    if ref_0v_folder is None:
        raise ValueError(f"Could not locate 0V superconducting reference folder (bias_0v00) in {base_dir}")

    # Load 0V reference tones and FFT
    npz_0v_files = list(ref_0v_folder.glob("*.npz"))
    if not npz_0v_files:
        raise ValueError(f"No .npz file in reference folder {ref_0v_folder}")

    data_0v = np.load(npz_0v_files[0], allow_pickle=True)
    if "fs_sch" not in data_0v:
        raise ValueError(f"Missing 'fs_sch' frequencies in {npz_0v_files[0]}")

    fs_sch_ref = np.asarray(data_0v["fs_sch"], dtype=float)
    u0_fft = None
    if "fft_unroll" in data_0v:
        u0_fft = np.asarray(data_0v["fft_unroll"], dtype=complex)
    elif "raw_data" in data_0v:
        # Fallback demodulation
        fs_sample = float(data_0v.get("fs_global", 250000.0))
        raw_amps = arb_to_amp(np.asarray(data_0v["raw_data"], dtype=float))
        n_pts = len(raw_amps)
        freqs_fft = rfftfreq(n_pts, 1.0 / fs_sample)
        tau_shift = float(data_0v.get("tau_shift", 0.0))
        fft_dc = rfft(raw_amps - np.mean(raw_amps)) * np.exp(-2j * np.pi * freqs_fft * tau_shift)
        u0_fft = np.array([fft_dc[np.argmin(np.abs(freqs_fft - f))] for f in fs_sch_ref])
    else:
        raise ValueError(f"Neither 'fft_unroll' nor 'raw_data' present in {npz_0v_files[0]}")

    # 1.5. Extract or predict f_roll and L_nyq from 0V SC transfer function
    f_roll_0v = None
    if "f_roll" in data_0v:
        try:
            f_roll_0v = float(data_0v["f_roll"])
        except Exception:
            f_roll_0v = None

    if f_roll_0v is None and u0_fft is not None:
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
        except Exception as e:
            logger.warning(f"Could not fit f_roll from 0V reference data: {e}")

    predicted_lnyq_nh = None
    if f_roll_0v is not None and f_roll_0v > 0:
        # L = R_shunt / (2 * pi * f_roll)
        predicted_lnyq_nh = (rshunt_uohm * 1e3) / (2.0 * np.pi * f_roll_0v)

    # 2. Process all non-zero bias points
    R_shunt = rshunt_uohm * 1e-6
    L_nyq = (predicted_lnyq_nh * 1e-9) if predicted_lnyq_nh else (lnyq_nh * 1e-9)

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
            u_bias = None

            if "fft_unroll" in data_bias:
                u_bias = np.asarray(data_bias["fft_unroll"], dtype=complex)
            elif "raw_data" in data_bias:
                fs_sample = float(data_bias.get("fs_global", 250000.0))
                raw_amps = arb_to_amp(np.asarray(data_bias["raw_data"], dtype=float))
                n_pts = len(raw_amps)
                freqs_fft = rfftfreq(n_pts, 1.0 / fs_sample)
                tau_shift = float(data_bias.get("tau_shift", 0.0))
                fft_dc = rfft(raw_amps - np.mean(raw_amps)) * np.exp(-2j * np.pi * freqs_fft * tau_shift)
                u_bias = np.array([fft_dc[np.argmin(np.abs(freqs_fft - f))] for f in fs_bias])

            if u_bias is None or len(u_bias) != len(fs_bias):
                continue

            # Calculate complex impedance:
            # Transfer function ratio: I_bias / I_0 = Z_circuit / Z_total
            # where Z_circuit = R_shunt + 1j * omega * L_nyq
            # and Z_total = Z_tes + R_shunt + 1j * omega * L_nyq = Z_tes + Z_circuit
            Z_circuit = 1j * 2.0 * np.pi * fs_bias * L_nyq + R_shunt
            ratio_I = u_bias / u0_fft
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
                "u0_fft": u0_fft,
                "safe_ratio": safe_ratio,
                "valid_mask": np.isfinite(Z_tes),
            }
            bias_voltages.append(vb_val)
        except Exception as e:
            continue

    bias_voltages.sort()

    return {
        "raw_dir": base_dir,
        "ref_0v_folder": ref_0v_folder,
        "fs_sch": fs_sch_ref,
        "u0_fft": u0_fft,
        "f_roll_0v": f_roll_0v,
        "predicted_lnyq_nh": predicted_lnyq_nh,
        "rshunt_uohm": rshunt_uohm,
        "lnyq_nh": (predicted_lnyq_nh if predicted_lnyq_nh else lnyq_nh),
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
