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
    rbias: float = 2000.0,
    channel_id: int = 63,
    rshunt: float = 250e-6,
    correct_shift: bool = True,
) -> Dict[str, Any]:
    """
    Loads an IV curve (.npz) and extracts Vbias, Ites, Rtes, and normal resistance.
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

    # Determine Normal Resistance Rnormal
    rnormal = None
    pos_indices = np.where((rtes_full > 50e-6) & np.isfinite(rtes_full))[0]
    if len(pos_indices) > 0:
        vbias_pos = vbias_full[pos_indices]
        ites_pos = ites_full[pos_indices]
        rtes_pos = rtes_full[pos_indices]
        min_ites_idx = np.argmin(np.abs(ites_pos))
        rnormal = float(rtes_pos[min_ites_idx])
    else:
        valid_r = rtes_full[np.isfinite(rtes_full)]
        rnormal = float(np.max(valid_r)) if len(valid_r) > 0 else 0.003

    def lookup_op(vbias_target: float) -> Dict[str, float]:
        idx = np.argmin(np.abs(vbias_full - vbias_target))
        vb_match = float(vbias_full[idx])
        ites_match = float(ites_full[idx])
        rtes_match = float(rtes_full[idx])
        if not np.isfinite(rtes_match) or rtes_match <= 0:
            rtes_match = max(float(rnormal * (vb_match / max(vbias_full[0], 1e-3))), 1e-5)
        p0_match = float(ites_match**2 * rtes_match)
        rn_pct = float((rtes_match / rnormal) * 100.0) if rnormal > 0 else 0.0
        return {
            "vbias": vb_match,
            "ites": ites_match,
            "rtes": rtes_match,
            "p0": p0_match,
            "rn_pct": rn_pct,
            "rnormal": rnormal,
        }

    return {
        "vbias": vbias_full,
        "ites": ites_full,
        "ibias": ibias_full,
        "rtes": rtes_full,
        "rnormal": rnormal,
        "channel_id": channel_id,
        "lookup_op": lookup_op,
    }


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

    # 2. Process all non-zero bias points
    R_shunt = rshunt_uohm * 1e-6
    L_nyq = lnyq_nh * 1e-9

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

            # Calculate complex impedance
            Z_circuit = 1j * 2.0 * np.pi * fs_bias * L_nyq + R_shunt
            ratio_I = u_bias / u0_fft
            safe_ratio = np.where(np.abs(ratio_I) > 1e-12, ratio_I, np.nan + 1j * np.nan)
            Z_tes = Z_circuit * (1.0 / safe_ratio - 1.0)

            valid_mask = np.isfinite(Z_tes)

            # Flip polarity if requested or if Re(Z) is negative at DC
            is_flip = flip_bias or bool(data_bias.get("flip_bias", False))
            if is_flip or (np.sum(valid_mask) > 0 and np.real(Z_tes[valid_mask][0]) < 0):
                Z_tes = -Z_tes

            bias_points_dict[vb_val] = {
                "vbias": vb_val,
                "folder": bf,
                "file": npz_files[0],
                "f": fs_bias,
                "z_tes": Z_tes,
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
        "bias_points": bias_points_dict,
        "bias_voltages": bias_voltages,
        "num_biases": len(bias_voltages),
    }
