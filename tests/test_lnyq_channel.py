import os
import sys
from pathlib import Path
import numpy as np
import pytest

from ez_cz.data_io import load_cz_sweep_dataset, recompute_dataset_circuit_impedance, compute_lindeman_two_state_impedance
from ez_cz.models import get_model, SimpleModel


def test_lnyq_physics_formula():
    """Verify L = R_shunt / (2*pi*f_roll) relationship."""
    rshunt_uohm = 250.0  # 250 uOhm

    # Nominal 119.13 nH -> f_roll ~ 334 Hz
    f_roll_1 = 333.988  # Hz
    l_1 = (rshunt_uohm * 1e3) / (2.0 * np.pi * f_roll_1)
    assert abs(l_1 - 119.13) < 0.1

    # Nominal 220 nH -> f_roll ~ 180.85 Hz
    expected_froll_2 = (rshunt_uohm * 1e3) / (2.0 * np.pi * 220.0)
    assert abs(expected_froll_2 - 180.85) < 0.1
    l_2 = (rshunt_uohm * 1e3) / (2.0 * np.pi * expected_froll_2)
    assert abs(l_2 - 220.0) < 0.01


def test_multichannel_sc_tf_channel_independence():
    """
    Verify that in multichannel datasets, each channel's SC transfer function is extracted
    independently and predicts that channel's own f_roll and L_nyq without hardcoded assumptions.
    """
    raw_dir = Path("/data/20260714/Complex_Z/20260714_130145")
    if not raw_dir.exists():
        pytest.skip("Multichannel dataset 20260714_130145 not available")

    # Channel 5: fitted f_roll ~ 717.2 Hz -> ~55.5 nH
    ds_ch5 = load_cz_sweep_dataset(raw_dir, channel_id=5, rshunt_uohm=250.0)
    assert ds_ch5["is_multichannel"] is True
    assert ds_ch5["is_channel_mismatch"] is False
    assert ds_ch5["f_roll_0v"] is not None
    assert abs(ds_ch5["f_roll_0v"] - 717.2) < 5.0
    assert abs(ds_ch5["predicted_lnyq_nh"] - 55.5) < 1.0

    # Channel 63: fitted f_roll ~ 1613.2 Hz -> ~24.7 nH
    ds_ch63 = load_cz_sweep_dataset(raw_dir, channel_id=63, rshunt_uohm=250.0)
    assert ds_ch63["is_multichannel"] is True
    assert ds_ch63["is_channel_mismatch"] is False
    assert ds_ch63["f_roll_0v"] is not None
    assert abs(ds_ch63["f_roll_0v"] - 1613.2) < 5.0
    assert abs(ds_ch63["predicted_lnyq_nh"] - 24.7) < 1.0

    # Channel 4: fitted f_roll ~ 770.3 Hz -> ~51.7 nH
    ds_ch4 = load_cz_sweep_dataset(raw_dir, channel_id=4, rshunt_uohm=250.0)
    assert abs(ds_ch4["predicted_lnyq_nh"] - 51.7) < 1.0

    # Channels have distinct inductances derived purely from their own SC transfer functions!
    assert abs(ds_ch5["predicted_lnyq_nh"] - ds_ch63["predicted_lnyq_nh"]) > 20.0

    # Non-existent channel in multichannel file returns None and mismatch is flagged
    ds_ch99 = load_cz_sweep_dataset(raw_dir, channel_id=99, rshunt_uohm=250.0)
    assert ds_ch99["is_channel_mismatch"] is True
    assert ds_ch99["predicted_lnyq_nh"] is None


def test_external_sc_reference_loading():
    """
    Verify that an external SC reference folder or file can be specified to predict
    channel-specific L_nyq for sweeps that do not have that channel's SC TF.
    """
    raw_dir_single = Path("/data/20261030/Complex_Z/20261030_171549")
    sc_ref_multi = Path("/data/20260714/Complex_Z/20260714_130145/bias_0v00")
    if not (raw_dir_single.exists() and sc_ref_multi.exists()):
        pytest.skip("Required test datasets not available")

    # Load single channel sweep for Channel 5 with external SC reference
    ds = load_cz_sweep_dataset(raw_dir_single, channel_id=5, rshunt_uohm=250.0, sc_ref_path=sc_ref_multi)
    assert ds["is_channel_mismatch"] is False
    assert abs(ds["predicted_lnyq_nh"] - 55.5) < 1.0
    assert ds["num_biases"] > 0


def test_gui_channel_and_sc_ref():
    """Verify offscreen GUI updates L_nyq from loaded channel SC reference."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication
    from ez_cz.gui import CZMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    gui = CZMainWindow()
    raw_dir = Path("/data/20260714/Complex_Z/20260714_130145")
    if not raw_dir.exists():
        pytest.skip("Dataset 20260714_130145 not available")

    gui.edit_analyze_raw_dir.setText(str(raw_dir))

    # Channel 5 on multichannel sweep
    gui.spin_analyze_channel.setValue(5)
    gui.on_load_analyze_sweep()
    assert abs(gui.current_model_params["L_nyq"].value - 55.5) < 1.5

    # Switch to Channel 63 on same multichannel sweep
    gui.spin_analyze_channel.setValue(63)
    assert abs(gui.current_model_params["L_nyq"].value - 24.7) < 1.5

    # Test GUI buttons Predict G and Predict Tc
    assert hasattr(gui, "btn_predict_g")
    assert hasattr(gui, "btn_predict_ttes")
    assert gui.btn_predict_g.text() == "Predict G"
    assert gui.btn_predict_ttes.text() == "Predict Tc"

    # Set up known operating point
    gui.spin_analyze_tc.setValue(45.0)
    gui.analyze_iv_data = {
        "lookup_op": lambda vb: {"p0": 7.27e-12, "rtes": 0.008, "ites": 30e-6, "rn_pct": 35.0}
    }
    gui.current_selected_bias = 0.60
    gui.current_model_params["n_mem"].value = 3.5
    gui.current_model_params["T_bath"].value = 0.020

    # Calculate G from Tc
    gui.on_predict_g_clicked()
    g_val_nw = gui.current_model_params["G_tes_bath"].value
    assert abs(g_val_nw - 0.6006) < 0.01  # ~0.601 nW/K = 600.6 pW/K

    # Calculate Tc from G
    gui.spin_analyze_tc.setValue(80.0)  # offset Tc
    gui.on_predict_ttes_clicked()
    assert abs(gui.current_model_params["T_tes"].value - 0.045) < 1e-4
    assert abs(gui.spin_analyze_tc.value() - 45.0) < 0.1


def test_lindeman_two_state_calibration():
    """Verify empirical 2-state calibration method from Lindeman et al. (2007) [arXiv:0704.3066]."""
    f = np.array([10.0, 100.0, 1000.0, 5000.0])
    Rsh = 250e-6
    L = 120e-9
    Rn = 25e-3
    Z_eq_true = Rsh + 1j * 2 * np.pi * f * L

    # Arbitrary frequency-dependent amplifier gain & cable delay
    G_readout = (1200.0 / np.sqrt(1 + (f / 30000.0)**2)) * np.exp(-2j * np.pi * f * 18.5e-6)
    Vac = 0.02
    u_sc = (Vac / Z_eq_true) * G_readout
    u_normal = (Vac / (Z_eq_true + Rn)) * G_readout

    # Arbitrary complex device impedance in transition
    Z_tes_true = 8e-3 + 1j * 2 * np.pi * f * 60e-9
    u_bias = (Vac / (Z_eq_true + Z_tes_true)) * G_readout

    Z_tes_calc, Z_eq_calc = compute_lindeman_two_state_impedance(u_bias, u_sc, u_normal, Rn)

    assert np.allclose(Z_eq_calc, Z_eq_true, rtol=1e-12, atol=1e-15)
    assert np.allclose(Z_tes_calc, Z_tes_true, rtol=1e-12, atol=1e-15)


def test_single_channel_sc_tf_hardware_agnostic():
    """
    Verify that single-channel datasets derive L_nyq from whatever SC transfer function
    is present (hardware-agnostic), or preserve explicitly passed lnyq_nh.
    """
    raw_dir = Path("/data/20261030/Complex_Z/20261030_171549")
    if not raw_dir.exists():
        pytest.skip("Dataset 20261030_171549 not available")

    # Load with manual explicit Lnyq of 220 nH
    ds_manual = load_cz_sweep_dataset(raw_dir, channel_id=5, lnyq_nh=220.0)
    assert ds_manual["lnyq_nh"] == 220.0
    assert abs(ds_manual["predicted_lnyq_nh"] - 119.13) < 0.2

    # Load with lnyq_nh=None: uses predicted L_nyq directly from SC transfer function
    ds_pred = load_cz_sweep_dataset(raw_dir, channel_id=5, lnyq_nh=None)
    assert abs(ds_pred["predicted_lnyq_nh"] - 119.13) < 0.2
    assert abs(ds_pred["lnyq_nh"] - 119.13) < 0.2


def test_explicit_tc_and_g_calculation():
    """
    Verify explicit power balance calculations:
        P_tes = (G / n) * Tc * (1 - (T_bath / Tc)^n)
    predict_tc and predict_g are exact mathematical inverses with no arbitrary clamps or defaults.
    """
    from ez_cz.models import predict_tc, predict_g, predict_ttes

    P0 = 7.27e-12   # 7.27 pW
    Tb = 0.020      # 20 mK
    n = 3.5
    Tc_target = 0.045  # 45 mK

    # 1. Given Tc, calculate G explicitly
    G_calc = predict_g(P0, Tc_target, n, Tb)
    assert abs(G_calc * 1e12 - 600.6) < 1.0  # ~600.6 pW/K

    # 2. Given G, calculate Tc explicitly
    Tc_calc = predict_tc(P0, G_calc, n, Tb)
    assert abs(Tc_calc - Tc_target) < 1e-6   # exactly 45.0 mK

    # 3. Predict_ttes is identical explicit solver
    assert abs(predict_ttes(P0, G_calc, n, Tb) - Tc_target) < 1e-6

    # 4. Low power regime (Ch 5: P0 = 1.9 pW, Tc = 35 mK)
    P_low = 1.90e-12
    Tc_low = 0.035
    G_low = predict_g(P_low, Tc_low, n, Tb)
    Tc_low_rec = predict_tc(P_low, G_low, n, Tb)
    assert abs(Tc_low_rec - Tc_low) < 1e-6

    # 5. Given G = 220 pW/K and P0 = 7.27 pW, Tc is explicitly ~115.9 mK without artificial clipping
    G_legacy = 220e-12
    Tc_unclamped = predict_tc(P0, G_legacy, n, Tb)
    assert abs(Tc_unclamped * 1e3 - 115.9) < 0.5

