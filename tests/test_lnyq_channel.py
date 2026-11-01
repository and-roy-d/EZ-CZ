import os
import sys
from pathlib import Path
import numpy as np
import pytest

from ez_cz.data_io import load_cz_sweep_dataset, recompute_dataset_circuit_impedance
from ez_cz.models import get_model, SimpleModel


def test_lnyq_physics_formula():
    """Verify L = R_shunt / (2*pi*f_roll) for both channel categories."""
    rshunt_uohm = 250.0  # 250 uOhm

    # Channel >= 32 (~120 nH bonded):
    f_roll_high = 333.988  # Hz
    l_high = (rshunt_uohm * 1e3) / (2.0 * np.pi * f_roll_high)
    assert abs(l_high - 119.13) < 0.1

    # Channel < 32 (220 nH bonded):
    expected_froll_low = (rshunt_uohm * 1e3) / (2.0 * np.pi * 220.0)
    assert abs(expected_froll_low - 180.85) < 0.1
    l_low = (rshunt_uohm * 1e3) / (2.0 * np.pi * expected_froll_low)
    assert abs(l_low - 220.0) < 0.01


def test_models_channel_aware_lnyq():
    """Verify SimpleModel parameter defaults reflect bonded inductors by channel ID."""
    model = SimpleModel()

    # Channel 5 (< 32): should default to 220.0 nH
    p_ch5 = model.get_parameter_defs(channel=5)
    assert p_ch5["L_nyq"].value == 220.0

    # Channel 63 (>= 32): should default to 119.13 nH
    p_ch63 = model.get_parameter_defs(channel=63)
    assert p_ch63["L_nyq"].value == 119.13

    # Explicit override takes precedence
    p_custom = model.get_parameter_defs(channel=5, L_nyq=150.0)
    assert p_custom["L_nyq"].value == 150.0


def test_load_cz_sweep_channel_mismatch():
    """Verify single-channel dataset mismatch detection on real data."""
    raw_dir = Path("/data/20261030/Complex_Z/20261030_171549")
    if not raw_dir.exists():
        pytest.skip("Dataset 20261030_171549 not available")

    # Load for Channel 63: match
    ds_63 = load_cz_sweep_dataset(raw_dir, channel_id=63)
    assert ds_63["is_channel_mismatch"] is False
    assert abs(ds_63["lnyq_nh"] - 119.13) < 0.5
    assert abs(ds_63["f_roll_0v"] - 334.0) < 1.0

    # Load for Channel 5: mismatch (single-channel Ch63 data loaded for Ch5)
    ds_5 = load_cz_sweep_dataset(raw_dir, channel_id=5)
    assert ds_5["is_channel_mismatch"] is True
    assert ds_5["nominal_lnyq_nh"] == 220.0
    assert ds_5["lnyq_nh"] == 220.0  # Uses 220.0 nH instead of 119.1 nH!


def test_gui_channel_switching_lnyq():
    """Verify GUI dynamically manages L_nyq when channel changes."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication
    from ez_cz.gui import CZMainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    gui = CZMainWindow()
    raw_dir = Path("/data/20261030/Complex_Z/20261030_171549")
    if not raw_dir.exists():
        pytest.skip("Dataset 20261030_171549 not available")

    gui.edit_analyze_raw_dir.setText(str(raw_dir))

    # Channel 63
    gui.spin_analyze_channel.setValue(63)
    gui.on_load_analyze_sweep()
    assert abs(gui.current_model_params["L_nyq"].value - 119.13) < 0.5

    # Switch to Channel 5
    gui.spin_analyze_channel.setValue(5)
    assert abs(gui.current_model_params["L_nyq"].value - 220.0) < 0.01

    # Switch back to Channel 63
    gui.spin_analyze_channel.setValue(63)
    assert abs(gui.current_model_params["L_nyq"].value - 119.13) < 0.5
