# EZ-CZ (Complex Impedance Measure & Analyze)

`ez-cz` is a standalone Python package for measuring and analyzing the complex impedance ($Z_{TES}$) of Transition Edge Sensor (TES) devices. It interfaces with SRS SIM900/928 voltage boxes, Keysight function generators, and the DASTARD data acquisition client, and fits the measured impedance curves to thermal model equations.

## Installation

### 1. Prerequisites
You must have the `uv` Python package manager installed. If you do not have it, install it via:
```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment and install
Clone this repository (or navigate to its folder) and run:
```bash
# Initialize/Create virtual environment
uv venv

# Install the package in editable mode
uv pip install -e .
```
This command automatically installs all dependencies.

### Alternative: Running without Installation
If you prefer not to install the package, you can run the source scripts directly using standard Python (provided you have the dependencies like `numpy`, `scipy`, `matplotlib`, `lmfit`, and `pandas` installed in your environment):

```bash
# Run measurements directly
python src/ez_cz/measure.py --channel 13

# Run analysis directly
python src/ez_cz/analyze.py -c 13
```
In this direct mode, the scripts will still automatically check for the `run_settings.json` file in the directory where they are run.

---

## Configuration File (`run_settings.json`)

Both CLI commands can load their default parameters from a single JSON settings file. By default, they will search for a file named `run_settings.json` in the directory where they are executed.

To use a custom settings file, pass the `--config` (or `-cfg`) flag:
```bash
ez-cz-measure --config custom_settings.json
ez-cz-analyze -cfg custom_settings.json
```

*Note: Command-line arguments always override values specified in the configuration file.*

---

## Command Line Interface (CLI)

The package registers three command-line scripts: `ez-cz-measure`, `ez-cz-analyze`, and `ez-cz-gui`.

### 1. Measurements: `ez-cz-measure`
To run measurement sweeps, configure the waveform generator, and collect DASTARD traces:
```bash
# Default: Using Keysight AWG Ch2 DC offset (cztest4 / cz_test_AWG-DC style)
ez-cz-measure --channel 13 --dc-source awg

# Alternative: Using SRS SIM900/SIM928 DC voltage box (cztest3 / cz_test_SRS900-DC style)
ez-cz-measure --channel 13 --dc-source srs900 --srs-url "serial:///dev/ttyUSB1?baudrate=9600"
```

#### Key Arguments:
* `--dc-source`: DC bias voltage source: `awg` (default) or `srs900`.
* `--channel` / `-c`: Dastard channel ID to read data from (default: `13`).
* `--config-cfg`: Path to `experiment.cfg` to auto-read `time_per_flux_ramp` (default: `/data/experiment.cfg`).
* `--time-per-flux-ramp`: Manual time duration per flux ramp in seconds (e.g. `8.192e-6`).
* `--v-norm`: Bias normalization voltage in Volts (default: `9.5`).
* `--normal-first`: Drive TES normal for 1.0 s after 0V calibration before sweep.
* `--awg-rate`: AWG sample rate in Hz (default: `5000000.0`).
* `--target-duration`: Nominal measurement duration window in seconds (default: `0.2`).
* `--f0`: Base frequency comb spacing in Hz (default: `20.0`).
* `--amp-mult`: Amplitude multiplier per sine tone (default: `0.04`).
* `--vpp`: Final AWG AC amplitude in Vpp (default: `0.02`).
* `--awg-channel`: AWG output channel number (default: `2`).
* `--awg-load`: AWG termination load: `INFinity` (default, ±10 V range) or `50` (±5 V range).
* `--keysight-url`: Keysight AWG connection address or URL (default: `USB::2391::22279::INSTR`).
* `--srs-url`: SRS SIM900 mainframe serial connection string (default: `"serial:///dev/ttyUSB1?baudrate=9600"`).
* `--srs-port`: SRS SIM928 card slot port (default: `2`).
* `--bias-dense`: Start, stop, and step size for dense bias sweeps (default: `0.5 1.5 0.01`).
* `--bias-medium`: Start, stop, and step size for medium bias sweeps (default: `1.5 5.0 0.1`).
* `--output-dir`: Custom path to save raw `.npz` records and sweeps.
* `--dry-run` / `--simulation`: Run in simulation mode without physical hardware.

---

### 2. Analysis: `ez-cz-analyze`
To process measurement sweeps, compute the complex impedance $Z_{TES}$, and fit the hanging body thermal model:
```bash
ez-cz-analyze --channel-id 13 --raw-dir ./data/20250807/Complex_Z/20250807_141314 --iv-file 20250807_150622_iv.npz
```

#### Key Arguments:
* `--raw-dir` / `-d`: Directory containing raw `.npz` sweeps (e.g. `bias_0v00`).
* `--iv-file` / `-i`: Reference IV curve `.npz` filename.
* `--channel-id` / `-c`: Dastard channel ID to analyze (default: `13`).
* `--zero-bias-time-shift` / `-t`: Superconducting time shift in seconds (default: `-2.70625e-2`).
* `--tc`: Device critical temperature $T_c$ in Kelvin (default: `120e-3`).
* `--g-data-file`: CSV file path containing precomputed G data.
* `--dummy-g-tes-b`: Fallback $g_{tes,b}$ in W/K if G CSV is not found (default: `200e-9`).
* `--pixel-map`: Space-separated list of `channel_id:pixel_number` mappings (default: `13:13`).
* `--rshunt-uohm`: Shunt resistance $R_{shunt}$ in micro-Ohms (default: `250.0`).
* `--l-nyq-nh`: Nyquist inductor $L_{nyq}$ in nano-Henries (default: `74.8`).
* `--output-base-dir` / `-o`: Directory to save plots and CSV parameter tables.
* `--no-save-plots`: Do not save plot files on disk (only displays them on screen).

---

## Graphical User Interface (GUI)

The package includes a modern, high-performance Qt5 dashboard (`ez-cz-gui`).

### Running the GUI

If the package is installed:
```bash
ez-cz-gui
```

Alternatively, run the GUI script directly without installation:
```bash
python src/ez_cz/gui.py
# or
python src/cztest/cz_gui.py
```

### Key Features
* **Selectable DC Bias Source**:
  - Rolldown menu (`DC source: AWG` or `DC source: SRS900`).
  - **`DC source: AWG` (Default)**: Generates the DC bias offset directly via Channel 2 of the Keysight 33600A AWG (eliminates the external voltage box).
  - **`DC source: SRS900`**: Uses the SRS SIM900/SIM928 DC voltage box (incorporating `cz_test_SRS900-DC` logic).
* **Flux-Ramp Synchronized Timing**: Auto-reads `flux_ramp_points` from `experiment.cfg` and harmonically locks all tones to eliminate spectral leakage.
* **Superconducting (0V) Health Check & Abort**: Fits the single-pole $L/R_s$ transfer function and phase delay $\tau_{shift}$ at 0 V. Automatically aborts or pauses for review if the fit is unphysical.
* **Live 4-Panel Matplotlib Display**: Real/Imaginary, Nyquist, Magnitude, and Phase responses update in real time at each bias point.
* **Safe Instant ABORT**: Safely zeroes the DC voltage on the AWG and/or SRS SIM928 immediately.
* **Light and Dark Modes**: Clean toggle between light and dark UI themes.

---

## Test & Reference Scripts (`src/cztest/`)

Standalone test scripts for experimental validation and bench testing:
* `src/cztest/cz_test_AWG-DC.py` (formerly `cz_test4.py`): AWG-direct DC offset measurement script.
* `src/cztest/cz_test_SRS900-DC.py` (formerly `cz_test3.py`): SRS SIM900/SIM928 DC bias measurement script.
* `src/cztest/cz_gui.py`: Standalone GUI entrypoint pointing to `ez_cz.gui`.
* `src/cztest/analyze_complexZ3.py`: Reference complex impedance analysis script.

---

## Customizing inside Python scripts
You can also import modules directly in Python:
```python
from ez_cz.measure import derive_timing_from_flux_ramp, generate_stimulus_tones
from ez_cz.analyze import calculate_z_hanging_model_complex
from ez_cz.gui import CZMainWindow
```
