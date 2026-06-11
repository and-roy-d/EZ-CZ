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

The package registers two command-line scripts: `ez-cz-measure` and `ez-cz-analyze`.

### 1. Measurements: `ez-cz-measure`
To run the measurement sweeps, configure the waveform generator, and collect DASTARD traces:
```bash
ez-cz-measure --channel 13 --fs 250000 --trigger-awg
```

#### Key Arguments:
* `--fs`: Digitization sample rate in Hz (default: `250000`).
* `--channel` / `-c`: Dastard channel ID to read data from (default: `13`).
* `--v-norm`: Bias normalization voltage in Volts (default: `9.5`).
* `--rs-uohm`: Shunt resistance $R_s$ in micro-Ohms (default: `250.0`).
* `--l-nh`: Nyquist inductor $L$ in nano-Henries (default: `1000.0`).
* `--trigger-awg`: Connect to the Keysight AWG and load the waveforms (default: `False`).
* `--srs-url`: SRS SIM900 mainframe serial connection string (default: `"serial:///dev/ttyUSB1?baudrate=9600"`).
* `--keysight-url`: Keysight function generator connection URL (default: `"vxi11://192.168.0.112:2021"`).
* `--bias-dense`: Start, stop, and step size for dense bias sweeps (default: `0.5 1.5 0.01`).
* `--bias-medium`: Start, stop, and step size for medium bias sweeps (default: `1.5 5.0 0.1`).
* `--output-dir`: Custom path to save raw `.npz` records and sweeps.

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
* `--g-data-file`: CSV file path containing precomputed G data (default: `/home/pcuser/Runs/Cooldown_A18/Results/G_parameter_summary.csv`).
* `--dummy-g-tes-b`: Fallback $g_{tes,b}$ in W/K if G CSV is not found or mapping is missing (default: `200e-9`).
* `--pixel-map`: Space-separated list of `channel_id:pixel_number` mappings (default: `13:13`).
* `--rshunt-uohm`: Shunt resistance $R_{shunt}$ in micro-Ohms (default: `250.0`).
* `--l-nyq-nh`: Nyquist inductor $L_{nyq}$ in nano-Henries (default: `74.8`).
* `--output-base-dir` / `-o`: Directory to save plots and CSV parameter tables.
* `--no-save-plots`: Do not save plot files on disk (only displays them on screen).

---

## Graphical User Interface (GUI)

The package includes a lightweight, real-time GUI built using `tkinter` and `matplotlib`. It allows you to configure measurement settings, trigger sweeps, and visualize results dynamically.

### Running the GUI

If the package is installed:
```bash
ez-cz-gui
```

Alternatively, you can run the GUI script directly without installation:
```bash
python src/ez_cz/gui.py
```

### Features
* **Interactive Control Board**: Customize channel ID, sample rate, shunt resistance, Nyquist inductor, bias range (dense and medium sweeps), and AWG URL.
* **Save Default Settings**: Save your current inputs to the local `run_settings.json` file.
* **Dry Run / Simulation Mode**: Test and verify the GUI offline. It synthesizes signals using realistic TES models when laboratory instruments are not present.
* **4-Panel Real-time Plotting**: Shows the Real/Imaginary, Nyquist, Magnitude, and Phase response curves updating live at each bias step.

---

## Customizing inside Python scripts
You can also import modules directly in Python:
```python
from ez_cz.measure import generate_freq_mutiples_logspaced, cos_superimposed
from ez_cz.analyze import calculate_z_hanging_model_complex
```
