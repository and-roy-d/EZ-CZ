# EZ-CZ: Complex Impedance ($Z_{TES}$) Measurement & Analysis

**EZ-CZ** is a Python suite for measuring, visualizing, and analyzing the complex impedance of Transition Edge Sensor (TES) microcalorimeters and bolometers. It automates multi-frequency sine-comb excitation, synchronous DASTARD acquisition, superconducting 0 V transfer function health checks, and two-fluid hanging-body thermal model fitting.

---

## Quickstart (30 Seconds)

```bash
# 1. Launch the interactive real-time GUI
ez-cz-gui

# 2. Run a measurement sweep from the terminal (default: AWG Ch2 DC bias)
ez-cz-measure --channel 13

# 3. Test offline on your laptop (simulation / dry-run mode, no hardware needed)
ez-cz-measure --channel 13 --dry-run

# 4. Analyze an existing dataset and fit thermal parameters
ez-cz-analyze --channel-id 13 --raw-dir ./data/20250807/Complex_Z/run_01 --iv-file 20250807_iv.npz
```

> **Running without installation?**  
> You can also run the scripts directly:
> ```bash
> python src/ez_cz/gui.py
> python src/ez_cz/measure.py --channel 13 --dry-run
> python src/ez_cz/analyze.py --channel-id 13
> ```

---

## How It Works: Two DC Bias Modes

EZ-CZ supports two methods for setting the TES DC bias voltage. You can toggle between them using the GUI dropdown menu or the `--dc-source` CLI flag:

| Mode | AC Stimulus Comb | DC Bias Source | Lab Hardware Required |
|:---|:---|:---|:---|
| **`awg`** <br>*(Default)* | Keysight 33600A Ch2 | **Keysight 33600A Ch2** (Hardware DC offset) | **1 BNC cable directly from AWG Ch2.** No external DC voltage box or hardware summing amplifier needed. |
| **`srs900`** | Keysight 33600A Ch2 | **SRS SIM928** (via SIM900 mainframe) | Keysight AWG + SRS SIM900/928 mainframe + hardware summing amplifier. |

### Flux-Ramp Synchronized Timing
To eliminate spectral leakage without windowing, EZ-CZ harmonizes all excitation frequencies with DASTARD's flux-ramp demodulation rate:
1. It reads `flux_ramp_points` from `/data/experiment.cfg` (e.g. $32 \times 64 = 2048$ ticks $\rightarrow 524.288\text{ }\mu\text{s}$).
2. The fundamental period $T_{\text{meas}} \approx 0.2\text{ s}$ is rounded to an exact integer multiple of flux-ramp cycles.
3. Every tone in the multi-sine comb is placed on an exact Fourier frequency bin $f_k = k \cdot f_{\text{fund}}$, guaranteeing orthogonal, leakage-free FFT extraction.

---

## Graphical User Interface (`ez-cz-gui`)

The PyQt5 Dashboard provides complete experiment control with real-time feedback:

```
+-----------------------------------------------------------------------------------------+
| EZ-CZ Dashboard                                                       [ Light / Dark ]  |
+---------------------------------------+-------------------------------------------------+
| Experiment Settings                   | Live 4-Panel Visualization                      |
|  - Channel: [ 13 ]                    |  1. Re(Z) & Im(Z) vs. Frequency (semilog-x)     |
|  - DC Source: [ DC source: AWG v ]   |  2. Nyquist Plot: Im(Z) vs. Re(Z)               |
|  - AWG Vpp: [ 20 mV ]                 |  3. Magnitude |Z(f)| with L/Rs Fit (log-log)   |
|  - Bias Points: 0V, Dense, Medium     |  4. Phase vs. Frequency with Tau Delay Fit      |
|                                       +-------------------------------------------------+
| Controls                              | Diagnostic Log                                  |
|  [ START SWEEP ]   [ TEST 0V SC ]     |  [14:02:10] SC Health: PASS (R² = 0.9998)      |
|  [ CHECK LIMITS ]  [ ABORT (SAFE) ]   |  [14:02:11] Voltage headroom: +4.99 V (PASS)    |
+---------------------------------------+-------------------------------------------------+
```

### Key Safety & Diagnostic Features
- **0 V Superconducting Health Check**: Before sweeping bias, the system measures $Z(f)$ at $0\text{ V}$ and fits the single-pole $L/R_s$ transfer function. If the response is unphysical ($R^2 < 0.90$, abnormal roll-off, or corrupted delay), the run aborts immediately to protect data integrity.
- **Voltage Headroom Verification**: Verifies that combined AC + DC voltage stays within the AWG's allowable range ($\pm 10\text{ V}$ into High-$Z$, or $\pm 5\text{ V}$ into $50\,\Omega$).
- **Instant Safe ABORT**: Clicking **ABORT** or pressing `Ctrl+C` immediately ramps down and zeroes the DC voltage on both the AWG and SRS SIM928.
- **Offline Dry Run**: Automatically switches to simulated hardware if `qsdumux` or DASTARD libraries are not detected.

---

## Command-Line Usage

### 1. Measurement Runner (`ez-cz-measure`)

```bash
ez-cz-measure [OPTIONS]
```

#### Common Options:
| Flag | Description | Default |
|:---|:---|:---|
| `--dc-source {awg,srs900}` | Choose DC bias source: direct AWG offset or SRS SIM928 | `awg` |
| `--channel`, `-c` | DASTARD detector channel number | `13` |
| `--vpp` | AC multi-sine stimulus amplitude in $V_{\text{pp}}$ | `0.02` (20 mV) |
| `--v-norm` | TES normal-state bias pulse voltage in Volts | `9.5` |
| `--normal-first` | Pulse TES to normal state for 1 s after 0V calibration | `False` |
| `--bias-dense` | `START STOP STEP` for dense transition bias sweep | `0.5 1.5 0.01` |
| `--bias-medium` | `START STOP STEP` for medium transition bias sweep | `1.5 5.0 0.1` |
| `--dry-run` | Run in simulation mode without hardware | `False` |
| `--config`, `-cfg` | Path to JSON settings file | `run_settings.json` |

---

### 2. Analysis & Thermal Model Fitting (`ez-cz-analyze`)

```bash
ez-cz-analyze [OPTIONS]
```

Processes measured $Z(f)$ across all bias points and fits the two-fluid hanging-body thermal model:

$$Z_{\text{hanging}}(\omega) = R_0 (1 + \beta) + \frac{R_0 \mathcal{L}}{(1 - \mathcal{L})} \frac{(2 + \beta)}{1 + i \omega \tau_I - \frac{g_1 / (g_1 + g_b)(1 - \mathcal{L})}{1 + i \omega \tau_1}}$$

#### Common Options:
| Flag | Description | Default |
|:---|:---|:---|
| `--raw-dir`, `-d` | Path to folder containing measured `bias_*` subfolders | *(from config)* |
| `--iv-file`, `-i` | Corresponding IV curve `.npz` file for $R_0$ and $I_{\text{TES}}$ extraction | *(from config)* |
| `--channel-id`, `-c` | Detector channel ID to analyze | `13` |
| `--tc` | Device critical temperature $T_c$ in Kelvin | `0.12` |
| `--tbath` | Cryostat bath temperature $T_{\text{bath}}$ in Kelvin | `0.02` |
| `--rshunt-uohm` | Shunt resistance $R_s$ in micro-Ohms | `250.0` ($\mu\Omega$) |
| `--l-nyq-nh` | Nyquist circuit inductance $L$ in nano-Henries | `74.8` (nH) |
| `--output-base-dir`, `-o`| Directory where fit plots and summary CSVs are saved | `/data/.../Results/` |

---

## Configuration File (`run_settings.json`)

Both CLI tools and the GUI can load and save settings from `run_settings.json`. Command-line arguments always override values in this file:

```json
{
  "measure": {
    "dc_source": "awg",
    "channel": 13,
    "v_norm": 9.5,
    "vpp": 0.02,
    "bias_dense": [0.5, 1.5, 0.01],
    "bias_medium": [1.5, 5.0, 0.1],
    "keysight_url": "vxi11://192.168.0.112:2021",
    "srs_url": "serial:///dev/ttyUSB1?baudrate=9600"
  },
  "analyze": {
    "channel_id": 13,
    "tc": 0.12,
    "tbath": 0.02,
    "rshunt_uOhm": 250.0,
    "l_nyq_nH": 74.8,
    "save_plots": true
  }
}
```

---

## Project Structure

```
EZ-CZ/
├── run_settings.json         # Default hardware, sweep, and analysis parameters
├── pyproject.toml            # Package configuration and dependencies
├── README.md                 # This documentation
└── src/
    └── ez_cz/                # Consolidated Python package
        ├── __init__.py       # Package metadata and helper exports
        ├── gui.py            # Unified PyQt5 real-time measurement dashboard
        ├── measure.py        # Automated measurement CLI and hardware control
        └── analyze.py        # Thermal model fitting and parameter extraction
```

---

## Installation & Requirements

### Recommended Setup (using `uv`)

```bash
# Clone the repository
git clone https://github.com/and-roy-d/EZ-CZ.git
cd EZ-CZ

# Create virtual environment and install in editable mode
uv sync
```

### Lab Hardware Libraries
On data-acquisition measurement PCs, install the NIST instrument libraries:
- `qsdumux`: Keysight 33600A and SRS SIM900/928 driver wrappers.
- `dastardcommander`: Client interface for streaming DASTARD records.

*(When these drivers are absent, EZ-CZ runs automatically in mock simulation mode for offline analysis and UI testing.)*
