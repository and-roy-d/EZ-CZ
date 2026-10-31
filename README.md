# EZ-CZ: Complex Impedance (Z<sub>TES</sub>) Measurement & Analysis

**EZ-CZ** is a Python suite for measuring, visualizing, and analyzing the complex impedance of Transition Edge Sensor (TES) microcalorimeters and bolometers. It automates multi-frequency sine-comb excitation, synchronous DASTARD acquisition, superconducting 0 V transfer function checks, and thermal model fitting.

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
1. It reads `flux_ramp_points` from `/data/experiment.cfg` (e.g. 32 × 64 = 2048 ticks → 524.288 µs).
2. The fundamental period `T_meas ≈ 0.2 s` is rounded to an exact integer multiple of flux-ramp cycles.
3. Every tone in the multi-sine comb is placed on an exact Fourier frequency bin `f_k = k * f_fund`, guaranteeing orthogonal, leakage-free FFT extraction.

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
- **0 V Superconducting Health Check**: Before sweeping bias, the system measures `Z(f)` at 0 V and fits the single-pole `L/R_s` transfer function. If the response is unphysical (`R² < 0.90`, abnormal roll-off, or corrupted delay), the run aborts immediately to protect data integrity.
- **Voltage Headroom Verification**: Verifies that combined AC + DC voltage stays within the AWG's allowable range (±10 V into High-Z, or ±5 V into 50 Ω).
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
| `--vpp` | AC multi-sine stimulus amplitude in V<sub>pp</sub> | `0.02` (20 mV) |
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

Processes measured `Z(f)` across all bias points and fits the two-fluid hanging-body thermal model:

```text
Z_hanging(ω) = R_0 * (1 + β) + [R_0 * ℒ / (1 - ℒ)] * (2 + β) / [1 + i * ω * τ_I - (g_1 / (g_1 + g_b) * (1 - ℒ)) / (1 + i * ω * τ_1)]
```

#### Common Options:
| Flag | Description | Default |
|:---|:---|:---|
| `--raw-dir`, `-d` | Path to folder containing measured `bias_*` subfolders | *(from config)* |
| `--iv-file`, `-i` | Corresponding IV curve `.npz` file for `R_0` and `I_TES` extraction | *(from config)* |
| `--channel-id`, `-c` | Detector channel ID to analyze | `13` |
| `--tc` | Device critical temperature `T_c` in Kelvin | `0.12` |
| `--tbath` | Cryostat bath temperature `T_bath` in Kelvin | `0.02` |
| `--rshunt-uohm` | Shunt resistance `R_s` in micro-Ohms | `250.0` (µΩ) |
| `--l-nyq-nh` | Nyquist circuit inductance `L` in nano-Henries | `74.8` (nH) |
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
├── setup.py                  # Standard pip editable packaging
├── README.md                 # This documentation
└── src/
    └── ez_cz/                # Consolidated Python package
        ├── __init__.py       # Package metadata and helper exports
        ├── data_io.py        # Sweep dataset parser & IV operating point extraction
        ├── gui.py            # Unified PyQt5 measurement & analysis dashboard
        ├── mdt3/             # Embedded microcal_design_tools thermal models
        ├── measure.py        # Automated measurement CLI and hardware control
        ├── models.py         # Vectorized thermal impedance models & bounded fitter
        └── analyze.py        # Batch thermal model fitting and parameter extraction
```

---

## Installation

Installation is as simple as running:

```bash
git clone https://github.com/and-roy-d/EZ-CZ.git
cd EZ-CZ
pip install -e .
```

This installs EZ-CZ into your active Python environment in editable mode and registers the command-line entry points:
- `ez-cz-gui`: Interactive graphical user interface (Measure & Analyze tabs)
- `ez-cz-measure`: Terminal automation for multi-sine impedance sweeps
- `ez-cz-analyze`: Thermal impedance analysis and batch fitting

*(Optional: if you prefer using `uv`, you can run `uv pip install -e .` or `uv sync`)*.

---

## Thermal Impedance Models (MDT Integration)

EZ-CZ includes self-contained implementations of the thermal models from the NIST `microcal_design_tools` (MDT) suite, eliminating any private repository dependencies. All models are available directly in the Analyze GUI dropdown and CLI:

1. **Simple (1-body)** (`TES_Simple`):  
   TES heat capacity *C*<sub>tes</sub> coupled directly to the thermal bath via thermal conductance *G*<sub>tes-bath</sub>.
2. **Dangling / Hanging (2-body)** (`TES_Dangling`):  
   TES heat capacity *C*<sub>tes</sub> coupled to bath with an internal dangling absorber heat capacity *C*<sub>abs</sub> connected via internal conductance *G*<sub>abs-tes</sub>.
3. **Intervening (2-body)** (`TES_Intervening`):  
   TES heat capacity *C*<sub>tes</sub> coupled to an intermediate stage *C*<sub>int</sub>, which in turn connects to the bath via *G*<sub>int-bath</sub>.
4. **Intermediate + Dangling (3-body)** (`TES_Inter_Dangle`):  
   3-body model incorporating TES, intermediate thermal stage, and hanging absorber.
5. **Two-Fluid Model** (`TES_TwoFluid`):  
   Phenomenological two-fluid model parameterizing superconducting fraction *w*<sub>L</sub> and dynamic resistance.

### Interactive Model Fitting in the GUI
- **Dynamic Parameter Table**: Text boxes and numeric controls dynamically adapt to the selected model.
- **Freeze / Float Flexibility**: Every parameter (*C*, *G*, *L*, *R*<sub>shunt</sub>, α, β, *T*<sub>c</sub>, *T*<sub>bath</sub>, *n*, *R*<sub>0</sub>) has an individual **Freeze** checkbox to lock or float it during optimization.
- **Physical Bounds**: User-specifiable **Min** and **Max** limits for every variable.
- **Instantaneous Model Prediction**: The predicted theoretical curve updates on the plot in real time (< 1 ms) as any parameter is adjusted.
- **Fast, Stable Fitter**: Bounded non-linear least squares converges in 10–30 ms with 1-sigma uncertainty (±) estimates.
- **Automatic Output Archiving**: Fit summaries (CSV), high-resolution figures (PNG), and parameter exports (JSON) can be automatically saved directly inside `<Complex_Z_Dir>/analysis/`.

### Lab Hardware Libraries
On data-acquisition measurement PCs, install the NIST instrument libraries:
- `qsdumux`: Keysight 33600A and SRS SIM900/928 driver wrappers.
- `dastardcommander`: Client interface for streaming DASTARD records.

*(When these drivers are absent, EZ-CZ runs automatically in mock simulation mode for offline analysis and UI testing.)*

---

## Technical Reference & Recent Fixes

### 1. Multi-Tone Comb Scaling (100+ Tones) Without AWG Clipping
* **The Problem (PAPR / Crest Factor)**:  
  In standard multi-sine combs, all cosines start in-phase (`ϕ_k = 0`). At `t = 0`, all `N` tones align at maximum amplitude, causing the peak voltage to scale as `N * V_tone`. For 100 tones with an amplitude multiplier of 0.04, the peak voltage exceeded 4.04 V. This triggered Keysight SCPI error `+881, "Arb: Values are out of range"` because arbitrary waveform volatile memory is strictly bounded to ±1.0 V.
* **The Fix (Schroeder Phase Algorithm)**:  
  We implemented Schroeder crest-factor minimization:
  ```text
  ϕ_k = -π * k * (k - 1) / N
  ```
  where `k ∈ [0, N - 1]` is the tone index. This disperses the tone phases quadratically, reducing peak-to-peak voltage by >7× (down to ≈ 0.64 V for 100 tones at 0.04 V per tone). Waveforms are auto-normalized to guarantee peak amplitude ≤ 1.0 V without clipping.
* **Phase Derotation at Demodulation**:  
  To measure true complex impedance, the received FFT bin at each tone frequency is derotated by its stimulus phase:
  ```text
  tone_data_raw[k] = fft_data_raw[f_k] * exp(-j * ϕ_k)
  ```
  This cancels `ϕ_k` analytically, restoring the device frequency response `H(f)` to numerical precision (< 10⁻¹³).

---

### 2. Phase Estimation Fix: Coherent Matched-Filter Delay Recovery
* **The Problem (Why 100 Tones Wrecked Delay Fitting)**:  
  During the 0 V superconducting baseline calibration, EZ-CZ fits the cable/readout time delay `τ_shift`:
  ```text
  Z_model(f) = [amp0 / (1 + j * f / f_roll)] * exp(-2 * π * j * f * τ)
  ```
  Adding 100 tones broke the previous implementation in two places:
  1. `real_model` used raw `fft_data_raw` instead of derotated `tone_data_raw`. It was trying to fit a single linear delay `τ_shift` to cancel a quadratic Schroeder phase chirp `ϕ_k = -π * k * (k - 1) / N`, which is mathematically impossible.
  2. `tau_guess = -np.argmax(data) / fs_global` searched for an impulse pulse in the raw time-domain signal. Because Schroeder phase intentionally flattens the time envelope, there is no pulse at `t = τ`, so `argmax` picked a random noise ripple milliseconds away.
  3. At 20 kHz, the cosine period is only 50 µs. Starting milliseconds away caused gradient descent to get trapped in cycle-slip local minima, yielding corrupted delays (e.g. -6.8 ms instead of -15 µs). Unrolling with this corrupted delay spun the phase by >6 full revolutions (>2200°).
* **The Fix (Coherent Matched Filtering)**:  
  We introduced equalized coherent matched filtering to find the true delay regardless of tone count or phase distribution:
  1. Equalize the derotated tones with the fitted 0 V magnitude model:
     ```text
     H_eq[k] = tone_data_raw[k] / Z_model_pure[k]
     ```
  2. Normalize amplitudes to unit magnitude across all tones:
     ```text
     comb_eq[f_k] = H_eq[k] / |H_eq[k]|
     ```
  3. Take the inverse FFT to reconstruct the equalized impulse response:
     ```text
     time_impulse = IFFT(comb_eq)
     ```
     Because stimulus phases and device roll-off are removed, all 100 tones constructively interfere at the physical delay `τ_delay`, creating a sharp spike.
  4. `peak_sample = np.argmax(time_impulse)` recovers the delay within ±1 µs over any range (±100 ms).
  5. Bounded LMfit fine-tuning around `tau_guess` refines the delay to sub-nanosecond precision:
     ```python
     def real_model(fs, tau_shift):
         return np.real(sc_tone_data_raw * np.exp(-2j * np.pi * fs_sch * tau_shift))

     params_real.add("tau_shift", value=tau_guess, min=tau_guess - 25e-6, max=tau_guess + 25e-6)
     res_real = model_real.fit(np.real(Z_model_pure), fs=fs_sch, params=params_real)
     tau_shift_0V = float(res_real.params["tau_shift"].value)
     ```

---

### 3. Dynamic Bias Schedule Table (Start, Stop, Step)
* Replaced the hardcoded "Dense" and "Medium" regions with an expandable `QTableWidget` (`tbl_bias_ranges`).
* **Default**: Starts with **1 row** by default (`[0.5, 1.5, 0.01]`).
* **`+ Add Bias Range Row`**: Appends a new region at the bottom with editable `Start (V)`, `Stop (V)`, and `Step (V)` spinboxes.
* **Delete Button (`Remove`)**: Removes any row and automatically renumbers remaining regions.
* Total points and the live bias voltage preview table update dynamically on any edit.
* Saved to and loaded from `run_settings.json` under `"bias_ranges"`.
* Supported in CLI via `--bias-range START STOP STEP` (can be specified multiple times).

---

### 4. Non-Blocking Safe Abort & Emergency Zeroing
* **Worker Polling**: `sleep_interruptible(seconds)` checks `_is_aborted` every 20–40 ms during settle delays and DASTARD acquisition loops.
* **Watchdog Termination**: If DASTARD is blocked inside an underlying C/ZMQ network call (`c.getNewData()`), clicking **ABORT** waits 1.0 s and then terminates the worker thread safely (`worker.terminate()`).
* **Main-Thread Hardware Clamp**: `_emergency_zero_hardware()` runs on the GUI thread: immediately forces Keysight AWG DC offset to 0.0 V, shuts off channel output (`OUTPUT OFF`), and zeroes the SRS SIM928 bias card.

---

### 5. Analysis Pipeline (`analyze.py`) Dataset Compatibility
* Automatically reads and applies `tau_shift` from the 0 V dataset file instead of relying on a hardcoded constant from an older run.
* Supports both `"raw_data"` (direct NumPy array) and legacy `"data"` (channel dictionary) formats in `.npz` files.

---

### 6. Power Line Frequency (60 Hz) Harmonic Rejection
* **The Problem (AC Mains Noise Contamination)**:  
  Multi-tone comb frequencies `f_m = m * f_0` frequently align with AC mains harmonics (`k * f_line = 60, 120, 180, 240, 300, 360, ... Hz`). When `f_0 = 20 Hz`, every 3rd tone (3, 6, 9, 12, ...) coincides exactly with a line harmonic, introducing severe 60 Hz pickup into the demodulated response and distorting impedance measurements.
* **The Solution**:  
  Added an automated rejection filter to `generate_stimulus_tones`:
  - Candidate frequencies are checked against the line frequency:
    ```text
    |f - round(f / f_line) * f_line| >= tolerance
    ```
    (default tolerance: ±2.0 Hz, configurable line frequency: 60.0 Hz).
  - Multiplier generation oversamples candidates by 2.5× so that after removing all harmonic collisions, exactly the desired number of clean tones (e.g. 100) are selected.
  - **GUI Control**: Accessible under the **Timing / Flux Ramp** tab with the `Skip Line Harmonics` checkbox and `Line Freq (Hz)` spinbox (persisted in `run_settings.json`).
  - **CLI Flags**: `--skip-line-harmonics` (default), `--no-skip-line-harmonics`, and `--line-freq 60.0`.

---

### 7. Matplotlib Multi-Threading Crash Fix (`ParseException` & Exit Code 139)
* **The Problem (`ParseException: Expected end of text, found '$'`)**:  
  Running a wide bias sweep (such as 0 V to 7 V) previously crashed the program with:
  ```text
  ValueError: $\mathdefault{4\times10^{6}}$
  ParseException: Expected end of text, found '$' (at char 0), (line:1, col:1)
    at self.fig.tight_layout(rect=[0, 0, 1, 0.96])
  ```
  or terminated abruptly with segmentation fault (exit code 139).
* **Root Cause**:  
  `CZWorker` was running `_save_bias_plot(...)` inside a background `QThread` while the main GUI thread simultaneously executed `self.canvas.update_plots()`. Matplotlib and Python's `pyparsing` packrat cache are not thread-safe. Concurrent execution caused race conditions in the mathtext parser when formatting scientific notation log tick labels (`4×10⁶`).
* **The Fix**:  
  1. **Thread Isolation**: Removed all `matplotlib.pyplot` figure generation and file saving from the background worker thread. All plot exports (`self.canvas.fig.savefig`) are executed strictly on the main Qt GUI thread inside `on_worker_step_completed`.
  2. **Mathtext Protection**: Configured `matplotlib.rcParams['axes.formatter.use_mathtext'] = False` to prevent pyparsing formatting failures on log ticks.
  3. **Defensive Layout Engine**: Wrapped `tight_layout()` and `canvas.draw()` in defensive exception handling to ensure any transient rendering warnings never crash the application.

---

### 8. Dynamic Acquisition Window Harmonization & Comb Deduplication
* **The Problem (Why a 7 Hz Fundamental Dropped Low-Frequency Amplitude)**:  
  Previously, the acquisition window duration was hardcoded to `T ≈ 0.200 s`, creating fixed 5.0 Hz FFT bins. Setting `f_0 = 7.0 Hz` caused non-integer cycle truncation (`0.2 s * 7 Hz = 1.4 cycles`), landing 7 Hz between bins 1 and 2. Scalloping loss and spectral leakage dropped the extracted amplitude by 35% to 50%, while higher tones that happened to align with 5 Hz bins (35 Hz, 70 Hz) remained tall. Furthermore, setting `f_0 = f_low` injected duplicate 7 Hz tones into the comb.
* **The Fix**:  
  1. **Dynamic Duration Harmonization**: `derive_timing_from_flux_ramp` automatically harmonizes the measurement window `T` to an exact integer number of cycles `k = max(1, round(T_target * f_0))` of the requested fundamental:
     ```text
     T = round((k / f_0) / t_flux_ramp) * t_flux_ramp
     ```
     For `f_0 = 7.0 Hz`, `T ≈ 0.142856 s` (34,877 flux ramps), ensuring `Δf = 7.000047 Hz` where every tone `m * f_0` sits identically on a Fourier bin center with **100.00% coherent amplitude recovery and zero leakage**.
  2. **Frequency Deduplication**: Added `np.unique` sorting in `generate_stimulus_tones` to guarantee every frequency is distinct.

---

### 9. SQUID Flux-Jump Prevention & Demodulation Unlock Detection
* **The Problem (Noise at Higher Bias Points)**:  
  When stepping DC bias across the superconducting-to-normal transition (`V_bias ~ 0.5 - 0.6 V`), the sudden change in resistance causes an abrupt drop in SQUID input coil current (`ΔI ~ 300 µA`), generating a step of hundreds of flux quanta (`Φ_0`). Instantaneous voltage steps from the AWG exceeded DASTARD's flux-ramp tracking slew rate, causing the SQUID demodulator to **lose lock / unlock** (`σ_raw ~ 10,000`, railed ADC levels). Once unlocked, all subsequent points in the sweep measure as pure broadband noise.
* **The Fix**:  
  1. **Slew-Rate Controlled Ramping (`ramp_awg_offset`)**: Replaced instantaneous voltage jumps with controlled software ramping (≤ 10 mV steps with 5 ms delay → ~2 V/s slew rate). When transitioning between bias points or returning from `V_norm`, voltage transitions are smooth and fully trackable by the SQUID feedback loop.
  2. **Automated Unlock Warning**: Added real-time diagnostics on raw DASTARD frame statistics. If `σ_raw > 3000`, a high-priority warning is emitted to alert the operator immediately.
  3. **Bidirectional (Descending) Bias Schedule**: Enabled descending sweeps (e.g. `Start: 1.5 V, Stop: 0.2 V, Step: 0.02 V`). Sweeping downward from normal into transition avoids the abrupt critical-current latching quench that occurs when ramping upward from zero.

---

### 10. High-Frequency Cutoff & Strict Nyquist Tone Protection
* **Motivation & Requirement**:  
  When scaling up tone count (e.g. 100 to 200 tones), excitation frequencies must never approach or exceed the Nyquist frequency (`f_Nyquist = f_DAQ / 2`). Digital reconstruction filters distort tones near Nyquist, and `f = f_Nyquist` creates aliasing ambiguities (`exp(j * π * n) = (-1)^n`). Furthermore, measuring far beyond the detector's electrical/thermal bandwidth (`L/R ~ 1 kHz`) wastes AWG dynamic range.
* **The Implementation**:  
  1. **Strict Internal Nyquist Rejection**: In `generate_stimulus_tones`, any candidate multiplier resulting in `f >= f_Nyquist` is strictly rejected:
     ```text
     f < f_Nyquist - f_0 / 2
     ```
     All tone frequencies in `all_freqs` are explicitly filtered by `f < nyquist`.
  2. **Configurable High-Frequency Cutoff**: Added a user-defined cutoff parameter `high_f_cutoff` (default: 50.0 kHz, clamped strictly below Nyquist):
     ```text
     f_max = min(f_cutoff, f_Nyquist - f_0 / 2)
     ```
  3. **GUI & CLI Controls**:
     - **GUI**: `High-Freq Cutoff (Hz)` double spinbox under the **AWG Stimulus** tab (persisted in `run_settings.json`).
     - **CLI**: `--high-f-cutoff` / `--max-freq` flags (default: 50,000 Hz).

---

### 11. Dynamic FPGA Firmware Clock Discovery (`TIME_PER_FLUX_RAMP_POINT_S`)
* **The Question**:  
  *"`TIME_PER_FLUX_RAMP_POINT_S = 256e-9` — can we find this in the firmware so we don't have to hardcode it?"*
* **Where It Lives in the Firmware & Hardware**:  
  Yes! The 256 ns clock period is not arbitrary; it is burned directly into the FPGA IP-XACT core metadata ROM registers of the polyphase filter bank channelizer (`biblk/fset0/frontend`, core `QSG-NIST:qsg-nist:chpfbfft:0.15` at register base `0x08800000`).
  The firmware parameters define:
  - **ADC Sampling Clock**:
    ```text
    f_ADC = 10^9 * (TSDENOM / TSNUM) * SAMPLESPERCLK = 10^9 * (1 / 4) * 4 = 1.0 GHz (1 GSPS)
    ```
  - **Polyphase Filter Bank Window Length**: `WINDOWLEN = 2048`
  - **Active Sub-filter Sets**: `NSETS = 8`
  - **Fine DDC Output Sample Rate**:
    ```text
    f_fineddc = (f_ADC / WINDOWLEN) * NSETS = (10^9 Hz / 2048) * 8 = 3,906,250 Hz = 3.90625 MHz
    ```
  - **Time Per Flux Ramp Point**:
    ```text
    t_point = 1 / f_fineddc = 1 / 3,906,250 Hz = 256 * 10^-9 s = 256 ns
    ```
* **Live Dynamic Firmware Query (`query_time_per_flux_ramp_point`)**:  
  Implemented `query_time_per_flux_ramp_point(cfg_path)` in `measure.py` and `gui.py`:
  1. Inspects `/data/experiment.cfg` for the FPGA control interface URL (e.g. `ctrl = udp://10.0.15.4`) and active front-end path (`fset0`).
  2. Directly reads the live IP-XACT core parameters (`frontend.samplerate`, `frontend.get_windowlen()`, `frontend.get_nsets()`) from the FPGA hardware registers via `qsdumux.user.abaco_setup`.
  3. Computes the exact time per point dynamically:
     ```python
     rate = frontend.samplerate / frontend.get_windowlen() * frontend.get_nsets()
     dt = 1.0 / rate  # 2.56e-7 s (256.000 ns)
     ```
  4. Automatically incorporates any future firmware modifications (e.g. changing window decimation or ADC clock rate).
  5. **Clean Safe Fallback**: If the FPGA hardware is offline, network unreachable, or the tool is running in simulation/dry-run mode, it catches the error and cleanly falls back to nominal `256e-9 s` with zero crash risk.

---

### 12. Flip Bias Polarity Toggle & Re(Z) > 0 at DC
* **The Problem (Fitter Divergence on Inverted Bias / SQUID Polarities)**:  
  In TES setups where the bias current leads are reversed, negative bias sweeps are applied, or SQUID flux ramp demodulation operates on a negative slope (`dΦ / dI < 0`), the demodulated current response `ΔI` arrives with an overall minus sign (`Re < 0` at DC).
  Because the magnitude fit always estimates positive amplitude (`amp0 = |amp0| > 0`), generating the single-pole `L/R` model with positive sign created an irreconcilable sign contradiction:
  - `Re(data) ≈ -amp0 < 0`
  - `Re(Z_model) = +amp0 > 0`
  The matched filter delay estimator and `lmfit` `model_real.fit(...)` tried to fit a negative signal to a positive target, causing delay estimation failure, low `R²`, or fitter crashes. Furthermore, unrolled impedance appeared in the left-half complex plane (`Re(Z) < 0`), violating physical passive resistance (`R > 0`).
* **The Solution**:  
  Added a dedicated **`Flip bias?`** toggle across the GUI and measurement runners:
  1. **Opposite-Sign Model Generation**:  
     When `flip_bias` is enabled, the 0 V superconducting transfer function model is generated with inverted sign:
     ```text
     Z_model_fit(f) = -amp0 / (1 + j * f / f_roll)
     ```
     This aligns the target sign with the inverted detector response, allowing coherent matched filtering and fine `τ` delay fitting to converge cleanly without crashing.
  2. **Re(Z) > 0 at DC Enforcement**:  
     Because physical impedance `Z = ΔV / ΔI` has an overall minus sign when bias polarity is flipped, unrolled response data is harmonized:
     ```text
     fft_unroll = -raw_unroll  ==>  Re(Z) > 0 at DC
     ```
     This guarantees that both measured and modeled `Re(Z)` land strictly on the **positive real axis at DC** in both Bode and Nyquist plots.
  3. **UI & Configuration Integration**:
     - **GUI**: Synced `Flip bias?` checkboxes in both the **Bias & DC Source** tab and **SC Health Check** tab.
     - **CLI**: `--flip-bias` flag in `measure.py` and `analyze.py`.
     - **JSON**: Saved and restored automatically via `"flip_bias": true/false` in `run_settings.json`.
     - **NPZ Output**: Records `flip_bias` inside each step's saved `vbias_*_data.npz` file.
