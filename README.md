# EZ-CZ: Complex Impedance (Z<sub>TES</sub>) Measurement & Analysis

**EZ-CZ** is a Python suite for measuring, visualizing, and analyzing the complex impedance of Transition Edge Sensor (TES) microcalorimeters and bolometers. It automates multi-frequency sine-comb excitation, synchronous DASTARD acquisition, superconducting 0 V transfer function calibration, hardware-agnostic inductance derivation, and thermal model fitting.

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
ez-cz-analyze --channel-id 13 --raw-dir ./data/20261030/Complex_Z/20261030_171549 --iv-file iv_curve.npz
```

Running without installation:
```bash
python src/ez_cz/gui.py
python src/ez_cz/measure.py --channel 13 --dry-run
python src/ez_cz/analyze.py --channel-id 13
```

---

## Hardware Interfacing & DC Bias Modes

EZ-CZ supports two methods for setting the TES DC bias voltage. You can toggle between them using the GUI dropdown menu or the `--dc-source` CLI flag:

| Mode | AC Stimulus Comb | DC Bias Source | Lab Hardware Required |
|:---|:---|:---|:---|
| **`awg`** <br>*(Default)* | Keysight 33600A Ch2 | **Keysight 33600A Ch2** (Hardware DC offset) | **1 BNC cable directly from AWG Ch2.** No external DC voltage box or hardware summing amplifier needed. |
| **`srs900`** | Keysight 33600A Ch2 | **SRS SIM928** (via SIM900 mainframe) | Keysight AWG + SRS SIM900/928 mainframe + hardware summing amplifier. |

### Safety & Diagnostic Controls
- **0 V Superconducting Calibration**: Before sweeping bias, the system measures $Z(f)$ at 0 V and fits the single-pole transfer function to determine circuit inductance and readout delay. If the response is unphysical ($R^2 < 0.90$ or abnormal roll-off), the run aborts immediately to protect data integrity.
- **Voltage Headroom Verification**: Verifies that combined AC + DC voltage stays strictly within the AWG allowable output range ($\pm 10\text{ V}$ into High-Z, or $\pm 5\text{ V}$ into $50\ \Omega$).
- **Safe Abort & Emergency Zeroing**: Clicking **ABORT** or pressing `Ctrl+C` immediately ramps down and zeroes the DC voltage on both the AWG and SRS SIM928.
- **Offline Dry Run**: Automatically switches to simulated hardware if instrument drivers or DASTARD libraries are not detected.

---

## Graphical User Interface (`ez-cz-gui`)

The PyQt5 Dashboard provides complete experiment control with real-time feedback across two primary workflows: **Measure** and **Analyze**.

```
+-----------------------------------------------------------------------------------------+
| EZ-CZ Dashboard                                                                         |
+---------------------------------------+-------------------------------------------------+
| Experiment Settings                   | Live 4-Panel Visualization                      |
|  - Channel: [ 13 ]                    |  1. Re(Z) & Im(Z) vs. Frequency (semilog-x)     |
|  - DC Source: [ DC source: AWG v ]   |  2. Nyquist Plot: Im(Z) vs. Re(Z)               |
|  - AWG Vpp: [ 20 mV ]                 |  3. Magnitude |Z(f)| with L/Rs Fit (log-log)   |
|  - Bias Ranges Table                  |  4. Phase vs. Frequency with Tau Delay Fit      |
|                                       +-------------------------------------------------+
| Controls                              | Diagnostic Log                                  |
|  [ START SWEEP ]   [ TEST 0V SC ]     |  [14:02:10] SC Health: PASS (R² = 0.9998)      |
|  [ CHECK LIMITS ]  [ ABORT (SAFE) ]   |  [14:02:11] Voltage headroom: +4.99 V (PASS)    |
+---------------------------------------+-------------------------------------------------+
```

### Analyze Workflow Features
- **Dataset Loading**: Supports both modern multichannel datasets and legacy single-channel sweeps without requiring reformatting.
- **Dynamic Parameter Table**: Displays active model parameters ($C_{\text{tes}}$, $C_{\text{abs}}$, $G_{\text{tes-bath}}$, $\alpha$, $\beta$, $R_0$, $P_0$, $R_{\text{shunt}}$, $L_{\text{nyq}}$, $T_c$, $T_{\text{bath}}$, $n$).
- **Predict Tc & Predict G**: Automatically solves the non-linear power balance equation to determine $T_c$ from measured operating point power $P_0$ and $G$, or computes $G$ directly from $T_c$ in closed form.
- **Manual $R_n$ Override**: Direct input field for normal resistance when IV curves do not extend sufficiently into the normal branch.
- **Interactive Model Curve Preview**: Instant theoretical curve calculation on plot updates upon any parameter change.
- **Bounded Fitter**: Non-linear least squares fit with parameter limits and freeze/float toggles converging in 10-30 ms.

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
| `--bias-range` | `START STOP STEP` for bias ranges (can be repeated) | `0.5 1.5 0.01` |
| `--flip-bias` | Invert stimulus phase / demodulation polarity | `False` |
| `--save-raw-all` | Also save the averaged time record of every channel (float32) | `False` |
| `--dry-run` | Run in simulation mode without hardware | `False` |
| `--config`, `-cfg` | Path to JSON settings file | `run_settings.json` |

---

### 2. Analysis & Thermal Model Fitting (`ez-cz-analyze`)

```bash
ez-cz-analyze [OPTIONS]
```

#### Common Options:
| Flag | Description | Default |
|:---|:---|:---|
| `--raw-dir`, `-d` | Path to folder containing measured `bias_*` subfolders | *(from config)* |
| `--iv-file`, `-i` | Corresponding IV curve `.npz` file for operating point extraction | *(from config)* |
| `--channel-id`, `-c` | Detector channel ID to analyze | `13` |
| `--tc` | Device critical temperature $T_c$ in Kelvin | `0.045` |
| `--tbath` | Cryostat bath temperature $T_{\text{bath}}$ in Kelvin | `0.020` |
| `--rshunt-uohm` | Shunt resistance $R_{\text{shunt}}$ in micro-Ohms | `250.0` |
| `--l-nyq-nh` | Nyquist circuit inductance $L_{\text{nyq}}$ in nano-Henries (or fit from SC TF) | *(auto-fit)* |
| `--output-base-dir`, `-o`| Directory where fit plots and summary CSVs are saved | `<raw_dir>/analysis/` |

---

## Configuration File (`run_settings.json`)

Both CLI tools and the GUI synchronize settings via `run_settings.json`:

```json
{
  "measure": {
    "dc_source": "awg",
    "channel": 13,
    "v_norm": 9.5,
    "vpp": 0.02,
    "bias_ranges": [
      [0.5, 1.5, 0.01],
      [1.5, 5.0, 0.1]
    ],
    "keysight_url": "vxi11://192.168.0.112:2021",
    "srs_url": "serial:///dev/ttyUSB1?baudrate=9600",
    "flip_bias": true
  },
  "analyze": {
    "channel_id": 13,
    "tc": 0.045,
    "tbath": 0.02,
    "rshunt_uOhm": 250.0,
    "l_nyq_nH": 119.13,
    "model": "Simple (1-body)",
    "save_in_raw": true
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
├── README.md                 # System architecture and technical documentation
├── src/
│   └── ez_cz/                # Core Python package
│       ├── __init__.py       # Package metadata and helper exports
│       ├── data_io.py        # Dataset loader, FFT extraction, and impedance derivation
│       ├── gui.py            # Unified PyQt5 measurement and analysis dashboard
│       ├── mdt3/             # Embedded microcal_design_tools thermal models
│       ├── measure.py        # Automated measurement CLI and hardware control
│       ├── models.py         # Vectorized thermal impedance models and power balance
│       └── analyze.py        # Batch thermal model fitting and parameter extraction
└── tests/
    └── test_lnyq_channel.py  # Unit and regression test suite
```

---

## Installation

```bash
git clone https://github.com/and-roy-d/EZ-CZ.git
cd EZ-CZ
pip install -e .
```

Installs EZ-CZ into your active Python environment in editable mode and registers command-line entry points:
- `ez-cz-gui`: Interactive graphical user interface
- `ez-cz-measure`: Terminal automation for multi-sine impedance sweeps
- `ez-cz-analyze`: Thermal impedance analysis and batch fitting

*(Alternatively with `uv`: `uv pip install -e .` or `uv sync`)*.

---

## Thermal Impedance Models

EZ-CZ includes self-contained implementations of the thermal models from the NIST `microcal_design_tools` (MDT) suite, eliminating external dependencies:

1. **Simple (1-body)** (`TES_Simple`):  
   TES heat capacity $C_{\text{tes}}$ coupled directly to the thermal bath via thermal conductance $G_{\text{tes-bath}}$.
2. **Dangling / Hanging (2-body)** (`TES_Dangling`):  
   TES heat capacity $C_{\text{tes}}$ coupled to bath with an internal dangling absorber heat capacity $C_{\text{abs}}$ connected via internal conductance $G_{\text{abs-tes}}$.
3. **Intervening (2-body)** (`TES_Intervening`):  
   TES heat capacity $C_{\text{tes}}$ coupled to an intermediate stage $C_{\text{int}}$, which connects to bath via $G_{\text{int-bath}}$.
4. **Intermediate + Dangling (3-body)** (`TES_Inter_Dangle`):  
   3-body model incorporating TES, intermediate thermal stage, and hanging absorber.
5. **Two-Fluid Model** (`TES_TwoFluid`):  
   Phenomenological two-fluid model parameterizing superconducting fraction $w_L$ and dynamic resistance.

---

## Theory of Operation & Signal Processing Pipeline

### 1. Multi-Tone Stimulus Generation
To measure complex impedance $Z_{\text{tes}}(f)$ rapidly across decades of frequency without perturbing the TES operating point, EZ-CZ synthesizes a multi-frequency sinusoidal excitation comb.

- **Flux-Ramp Harmonization**: The acquisition duration $T_{\text{meas}}$ is synchronized to an integer multiple of the SQUID flux-ramp demodulation period $t_{\text{FR}}$ ($t_{\text{FR}} = 256\text{ ns} \times N_{\text{points}}$). The discrete frequency bin spacing is:
  $$\Delta f = f_0 \approx \frac{1}{T_{\text{meas}}}$$
  Every excitation frequency $f_k$ is placed precisely on a Fourier bin center:
  $$f_k = m_k \Delta f, \quad m_k \in \mathbb{Z}^+$$
  This ensures that all tones undergo an exact integer number of cycles over the measurement window, eliminating spectral leakage and scallop loss without needing windowing functions.
- **Schroeder Phase Quadratic Dispersion**: When summing $N$ sinusoidal tones in phase ($\phi_k = 0$), the peak amplitude scales as $N \cdot V_{\text{tone}}$, resulting in a high Peak-to-Average Power Ratio (crest factor) that clips the AWG DAC memory ($\pm 1.0\text{ V}$ limit). EZ-CZ applies Schroeder phase dispersion:
  $$\phi_k = -\frac{\pi k (k - 1)}{N}, \quad k \in [0, N - 1]$$
  This spreads the individual tone peaks evenly across time, decreasing the crest factor by more than $7\times$ and allowing 100+ simultaneous tones at full signal-to-noise ratio.
- **Power Line Harmonic Rejection**: Multi-tone candidates are checked against AC power line harmonics ($k \cdot 60\text{ Hz}$). Frequencies satisfying:
  $$|f - \text{round}(f / f_{\text{line}}) \cdot f_{\text{line}}| < \delta f_{\text{tol}}$$
  (with $\delta f_{\text{tol}} = 2.0\text{ Hz}$) are automatically filtered out, preventing 60 Hz pickup from distorting low-frequency impedance bins.
- **Nyquist Bandwidth Protection**: Tones are bounded below the DAQ Nyquist frequency ($f_k < f_{\text{Nyquist}} - f_0/2$) and constrained by a user-configurable high-frequency cutoff (e.g. 50 kHz) to concentrate AWG dynamic range within the detector's electrical and thermal response band.

---

### 2. Tone Summation and Hardware Excitation
- **Waveform Synthesis & Normalization**: The individual tones are summed digitally:
  $$V_{\text{comb}}(t) = \sum_{k=0}^{N-1} A_k \cos(2\pi f_k t + \phi_k)$$
  The compiled waveform is normalized such that $\max |V_{\text{comb}}| \le 1.0\text{ V}$ and transferred to volatile arbitrary waveform memory on the Keysight 33600A AWG.
- **DC Bias Coupling**:
  - In `awg` mode, AWG Channel 2 applies both the AC stimulus comb and the DC bias offset simultaneously through hardware summing, routing to the cryostat via a single BNC line.
  - In `srs900` mode, an SRS SIM928 DC voltage source supplies the DC bias, which is combined with the AWG AC signal via an external hardware summing amplifier.
- **Slew-Rate Controlled Ramping**: Stepping DC bias across the superconducting transition can cause large current jumps ($\Delta I \sim 300\ \mu\text{A}$), generating flux jumps that exceed the SQUID feedback tracking rate. Voltage transitions use software-ramped steps ($\le 10\text{ mV}$ increments with 5 ms dwell, $\sim 2\text{ V/s}$) to guarantee continuous tracking lock.

---

### 3. Synchronous Acquisition and Coherent Demodulation
- **DASTARD Streaming**: Raw SQUID feedback counts are acquired synchronously over an exact integer number of flux-ramp cycles.
- **All Channels Saved**: Every channel DASTARD returns is processed at every bias. Each `vbias_*_data.npz` keeps the `--channel` response in `fft_unroll`/`raw_data` and adds, keyed `chanNN`, the tone responses of all channels (`fft_real`, `fft_imag`), their DC levels (`dc_mean_all`, the TES current for the operating point) and spreads (`raw_std_all`), plus `channel` and `channels`. The readout delay from the 0 V calibration of `--channel` is applied to all channels. `ez-cz-analyze --channel-id N` reads any recorded channel.
- **Coherent Matched-Filter Delay Recovery**: Readout cables, room-temperature electronics, and digital decimation introduce a propagation delay $\tau$. EZ-CZ recovers $\tau$ during the zero-bias superconducting calibration using an equalized matched filter:
  1. Derotate the raw FFT by the stimulus phases:
     $$u_{\text{raw}}[k] = \text{FFT}(I)[f_k] \cdot \exp(-j \phi_k)$$
  2. Equalize the magnitude using the fitted single-pole response:
     $$H_{\text{eq}}[k] = \frac{u_{\text{raw}}[k]}{Z_{\text{model}}[k]}$$
  3. Form unit-magnitude complex tones and compute the inverse FFT:
     $$h_{\text{impulse}}(t) = \text{IFFT}\left(\frac{H_{\text{eq}}}{|H_{\text{eq}}|}\right)$$
  Because Schroeder phases and device rolloff are equalized out, all tones constructively interfere into a sharp impulse at $t = \tau$, providing an unambiguous delay estimate that is subsequently refined via non-linear least squares.
- **Tone Extraction with Phase Derotation**: At each bias point, tone responses are extracted by Fourier demodulation and phase-derotated:
  $$u(f_k) = \text{FFT}(I)[f_k] \cdot \exp(-j \phi_k) \cdot \exp(-2\pi j f_k \tau)$$
  This analytically cancels stimulus dispersion and cable delay, yielding the calibrated complex transfer function $H(f)$.
- **Polarity Alignment (`flip_bias`)**: When current lead routing or SQUID demodulation slopes ($d\Phi/dI < 0$) invert the response sign, the polarity toggle inverts the demodulated response so that $\text{Re}(Z) > 0$ at DC.

---

### 4. Complex Impedance Calculation
- **Transfer Function Ratio Method**:
  In the superconducting state ($Z_{\text{tes}} = 0$), the circuit impedance is purely passive:
  $$Z_{\text{circuit}}(f) = R_{\text{shunt}} + 2\pi j f L_{\text{nyq}}$$
  The ratio of the measured transfer function at bias $V_{\text{bias}}$ to the zero-bias superconducting reference $u_0(f)$ relates directly to total impedance:
  $$\frac{u_{\text{bias}}(f)}{u_0(f)} = \frac{Z_{\text{circuit}}(f)}{Z_{\text{total}}(f)}$$
  $$Z_{\text{total}}(f) = \frac{Z_{\text{circuit}}(f)}{u_{\text{bias}}(f) / u_0(f)}$$
  $$Z_{\text{tes}}(f) = Z_{\text{total}}(f) - Z_{\text{circuit}}(f)$$
- **Lindeman Two-State Empirical Calibration**:
  For systems with frequency-dependent readout gains or distributed parasitics, EZ-CZ supports the empirical calibration method of Lindeman et al. (2007) [arXiv:0704.3066] using superconducting $u_{\text{sc}}(f)$ and normal-state $u_{\text{normal}}(f)$ transfer functions with known normal resistance $R_n$:
  $$Z_{\text{eq}}(f) = \frac{R_n}{(u_{\text{sc}}(f) / u_{\text{normal}}(f)) - 1}$$
  $$Z_{\text{tes}}(f) = Z_{\text{eq}}(f) \left(\frac{u_{\text{sc}}(f)}{u_{\text{bias}}(f)} - 1\right)$$

---

### 5. Hardware-Agnostic Circuit Inductance ($L_{\text{nyq}}$) Derivation
- **Roll-off Frequency Fit**:
  Rather than assuming a fixed inductance, EZ-CZ fits the zero-bias superconducting transfer function:
  $$|u_0(f)| = \frac{A_0}{\sqrt{1 + (f / f_{\text{roll}})^2}}$$
  The circuit Nyquist inductance is computed directly from the roll-off frequency:
  $$L_{\text{nyq}} = \frac{R_{\text{shunt}}}{2\pi f_{\text{roll}}}$$
  This formulation is hardware-agnostic: whether a channel has a nominal 120 nH bond, a 220 nH bond, or an external reference dataset, $L_{\text{nyq}}$ is derived directly from the observed transfer function for that channel. Explicit manual overrides in the GUI and CLI are also supported.

---

### 6. TES Operating Point & Explicit Power Balance
- **Normal Resistance ($R_n$) Determination**:
  Derived directly from the ratio of the superconducting ohmic slope $m_{\text{sc}} = (dI/dV)_{\text{sc}}$ and normal-state slope $m_n = (dI/dV)_n$ on the IV curve:
  $$R_n = R_{\text{shunt}} \left(\frac{m_{\text{sc}}}{m_n} - 1\right)$$
  A dedicated input field allows manual override if IV data lacks sufficient ohmic coverage.
- **Explicit Thermal Power Balance**:
  Steady-state power balance relates bias power $P_0 = I_0^2 R_0$ to bath temperature $T_{\text{bath}}$, critical temperature $T_c$, thermal conductance $G$, and thermal power index $n$:
  $$P_0 = \frac{G}{n} T_c \left[1 - \left(\frac{T_{\text{bath}}}{T_c}\right)^n\right]$$
  - **Predict $T_c$**: Solves the root equation $f(T_c) = 0$ numerically via Brent's method (`scipy.optimize.brentq`), calculating the true physical transition temperature consistent with measured power $P_0$ and conductance $G$.
  - **Predict $G$**: Inverts the relation in closed form:
    $$G = \frac{n P_0}{T_c \left[1 - (T_{\text{bath}} / T_c)^n\right]}$$
  Both calculations are exact mathematical inverses without arbitrary clamping or hardcoded temperature defaults.

---

### 7. FPGA Timing & Polyphase Filter Bank Architecture
The flux-ramp sample clock period ($t_{\text{point}} = 256\text{ ns}$) originates directly from the FPGA firmware polyphase filter bank channelizer:
- **ADC Sampling Rate**:
  $$f_{\text{ADC}} = 1.0\text{ GHz}$$
- **Filter Bank Parameters**: $\text{WINDOWLEN} = 2048$, $\text{NSETS} = 8$
- **Fine DDC Output Rate**:
  $$f_{\text{fineddc}} = \frac{f_{\text{ADC}}}{\text{WINDOWLEN}} \times \text{NSETS} = \frac{10^9\text{ Hz}}{2048} \times 8 = 3.90625\text{ MHz}$$
- **Time per Point**:
  $$t_{\text{point}} = \frac{1}{f_{\text{fineddc}}} = 256\text{ ns}$$
EZ-CZ dynamically queries these IP-XACT core parameters from FPGA hardware registers over UDP when connected, and gracefully falls back to $256\text{ ns}$ when operating offline or in simulation mode.
