import numpy as np
import time
import pylab as plt
import scipy
from scipy.fft import rfft, rfftfreq
from lmfit import Model, Parameters
import os
from datetime import datetime

import qsdumux
import qsdumux.instruments.srs_sim900 as sim900
import qsdumux.instruments.srs_sim928 as sim928
from dastardcommander import EasyClientDastard


#Target directory setup
from pathlib import Path  # optional, cleaner than os.path sometimes

run_start_time = datetime.now()
run_timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
run_date = run_start_time.strftime("%Y%m%d")

base_dir = Path("/data") / run_date / "Complex_Z" / run_timestamp
base_dir.mkdir(parents=True, exist_ok=True)

# Plot setup
plt.ion()
plt.close("all")

# Constants
fs_global = 250e3
output_rate_hz = 5e6  # AWG sample rate
f_low = 5
f0 = 20.0
n_periods_low_for_duration = 1
amplitude_multiplier_per_sine = 0.04
final_awg_amplitude_Vpp = 0.02

N_samples = int(fs_global * n_periods_low_for_duration / f_low)
ch_global = 13
# bias_values = [0.0]
dense = np.arange(0.5, 1.5, 0.01)

medium = np.arange(1.5, 5, 0.1)

# sparse = np.arange(3.5, 5, 0.1)

bias_values = np.concatenate([[0.0], dense, medium])

print(bias_values)

assert bias_values[0] == 0

# Voltage Box Setup
SRS_MAINFRAME_URL = "serial:///dev/ttyUSB1?baudrate=9600"
VBOX_PORT = 2
mainframe = sim900.SRS_SIM900(url=SRS_MAINFRAME_URL)
vbox = sim928.SRS_SIM928(mainframe, VBOX_PORT)

def apply_bias(voltage_box, voltage, v_norm=9.5):
    if voltage_box:
        print(f"Applying bias: {voltage:.3f} V")
        voltage_box.setvolt(v_norm)
        time.sleep(0.1)
        voltage_box.setvolt(voltage)
        time.sleep(0.5)
    else:
        print("Voltage box not initialized.")

# Generate waveform functions
def generate_freq_mutiples_logspaced(f0=20, start_exp=0, stop_exp=3, num_high_freq_tones=20):
    max_desired_freq = fs_global / 2
    max_allowable_multiplier = int(max_desired_freq / f0)
    multipliers = np.unique(np.int64(np.logspace(start_exp, stop_exp, num_high_freq_tones)))
    return [m for m in multipliers if 1 <= m <= max_allowable_multiplier]

def cos_superimposed(fmults, f0, f_low, n_periods_of_f_low, output_rate_hz, multiplier):
    t_duration = n_periods_of_f_low / f_low
    npts = int(round(t_duration * output_rate_hz))
    t = np.arange(npts) / output_rate_hz
    v_superimposed = np.zeros(npts)
    all_freqs = np.insert(np.array(fmults)*f0, 0, f_low)
    for f in all_freqs:
        v_superimposed += np.cos(2 * np.pi * t * f)
    return all_freqs, v_superimposed * multiplier

def sendArbFloatBinaryBlock2(instr, data, source, arb_name='myArb'):
    instr._checkSource(source)
    instr.clearVolatileMem(source)
    time.sleep(0.1)
    instr.setByteOrder(swap_bytes=True)
    packed_bytes = np.array(data, dtype=np.float32).tobytes()
    data_length = len(packed_bytes)
    header = f'SOURce{source}:DATA:ARB {arb_name}, #{len(str(data_length))}{data_length}'.encode('ascii')
    instr._write_raw(header + packed_bytes + b"\n")

def send_waveform(vals, output_num=2):
    fg = qsdumux.instruments.Keysight33600A_usb(url="vxi11://192.168.0.112:2021")
    sendArbFloatBinaryBlock2(fg, vals, output_num, "newone")
    fg.setFunction(waveform_function='ARB', source=output_num)
    fg.setArbFile(output_num, "newone")
    fg.setArbSampleRate(output_num, output_rate_hz)
    fg.setArbFilter(output_num, filter_state='STEP')
    fg.setAmplitude(output_num, final_awg_amplitude_Vpp)
    fg.setOutputState(source=output_num, output_state='ON')
    time.sleep(2)
    for _ in range(100):
        fg._write_raw(b"SYST:ERR?")
        if fg.readline().startswith("+0"):
            break


def save_all_data_npz(save_dir, bias_val, v_stim, data_all, N_samples, N_averages,
                      amplitude_vpp, fs_sch, fft_freqs, tau_shift, fs_global, multiplier):
    
    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs(save_dir, exist_ok=True)

    bias_str = f"{bias_val:.2f}".replace(".", "v")
    amp_str = f"{amplitude_vpp:.3f}".replace(".", "v")
    filename = f"vbias_{bias_str}_vac_{amp_str}vpp.npz"
    full_path = save_dir / filename

    data_dict = {}
    fft_real_dict = {}
    fft_imag_dict = {}

    # Compute applied voltage at each frequency
    true_voltage_at_frequencies = np.array([multiplier * amplitude_vpp for _ in fft_freqs])

    f_sch_idx = np.searchsorted(fft_freqs, fs_sch)

    for k, v in data_all.items():
        if k.startswith("chan"):
            avg = v[:N_samples * N_averages].reshape((N_averages, N_samples)).mean(axis=0)
            
            firstFrameIndexMod = data_all["firstFrameIndex"][0] % N_samples
            data = np.roll(avg, firstFrameIndexMod)
            data_dict[k] = data

            # Unroll FFT phase
            fft_data = rfft(data)
            fft_unrolled = fft_data * np.exp(-2j * np.pi * fft_freqs * tau_shift)
            fft_unrolled_selected = fft_unrolled[f_sch_idx]

            fft_real_dict[k] = np.real(fft_unrolled_selected)
            fft_imag_dict[k] = np.imag(fft_unrolled_selected)

    np.savez(
        full_path,
        data=data_dict,
        V_stim=v_stim,
        fft_real=fft_real_dict,
        fft_imag=fft_imag_dict,
        fs_sch=fs_sch,
        true_voltage_at_frequencies=true_voltage_at_frequencies  # Save the applied voltage at each frequency
    )
    print(f"Saved full dataset: {full_path}")



# Try connecting to DASTARD
for _ in range(10):
    try:
        c = EasyClientDastard()
        break
    except:
        print("DASTARD connection failed. Retrying...")
        time.sleep(0.1)



trigger_function_generator = False
tau_shift_0V = None
fft_unrolls_by_bias = {}
raw_data_by_bias = {}
V_stim = amplitude_multiplier_per_sine*final_awg_amplitude_Vpp

for idx, bias in enumerate(bias_values):
    apply_bias(vbox, bias)
    print(f"Now starting {bias:.3f} V bias point. Step {idx+1} of {len(bias_values)}....")

    if idx == 0:
        f_mults_list = generate_freq_mutiples_logspaced(f0=f0)
        fs_sch, v_sch = cos_superimposed(f_mults_list, f0, f_low, n_periods_low_for_duration, output_rate_hz, amplitude_multiplier_per_sine)
        # send_waveform(v_sch)

    print(f"[{idx+1}/{len(bias_values)}] Requesting {N_samples} samples from DASTARD at bias = {bias:.3f} V")
    data_all = c.getNewData(N_samples * 4)
    data_raw = data_all[f"chan{ch_global}"] * 1.0
    data_mean = data_raw[:N_samples * 4].reshape((4, N_samples)).mean(axis=0)
    firstFrameIndexMod = data_all["firstFrameIndex"][0] % N_samples
    data = np.roll(data_mean, firstFrameIndexMod)
    raw_data_by_bias[bias] = data

    fft_data_raw = scipy.fft.rfft(data)
    fft_freqs = scipy.fft.rfftfreq(N_samples, d=1/fs_global)
    f_sch_idx = np.searchsorted(fft_freqs, fs_sch)

    # Circuit model
    Rs = 250e-6
    L = 1000e-9

    omega_fft_model = 2 * np.pi * fs_sch
    Z_L = 1j * omega_fft_model * L
    Z_model = Rs/(Rs + Z_L)*9.6e6

    if idx == 0:
        
        def real_model(fs, tau_shift):
            return np.real((fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift))[f_sch_idx])
        def magnitude_model(fs, amp0, f_roll):
            return amp0/np.sqrt(1+(fs/f_roll)**2)

        model_mag = Model(magnitude_model)
        params_mag = Parameters()
        params_mag.add('amp0', value=np.abs(fft_data_raw[0]), vary=True)
        params_mag.add('f_roll', value=1e3, vary=True)
        result_mag = model_mag.fit(np.abs(fft_data_raw[f_sch_idx]), fs=fs_sch, params=params_mag)
        result_mag.plot()
        amp0 = result_mag.params["amp0"].value
        f_roll = result_mag.params["f_roll"].value

        Z_model_pure = amp0/(1-1j*fs_sch/f_roll)


        model_real = Model(real_model)
        tau_guess = -np.argmax(data) / fs_global
        params = Parameters()
        params.add('tau_shift', value=tau_guess, vary=True)
        result_real = model_real.fit(np.real(Z_model_pure), fs=fs_sch, params=params)
        result_real.plot()
        print(result_real.fit_report())
        tau_shift_0V = result_real.params['tau_shift'].value
        

    fft_unroll = (fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift_0V))[f_sch_idx]
    fft_unrolls_by_bias[bias] = fft_unroll
    bias_str = f"{bias:.2f}".replace(".", "v")

    # Save plots but only show for 0 V
    bias_dir = base_dir / f"bias_{bias_str}"
    bias_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Bias = {bias:.2f} V, τ_shift = {tau_shift_0V:.6e} s", fontsize=14)
    axs = axs.flatten()

    axs[0].semilogx(fs_sch, np.real(fft_unroll), ".-", label="real")
    axs[0].semilogx(fs_sch, np.imag(fft_unroll), ".-", label="imag")
    axs[0].semilogx(fs_sch, np.real(Z_model_pure), ".-", label="real model")
    axs[0].semilogx(fs_sch, np.imag(Z_model_pure), ".-", label="imag model")
    axs[0].set_title("Real and Imaginary Parts")
    axs[0].legend()
    axs[0].grid(True, which="both")

    axs[1].plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", label="data")
    axs[1].plot(np.real(Z_model_pure), np.imag(Z_model_pure), ".-", label="model")
    axs[1].set_title("Nyquist Plot")
    axs[1].grid(True)
    axs[1].axis("equal")

    axs[2].loglog(fs_sch, np.abs(fft_unroll), ".-", label="data")
    axs[2].loglog(fs_sch, np.abs(Z_model_pure), ".-", label="model")
    axs[2].set_title("Magnitude Response")
    axs[2].grid(True, which="both")

    axs[3].semilogx(fs_sch, np.angle(fft_unroll)/np.pi, ".-", label="data")
    axs[3].semilogx(fs_sch, np.angle(Z_model_pure)/np.pi, ".-", label="model")
    axs[3].set_title("Phase Response")
    axs[3].grid(True, which="both")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_filename = f"vbias_{bias_str}_impedance_plot.png"
    plot_path = bias_dir / plot_filename
    fig.savefig(plot_path)



 
    plt.close("all")

    save_all_data_npz(
        save_dir=bias_dir,
        bias_val=bias,
        v_stim=V_stim,
        data_all=data_all,
        N_samples=N_samples,
        N_averages=4,
        amplitude_vpp=final_awg_amplitude_Vpp,
        fs_sch=fs_sch,
        fft_freqs=fft_freqs,
        tau_shift=tau_shift_0V,
        fs_global=fs_global,
        multiplier=amplitude_multiplier_per_sine,
    )

    print(f"Bias {bias:.3f} V done. Saved to {bias_dir}")
    print("\n---------------------------------------------------------------------------------------\n")



