import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import queue
import threading
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lmfit import Model, Parameters

# Optional imports for hardware control
try:
    import qsdumux
    import qsdumux.instruments.srs_sim900 as sim900
    import qsdumux.instruments.srs_sim928 as sim928
    HAS_HARDWARE_LIBS = True
except ImportError:
    HAS_HARDWARE_LIBS = False

try:
    from dastardcommander import EasyClientDastard
    HAS_DASTARD_LIB = True
except ImportError:
    HAS_DASTARD_LIB = False


# Helper function to generate logspaced freq multiples
def generate_freq_mutiples_logspaced(f0=20, start_exp=0, stop_exp=3, num_high_freq_tones=20, fs_global=250e3):
    max_desired_freq = fs_global / 2
    max_allowable_multiplier = int(max_desired_freq / f0)
    multipliers = np.unique(np.int64(np.logspace(start_exp, stop_exp, num_high_freq_tones)))
    return [m for m in multipliers if 1 <= m <= max_allowable_multiplier]


# Helper function to generate superimposed sines
def cos_superimposed(fmults, f0, f_low, n_periods_of_f_low, output_rate_hz, multiplier):
    t_duration = n_periods_of_f_low / f_low
    npts = int(round(t_duration * output_rate_hz))
    t = np.arange(npts) / output_rate_hz
    v_superimposed = np.zeros(npts)
    all_freqs = np.insert(np.array(fmults) * f0, 0, f_low)
    for f in all_freqs:
        v_superimposed += np.cos(2 * np.pi * t * f)
    return all_freqs, v_superimposed * multiplier


# ----------------------------------------------------------------------
# Mock Hardware Classes for Simulation Mode
# ----------------------------------------------------------------------
class MockEasyClientDastard:
    def __init__(self, channel, fs_global, f_low, f0, amp_mult, vpp, rs_uohm, l_nh):
        self.channel = channel
        self.fs_global = fs_global
        self.f_low = f_low
        self.f0 = f0
        self.amp_mult = amp_mult
        self.vpp = vpp
        self.rs_uohm = rs_uohm
        self.l_nh = l_nh
        self.current_bias = 0.0
        
        # Calculate stimulus frequencies
        f_mults = generate_freq_mutiples_logspaced(self.f0, 0, 3, 20, self.fs_global)
        self.fs_sch = np.insert(np.array(f_mults) * self.f0, 0, self.f_low)

    def set_bias(self, bias):
        self.current_bias = bias

    def getNewData(self, num_samples):
        # Generate time array
        t = np.arange(num_samples) / self.fs_global
        
        # Cable/digital delay simulation
        tau_delay = -2.70625e-2
        t_delayed = t + tau_delay

        # Rnormal of simulated TES is 12 mOhm
        Rnormal = 12e-3
        Rshunt = self.rs_uohm * 1e-6
        L = self.l_nh * 1e-9

        # Simulated superconducting transition profile
        if self.current_bias == 0.0:
            R0 = 0.0
        else:
            R0 = Rnormal / (1.0 + np.exp(-(self.current_bias - 1.2) / 0.2))

        Vpp_tone = self.amp_mult * self.vpp
        data = np.zeros(num_samples)
        
        # Sum stimulus tone currents (I = V / Z)
        for f in self.fs_sch:
            Z = (R0 + Rshunt) + 1j * 2 * np.pi * f * L
            data += np.real((Vpp_tone / Z) * np.exp(1j * (2 * np.pi * f * t_delayed)))

        # Add simulated digitizer noise
        data += np.random.normal(0, 5e-7, num_samples)

        return {
            f"chan{self.channel}": data,
            "firstFrameIndex": [0]
        }


class MockSRS_SIM928:
    def __init__(self):
        pass
    def setvolt(self, voltage):
        print(f"[Simulation] SIM928 set bias voltage to: {voltage:.3f} V")


class MockKeysight33600A:
    def __init__(self):
        pass
    def setFunction(self, *args, **kwargs): pass
    def setArbFile(self, *args, **kwargs): pass
    def setArbSampleRate(self, *args, **kwargs): pass
    def setArbFilter(self, *args, **kwargs): pass
    def setAmplitude(self, *args, **kwargs): pass
    def setOutputState(self, *args, **kwargs): pass
    def clearVolatileMem(self, *args, **kwargs): pass
    def setByteOrder(self, *args, **kwargs): pass
    def _write_raw(self, *args, **kwargs): pass
    def readline(self): return b"+0"
    def _checkSource(self, *args, **kwargs): pass


# ----------------------------------------------------------------------
# Tkinter GUI Main Class
# ----------------------------------------------------------------------
class EZCZGui:
    def __init__(self, root):
        self.root = root
        self.root.title("EZ-CZ Complex Impedance Live Sweeper")
        self.root.geometry("1200x800")
        
        # Theme configuration (Dark Slate Premium Theme)
        self.bg_color = "#1e1e1e"
        self.fg_color = "#e0e0e0"
        self.accent_color = "#007acc"
        self.danger_color = "#d9534f"
        
        self.root.configure(bg=self.bg_color)
        
        # Setup UI styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background=self.bg_color, foreground=self.fg_color, fieldbackground="#333333")
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TEntry", fieldbackground="#333333", foreground=self.fg_color, font=("Segoe UI", 10))
        self.style.configure("TCheckbutton", background=self.bg_color, foreground=self.fg_color)
        
        # Threading control
        self.running_thread = None
        self.stop_requested = threading.Event()
        self.update_queue = queue.Queue()
        
        # Settings file location
        self.settings_file = Path("run_settings.json")
        self.settings = self.load_settings()

        # Build GUI Panels
        self.build_gui()
        
        # Load variables from configuration file
        self.load_variables_into_inputs()
        
        # Start periodic GUI queue polling
        self.root.after(100, self.process_queue)

    def load_settings(self):
        defaults = {
            "fs": 250000.0,
            "channel": 13,
            "v_norm": 9.5,
            "awg_rate": 5000000.0,
            "f_low": 5.0,
            "f0": 20.0,
            "amp_mult": 0.04,
            "vpp": 0.02,
            "rs_uOhm": 250.0,
            "l_nH": 1000.0,
            "trigger_awg": False,
            "srs_url": "serial:///dev/ttyUSB1?baudrate=9600",
            "srs_port": 2,
            "keysight_url": "vxi11://192.168.0.112:2021",
            "bias_dense": [0.5, 1.5, 0.01],
            "bias_medium": [1.5, 5.0, 0.1],
            "output_dir": ""
        }
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                    return {**defaults, **data.get("measure", {})}
            except Exception as e:
                print(f"Failed to load configuration file: {e}")
        return defaults

    def save_settings_to_file(self):
        # Fetch current UI inputs
        measure_settings = self.get_settings_from_inputs()
        if not measure_settings:
            return
        
        full_settings_data = {}
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    full_settings_data = json.load(f)
            except Exception:
                pass
        
        full_settings_data["measure"] = measure_settings
        
        try:
            with open(self.settings_file, "w") as f:
                json.dump(full_settings_data, f, indent=2)
            messagebox.showinfo("Config Saved", f"Successfully saved defaults to {self.settings_file.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def build_gui(self):
        # Master grid setup
        self.root.columnconfigure(0, weight=0, minsize=380)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ----------------------------------------------------------------------
        # Left Panel (Inputs and Controls)
        # ----------------------------------------------------------------------
        left_frame = tk.Frame(self.root, bg=self.bg_color, padx=15, pady=15, highlightthickness=1, highlightbackground="#333333")
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.columnconfigure(1, weight=1)

        # Header Label
        header = tk.Label(left_frame, text="EZ-CZ CONTROL BOARD", font=("Segoe UI", 14, "bold"), bg=self.bg_color, fg="#ff9900")
        header.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky="w")

        # Variables Dict for UI Inputs
        self.entries = {}
        self.trigger_awg_var = tk.BooleanVar()
        self.simulation_var = tk.BooleanVar(value=not (HAS_HARDWARE_LIBS and HAS_DASTARD_LIB))

        # Inputs mapping config
        inputs_config = [
            ("channel", "Dastard Channel ID:", "13"),
            ("fs", "Digi Sample Rate (Hz):", "250000.0"),
            ("rs_uOhm", "Rs (uOhm):", "250.0"),
            ("l_nH", "L (nH):", "1000.0"),
            ("v_norm", "Normalization Volt (V):", "9.5"),
            ("vpp", "Tone Amplitude Vpp:", "0.02"),
            ("amp_mult", "Tone Amp Multiplier:", "0.04"),
            ("awg_rate", "AWG Sample Rate (Hz):", "5000000.0"),
            ("f_low", "Low Freq Tone (Hz):", "5.0"),
            ("f0", "Base Tone Freq F0 (Hz):", "20.0"),
            ("srs_url", "SRS URL / Port:", "serial:///dev/ttyUSB1?baudrate=9600"),
            ("keysight_url", "Keysight AWG URL:", "vxi11://192.168.0.112:2021"),
        ]

        row = 1
        for key, label_text, default_val in inputs_config:
            lbl = ttk.Label(left_frame, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", pady=4)
            
            ent = ttk.Entry(left_frame)
            ent.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)
            self.entries[key] = ent
            row += 1

        # SRS Port (Slot) Entry (small next to URL)
        # We replace URL entry layout to hold both url and port side-by-side
        self.entries["srs_url"].grid_forget()
        srs_subframe = tk.Frame(left_frame, bg=self.bg_color)
        srs_subframe.grid(row=11, column=1, sticky="ew", padx=(10, 0), pady=4)
        srs_subframe.columnconfigure(0, weight=3)
        srs_subframe.columnconfigure(1, weight=1)

        ent_url = ttk.Entry(srs_subframe)
        ent_url.pack(side="left", fill="x", expand=True)
        self.entries["srs_url"] = ent_url

        ent_port = ttk.Entry(srs_subframe, width=5)
        ent_port.pack(side="right", padx=(5, 0))
        self.entries["srs_port"] = ent_port

        # Bias Sweeps Inputs
        lbl_dense = ttk.Label(left_frame, text="Dense Bias [Start, Stop, Step]:")
        lbl_dense.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 2))
        row += 1

        dense_frame = tk.Frame(left_frame, bg=self.bg_color)
        dense_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        dense_frame.columnconfigure((0, 1, 2), weight=1)
        self.entries["bias_dense_start"] = ttk.Entry(dense_frame)
        self.entries["bias_dense_start"].grid(row=0, column=0, padx=2)
        self.entries["bias_dense_stop"] = ttk.Entry(dense_frame)
        self.entries["bias_dense_stop"].grid(row=0, column=1, padx=2)
        self.entries["bias_dense_step"] = ttk.Entry(dense_frame)
        self.entries["bias_dense_step"].grid(row=0, column=2, padx=2)
        row += 1

        lbl_med = ttk.Label(left_frame, text="Medium Bias [Start, Stop, Step]:")
        lbl_med.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        row += 1

        med_frame = tk.Frame(left_frame, bg=self.bg_color)
        med_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        med_frame.columnconfigure((0, 1, 2), weight=1)
        self.entries["bias_med_start"] = ttk.Entry(med_frame)
        self.entries["bias_med_start"].grid(row=0, column=0, padx=2)
        self.entries["bias_med_stop"] = ttk.Entry(med_frame)
        self.entries["bias_med_stop"].grid(row=0, column=1, padx=2)
        self.entries["bias_med_step"] = ttk.Entry(med_frame)
        self.entries["bias_med_step"].grid(row=0, column=2, padx=2)
        row += 1

        # Checkboxes Frame
        chk_frame = tk.Frame(left_frame, bg=self.bg_color)
        chk_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=10)
        row += 1

        chk_awg = ttk.Checkbutton(chk_frame, text="Trigger Keysight AWG", variable=self.trigger_awg_var)
        chk_awg.pack(anchor="w", pady=2)

        chk_sim = ttk.Checkbutton(chk_frame, text="Simulation (Dry Run Mode)", variable=self.simulation_var)
        chk_sim.pack(anchor="w", pady=2)
        if not (HAS_HARDWARE_LIBS and HAS_DASTARD_LIB):
            chk_sim.configure(state="disabled")

        # Save Defaults Config Button
        btn_save = tk.Button(left_frame, text="Save Settings as Default", command=self.save_settings_to_file,
                             bg="#333333", fg=self.fg_color, relief="flat", activebackground="#444444", activeforeground=self.fg_color)
        btn_save.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        row += 1

        # Separation Bar
        sep = ttk.Separator(left_frame, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        row += 1

        # Sweep Status Labels
        status_header = ttk.Label(left_frame, text="Status Panel:", font=("Segoe UI", 11, "bold"))
        status_header.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_lbl = tk.Label(left_frame, text="Idle", font=("Segoe UI", 11, "bold"), fg="cyan", bg=self.bg_color)
        self.status_lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.progress_lbl = ttk.Label(left_frame, text="Click GO to start bias sweeps.")
        self.progress_lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        # Run Controls Buttons
        ctrl_frame = tk.Frame(left_frame, bg=self.bg_color)
        ctrl_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=15)
        ctrl_frame.columnconfigure((0, 1), weight=1)

        self.btn_go = tk.Button(ctrl_frame, text="GO", command=self.start_sweep,
                                bg="#28a745", fg="white", font=("Segoe UI", 12, "bold"), relief="flat", activebackground="#218838")
        self.btn_go.grid(row=0, column=0, padx=5, sticky="ew")

        self.btn_stop = tk.Button(ctrl_frame, text="STOP", command=self.stop_sweep, state="disabled",
                                  bg=self.danger_color, fg="white", font=("Segoe UI", 12, "bold"), relief="flat", activebackground="#c9302c")
        self.btn_stop.grid(row=0, column=1, padx=5, sticky="ew")

        # ----------------------------------------------------------------------
        # Right Panel (Real-time Matplotlib Plot Canvas)
        # ----------------------------------------------------------------------
        right_frame = tk.Frame(self.root, bg="#111111", padx=10, pady=10)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        # Style Matplotlib to match Dark Theme
        plt.style.use('dark_background')
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.patch.set_facecolor('#111111')
        for ax in self.axs.flat:
            ax.set_facecolor('#1e1e1e')
            ax.grid(True, which="both", color="#333333", linestyle=":")
        self.fig.suptitle("EZ-CZ Live Sweeper Plot", color="#ff9900", fontsize=14, weight="bold")
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Embed figure in Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_variables_into_inputs(self):
        # Load scalar settings
        for key, ent in self.entries.items():
            if key in ["srs_port", "bias_dense_start", "bias_dense_stop", "bias_dense_step", 
                       "bias_med_start", "bias_med_stop", "bias_med_step"]:
                continue
            ent.insert(0, str(self.settings.get(key, "")))

        self.entries["srs_port"].insert(0, str(self.settings.get("srs_port", 2)))

        # Load bias dense
        bd = self.settings.get("bias_dense", [0.5, 1.5, 0.01])
        self.entries["bias_dense_start"].insert(0, str(bd[0]))
        self.entries["bias_dense_stop"].insert(0, str(bd[1]))
        self.entries["bias_dense_step"].insert(0, str(bd[2]))

        # Load bias med
        bm = self.settings.get("bias_medium", [1.5, 5.0, 0.1])
        self.entries["bias_med_start"].insert(0, str(bm[0]))
        self.entries["bias_med_stop"].insert(0, str(bm[1]))
        self.entries["bias_med_step"].insert(0, str(bm[2]))

        # Load checkboxes
        self.trigger_awg_var.set(self.settings.get("trigger_awg", False))

    def get_settings_from_inputs(self):
        try:
            dense_vals = [
                float(self.entries["bias_dense_start"].get()),
                float(self.entries["bias_dense_stop"].get()),
                float(self.entries["bias_dense_step"].get())
            ]
            med_vals = [
                float(self.entries["bias_med_start"].get()),
                float(self.entries["bias_med_stop"].get()),
                float(self.entries["bias_med_step"].get())
            ]
            
            return {
                "channel": int(self.entries["channel"].get()),
                "fs": float(self.entries["fs"].get()),
                "rs_uOhm": float(self.entries["rs_uOhm"].get()),
                "l_nH": float(self.entries["l_nH"].get()),
                "v_norm": float(self.entries["v_norm"].get()),
                "vpp": float(self.entries["vpp"].get()),
                "amp_mult": float(self.entries["amp_mult"].get()),
                "awg_rate": float(self.entries["awg_rate"].get()),
                "f_low": float(self.entries["f_low"].get()),
                "f0": float(self.entries["f0"].get()),
                "srs_url": self.entries["srs_url"].get(),
                "srs_port": int(self.entries["srs_port"].get()),
                "keysight_url": self.entries["keysight_url"].get(),
                "bias_dense": dense_vals,
                "bias_medium": med_vals,
                "trigger_awg": self.trigger_awg_var.get(),
                "is_sim": self.simulation_var.get(),
                "output_dir": ""
            }
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Please enter valid numeric parameters.\nDetail: {e}")
            return None

    def start_sweep(self):
        config = self.get_settings_from_inputs()
        if not config:
            return
        
        # UI Toggles
        self.btn_go.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_lbl.configure(text="Running", fg="#28a745")
        
        self.stop_requested.clear()
        
        # Start background sweep thread
        self.running_thread = threading.Thread(target=self.run_sweep_loop, args=(config,))
        self.running_thread.daemon = True
        self.running_thread.start()

    def stop_sweep(self):
        self.stop_requested.set()
        self.status_lbl.configure(text="Stopping...", fg="#ffc107")
        self.progress_lbl.configure(text="Terminating sweep after current step...")
        self.btn_stop.configure(state="disabled")

    # ----------------------------------------------------------------------
    # Core Measurement Sweep Thread Loop
    # ----------------------------------------------------------------------
    def run_sweep_loop(self, config):
        try:
            is_sim = config.get("is_sim", False)
            
            # Setup sweep bias values
            dense = np.arange(config["bias_dense"][0], config["bias_dense"][1], config["bias_dense"][2])
            medium = np.arange(config["bias_medium"][0], config["bias_medium"][1], config["bias_medium"][2])
            bias_values = np.concatenate([[0.0], dense, medium])
            
            # Initialize hardware instruments
            vbox = None
            dastard_client = None

            if is_sim:
                self.update_queue.put(("log", "[Simulation Mode] Connecting mock instruments..."))
                vbox = MockSRS_SIM928()
                dastard_client = MockEasyClientDastard(
                    channel=config["channel"],
                    fs_global=config["fs"],
                    f_low=config["f_low"],
                    f0=config["f0"],
                    amp_mult=config["amp_mult"],
                    vpp=config["vpp"],
                    rs_uohm=config["rs_uOhm"],
                    l_nh=config["l_nH"]
                )
            else:
                self.update_queue.put(("log", "Connecting to hardware instruments..."))
                try:
                    mainframe = sim900.SRS_SIM900(url=config["srs_url"])
                    vbox = sim928.SRS_SIM928(mainframe, config["srs_port"])
                except Exception as e:
                    self.update_queue.put(("error", f"Failed to connect to SRS: {e}"))
                    return

                try:
                    dastard_client = EasyClientDastard()
                except Exception as e:
                    self.update_queue.put(("error", f"Failed to connect to Dastard: {e}"))
                    return
            
            n_periods_low_for_duration = 1
            N_samples = int(config["fs"] * n_periods_low_for_duration / config["f_low"])
            V_stim = config["amp_mult"] * config["vpp"]
            
            tau_shift_0V = None
            fs_sch = None
            Z_model_pure = None
            fft_freqs = None
            f_sch_idx = None
            
            total_steps = len(bias_values)
            
            # Outer Sweep Loop (Infinite run loop requested: "keeps making bias sweeps")
            sweep_count = 1
            while not self.stop_requested.is_set():
                self.update_queue.put(("log", f"Starting Bias Sweep #{sweep_count} ({total_steps} points)..."))
                
                for idx, bias in enumerate(bias_values):
                    if self.stop_requested.is_set():
                        break
                    
                    self.update_queue.put(("log", f"Sweep #{sweep_count} | Step {idx+1}/{total_steps} | Bias = {bias:.3f} V"))
                    
                    # Apply bias
                    if is_sim:
                        vbox.setvolt(bias)
                        dastard_client.set_bias(bias)
                    else:
                        if vbox:
                            vbox.setvolt(config["v_norm"])
                            time.sleep(0.1)
                            vbox.setvolt(bias)
                            time.sleep(0.5)
                    
                    # Set waveform on AWG for the 0V step
                    if idx == 0:
                        f_mults_list = generate_freq_mutiples_logspaced(
                            f0=config["f0"], start_exp=0, stop_exp=3, num_high_freq_tones=20, fs_global=config["fs"]
                        )
                        fs_sch, v_sch = cos_superimposed(
                            f_mults_list, config["f0"], config["f_low"], n_periods_low_for_duration, config["awg_rate"], config["amp_mult"]
                        )
                        
                        if config["trigger_awg"] and not is_sim:
                            # Direct execution keysight call
                            try:
                                fg = qsdumux.instruments.Keysight33600A_usb(url=config["keysight_url"])
                                # Setup arb block
                                fg._checkSource(2)
                                fg.clearVolatileMem(2)
                                time.sleep(0.1)
                                fg.setByteOrder(swap_bytes=True)
                                packed_bytes = np.array(v_sch, dtype=np.float32).tobytes()
                                header = f'SOURce2:DATA:ARB newone, #{len(str(len(packed_bytes)))}{len(packed_bytes)}'.encode('ascii')
                                fg._write_raw(header + packed_bytes + b"\n")
                                
                                fg.setFunction(waveform_function='ARB', source=2)
                                fg.setArbFile(2, "newone")
                                fg.setArbSampleRate(2, config["awg_rate"])
                                fg.setArbFilter(2, filter_state='STEP')
                                fg.setAmplitude(2, config["vpp"])
                                fg.setOutputState(source=2, output_state='ON')
                                time.sleep(2)
                                for _ in range(100):
                                    fg._write_raw(b"SYST:ERR?")
                                    if fg.readline().startswith(b"+0"):
                                        break
                            except Exception as e:
                                self.update_queue.put(("log", f"Warning: Keysight error: {e}"))
                    
                    # Fetch Dastard current data
                    data_all = dastard_client.getNewData(N_samples * 4)
                    data_raw = data_all[f"chan{config['channel']}"] * 1.0
                    data_mean = data_raw[:N_samples * 4].reshape((4, N_samples)).mean(axis=0)
                    firstFrameIndexMod = data_all["firstFrameIndex"][0] % N_samples
                    data = np.roll(data_mean, firstFrameIndexMod)
                    
                    # FFT
                    fft_data_raw = scipy.fft.rfft(data)
                    fft_freqs = scipy.fft.rfftfreq(N_samples, d=1/config["fs"])
                    f_sch_idx = np.searchsorted(fft_freqs, fs_sch)
                    
                    # Fit 0V pole to get time shift
                    if idx == 0:
                        def magnitude_model(fs, amp0, f_roll):
                            return amp0 / np.sqrt(1 + (fs / f_roll)**2)

                        model_mag = Model(magnitude_model)
                        params_mag = Parameters()
                        params_mag.add('amp0', value=np.abs(fft_data_raw[0]), vary=True)
                        params_mag.add('f_roll', value=1e3, vary=True)
                        result_mag = model_mag.fit(np.abs(fft_data_raw[f_sch_idx]), fs=fs_sch, params=params_mag)
                        amp0 = result_mag.params["amp0"].value
                        f_roll = result_mag.params["f_roll"].value

                        Z_model_pure = amp0 / (1 + 1j * fs_sch / f_roll)

                        def real_model(fs, tau_shift):
                            return np.real((fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift))[f_sch_idx])

                        model_real = Model(real_model)
                        tau_guess = -np.argmax(data) / config["fs"]
                        params = Parameters()
                        params.add('tau_shift', value=tau_guess, vary=True)
                        result_real = model_real.fit(np.real(Z_model_pure), fs=fs_sch, params=params)
                        tau_shift_0V = result_real.params['tau_shift'].value
                    
                    # Unroll data phase
                    fft_unroll = (fft_data_raw * np.exp(-2j * np.pi * fft_freqs * tau_shift_0V))[f_sch_idx]
                    
                    # Post data update back to Tkinter UI thread
                    plot_data = {
                        "bias": bias,
                        "tau_shift": tau_shift_0V,
                        "fs_sch": fs_sch,
                        "fft_unroll": fft_unroll,
                        "Z_model_pure": Z_model_pure
                    }
                    self.update_queue.put(("plot", plot_data))
                    
                    # Sleep slightly between points to keep background loop friendly
                    time.sleep(0.1)
                
                sweep_count += 1
                
            self.update_queue.put(("finished", None))
        except Exception as e:
            self.update_queue.put(("error", f"An error occurred in background sweep:\n{e}"))

    # ----------------------------------------------------------------------
    # UI Thread Communication and Matplotlib Plot Redrawing
    # ----------------------------------------------------------------------
    def process_queue(self):
        try:
            while True:
                msg_type, payload = self.update_queue.get_nowait()
                if msg_type == "log":
                    self.progress_lbl.configure(text=payload)
                elif msg_type == "error":
                    messagebox.showerror("Sweep Error", payload)
                    self.reset_ui()
                elif msg_type == "finished":
                    self.reset_ui()
                elif msg_type == "plot":
                    self.redraw_plots(payload)
        except queue.Empty:
            pass
        # Keep polling queue
        self.root.after(100, self.process_queue)

    def redraw_plots(self, data):
        # Extract data payload
        bias = data["bias"]
        tau = data["tau_shift"]
        fs_sch = data["fs_sch"]
        fft_unroll = data["fft_unroll"]
        Z_model_pure = data["Z_model_pure"]
        
        # Clear plots
        for ax in self.axs.flat:
            ax.clear()
            ax.grid(True, which="both", color="#333333", linestyle=":")
            
        self.fig.suptitle(f"Bias = {bias:.3f} V, τ_shift = {tau:.6e} s", color="#ff9900", fontsize=14, weight="bold")
        
        # Plot 0: Real/Imag parts
        self.axs[0, 0].semilogx(fs_sch, np.real(fft_unroll), ".-", label="Real Data", color="#1f77b4")
        self.axs[0, 0].semilogx(fs_sch, np.imag(fft_unroll), ".-", label="Imag Data", color="#ff7f0e")
        self.axs[0, 0].semilogx(fs_sch, np.real(Z_model_pure), "--", label="Real Model", color="#2ca02c")
        self.axs[0, 0].semilogx(fs_sch, np.imag(Z_model_pure), "--", label="Imag Model", color="#d62728")
        self.axs[0, 0].set_title("Real & Imaginary Parts")
        self.axs[0, 0].legend(fontsize='x-small')
        
        # Plot 1: Nyquist
        self.axs[0, 1].plot(np.real(fft_unroll), np.imag(fft_unroll), ".-", label="Data", color="#1f77b4")
        self.axs[0, 1].plot(np.real(Z_model_pure), np.imag(Z_model_pure), "--", label="Model", color="#2ca02c")
        self.axs[0, 1].set_title("Nyquist Plot")
        self.axs[0, 1].axis("equal")
        self.axs[0, 1].legend(fontsize='x-small')
        
        # Plot 2: Magnitude Response
        self.axs[1, 0].loglog(fs_sch, np.abs(fft_unroll), ".-", label="Data", color="#1f77b4")
        self.axs[1, 0].loglog(fs_sch, np.abs(Z_model_pure), "--", label="Model", color="#2ca02c")
        self.axs[1, 0].set_title("Magnitude Response")
        self.axs[1, 0].legend(fontsize='x-small')
        
        # Plot 3: Phase Response
        self.axs[1, 1].semilogx(fs_sch, np.angle(fft_unroll)/np.pi, ".-", label="Data", color="#1f77b4")
        self.axs[1, 1].semilogx(fs_sch, np.angle(Z_model_pure)/np.pi, "--", label="Model", color="#2ca02c")
        self.axs[1, 1].set_title("Phase Response (π rad)")
        self.axs[1, 1].legend(fontsize='x-small')
        
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()

    def reset_ui(self):
        self.btn_go.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_lbl.configure(text="Idle", fg="cyan")
        self.progress_lbl.configure(text="Sweep completed.")

def main():
    root = tk.Tk()
    app = EZCZGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
