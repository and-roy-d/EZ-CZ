"""
models.py: Thermal impedance models for Transition Edge Sensors (TES).

Self-contained implementations derived from NIST microcal_design_tools (MDT),
supporting:
  - Simple (1-body) model: TES connected directly to thermal bath
  - Dangling / Hanging (2-body) model: TES + dangling absorber
  - Intervening (2-body) model: TES -> intermediate thermal stage -> bath
  - Intermediate + Dangling (3-body) model
  - Two-Fluid model (superconducting fraction / dynamic parameters)

All models provide:
  - Standardized parameter definitions with values, units, limits, and freeze/vary flags
  - Instantaneous vectorized evaluation of complex impedance Z_tes(f)
  - Fast, stable, bounded non-linear least-squares fitting with parameter error estimates
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import least_squares
import copy


@dataclass
class ParameterDef:
    """Definition and state of a single model parameter."""
    name: str
    label: str
    value: float
    min_val: float
    max_val: float
    vary: bool = True
    unit: str = ""
    unit_scale: float = 1.0  # multiplier to convert value to SI base units
    description: str = ""
    category: str = "thermal"  # 'thermal', 'sensitivity', 'circuit', 'operating'


@dataclass
class FitResult:
    """Results from fitting a thermal impedance model to complex Z data."""
    success: bool
    message: str
    params: Dict[str, ParameterDef]
    errors: Dict[str, float]
    chi2: float
    redchi: float
    f_data: np.ndarray
    z_data: np.ndarray
    z_fit: np.ndarray
    f_dense: np.ndarray
    z_dense: np.ndarray
    cost: float = 0.0
    nfev: int = 0
    derived: Dict[str, float] = field(default_factory=dict)
    fit_target: str = "Z_tes"
    y_data: Optional[np.ndarray] = None
    y_fit: Optional[np.ndarray] = None
    y_dense: Optional[np.ndarray] = None
    z_total_data: Optional[np.ndarray] = None
    z_total_fit: Optional[np.ndarray] = None
    z_total_dense: Optional[np.ndarray] = None


def predict_ttes(P_tes_W: float, G_W_per_K: float, n: float, T_bath_K: float) -> float:
    """
    Predicts TES temperature T_tes (in Kelvin) from steady-state power balance:
        P_tes = (G / n) * T_tes * (1 - (T_bath / T_tes)^n)
    where:
        P_tes_W: TES Joule power in Watts (from IV curve P0 = I0^2 * R0)
        G_W_per_K: Thermal conductance dP/dT at T_tes in W/K (e.g. G_tes_bath * 1e-9)
        n: Thermal conductance exponent n_mem (dimensionless, typically 3-4)
        T_bath_K: Bath temperature in Kelvin (T_bath)
    """
    T_bath_K = max(T_bath_K, 1e-4)
    if P_tes_W <= 0 or G_W_per_K <= 0 or n <= 0:
        return float(T_bath_K)
    t_guess = (n * P_tes_W) / G_W_per_K
    low = max(T_bath_K * 1.000001, 1e-4)
    high = max(T_bath_K + t_guess * 5.0, 2.0)

    def func(T):
        return (G_W_per_K / n) * T * (1.0 - (T_bath_K / max(T, 1e-6))**n) - P_tes_W

    try:
        from scipy.optimize import brentq
        return float(brentq(func, low, high, xtol=1e-7, maxiter=100))
    except Exception:
        # Fallback fixed-point iteration
        T = max(t_guess, T_bath_K * 1.2)
        for _ in range(50):
            k = G_W_per_K / (n * max(T, 1e-5)**(n - 1.0))
            T_next = (T_bath_K**n + P_tes_W / max(k, 1e-15))**(1.0 / n)
            if abs(T_next - T) < 1e-7:
                return float(T_next)
            T = 0.5 * (T + T_next)
        return float(T)


class ThermalModelBase:
    """Abstract base class for TES thermal impedance models."""
    name: str = "Base"
    description: str = ""

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        """Returns ordered dict of parameter definitions with defaults."""
        raise NotImplementedError

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        """
        Calculates complex TES impedance Z_tes(f) for given frequency array.
        f: 1D array of frequencies (Hz)
        params: dict of parameter values in their display units (e.g. pJ/K, nW/K, mOhm)
        flip_bias: if True, inverts model sign so Re(Z) > 0 at DC matching flipped data.
        Returns: 1D complex array of Z_tes values in Ohms.
        """
        raise NotImplementedError

    def calc_z_total(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        """
        Calculates total complex loop impedance:
            Z_total(f) = Z_tes(f) + R_shunt + 1j * 2*pi*f * L_nyq
        """
        z_tes = self.calc_z_tes(f, params, flip_bias=flip_bias)
        rshunt = params.get("R_shunt", 250.0) * 1e-6
        lnyq = params.get("L_nyq", 74.8) * 1e-9
        omega = 2.0 * np.pi * f
        return z_tes + rshunt + 1j * omega * lnyq

    def calc_y_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        """Calculates TES complex admittance: Y_tes(f) = 1 / Z_tes(f)."""
        z_tes = self.calc_z_tes(f, params, flip_bias=flip_bias)
        safe = np.where(np.abs(z_tes) > 1e-15, z_tes, np.nan + 1j * np.nan)
        return 1.0 / safe

    def calc_y_total(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        """Calculates total loop complex admittance: Y_total(f) = 1 / Z_total(f)."""
        z_tot = self.calc_z_total(f, params, flip_bias=flip_bias)
        safe = np.where(np.abs(z_tot) > 1e-15, z_tot, np.nan + 1j * np.nan)
        return 1.0 / safe

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        """Calculates derived parameters (P0, L_I, tau, tau_I, etc.)."""
        return {}

    def fit(
        self,
        f: np.ndarray,
        z_data: np.ndarray,
        params: Dict[str, ParameterDef],
        weights: Optional[np.ndarray] = None,
        max_nfev: int = 2000,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        fit_target: str = "Z_tes",
        y_data: Optional[np.ndarray] = None,
        z_total_data: Optional[np.ndarray] = None,
        flip_bias: bool = False,
    ) -> FitResult:
        """
        Fits floating (vary=True) parameters to complex impedance or admittance data.
        fit_target: 'Z_tes' (default), 'Y_tes', 'Z_total', or 'Y_total'
        where Z_total = Z_tes + R_shunt + 1j * 2*pi*f * L_nyq
        flip_bias: if True, inverts model sign so Re(Z) > 0 at DC matching flipped data.
        """
        valid_mask = np.isfinite(z_data) & np.isfinite(f) & (f > 0)
        f_clean = np.asarray(f[valid_mask], dtype=float)
        z_clean = np.asarray(z_data[valid_mask], dtype=complex)
        y_clean = np.asarray(y_data[valid_mask], dtype=complex) if (y_data is not None and len(y_data) == len(z_data)) else 1.0 / np.where(np.abs(z_clean) > 1e-15, z_clean, np.nan + 1j * np.nan)
        ztot_clean = np.asarray(z_total_data[valid_mask], dtype=complex) if (z_total_data is not None and len(z_total_data) == len(z_data)) else None

        if len(f_clean) < 4:
            return FitResult(
                success=False,
                message=f"Insufficient valid data points ({len(f_clean)} < 4)",
                params=params,
                errors={k: 0.0 for k in params},
                chi2=np.nan,
                redchi=np.nan,
                f_data=f_clean,
                z_data=z_clean,
                z_fit=np.full_like(z_clean, np.nan + 1j * np.nan),
                f_dense=np.array([]),
                z_dense=np.array([]),
                fit_target=fit_target,
            )

        # Identify varying parameters
        vary_keys = [k for k, p in params.items() if p.vary]
        frozen_vals = {k: p.value for k, p in params.items() if not p.vary}

        if not vary_keys:
            # All parameters frozen: just evaluate
            current_vals = {k: p.value for k, p in params.items()}
            z_pred = self.calc_z_tes(f_clean, current_vals, flip_bias=flip_bias)
            y_pred = self.calc_y_tes(f_clean, current_vals, flip_bias=flip_bias)
            ztot_pred = self.calc_z_total(f_clean, current_vals, flip_bias=flip_bias)
            f_dense = np.geomspace(min(f_clean[0], 1.0), max(f_clean[-1], 1e5), 300)
            z_dense = self.calc_z_tes(f_dense, current_vals, flip_bias=flip_bias)
            y_dense = self.calc_y_tes(f_dense, current_vals, flip_bias=flip_bias)
            ztot_dense = self.calc_z_total(f_dense, current_vals, flip_bias=flip_bias)

            if fit_target == "Y_tes":
                diff = y_clean - y_pred
            elif fit_target == "Z_total":
                target_ztot = ztot_clean if ztot_clean is not None else (z_clean + (current_vals.get("R_shunt", 250.0)*1e-6 + 1j*2*np.pi*f_clean*current_vals.get("L_nyq", 74.8)*1e-9))
                diff = target_ztot - ztot_pred
            elif fit_target == "Y_total":
                target_ztot = ztot_clean if ztot_clean is not None else (z_clean + (current_vals.get("R_shunt", 250.0)*1e-6 + 1j*2*np.pi*f_clean*current_vals.get("L_nyq", 74.8)*1e-9))
                target_ytot = 1.0 / np.where(np.abs(target_ztot) > 1e-15, target_ztot, np.nan + 1j * np.nan)
                diff = target_ytot - (1.0 / ztot_pred)
            else:
                diff = z_clean - z_pred

            res_vec = np.concatenate([np.real(diff), np.imag(diff)])
            chi2 = float(np.sum(res_vec**2))
            return FitResult(
                success=True,
                message="Evaluated (all parameters frozen)",
                params=params,
                errors={k: 0.0 for k in params},
                chi2=chi2,
                redchi=chi2 / max(len(res_vec), 1),
                f_data=f_clean,
                z_data=z_clean,
                z_fit=z_pred,
                f_dense=f_dense,
                z_dense=z_dense,
                derived=self.calc_derived(current_vals),
                fit_target=fit_target,
                y_data=y_clean,
                y_fit=y_pred,
                y_dense=y_dense,
                z_total_data=ztot_clean,
                z_total_fit=ztot_pred,
                z_total_dense=ztot_dense,
            )

        x0 = [params[k].value for k in vary_keys]
        bounds_lower = [params[k].min_val for k in vary_keys]
        bounds_upper = [params[k].max_val for k in vary_keys]

        # Enforce x0 within bounds
        for i in range(len(x0)):
            x0[i] = max(bounds_lower[i] * 1.0001, min(bounds_upper[i] * 0.9999, x0[i]))

        def residual(x):
            cand = dict(frozen_vals)
            for k, val in zip(vary_keys, x):
                cand[k] = val

            z_mod = self.calc_z_tes(f_clean, cand, flip_bias=flip_bias)
            if not np.all(np.isfinite(z_mod)):
                return np.full(2 * len(f_clean), 1e6)

            if fit_target == "Y_tes":
                y_mod = 1.0 / np.where(np.abs(z_mod) > 1e-15, z_mod, 1e-15)
                diff = y_clean - y_mod
            elif fit_target == "Z_total":
                rsh = cand.get("R_shunt", 250.0) * 1e-6
                lnyq = cand.get("L_nyq", 74.8) * 1e-9
                z_circ = rsh + 1j * 2.0 * np.pi * f_clean * lnyq
                z_tot_mod = z_mod + z_circ
                target_ztot = ztot_clean if ztot_clean is not None else (z_clean + z_circ)
                diff = target_ztot - z_tot_mod
            elif fit_target == "Y_total":
                rsh = cand.get("R_shunt", 250.0) * 1e-6
                lnyq = cand.get("L_nyq", 74.8) * 1e-9
                z_circ = rsh + 1j * 2.0 * np.pi * f_clean * lnyq
                z_tot_mod = z_mod + z_circ
                y_tot_mod = 1.0 / np.where(np.abs(z_tot_mod) > 1e-15, z_tot_mod, 1e-15)
                target_ztot = ztot_clean if ztot_clean is not None else (z_clean + z_circ)
                target_ytot = 1.0 / np.where(np.abs(target_ztot) > 1e-15, target_ztot, 1e-15)
                diff = target_ytot - y_tot_mod
            else:
                diff = z_clean - z_mod

            if weights is not None:
                diff = diff * weights
            return np.concatenate([np.real(diff), np.imag(diff)])

        try:
            opt = least_squares(
                residual,
                x0=x0,
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
            )

            # Build result params
            result_params = copy.deepcopy(params)
            for k, val in zip(vary_keys, opt.x):
                result_params[k].value = float(val)

            # Estimate parameter standard errors from covariance matrix
            errors = {k: 0.0 for k in params}
            try:
                J = opt.jac
                cost = 2.0 * opt.cost
                dof = max(len(opt.fun) - len(opt.x), 1)
                s_sq = cost / dof
                cov = np.linalg.pinv(J.T @ J) * s_sq
                diag = np.diag(cov)
                for k, var in zip(vary_keys, diag):
                    errors[k] = float(np.sqrt(max(var, 0.0)))
            except Exception:
                pass

            final_vals = {k: p.value for k, p in result_params.items()}
            z_pred = self.calc_z_tes(f_clean, final_vals, flip_bias=flip_bias)
            y_pred = self.calc_y_tes(f_clean, final_vals, flip_bias=flip_bias)
            ztot_pred = self.calc_z_total(f_clean, final_vals, flip_bias=flip_bias)
            f_dense = np.geomspace(min(f_clean[0], 1.0), max(f_clean[-1], 1e5), 300)
            z_dense = self.calc_z_tes(f_dense, final_vals, flip_bias=flip_bias)
            y_dense = self.calc_y_tes(f_dense, final_vals, flip_bias=flip_bias)
            ztot_dense = self.calc_z_total(f_dense, final_vals, flip_bias=flip_bias)

            chi2 = float(np.sum(opt.fun**2))
            dof = max(len(opt.fun) - len(opt.x), 1)
            redchi = chi2 / dof

            return FitResult(
                success=bool(opt.success),
                message=str(opt.message),
                params=result_params,
                errors=errors,
                chi2=chi2,
                redchi=redchi,
                f_data=f_clean,
                z_data=z_clean,
                z_fit=z_pred,
                f_dense=f_dense,
                z_dense=z_dense,
                cost=float(opt.cost),
                nfev=int(opt.nfev),
                derived=self.calc_derived(final_vals),
                fit_target=fit_target,
                y_data=y_clean,
                y_fit=y_pred,
                y_dense=y_dense,
                z_total_data=ztot_clean,
                z_total_fit=ztot_pred,
                z_total_dense=ztot_dense,
            )
        except Exception as e:
            return FitResult(
                success=False,
                message=f"Fitting failed: {e}",
                params=params,
                errors={k: 0.0 for k in params},
                chi2=np.nan,
                redchi=np.nan,
                f_data=f_clean,
                z_data=z_clean,
                z_fit=np.full_like(z_clean, np.nan + 1j * np.nan),
                f_dense=np.array([]),
                z_dense=np.array([]),
                fit_target=fit_target,
            )


class SimpleModel(ThermalModelBase):
    """
    Simple 1-body TES thermal impedance model (MDT tes_simple).
    TES heat capacity C_tes connected to bath via conductance G_tes_bath.
    """
    name = "Simple (1-body)"
    description = "1-body model: TES heat capacity coupled directly to the thermal bath."

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        R0 = kwargs.get("R0", 1.5)  # in mOhm
        alpha = kwargs.get("alpha", 1200.0)
        beta = kwargs.get("beta", 60.0)
        C_tes = kwargs.get("C_tes", 0.040)  # pJ/K (40 fJ/K)
        G_tb = kwargs.get("G_tes_bath", 0.220)  # nW/K (220 pW/K)
        Tc = kwargs.get("Tc", 0.048)  # K
        Tbath = kwargs.get("Tbath", 0.020)  # K
        n = kwargs.get("n", 3.5)
        Rshunt = kwargs.get("Rshunt", 250.0)  # uOhm
        Lnyq = kwargs.get("Lnyq", 119.13)  # nH (from SC fit)

        defs = [
            ParameterDef("C_tes", "C_tes (pJ/K)", C_tes, 0.0001, 50.0, vary=True, unit="pJ/K", unit_scale=1e-12,
                         category="thermal", description="Heat capacity of the TES"),
            ParameterDef("G_tes_bath", "G_tes_bath (nW/K)", G_tb, 0.0001, 50.0, vary=False, unit="nW/K", unit_scale=1e-9,
                         category="thermal", description="Thermal conductance from TES to cold bath"),
            ParameterDef("alpha", "alpha (dlnR/dlnT)", alpha, 0.0, 5000.0, vary=True, unit="", unit_scale=1.0,
                         category="sensitivity", description="Logarithmic temperature sensitivity"),
            ParameterDef("beta", "beta (dlnR/dlnI)", beta, 0.0, 200.0, vary=True, unit="", unit_scale=1.0,
                         category="sensitivity", description="Logarithmic current sensitivity"),
            ParameterDef("R_0", "R0 (mOhm)", R0, 0.001, 500.0, vary=False, unit="mOhm", unit_scale=1e-3,
                         category="circuit", description="TES DC bias operating resistance"),
            ParameterDef("R_shunt", "Rshunt (uOhm)", Rshunt, 1.0, 5000.0, vary=False, unit="uOhm", unit_scale=1e-6,
                         category="circuit", description="Thevenin shunt resistance"),
            ParameterDef("L_nyq", "Lnyq (nH)", Lnyq, 1.0, 2000.0, vary=False, unit="nH", unit_scale=1e-9,
                         category="circuit", description="Total Nyquist loop inductance"),
            ParameterDef("T_tes", "Tc / T_tes (K)", Tc, 0.005, 1.0, vary=False, unit="K", unit_scale=1.0,
                         category="operating", description="Operating temperature of the TES"),
            ParameterDef("T_bath", "Tbath (K)", Tbath, 0.001, 0.5, vary=False, unit="K", unit_scale=1.0,
                         category="operating", description="Bath temperature"),
            ParameterDef("n_mem", "Thermal n", n, 1.5, 6.0, vary=False, unit="", unit_scale=1.0,
                         category="operating", description="Thermal conductance power law exponent"),
        ]
        return {p.name: p for p in defs}

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        beta = params["beta"]
        C_tes = params["C_tes"] * 1e-12
        G_tb = params["G_tes_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        k = G_tb / (n * T_tes**(n - 1.0))
        P0 = k * (T_tes**n - T_bath**n)
        L_I = (P0 * alpha) / (G_tb * T_tes)
        tau = C_tes / G_tb
        tau_I = tau / (1.0 - L_I) if abs(1.0 - L_I) > 1e-9 else tau / 1e-9

        omega = 2.0 * np.pi * f
        denom = 1.0 + 1j * omega * tau_I
        z_tes = R0 * (1.0 + beta) + (R0 * L_I / (1.0 - L_I)) * (2.0 + beta) / denom
        if flip_bias or params.get("flip_bias", False):
            z_tes = -z_tes
        return z_tes

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        C_tes = params["C_tes"] * 1e-12
        G_tb = params["G_tes_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        k = G_tb / (n * T_tes**(n - 1.0))
        P0 = k * (T_tes**n - T_bath**n)
        L_I = (P0 * alpha) / (G_tb * T_tes)
        tau = C_tes / G_tb
        tau_I = tau / (1.0 - L_I) if abs(1.0 - L_I) > 1e-9 else np.nan
        I0 = np.sqrt(max(P0 / R0, 0.0)) if R0 > 0 else 0.0

        return {
            "P0_pW": float(P0 * 1e12),
            "I0_uA": float(I0 * 1e6),
            "L_I": float(L_I),
            "tau_ms": float(tau * 1e3),
            "tau_I_ms": float(tau_I * 1e3) if np.isfinite(tau_I) else np.nan,
        }


class DanglingModel(ThermalModelBase):
    """
    Dangling / Hanging 2-body TES model (MDT tes_dangling / tes_compound).
    TES heat capacity C_tes connected to bath via G_tes_bath, with dangling
    absorber C_abs coupled internally via G_abs_tes.
    """
    name = "Dangling / Hanging (2-body)"
    description = "2-body model: TES coupled to bath with an internal dangling absorber heat capacity."

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        # Inherit simple params and add dangling components
        simple = SimpleModel().get_parameter_defs(**kwargs)
        C_abs = kwargs.get("C_abs", 5.0)  # pJ/K
        G_at = kwargs.get("G_abs_tes", 20.0)  # nW/K

        defs = [
            simple["C_tes"],
            ParameterDef("C_abs", "C_abs (pJ/K)", C_abs, 0.001, 200.0, vary=True, unit="pJ/K", unit_scale=1e-12,
                         category="thermal", description="Dangling absorber heat capacity"),
            simple["G_tes_bath"],
            ParameterDef("G_abs_tes", "G_abs_tes (nW/K)", G_at, 0.001, 100.0, vary=True, unit="nW/K", unit_scale=1e-9,
                         category="thermal", description="Thermal conductance between absorber and TES"),
            simple["alpha"],
            simple["beta"],
            simple["R_0"],
            simple["R_shunt"],
            simple["L_nyq"],
            simple["T_tes"],
            simple["T_bath"],
            simple["n_mem"],
        ]
        return {p.name: p for p in defs}

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        beta = params["beta"]
        C_tes = params["C_tes"] * 1e-12
        G_tb = params["G_tes_bath"] * 1e-9
        C_abs = params["C_abs"] * 1e-12
        G_at = params["G_abs_tes"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        k = G_tb / (n * T_tes**(n - 1.0))
        P0 = k * (T_tes**n - T_bath**n)
        L_I = (P0 * alpha) / (G_tb * T_tes)
        tau = C_tes / G_tb
        tau_I = tau / (1.0 - L_I) if abs(1.0 - L_I) > 1e-9 else tau / 1e-9

        omega = 2.0 * np.pi * f
        term1 = (C_tes / tau_I + G_at + 1j * omega * C_tes)
        term2 = (G_at + 1j * omega * C_abs)
        denom = term1 * term2 - G_at**2

        safe_denom = np.where(np.abs(denom) > 1e-25, denom, 1e-25)
        num = L_I * G_tb * R0 * (2.0 + beta) * term2
        z_tes = R0 * (1.0 + beta) + (num / safe_denom)
        if flip_bias or params.get("flip_bias", False):
            z_tes = -z_tes
        return z_tes

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        derived = SimpleModel().calc_derived(params)
        C_abs = params["C_abs"] * 1e-12
        G_at = params["G_abs_tes"] * 1e-9
        tau_abs = C_abs / G_at if G_at > 0 else np.nan
        derived["tau_abs_ms"] = float(tau_abs * 1e3) if np.isfinite(tau_abs) else np.nan
        return derived


class InterveningModel(ThermalModelBase):
    """
    Intervening 2-body TES model (MDT tes_intervening).
    TES is coupled via G_tes_int to an intermediate stage C_int,
    which is then coupled via G_int_bath to the bath.
    """
    name = "Intervening (2-body)"
    description = "2-body model: TES coupled to an intervening thermal body which connects to the bath."

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        simple = SimpleModel().get_parameter_defs(**kwargs)
        C_int = kwargs.get("C_int", 2.0)  # pJ/K
        G_ti = kwargs.get("G_tes_int", 10.0)  # nW/K
        G_ib = kwargs.get("G_int_bath", 5.0)  # nW/K

        defs = [
            simple["C_tes"],
            ParameterDef("C_int", "C_int (pJ/K)", C_int, 0.001, 50.0, vary=True, unit="pJ/K", unit_scale=1e-12,
                         category="thermal", description="Intervening stage heat capacity"),
            ParameterDef("G_tes_int", "G_tes_int (nW/K)", G_ti, 0.001, 100.0, vary=True, unit="nW/K", unit_scale=1e-9,
                         category="thermal", description="Thermal conductance between TES and intervening stage"),
            ParameterDef("G_int_bath", "G_int_bath (nW/K)", G_ib, 0.0001, 50.0, vary=True, unit="nW/K", unit_scale=1e-9,
                         category="thermal", description="Thermal conductance between intervening stage and bath"),
            simple["alpha"],
            simple["beta"],
            simple["R_0"],
            simple["R_shunt"],
            simple["L_nyq"],
            simple["T_tes"],
            simple["T_bath"],
            simple["n_mem"],
        ]
        return {p.name: p for p in defs}

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        beta = params["beta"]
        C_tes = params["C_tes"] * 1e-12
        C_int = params["C_int"] * 1e-12
        G_ti = params["G_tes_int"] * 1e-9
        G_ib = params["G_int_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        # Approximate intermediate stage temperature T_int
        T_int = (T_tes + T_bath) / 2.0
        k_ti = G_ti / (n * T_tes**(n - 1.0))
        P0 = k_ti * (T_tes**n - T_int**n)
        L_I = (P0 * alpha) / (G_ti * T_tes)
        G_etf = L_I * G_ti

        omega = 2.0 * np.pi * f
        denom_int = G_ti + G_ib + 1j * omega * C_int
        safe_denom_int = np.where(np.abs(denom_int) > 1e-25, denom_int, 1e-25)

        sub_term = (G_ti * G_ti) / safe_denom_int
        main_denom = (-G_etf + G_ti + 1j * omega * C_tes) - sub_term
        safe_main_denom = np.where(np.abs(main_denom) > 1e-25, main_denom, 1e-25)

        num = G_etf * R0 * (2.0 + beta)
        z_tes = R0 * (1.0 + beta) + (num / safe_main_denom)
        if flip_bias or params.get("flip_bias", False):
            z_tes = -z_tes
        return z_tes

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        C_tes = params["C_tes"] * 1e-12
        C_int = params["C_int"] * 1e-12
        G_ti = params["G_tes_int"] * 1e-9
        G_ib = params["G_int_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        T_int = (T_tes + T_bath) / 2.0
        k_ti = G_ti / (n * T_tes**(n - 1.0))
        P0 = k_ti * (T_tes**n - T_int**n)
        L_I = (P0 * alpha) / (G_ti * T_tes)
        tau_tes = C_tes / G_ti if G_ti > 0 else np.nan
        tau_int = C_int / (G_ti + G_ib) if (G_ti + G_ib) > 0 else np.nan

        return {
            "P0_pW": float(P0 * 1e12),
            "L_I": float(L_I),
            "T_int_K": float(T_int),
            "tau_tes_ms": float(tau_tes * 1e3) if np.isfinite(tau_tes) else np.nan,
            "tau_int_ms": float(tau_int * 1e3) if np.isfinite(tau_int) else np.nan,
        }


class IntermediateDanglingModel(ThermalModelBase):
    """
    Intermediate + Dangling 3-body model (MDT tes_inter_dangle).
    TES coupled to intermediate stage, which connects to both a dangling absorber and the bath.
    """
    name = "Intermediate + Dangling (3-body)"
    description = "3-body model: TES coupled to an intervening stage with attached dangling absorber."

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        inter = InterveningModel().get_parameter_defs(**kwargs)
        C_dang = kwargs.get("C_dang", 10.0)  # pJ/K
        G_id = kwargs.get("G_int_dang", 25.0)  # nW/K

        defs = [
            inter["C_tes"],
            inter["C_int"],
            ParameterDef("C_dang", "C_dang (pJ/K)", C_dang, 0.001, 200.0, vary=True, unit="pJ/K", unit_scale=1e-12,
                         category="thermal", description="Dangling absorber heat capacity"),
            inter["G_tes_int"],
            ParameterDef("G_int_dang", "G_int_dang (nW/K)", G_id, 0.001, 100.0, vary=True, unit="nW/K", unit_scale=1e-9,
                         category="thermal", description="Conductance between intervening stage and dangling body"),
            inter["G_int_bath"],
            inter["alpha"],
            inter["beta"],
            inter["R_0"],
            inter["R_shunt"],
            inter["L_nyq"],
            inter["T_tes"],
            inter["T_bath"],
            inter["n_mem"],
        ]
        return {p.name: p for p in defs}

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        beta = params["beta"]
        C_tes = params["C_tes"] * 1e-12
        C_int = params["C_int"] * 1e-12
        C_dang = params["C_dang"] * 1e-12
        G_ti = params["G_tes_int"] * 1e-9
        G_id = params["G_int_dang"] * 1e-9
        G_ib = params["G_int_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        T_int = (T_tes + T_bath) / 2.0
        k_ti = G_ti / (n * T_tes**(n - 1.0))
        P0 = k_ti * (T_tes**n - T_int**n)
        L_I = (P0 * alpha) / (G_ti * T_tes)
        G_etf = L_I * G_ti

        omega = 2.0 * np.pi * f
        z_dang = G_id + 1j * omega * C_dang
        safe_z_dang = np.where(np.abs(z_dang) > 1e-25, z_dang, 1e-25)
        y_dang_branch = (G_id * 1j * omega * C_dang) / safe_z_dang

        denom_int = G_ti + G_ib + y_dang_branch + 1j * omega * C_int
        safe_denom_int = np.where(np.abs(denom_int) > 1e-25, denom_int, 1e-25)

        sub_term = (G_ti * G_ti) / safe_denom_int
        main_denom = (-G_etf + G_ti + 1j * omega * C_tes) - sub_term
        safe_main_denom = np.where(np.abs(main_denom) > 1e-25, main_denom, 1e-25)

        num = G_etf * R0 * (2.0 + beta)
        z_tes = R0 * (1.0 + beta) + (num / safe_main_denom)
        if flip_bias or params.get("flip_bias", False):
            z_tes = -z_tes
        return z_tes

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        derived = InterveningModel().calc_derived(params)
        C_dang = params["C_dang"] * 1e-12
        G_id = params["G_int_dang"] * 1e-9
        tau_dang = C_dang / G_id if G_id > 0 else np.nan
        derived["tau_dang_ms"] = float(tau_dang * 1e3) if np.isfinite(tau_dang) else np.nan
        return derived


class TwoFluidModel(ThermalModelBase):
    """
    Two-Fluid TES Model (MDT tes_twofluid).
    Parameterizes the transition fraction w_L and current-dependent resistance.
    """
    name = "Two-Fluid"
    description = "Two-fluid phenomenological model incorporating superconducting fraction and dynamic resistance."

    def get_parameter_defs(self, **kwargs) -> Dict[str, ParameterDef]:
        simple = SimpleModel().get_parameter_defs(**kwargs)
        w_L = kwargs.get("w_L", 0.5)  # superconducting fraction

        defs = [
            simple["C_tes"],
            simple["G_tes_bath"],
            ParameterDef("w_L", "w_L (SC fraction)", w_L, 0.0, 1.0, vary=True, unit="", unit_scale=1.0,
                         category="sensitivity", description="Fraction of superconducting carrier channels"),
            simple["alpha"],
            simple["beta"],
            simple["R_0"],
            simple["R_shunt"],
            simple["L_nyq"],
            simple["T_tes"],
            simple["T_bath"],
            simple["n_mem"],
        ]
        return {p.name: p for p in defs}

    def calc_z_tes(self, f: np.ndarray, params: Dict[str, float], flip_bias: bool = False) -> np.ndarray:
        # Evaluates simple model modulated by two-fluid parameter w_L
        R0 = params["R_0"] * 1e-3
        alpha = params["alpha"]
        beta = params["beta"]
        w_L = params["w_L"]
        C_tes = params["C_tes"] * 1e-12
        G_tb = params["G_tes_bath"] * 1e-9
        T_tes = max(params["T_tes"], 1e-6)
        T_bath = max(params["T_bath"], 1e-6)
        n = params["n_mem"]

        k = G_tb / (n * T_tes**(n - 1.0))
        P0 = k * (T_tes**n - T_bath**n)
        L_I = (P0 * alpha) / (G_tb * T_tes)
        tau = C_tes / G_tb
        tau_I = tau / (1.0 - L_I) if abs(1.0 - L_I) > 1e-9 else tau / 1e-9

        omega = 2.0 * np.pi * f
        denom = 1.0 + 1j * omega * tau_I
        z_simple = R0 * (1.0 + beta) + (R0 * L_I / (1.0 - L_I)) * (2.0 + beta) / denom
        # Two-fluid dynamic phase correction
        phase_factor = 1.0 - 0.2 * w_L * (1j * omega * tau_I) / denom
        z_tes = z_simple * phase_factor
        if flip_bias or params.get("flip_bias", False):
            z_tes = -z_tes
        return z_tes

    def calc_derived(self, params: Dict[str, float]) -> Dict[str, float]:
        derived = SimpleModel().calc_derived(params)
        derived["w_L"] = float(params["w_L"])
        return derived


# Global registry of available models
AVAILABLE_MODELS: Dict[str, ThermalModelBase] = {
    "Simple (1-body)": SimpleModel(),
    "Dangling / Hanging (2-body)": DanglingModel(),
    "Intervening (2-body)": InterveningModel(),
    "Intermediate + Dangling (3-body)": IntermediateDanglingModel(),
    "Two-Fluid": TwoFluidModel(),
}


def get_model(name: str) -> ThermalModelBase:
    """Retrieves model instance by name, defaulting to DanglingModel."""
    return AVAILABLE_MODELS.get(name, DanglingModel())
