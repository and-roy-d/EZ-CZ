"""
Simple TES class of MDT Microcal Design Tools
Current Version Created on Jan 17, 2020
Original Version Created on Nov 14, 2009

MDT3 is written to modernize the code by switching to python3 and using newer modules for
things like fitting. It is a work in progress!
"""

import numpy as np
import cmath
import math
import scipy
from scipy import integrate

kB = 1.3806503e-23
ec = 1.60217646e-19

class TES_Simple(object):
    """Class for methods of the Simple TES model."""
    def __init__(self, tes):

        self.tes = tes

        #derived parmeters for stand input parameters
        self.k = None
        self.I0 = None
        self.PJ0 = None
        self.L_I = None

        #tcalculated time constants
        self.lambdaplus = None
        self.lambdaminus = None

        #state variables
        self.Ites = None
        self.Ttes = None

        self.simple_params ={}
        self.simple_params['k'] = self.k
        self.simple_params['I0'] = self.I0
        self.simple_params['PJ0'] = self.PJ0
        self.simple_params['L_I'] = self.L_I
        self.simple_params['lambdaplus'] = self.lambdaplus
        self.simple_params['lambdaminus'] = self.lambdaminus
        self.simple_params['Ites'] = self.Ites
        self.simple_params['Ttes'] = self.Ttes

    def calc_derived_params(self, tes_params_pd, set_index=0):
        ''' Calculate parameters derived from input parameters. '''

        L_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'L')
        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')

        tes_params = tes_params_pd.valuesdict()

        t = tes_params[f't_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        delta_T = tes_params[f'deltaT_{set_index}']
        #I0 = tes_params['I0']
        #L_I = tes_params['L_I']

        C_photon = C_unit_scale*tes_params[f'C_tes_{set_index}']

        if tes_params[f'initE_use_{set_index}'] is True:
            tes_params_pd[f'deltaT_{set_index}'].set(value=(tes_params[f'initE_{set_index}']*ec/C_photon))
        else:
            tes_params_pd[f'initE_{set_index}'].set(value=(tes_params[f'deltaT_{set_index}']*C_photon/ec))

        # Calculate currents, powers and temperatures
        alpha_I = tes_params[f'alpha_I_{set_index}']
        G_tes_bath = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']
        T_tes = tes_params[f'T_tes_{set_index}']
        T_bath = tes_params[f'T_bath_{set_index}']
        n_mem = tes_params[f'n_mem_{set_index}']

        k = G_tes_bath/(n_mem*T_tes**(n_mem-1))
        I0 = (k/R_0*(T_tes**n_mem-T_bath**n_mem))**0.5
        PJ0 = I0**2*R_0
        L_I = PJ0*alpha_I/(G_tes_bath*T_tes)

        self.simple_params['k'] = k
        self.simple_params['I0'] = I0
        self.simple_params['PJ0'] = PJ0
        self.simple_params['L_I'] = L_I

    def calc_L_crit(self, tes_params_pd, set_index=0):
        '''Solves for lambda in simple TES. Returns lambdaplus and lambdaminus, and maximum L for stable operation.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        C = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.simple_params['L_I']
        tau = C / G

        A = L_I*(3 + beta_I - R_L/R_0)
        B = (1 + beta_I + R_L/R_0)
        C = L_I*(2 + beta_I)*(L_I*(1 - R_L/R_0) + B)
        D = R_0 * tau / (L_I - 1)**2

        Lcritplus = (A + B + 2*np.sqrt(C))*D
        Lcritminus = (A + B - 2*np.sqrt(C))*D

        if L_I <= 1:
            Lstable = np.inf
        else:
            Lstable = tau / (L_I - 1) * (R_L + R_0*(1 + beta_I))

        # Do we want to stash this values somewhere
        self.simple_params['Lcritplus'] = Lcritplus
        self.simple_params['Lcritminus'] = Lcritminus
        self.simple_params['Lstable'] = Lstable

        return Lcritminus, Lcritplus, Lstable

    def calc_lambda(self, tes_params_pd, set_index=0):
        ''' Solves for lambda in simple TES. Returns lambdaplus and lambdaminus '''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        L_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'L')
        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        L = L_unit_scale*tes_params[f'L_{set_index}']
        C = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.simple_params['L_I']
        tau = C / G
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        lambdaplus = 1.0/(2.0*tau_el)+1.0/(2.0*tau_I)+1.0/2.0*cmath.sqrt((1.0/tau_el-1.0/tau_I)**2.0-4.0*R_0/L*L_I*(2.0+beta_I)/tau)
        lambdaminus = 1.0/(2.0*tau_el)+1.0/(2.0*tau_I)-1.0/2.0*cmath.sqrt((1.0/tau_el-1.0/tau_I)**2.0-4.0*R_0/L*L_I*(2.0+beta_I)/tau)

        #Do we want to stash this values somewhere
        self.simple_params['lambdaplus'] = lambdaplus
        self.simple_params['lambdaminus'] = lambdaminus

        return lambdaplus, lambdaminus

    def tsolve(self, tes_params_pd, set_index=0):
        ''' Solves the Simple TES for I and T. '''

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        C = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        t = tes_params[f't_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.simple_params['I0']
        L_I = self.simple_params['L_I']

        tau = C / G
        tau_I = tau/(1-L_I)

        lambdaplus, lambdaminus = self.calc_lambda(tes_params_pd, set_index)

        Ites = (tau_I*lambdaplus-1.0)*(tau_I*lambdaminus-1)*(1.0/(2.0+beta_I))*(C*deltaT/(I0*R_0*tau_I**2))*((cmath.exp(-t*lambdaplus)-cmath.exp(-t*lambdaminus))/(lambdaplus-lambdaminus))
        Ttes = (-(1.0/tau_I-lambdaplus)*cmath.exp(-t*lambdaminus)+(1.0/tau_I-lambdaminus)*cmath.exp(-t*lambdaplus))*deltaT/(lambdaplus-lambdaminus)

        #make sure the values are not complex
        Ites_r = np.real(Ites)
        Ttes_r = np.real(Ttes)

        self.simple_params['Ites'] = Ites_r
        self.simple_params['Ttes'] = Ttes_r

        return Ites_r, Ttes_r

    def ztes_solve(self, tes_params_pd, set_index=0):
        '''Returns the complex Z of the TES at the current frequency value.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        C1 = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G1 = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.simple_params['L_I']

        tau = C1 / G1
        tau_I = tau/(1-L_I)

        Zt = R_0*(1.0+beta_I)+R_0*L_I/(1.0-L_I)*(2.0+beta_I)/(1.0+1.0j*omega*tau_I)
        #Zt_real = Zt.real
        #Zt_imag = Zt.imag

        return Zt

    def responsivity_solve(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of a simple TES.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        #f = f_vals
        omega = 2.0*np.pi*f

        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        C = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        R_0 = tes_params[f'R_0_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']

        I0 = self.simple_params['I0']
        L_I = self.simple_params['L_I']

        tau = C / G
        tau_I = tau/(1-L_I)

        lambdaplus, lambdaminus = self.calc_lambda(tes_params_pd, set_index)

        sI = -(1./(I0*R_0))*(1./(2.+beta_I))*(1.-1./lambdaplus/tau_I)*(1.-1./lambdaminus/tau_I)/(1+1j*omega/lambdaplus)/(1+1j*omega/lambdaminus)

        return sI

    def current_noise_solve(self, tes_params_pd, set_index=0):
        ''' Returns the total and various current noises '''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        C_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')
        G_unit_scale = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')
        C1 = C_unit_scale*tes_params[f'C_tes_{set_index}']
        G1 = G_unit_scale*tes_params[f'G_tes_bath_{set_index}']

        f = tes_params[f'f_{set_index}']  # should be f_vals?
        #f = f_vals
        omega = 2.0*np.pi*f
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        T0 = tes_params[f'T_tes_{set_index}']
        Tb = tes_params[f'T_bath_{set_index}']
        n = tes_params[f'n_mem_{set_index}']
        M = tes_params[f'M_{set_index}']
        xi = tes_params[f'xi_{set_index}']

        #Tsh allows for a different temperature for the shunt resistor, for now we set it to Tb
        Tsh = Tb

        I0 = self.simple_params['I0']
        L_I = self.simple_params['L_I']

        tau = C1 / G1
        tau_I = tau/(1-L_I)

        lambdaplus, lambdaminus = self.calc_lambda(tes_params_pd, set_index)

        epsilon = 1+2.0*beta_I+xi*beta_I**2.
        #epsilon = 1
        G1atTb = G1*Tb**(n-1)/(T0**(n-1))

        sI = self.responsivity_solve(tes_params_pd, set_index)

        #SI2s are the current noise squared
        SI2_R0 = 4*kB*T0*R_0*epsilon*(1+M**2)*I0**2*(1+tau**2*omega**2)*sI*sI.conjugate()/L_I**2
        #SI2_R0 = 4*kB*T0*R_0*M*I0**2*(1+tau**2*omega**2)*sI*sI.conjugate()/L_I**2
        SI2_RL = 4*kB*Tsh*R_L*I0**2*(L_I-1)**2*(1+tau_I**2*omega**2)*sI*sI.conjugate()/L_I**2
        SI2_G1 = 2*kB*(T0**2*G1+Tb**2*G1atTb)*sI*sI.conjugate()
        SI2_squid = (tes_params[f'squid_noise_{set_index}'])**2
        SI2_back = (tes_params[f'background_noise_{set_index}'])**2*sI*sI.conjugate()

        SI2_total = SI2_R0+SI2_RL+SI2_G1+SI2_squid+SI2_back

        SI_total = math.sqrt(SI2_total.real)
        SI_R0 = math.sqrt(SI2_R0.real)
        SI_RL = math.sqrt(SI2_RL.real)
        SI_G1 = math.sqrt(SI2_G1.real)
        SI_squid = math.sqrt(SI2_squid)
        SI_back = math.sqrt(SI2_back.real)

        return SI_total, SI_R0, SI_RL, SI_G1, SI_squid, SI_back

    def nep2_solve(self, tes_params_pd, set_index=0):
        ''' Returns the total and various noise equivalent powers squared '''

        s_I = self.responsivity_solve(tes_params_pd, set_index)
        SI_total, SI_R0, SI_RL, SI_G1, SI_squid, SI_back  = self.current_noise_solve(tes_params_pd, set_index)
        NEP2_R0 = np.real(SI_R0**2/(s_I*s_I.conjugate()))
        NEP2_RL = np.real(SI_RL**2/(s_I*s_I.conjugate()))
        NEP2_G1 = np.real(SI_G1**2/(s_I*s_I.conjugate()))
        NEP2_squid = np.real(SI_squid**2/(s_I*s_I.conjugate()))
        NEP2_back = np.real(SI_back**2/(s_I*s_I.conjugate()))

        NEP2_total = NEP2_R0 + NEP2_RL + NEP2_G1 + NEP2_squid + NEP2_back

        return NEP2_total, NEP2_R0, NEP2_RL, NEP2_G1, NEP2_squid, NEP2_back


    def calc_pulse(self, tes_params_pd, t_vals, set_index=0):
        '''Calculate the current and temperature of the TES as a function of time.'''

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)

        for index in range(len(t_vals)):
            tes_params_pd[f't_{set_index}'].set(value=t_vals[index])
            Ites, Ttes = self.tsolve(tes_params_pd, set_index)
            Ites_vals[index] = Ites
            Ttes_vals[index] = Ttes

        return Ites_vals, Ttes_vals

    def calc_cz(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the impeadence of the TES as a function of frequnecy.'''

        Zt_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            Zt = self.ztes_solve(tes_params_pd, set_index)
            Zt_vals[index] = Zt

        return Zt_vals

    def calc_responsivity(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the responsivity of the TES as a function of frequency.'''

        sI_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            sI = self.responsivity_solve(tes_params_pd, set_index)
            sI_vals[index] = sI

        return sI_vals

    def calc_noise(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the noise of the TES as a function of frequency.'''

        SI_total_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_R0_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_RL_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_G1_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_squid_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_back_vals = np.zeros_like(f_vals, dtype=np.float64)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            SI_total, SI_R0, SI_RL, SI_G1, SI_squid, SI_back  = self.current_noise_solve(tes_params_pd, set_index)
            SI_total_vals[index] = SI_total
            SI_R0_vals[index] = SI_R0
            SI_RL_vals[index] = SI_RL
            SI_G1_vals[index] = SI_G1
            SI_squid_vals[index] = SI_squid
            SI_back_vals[index] = SI_back

        return SI_total_vals, SI_R0_vals, SI_RL_vals, SI_G1_vals, SI_squid_vals, SI_back_vals

    def calc_nep2(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the NEP squared of the TES as a function of frequency.'''

        #not update yet form simple

        NEP_total_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_R0_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_RL_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_G1_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_squid_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_back_vals = np.zeros_like(f_vals, dtype=np.float64)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            NEP_total, NEP_R0, NEP_RL, NEP_G1, NEP_squid, NEP_back  = self.nep2_solve(tes_params_pd,  set_index)
            NEP_total_vals[index] = NEP_total
            NEP_R0_vals[index] = NEP_R0
            NEP_RL_vals[index] = NEP_RL
            NEP_G1_vals[index] = NEP_G1
            NEP_squid_vals[index] = NEP_squid
            NEP_back_vals[index] = NEP_back

        return NEP_total_vals, NEP_R0_vals, NEP_RL_vals, NEP_G1_vals, NEP_squid_vals, NEP_back_vals
    
    def solve_energy_res_integral(self, frequency, tes_params_pd, set_index):
        '''The integral of that gives the energy resolution'''

        tes_params_pd[f'f_{set_index}'].set(value=frequency)

        NEP2_total, NEP2_R0, NEP2_RL, NEP2_G1, NEP2_squid, NEP2_back = self.nep2_solve(tes_params_pd, set_index)
        integral_val = 4.0/(NEP2_total.real*6.24150974e18**2)  #6.24e18 is to convert to eV before integrating
        #print frequency, integral_val

        return integral_val

    def calc_energy_res(self, tes_params_pd, set_index=0):
        '''Calculate minimum TES energy resolution'''

        integrated_val = integrate.quad(self.solve_energy_res_integral,0.0,scipy.inf,args=(tes_params_pd, set_index),limit=200)
        #print integrated_val
        deltaE = 2*math.sqrt(2*math.log(2))/math.sqrt(integrated_val[0])
        #print deltaE

        return deltaE

    def tsolve_numint_Tpulse(self, X, t, tes_params_pd, T_in_f, set_index):
        ''' Solves the Complex TES for I and T using numerical integration of
        non-linear TES equations and provides the capabilty to add heat to the 
        absorber as function of time by passing an funcion (possibly interpolated)'''

        T_in = T_in_f(t)

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        #C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        #G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        print('deltaT',deltaT)
        I0 = self.simple_params['I0']
        L_I = self.simple_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        return np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1],
                            I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I)*X[1]+ T_in])

    def calc_pulse_numint_Tpulse(self, tes_params_pd, t_vals, T_in_f, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations and
        provides the capabilty to add heat to the absorber as funmction of time by passing 
        a funcion (possibly interpolated)'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        deltaT = tes_params[f'deltaT_{set_index}']
        T0 = T_in_f(0.0)

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)

        X0 = np.array([0.0, T0])
        X, infodict = integrate.odeint(self.tsolve_numint_Ipulse, X0, t_vals, args=(tes_params_pd, T_in_f, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals = X.T

        return Ites_vals, Ttes_vals
    
    def tsolve_numint_Ipulse(self, X, t, tes_params_pd, I_in_f, set_index):
        ''' Solves the Complex TES for I and T using numerical integration of
        non-linear TES equations and provides the capabilty to add heat to the 
        absorber as function of time by passing an funcion (possibly interpolated)'''

        I_in = I_in_f(t)

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        #C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        #G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        #deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.simple_params['I0']
        L_I = self.simple_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        return np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1]-I_in/tau_el,
                            I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I)*X[1]])

    def calc_pulse_numint_Ipulse(self, tes_params_pd, t_vals, I_in_f, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations and
        provides the capabilty to add heat to the absorber as funmction of time by passing 
        a funcion (possibly interpolated)'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        I0 = I_in_f(0.0)

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)

        X0 = np.array([I0, 0.0])
        X, infodict = integrate.odeint(self.tsolve_numint_Ipulse, X0, t_vals, args=(tes_params_pd, I_in_f, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals = X.T

        return Ites_vals, Ttes_vals


