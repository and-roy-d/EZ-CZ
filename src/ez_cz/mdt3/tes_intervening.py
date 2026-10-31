"""
Inteerveing TES class of MDT Microcal Design Tools
Current Version Created on Jun 13, 2024
Equations from orginal version of MDT

MDT3 is written to modernize the code by switching to python3 and using newer modules for
things like fitting. It is a work in progress!
"""

import numpy as np
import cmath
import math
import scipy
from scipy import integrate
from scipy.optimize import fsolve

kB = 1.3806503e-23
ec = 1.60217646e-19

class TES_Intervening(object):
    """Class for methods of the Intervening TES model."""
    def __init__(self, tes):
                
        self.tes = tes

        #derived parmeters for stand input parameters
        self.k = None
        self.I0 = None
        self.PJ0 = None
        self.L_I = None

        #calculated time constants
        self.lambdaplus = None
        self.lambdaminus = None

        #state variables
        self.Ites = None
        self.Ttes = None
        self.T_int = None

        self.inter_params ={}
        self.inter_params['k'] = self.k
        self.inter_params['I0'] = self.I0
        self.inter_params['PJ0'] = self.PJ0
        self.inter_params['L_I'] = self.L_I
        self.inter_params['lambdaplus'] = self.lambdaplus
        self.inter_params['lambdaminus'] = self.lambdaminus
        self.inter_params['Ites'] = self.Ites
        self.inter_params['Ttes'] = self.Ttes
        self.inter_params['T_int'] = self.T_int

    def func_intervning_T_0(self, T_s, T_tes, T_bath, n_int1, n_int2 ,k_ratio):
        return (k_ratio*(T_tes**n_int1-T_s**n_int1)-(T_s**n_int2-T_bath**n_int2))

    def calc_derived_params(self, tes_params_pd, set_index=0):
        ''' Calculate parameters derived from input parameters. '''

        tes_params = tes_params_pd.valuesdict()

        #This could be either but currently set for photon strike to intervening
        C_photon = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']

        if tes_params[f'initE_use_{set_index}'] is True:
            tes_params_pd[f'deltaT_{set_index}'].set(value=(tes_params[f'initE_{set_index}']*ec/C_photon))
        else:
            tes_params_pd[f'initE_{set_index}'].set(value=(tes_params[f'deltaT_{set_index}']*C_photon/ec))
        
        # Calculate currents, powers and temperatures
        R_0 = tes_params[f'R_0_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        T_tes = tes_params[f'T_tes_{set_index}']
        T_bath = tes_params[f'T_bath_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        n_int2 = tes_params[f'n_int2_{set_index}']
        k_int_ratio = tes_params[f'k_int_ratio_{set_index}']
        background_power = tes_params[f'int_background_power_{set_index}']

        T_int_guess = (T_tes-T_bath)/2.0+T_bath
        T_int = float(fsolve(self.func_intervning_T_0,T_int_guess,args=(T_tes, T_bath, n_int1, n_int2 ,k_int_ratio)))
        k_tes_int = G_tes_int/(n_int1*T_tes**(n_int1-1.0))
        k_int_bath = k_tes_int/k_int_ratio
        G_int_bath = k_int_bath*n_int2*T_int**(n_int2-1.0)
        PJ0 = k_tes_int*(T_tes**n_int1-T_int**n_int1)
        I0 = (PJ0/R_0)**.5
        L_I = PJ0*alpha_I/(G_tes_int*T_tes)

        self.inter_params['I0'] = I0
        self.inter_params['PJ0'] = PJ0
        self.inter_params['L_I'] = L_I
        self.inter_params['G_int_bath'] = G_int_bath
        self.inter_params['T_int'] = T_int
        self.inter_params['k_int_bath'] = k_int_bath


    def calc_v_tes(self, tes_params_pd, set_index=0):
        '''Calculate v_tes for current params.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        R_0 = tes_params[f'R_0_{set_index}']
        I_0 = self.inter_params['I0']
        V_0 = R_0*I_0

        return V_0
    
    def calc_G_int_bath(self, tes_params_pd, set_index=0):
        '''Calculate G_int_bath for current params.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        G_int_bath = self.inter_params['G_int_bath']

        return G_int_bath

    def calc_T_int(self, tes_params_pd, set_index=0):
        '''Calculate T_int_bath for current params.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        T_int = self.inter_params['T_int']

        return T_int

    def calc_k_int_bath(self, tes_params_pd, set_index=0):
        '''Calculate k_int_bath for current params.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        k_int_bath = self.inter_params['k_int_bath']

        return k_int_bath

    def calc_lambda(self, tes_params_pd, set_index=0):
        ''' Solves for lambda in danglinf TES model.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        B = 1./tau_I + 1./tau_el + G_tes_int_Tint/C_int + G_int_bath/C_int
        C = (G_tes_int_Tint+G_int_bath)/(C_int*tau_el) - G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int) + (G_tes_int_Tint+G_int_bath)/(C_int*tau_I) + 1./(tau_el*tau_I) + L_I*R_0*(2+beta_I)/(tau*L)
        D = (G_tes_int_Tint+G_int_bath)/(tau_el*tau_I*C_int) + (G_tes_int_Tint+G_int_bath)*R_0*L_I*(2+beta_I)/(tau*L*C_int) - G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int*tau_el)
                
        q = 1.0/3.0*C-1.0/9.0*(B)**2
        r = 1.0/6.0*B*C-1.0/2.0*D-1.0/27.0*(B)**3        
        delta =  q**3+r**2
        

        if delta < 0:
            s1ps2 = 2*((r**2-delta)**0.5)**(1/3.)*math.cos(1/3.*math.atan2((-delta)**0.5,r))
            s1ms2 = 2*((r**2-delta)**0.5)**(1/3.)*math.sin(1/3.*math.atan2((-delta)**0.5,r))
            lambda1 = s1ps2 - B/3.
            lambda2 = -1/2.*s1ps2 + math.sqrt(3.)/2.*s1ms2-B/3.
            lambda3 = -1/2.*s1ps2 - math.sqrt(3.)/2.*s1ms2-B/3.
        else:
            if (r+(q**3+r**2)**0.5) < 0:
                s1 = -1*(abs(r+(q**3+r**2)**0.5))**(1/3.)
            else:    
                s1 = (r+(q**3+r**2)**0.5)**(1/3.)
            if (r-(q**3+r**2)**0.5) < 0:
                s2 = -1*(abs(r-(q**3+r**2)**0.5))**(1/3.)
            else:    
                s2 = (r-(q**3+r**2)**0.5)**(1/3.)
            lambda1 = s1+s2-B/3.
            lambda2 = -1/2.*(s1+s2)+math.sqrt(3)/2.*1j*(s1-s2)-B/3.
            lambda3 = -1/2.*(s1+s2)-math.sqrt(3)/2.*1j*(s1-s2)-B/3.
        
        lambda_s = (G_tes_int_Tint+G_int_bath)/C_int
        
        A1 = 1/(C+2*B*lambda1+3*lambda1**2)
        A2 = 1/(C+2*B*lambda2+3*lambda2**2)
        A3 = 1/(C+2*B*lambda3+3*lambda3**2)
        AT1 = (1/tau_el+lambda1)/(lambda_s+lambda1)
        AT2 = (1/tau_el+lambda2)/(lambda_s+lambda2)
        AT3 = (1/tau_el+lambda3)/(lambda_s+lambda3)
        
        return lambda1, lambda2, lambda3, A1, A2, A3, AT1, AT2, AT3, delta

    def calc_lambda_arb_init(self, tes_params_pd, set_index=0):
        ''' Solves for lambda in danglinf TES model.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        dI0 = 0.0
        # This is where you control where the phton strickes
        dT1 = 0.0 #Inital temperature step at the TES
        dT2 =  tes_params[f'deltaT_{set_index}'] #Inital temperature step at the interveing body
        
        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        B = 1./tau_I + 1./tau_el + G_tes_int_Tint/C_int + G_int_bath/C_int
        C = (G_tes_int_Tint+G_int_bath)/(C_int*tau_el) - G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int) + (G_tes_int_Tint+G_int_bath)/(C_int*tau_I) + 1./(tau_el*tau_I) + L_I*R_0*(2+beta_I)/(tau*L)
        D = (G_tes_int_Tint+G_int_bath)/(tau_el*tau_I*C_int) + (G_tes_int_Tint+G_int_bath)*R_0*L_I*(2+beta_I)/(tau*L*C_int) - G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int*tau_el)
                
        #for teting
        print('D1',(G_tes_int_Tint+G_int_bath)/(tau_el*tau_I*C_int))
        print('D2', (G_tes_int_Tint+G_int_bath)*R_0*L_I*(2+beta_I)/(tau*L*C_int) )
        print('D3',  G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int*tau_el))

        q = 1.0/3.0*C-1.0/9.0*(B)**2
        r = 1.0/6.0*B*C-1.0/2.0*D-1.0/27.0*(B)**3
        print('r, B ,C D',r, B, C, D)      
        delta =  q**3+r**2

        if delta < 0:
            s1ps2 = 2*((r**2-delta)**0.5)**(1/3.)*math.cos(1/3.*math.atan2((-delta)**0.5,r))
            s1ms2 = 2*((r**2-delta)**0.5)**(1/3.)*math.sin(1/3.*math.atan2((-delta)**0.5,r))
            lambda1 = s1ps2 - B/3.
            lambda2 = -1/2.*s1ps2 + math.sqrt(3.)/2.*s1ms2-B/3.
            lambda3 = -1/2.*s1ps2 - math.sqrt(3.)/2.*s1ms2-B/3.
        else:
            if (r+(q**3+r**2)**0.5) < 0:
                s1 = -1*(abs(r+(q**3+r**2)**0.5))**(1/3.)
            else:    
                s1 = (r+(q**3+r**2)**0.5)**(1/3.)
            if (r-(q**3+r**2)**0.5) < 0:
                s2 = -1*(abs(r-(q**3+r**2)**0.5))**(1/3.)
            else:    
                s2 = (r-(q**3+r**2)**0.5)**(1/3.)
            lambda1 = s1+s2-B/3.
            lambda2 = -1/2.*(s1+s2)+math.sqrt(3)/2.*1j*(s1-s2)-B/3.
            lambda3 = -1/2.*(s1+s2)-math.sqrt(3)/2.*1j*(s1-s2)-B/3.
        
        lambda_s = (G_tes_int_Tint+G_int_bath)/C_int
        
        v20 = -1*I0*L/(L_I*G_tes_int_Ttes)
        v30 = -1*I0*L/(L_I*C_int)
        v21 = v20*(1./tau_el+lambda1)
        v22 = v20*(1./tau_el+lambda2)
        v23 = v20*(1./tau_el+lambda3)
        v31 = v30*(1./tau_el+lambda1)/(lambda_s+lambda1)
        v32 = v30*(1./tau_el+lambda2)/(lambda_s+lambda2)
        v33 = v30*(1./tau_el+lambda3)/(lambda_s+lambda3)
        vdenom = v22*v31-v23*v31-v21*v32+v23*v32+v21*v33-v22*v33
        #Assuming t=0 
        A1 = -1*(-dT2*v22+dT2*v23+dT1*v32-dI0*v23*v32-dT1*v33+dI0*v22*v33)/vdenom
        A2 = -1*(dT2*v21-dT2*v23-dT1*v31+dI0*v23*v31+dT1*v33-dI0*v21*v33)/vdenom
        A3 = (dT2*v21-dT2*v22-dT1*v31+dI0*v22*v31+dT1*v32-dI0*v21*v32)/vdenom
        
        return lambda1, lambda2, lambda3, A1, A2, A3, delta


    def tsolve(self, tes_params_pd, set_index=0):
        ''' Solves the dangling TES for I and T. '''

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        t = tes_params[f't_{set_index}']

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']

        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        #print('delta_T, deltaT + T_int, T_tes', deltaT, deltaT+ T_int, T_tes)

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        lambda1, lambda2, lambda3, A1, A2, A3, AT1, AT2, AT3, delta = self.calc_lambda(tes_params_pd, set_index)

        I = -deltaT*L_I*G_tes_int_Ttes*G_tes_int_Tint/(I0*L*C_tes)*(A1*cmath.exp(lambda1*t)+A2*cmath.exp(lambda2*t)+A3*cmath.exp(lambda3*t))
        Ttes = deltaT*G_tes_int_Tint/C_tes*((1/tau_el+lambda1)*A1*cmath.exp(lambda1*t)+(1/tau_el+lambda2)*A2*cmath.exp(lambda2*t)+(1/tau_el+lambda3)*A3*cmath.exp(lambda3*t))
        Tint = deltaT*G_tes_int_Ttes*G_tes_int_Tint/(C_tes*C_int)*(AT1*A1*cmath.exp(lambda1*t)+AT2*A2*cmath.exp(lambda2*t)+AT3*A3*cmath.exp(lambda3*t))
        
        return I.real, Ttes.real, Tint.real

    def tsolve_t_at_int(self, tes_params_pd, set_index=0):
        ''' Solves the dangling TES for I and T. '''

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        t = tes_params[f't_{set_index}']

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        lambda1, lambda2, lambda3, As1, As2, As3, delta = self.calc_lambda_arb_init(tes_params_pd, set_index)
        print(t, lambda1, lambda2, lambda3, As1, As2, As3)
        print(lambda1*(t),math.exp(lambda1*(t)))

        lambda_s = (G_tes_int_Ttes+G_int_bath)/C_int
        v20 = I0*L/(L_I*G_tes_int_Ttes)
        v30 = I0*L/(L_I*C_int)
        
        #I = (As1*cmath.exp(lambda1*(t))+As2*cmath.exp(lambda2*(t))+As3*cmath.exp(lambda3*(t)))
        I = (As1*cmath.exp(lambda1*(t))+As2*cmath.exp(lambda2*(t))+As3*cmath.exp(lambda3*(t)))
        Ttes = v20*((1/tau_el+lambda1)*As1*cmath.exp(lambda1*(t))+(1/tau_el+lambda2)*As2*cmath.exp(lambda2*(t))+(1/tau_el+lambda3)*As3*cmath.exp(lambda3*(t)))
        Tint = v30*((1/tau_el+lambda1)/(lambda_s+lambda1)*As1*cmath.exp(lambda1*(t))+(1/tau_el+lambda2)/(lambda_s+lambda2)*As2*cmath.exp(lambda2*(t))+(1/tau_el+lambda3)/(lambda_s+lambda3)*As3*cmath.exp(lambda3*(t)))
           
        return I.real, Ttes.real, Tint.real

    def ztes_solve(self, tes_params_pd, set_index=0):
        '''Returns the complex Z of the TES at the current frequnecy value.
        The ztes for the dangling model is the same as for the complex model.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)
        
        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        n_int2 = tes_params[f'n_int2_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        T_bath = tes_params[f'T_bath_{set_index}']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)
        G_int_bath_Tbath = G_int_bath*T_bath**(n_int2-1)/(T_int**(n_int2-1))

        G_etf=L_I*G_tes_int_Ttes
        Zt = R_0*(1+beta_I)+(G_etf*R_0*(2+beta_I))/((-G_etf+G_tes_int_Ttes+1j*omega*C_tes)-(G_tes_int_Tint*G_tes_int_Tint)/(G_tes_int_Tint+G_int_bath_Tbath+1j*omega*C_int))

        return Zt   

    def responsivity_tes_solve(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of a intervening TES. Uses the
        1,2 element of the intenal Y-matrixies'''

        self.calc_derived_params(tes_params_pd, set_index)

        Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13 = self.y_solve(tes_params_pd, set_index)
        s_I  = Y_int_12
        
        return s_I

    def responsivity_tes_solve(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of a intervening TES. Uses the
        1,2 element of the intenal Y-matrixies'''
        
        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        #C_dang = self.tes.get_param_unit_scale(tes_params_pd,'C_dang')*tes_params[f'C_dang_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        #G_int_dang_Ttes = self.tes.get_param_unit_scale(tes_params_pd,'G_int_dang')*tes_params[f'G_int_dang_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        n_int2 = tes_params[f'n_int2_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        T_bath = tes_params[f'T_bath_{set_index}']
        G_int_bath = self.inter_params['G_int_bath']
        I_0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)
        G_int_bath_Tbath = G_int_bath*T_bath**(n_int2-1)/(T_int**(n_int2-1))

        #Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13 = self.y_solve(tes_params_pd, set_index)
        Y_int_12 = G_tes_int_Ttes*L_I*tau_el*(-C_int*omega + 1j*G_int_bath_Tbath + 1j*G_tes_int_Tint)/(I_0*(-C_int*C_tes*L*omega**3*tau_el + 1j*C_int*C_tes*L*omega**2 - 1j*C_int*G_tes_int_Ttes*L*L_I*omega**2*tau_el - C_int*G_tes_int_Ttes*L*L_I*omega + 1j*C_int*G_tes_int_Ttes*L*omega**2*tau_el + C_int*G_tes_int_Ttes*L*omega + C_int*G_tes_int_Ttes*L_I*R_0*beta_I*omega*tau_el + 2*C_int*G_tes_int_Ttes*L_I*R_0*omega*tau_el + 1j*C_tes*G_int_bath_Tbath*L*omega**2*tau_el + C_tes*G_int_bath_Tbath*L*omega + 1j*C_tes*G_tes_int_Tint*L*omega**2*tau_el + C_tes*G_tes_int_Tint*L*omega - G_int_bath_Tbath*G_tes_int_Ttes*L*L_I*omega*tau_el + 1j*G_int_bath_Tbath*G_tes_int_Ttes*L*L_I + G_int_bath_Tbath*G_tes_int_Ttes*L*omega*tau_el - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_0*beta_I*tau_el - 2*1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_0*tau_el - G_tes_int_Tint*G_tes_int_Ttes*L*L_I*omega*tau_el + 1j*G_tes_int_Tint*G_tes_int_Ttes*L*L_I - 1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_0*beta_I*tau_el - 2*1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_0*tau_el))
        #Y_int_12 = G_tes_int_Ttes*L_I*tau_el*(-C_int*omega + 1j*G_int_bath_Tbath + 1j*G_tes_int_Tint)/(I_0*(-C_int*C_tes*L*omega**3*tau_el + 1j*C_int*C_tes*L*omega**2 - 1j*C_int*G_tes_int_Ttes*L*L_I*omega**2*tau_el + 1j*C_int*G_tes_int_Ttes*L*omega**2*tau_el + C_int*G_tes_int_Ttes*L*omega + C_int*G_tes_int_Ttes*L_I*R_0*omega*tau_el - C_int*G_tes_int_Ttes*L_I*R_L*omega*tau_el + 1j*C_tes*G_int_bath_Tbath*L*omega**2*tau_el + C_tes*G_int_bath_Tbath*L*omega + 1j*C_tes*G_tes_int_Tint*L*omega**2*tau_el + C_tes*G_tes_int_Tint*L*omega - G_int_bath_Tbath*G_tes_int_Ttes*L*L_I*omega*tau_el + G_int_bath_Tbath*G_tes_int_Ttes*L*omega*tau_el - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_0*tau_el + 1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_L*tau_el - G_tes_int_Tint*G_tes_int_Ttes*L*L_I*omega*tau_el - 1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_0*tau_el + 1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_L*tau_el))
        s_I  = -1.0*Y_int_12

        return s_I

    def responsivity_int_solve(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of a intervening TES. Uses the
        1,2 element of the intenal Y-matrixies'''
        
        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        n_int2 = tes_params[f'n_int2_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        T_bath = tes_params[f'T_bath_{set_index}']
        G_int_bath = self.inter_params['G_int_bath']
        I_0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)
        G_int_bath_Tbath = G_int_bath*T_bath**(n_int2-1)/(T_int**(n_int2-1))

        Y_int_13 = 1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*tau_el/(I_0*(-C_int*C_tes*L*omega**3*tau_el + 1j*C_int*C_tes*L*omega**2 - 1j*C_int*G_tes_int_Ttes*L*L_I*omega**2*tau_el - C_int*G_tes_int_Ttes*L*L_I*omega + 1j*C_int*G_tes_int_Ttes*L*omega**2*tau_el + C_int*G_tes_int_Ttes*L*omega + C_int*G_tes_int_Ttes*L_I*R_0*beta_I*omega*tau_el + 2*C_int*G_tes_int_Ttes*L_I*R_0*omega*tau_el + 1j*C_tes*G_int_bath_Tbath*L*omega**2*tau_el + C_tes*G_int_bath_Tbath*L*omega + 1j*C_tes*G_tes_int_Tint*L*omega**2*tau_el + C_tes*G_tes_int_Tint*L*omega - G_int_bath_Tbath*G_tes_int_Ttes*L*L_I*omega*tau_el + 1j*G_int_bath_Tbath*G_tes_int_Ttes*L*L_I + G_int_bath_Tbath*G_tes_int_Ttes*L*omega*tau_el - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L - 1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_0*beta_I*tau_el - 2*1j*G_int_bath_Tbath*G_tes_int_Ttes*L_I*R_0*tau_el - G_tes_int_Tint*G_tes_int_Ttes*L*L_I*omega*tau_el + 1j*G_tes_int_Tint*G_tes_int_Ttes*L*L_I - 1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_0*beta_I*tau_el - 2*1j*G_tes_int_Tint*G_tes_int_Ttes*L_I*R_0*tau_el))
        s_I  = Y_int_13

        return s_I

    def responsivity_int_solve_old(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of a intervening TES. Uses the
        1,2 element of the intenal Y-matrixies'''

        self.calc_derived_params(tes_params_pd, set_index)

        Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13 = self.y_solve(tes_params_pd, set_index)
        s_I  = Y_int_13
        
        return s_I

    def y_solve(self, tes_params_pd, set_index=0):
        '''Returns the relevant elements of the Y-matrices for a intervening TES.
        some element of Y_int and Y_ext are differnet are different'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = self.inter_params['T_int']
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        Z22 = -L_I*G_tes_int_Ttes+G_tes_int_Ttes+1j*omega*C_tes
        Z33 = G_tes_int_Tint+G_int_bath+1j*omega*C_int
        Y0 = (L/tau_el+1j*omega*L)*(Z22*Z33-G_tes_int_Tint*G_tes_int_Ttes)+L_I*G_tes_int_Ttes*(R_0*(2+beta_I)*Z33)
        Zi22 = G_tes_int_Ttes+1j*omega*C_tes
        Y0i = (L/tau_el+1j*omega*L)*(Zi22*Z33-G_tes_int_Tint*G_tes_int_Ttes)-L_I*G_tes_int_Ttes*(((R_L-R_0)+1j*omega*L)*Z33)
        #print 'Y0, Y0i: ', Y0, Y0i #Checked and are equal as expected
        
        Y_int_11 = (Zi22*Z33-G_tes_int_Ttes*G_tes_int_Tint)/Y0i
        Y_ext_11 = (Z22*Z33-G_tes_int_Ttes*G_tes_int_Tint)/Y0
        Y_int_12 = (L_I*G_tes_int_Ttes/I0*Z33)/Y0i
        Y_ext_12 = (L_I*G_tes_int_Ttes/I0*Z33)/Y0
        Y_int_13 = ((L_I*G_tes_int_Ttes/I0)*(-1*G_tes_int_Tint))/Y0i

        return Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13

    def current_noise_solve(self, tes_params_pd, set_index=0):
        ''' Returns the total and various current noises.
        The the terms are the same as for the compund model since the Y
        matrix is the same.'''

        # not updated
        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']   
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        T0 = tes_params[f'T_tes_{set_index}']
        Tb = tes_params[f'T_bath_{set_index}']
        n = tes_params[f'n_mem_{set_index}']
        M = tes_params[f'M_{set_index}']
        xi = tes_params[f'xi_{set_index}']
        fcorr = tes_params[f'fcorr_{set_index}']
        squid_noise = tes_params[f'squid_noise_{set_index}']
        background_noise = tes_params[f'background_noise_{set_index}']
        abs_background_noise = tes_params[f'abs_background_noise_{set_index}']

        #Tsh allows for a different temperture for the shunt resistor for now we set it to Tb
        Tsh = Tb

        I0 = self.inter_params['I0']
        L_I = self.inter_params['L_I']

        tau = C1 / G1
        tau_I = tau/(1-L_I)

        epsilon = 1+2.0*beta_I+xi*beta_I**2.
        #We think this is okay, but we need to understand if beta can be negative or if epsilon can be negative
        epsilon = abs(epsilon)

        G1atTb = G1*Tb**(n-1)/(T0**(n-1))
        
        
        Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13 = self.y_solve(tes_params_pd, set_index)
        # Check this!!
        Y12m13 = Y_int_12*Y_int_12 + Y_int_13*Y_int_13 + 2*(1-2*fcorr)*Y_int_12*Y_int_13 
        #Y12m13 = Y_int_12-Y_int_13
        #Y12m13 = Y_int_12*Y_int_12 + Y_int_13*Y_int_13 - 2*Y_int_12*Y_int_13
        
        #SI2s are the current noise squared
        SI2_R0 = 4*kB*T0*R_0*epsilon*(1+M**2)*(Y_int_11*Y_int_11.conjugate())
        SI2_RL = 4*kB*Tsh*R_L*(Y_ext_11*Y_ext_11.conjugate())
        SI2_G1 = 2*kB*(T0**2*G1+Tb**2*G1atTb)*(Y_int_12*Y_int_12.conjugate())
        SI2_G2 = 2*2*kB*T0**2*G2*(Y12m13*Y12m13.conjugate())**0.5
        SI2_squid = (squid_noise)**2
        SI2_background = (background_noise)**2*(Y_int_12*Y_int_12.conjugate())
        SI2_abs_background = (abs_background_noise)**2*(Y_int_13*Y_int_13.conjugate())
        
        SI2_total = SI2_R0+SI2_RL+SI2_G1+SI2_G2+SI2_squid+SI2_background+SI2_abs_background
        
        SI_total = np.sqrt(SI2_total.real)
        SI_R0 = np.sqrt(SI2_R0.real)
        SI_RL = np.sqrt(SI2_RL.real)
        SI_G1 = np.sqrt(SI2_G1.real)
        SI_G2 = np.sqrt(SI2_G2.real)
        SI_squid = np.sqrt(SI2_squid)
        SI_background = np.sqrt(SI2_background.real)
        SI_abs_background = np.sqrt(SI2_abs_background.real)
        
        return SI_total, SI_R0, SI_RL, SI_G1, SI_G2, SI_squid, SI_background, SI_abs_background


    def nep2_solve(self, tes_params_pd, set_index=0):
        ''' Returns the total and various noise equivlent powers squared '''
        
        # not updated
        s_I = self.responsivity_solve(tes_params_pd, set_index=0)
        SI_total, SI_R0, SI_RL, SI_G1, SI_G2, SI_squid, SI_back, SI_abs_back  = self.current_noise_solve(tes_params_pd, set_index)
        NEP2_R0 = np.real(SI_R0**2/(s_I*s_I.conjugate()))
        NEP2_RL = np.real(SI_RL**2/(s_I*s_I.conjugate()))
        NEP2_G1 = np.real(SI_G1**2/(s_I*s_I.conjugate()))
        NEP2_G2 = np.real(SI_G2**2/(s_I*s_I.conjugate()))
        NEP2_squid = np.real(SI_squid**2/(s_I*s_I.conjugate()))
        NEP2_back = np.real(SI_back**2/(s_I*s_I.conjugate()))
        NEP2_abs_back = np.real(SI_abs_back**2/(s_I*s_I.conjugate()))
        
        NEP2_total = NEP2_R0 + NEP2_RL + NEP2_G1 + NEP2_G2 + NEP2_squid + NEP2_back + NEP2_abs_back
                
        return NEP2_total, NEP2_R0, NEP2_RL, NEP2_G1, NEP2_G2, NEP2_squid, NEP2_back, NEP2_abs_back


    def calc_pulse(self, tes_params_pd, t_vals, set_index=0):
        '''Calculate the current and temperatures of the TES as a function of time.'''

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tint_vals = np.zeros_like(t_vals)

        for index in range(len(t_vals)):
            tes_params_pd[f't_{set_index}'].set(value=t_vals[index])
            Ites, Ttes, Tint = self.tsolve(tes_params_pd, set_index)
            Ites_vals[index] = Ites
            Ttes_vals[index] = Ttes
            Tint_vals[index] = Tint

        return Ites_vals, Ttes_vals, Tint_vals

    def calc_pulse_tint(self, tes_params_pd, t_vals, set_index=0):
        '''Calculate the current and temperatures of the TES as a function of time.'''

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tint_vals = np.zeros_like(t_vals)

        for index in range(len(t_vals)):
            tes_params_pd[f't_{set_index}'].set(value=t_vals[index])
            Ites, Ttes, Tint = self.tsolve_t_at_int(tes_params_pd, set_index)
            Ites_vals[index] = Ites
            Ttes_vals[index] = Ttes
            Tint_vals[index] = Tint

        return Ites_vals, Ttes_vals, Tint_vals

    def calc_cz(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the impeadence of the TES as a function of frequnecy.'''

        Zt_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            Zt = self.ztes_solve(tes_params_pd, set_index)
            Zt_vals[index] = Zt
  
        return Zt_vals

    def calc_responsivity_tes(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the responsivity of power to the TES as a function of frequnecy.'''

        sI_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            sI = self.responsivity_tes_solve(tes_params_pd, set_index)
            sI_vals[index] = sI
  
        return sI_vals

    def calc_responsivity_int(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the responsivity of power to the interveing heat capcity
          as a function of frequnecy.'''

        sI_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            sI = self.responsivity_int_solve(tes_params_pd, set_index)
            sI_vals[index] = sI
  
        return sI_vals

    def calc_noise(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the impeadence of the TES as a function of frequnecy.'''

        #not update yet form simple

        SI_total_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_R0_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_RL_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_G1_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_G2_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_squid_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_back_vals = np.zeros_like(f_vals, dtype=np.float64)
        SI_abs_back_vals = np.zeros_like(f_vals, dtype=np.float64)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            SI_total, SI_R0, SI_RL, SI_G1, SI_G2, SI_squid, SI_back, SI_abs_back  = self.current_noise_solve(tes_params_pd, set_index)
            SI_total_vals[index] = SI_total
            SI_R0_vals[index] = SI_R0
            SI_RL_vals[index] = SI_RL
            SI_G1_vals[index] = SI_G1
            SI_G2_vals[index] = SI_G2
            SI_squid_vals[index] = SI_squid
            SI_back_vals[index] = SI_back
            SI_abs_back_vals[index] = SI_abs_back
  
        return SI_total_vals, SI_R0_vals, SI_RL_vals, SI_G1_vals, SI_G2_vals ,SI_squid_vals, SI_back_vals, SI_abs_back_vals

    def calc_nep2(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the NEP squared of the TES as a function of frequnecy.'''

        #not update yet form simple

        NEP_total_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_R0_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_RL_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_G1_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_G2_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_squid_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_back_vals = np.zeros_like(f_vals, dtype=np.float64)
        NEP_abs_back_vals = np.zeros_like(f_vals, dtype=np.float64)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            NEP_total, NEP_R0, NEP_RL, NEP_G1, NEP_G2, NEP_squid, NEP_back, NEP_abs_back  = self.nep2_solve(tes_params_pd, set_index)
            NEP_total_vals[index] = NEP_total
            NEP_R0_vals[index] = NEP_R0
            NEP_RL_vals[index] = NEP_RL
            NEP_G1_vals[index] = NEP_G1
            NEP_G2_vals[index] = NEP_G2
            NEP_squid_vals[index] = NEP_squid
            NEP_back_vals[index] = NEP_back
            NEP_abs_back_vals[index] = NEP_abs_back
  
        return NEP_total_vals, NEP_R0_vals, NEP_RL_vals, NEP_G1_vals, NEP_G2_vals ,NEP_squid_vals, NEP_back_vals, NEP_abs_back_vals

    def solve_energy_res_integral(self, frequency, tes_params_pd, set_index):
        '''The integral of that gives the energy resolution'''
        
        tes_params_pd[f'f_{set_index}'].set(value=frequency)

        NEP2_total, NEP2_R0, NEP2_RL, NEP2_G1, NEP2_G2, NEP2_squid, NEP2_back, NEP2_abs_back = self.nep2_solve(tes_params_pd, set_index)
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

    def tsolve_numint(self, X, t, tes_params_pd, set_index):
        ''' Solves the Complex TES for I and T using numerical integration of
        non-linear TES equations. '''

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        C_tes = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        C_int = self.tes.get_param_unit_scale(tes_params_pd,'C_int')*tes_params[f'C_int_{set_index}']
        G_tes_int = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_int')*tes_params[f'G_tes_int_{set_index}']
        n_int1 = tes_params[f'n_int1_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']

        L_I = self.inter_params['L_I']
        tau = C_tes / G_tes_int
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T_tes = tes_params[f'T_tes_{set_index}']
        T_int = float(self.inter_params['T_int'])
        G_int_bath = self.inter_params['G_int_bath']
        I0 = self.inter_params['I0']

        G_tes_int_Ttes = G_tes_int
        G_tes_int_Tint = G_tes_int_Ttes*T_tes**(n_int1-1)/T_int**(n_int1-1)

        out = np.array([ -1./tau_el*X[0]-L_I*G_tes_int_Ttes/(I0*L)*X[1] ,
                            I0*R_0*(2+beta_I)/C_tes*X[0] - (-1.0*L_I*G_tes_int_Ttes/C_tes + G_tes_int_Ttes/C_tes)*X[1] + G_tes_int_Tint/C_tes*X[2],
                            G_tes_int_Ttes/C_int*X[1] - (G_int_bath+G_tes_int_Tint)/C_int*X[2] ])

        return out

    def calc_pulse_numint(self, tes_params_pd, t_vals, X0=None, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        deltaT = tes_params[f'deltaT_{set_index}']

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        if X0 is None:
            X0 = np.array([0.0, 0.0, deltaT])
            print('Assume delta_T delviered to intervening body.')
        X, infodict = integrate.odeint(self.tsolve_numint, X0, t_vals, args=(tes_params_pd, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals,Tabs_vals = X.T

        return Ites_vals, Ttes_vals, Tabs_vals

