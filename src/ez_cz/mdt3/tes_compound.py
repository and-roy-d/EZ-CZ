"""
Compund TES class of MDT Microcal Design Tools
Current Version Created on Jan 17, 2020
Equations from orginal version of MDT

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

class TES_Compound(object):
    """Class for methods of the Simlpe TES model."""
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

        self.compund_params ={}
        self.compund_params['k'] = self.k
        self.compund_params['I0'] = self.I0
        self.compund_params['PJ0'] = self.PJ0
        self.compund_params['L_I'] = self.L_I
        self.compund_params['lambdaplus'] = self.lambdaplus
        self.compund_params['lambdaminus'] = self.lambdaminus
        self.compund_params['Ites'] = self.Ites
        self.compund_params['Ttes'] = self.Ttes

    def calc_derived_params(self, tes_params_pd, set_index=0):
        ''' Calculate parameters derived from input parameters. '''

        tes_params = tes_params_pd.valuesdict()

        #t = tes_params[f't_{set_index}']
        #C = tes_params[f'C_tes_{set_index}']
        #G = tes_params[f'G_tes_bath_{set_index}']
        #R_0 = tes_params[f'R_0_{set_index}']
        #beta_I = tes_params[f'beta_I_{set_index}']
        #delta_T = tes_params[f'deltaT_{set_index}']
        #I0 = tes_params['I0']
        #L_I = tes_params['L_I']




        C_photon = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']

        if tes_params[f'initE_use_{set_index}'] is True:
            tes_params_pd[f'deltaT_{set_index}'].set(value=(tes_params[f'initE_{set_index}']*ec/C_photon))
        else:
            tes_params_pd[f'initE_{set_index}'].set(value=(tes_params[f'deltaT_{set_index}']*C_photon/ec))
        
        # Calculate currents, powers and temperatures
        R_0 = tes_params[f'R_0_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        G_tes_bath = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        T_tes = tes_params[f'T_tes_{set_index}']
        T_bath = tes_params[f'T_bath_{set_index}']
        n_mem = tes_params[f'n_mem_{set_index}']

        k = G_tes_bath/(n_mem*T_tes**(n_mem-1))
        I0 = (k/R_0*(T_tes**n_mem-T_bath**n_mem))**0.5
        PJ0 = I0**2*R_0
        L_I = PJ0*alpha_I/(G_tes_bath*T_tes)

        self.compund_params['k'] = k
        self.compund_params['I0'] = I0
        self.compund_params['PJ0'] = PJ0
        self.compund_params['L_I'] = L_I


    def calc_lambda(self, tes_params_pd, set_index=0):
        ''' Solves for lambda in simple TES. Returns lambdaplus and lambdaminus '''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.compund_params['L_I']
        tau = C1 / G1
        tau_I = tau/(1-L_I)
        tau_el = L/(R_L+R_0*(1+beta_I))

        T0 = tes_params[f'T_tes_{set_index}']
        I0 = self.compund_params['I0']

        B = 1.0/tau_I + 1.0/tau_el + G2/C2 + G2/C1
        C = G2/(C2*tau_el) + G2/(C1*tau_el) + G2/(C2*tau_I) + 1/(tau_el*tau_I) + L_I*R_0*(2+beta_I)/(tau*L)
        D = G2/(tau_el*tau_I*C2) + G2*R_0*L_I*(2+beta_I)/(tau*L*C2)
        
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
        
        A1 = 1/(C+2*B*lambda1+3*lambda1**2)
        A2 = 1/(C+2*B*lambda2+3*lambda2**2)
        A3 = 1/(C+2*B*lambda3+3*lambda3**2)
        AT11 = A1*(1./tau_el+lambda1)
        AT12 = A2*(1./tau_el+lambda2)
        AT13 = A3*(1./tau_el+lambda3)
        AT21 = AT11/(G2/C2+lambda1)
        AT22 = AT12/(G2/C2+lambda2)
        AT23 = AT13/(G2/C2+lambda3)
        #AT1 = (C-G2/C2*(1/tau_el+1/tau_I))+(B-G2/C2)*lambda1+lambda1**2
        #AT2 = (C-G2/C2*(1/tau_el+1/tau_I))+(B-G2/C2)*lambda2+lambda2**2
        #AT3 = (C-G2/C2*(1/tau_el+1/tau_I))+(B-G2/C2)*lambda3+lambda3**2
        
        return lambda1, lambda2, lambda3, A1, A2, A3, AT11, AT12, AT13, AT21, AT22, AT23, delta

    def tsolve(self, tes_params_pd, set_index=0):
        ''' Solves the Simple TES for I and T. '''

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        t = tes_params[f't_{set_index}']
        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))

        lambda1, lambda2, lambda3, A1, A2, A3, AT11, AT12, AT13, AT21, AT22, AT23, delta = self.calc_lambda(tes_params_pd, set_index)

        I = -deltaT*L_I*G1*G2/(I0*L*C1)*(A1*np.exp(lambda1*t)+A2*np.exp(lambda2*t)+A3*np.exp(lambda3*t))
        T1 = deltaT*G2/C1*(AT11*np.exp(lambda1*t)+AT12*np.exp(lambda2*t)+AT13*np.exp(lambda3*t))
        T2 = deltaT*G2**2/(C1*C2)*(AT21*np.exp(lambda1*t)+AT22*np.exp(lambda2*t)+AT23*np.exp(lambda3*t))
        
        return I, T1, T2

    def ztes_solve(self, tes_params_pd, set_index=0):
        '''Returns the complex Z of the TES at the current frequnecy value.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)
        
        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']       
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.compund_params['L_I']

        tau = C1 / G1
        tau_I = tau/(1-L_I)       

        Zt = R_0*(1+beta_I)+(L_I*G1*R_0*(2+beta_I)*(G2+1j*omega*C2))/((C1/tau_I+G2+1j*omega*C1)*(G2+1j*omega*C2)-G2**2) 
        
        return Zt   

    def responsivity_solve(self, tes_params_pd, set_index=0):
        ''' Returns the responsivity of complex TES. Uses the
        1,3 element of the internal Y matrix.'''

        #tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13 = self.y_solve(tes_params_pd, set_index)
        s_I  = Y_int_13
        
        return s_I

    def y_solve(self, tes_params_pd, set_index=0):
        '''Returns the relevant elements of the Y-matrices for a comlpex TES.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        f = tes_params[f'f_{set_index}']
        omega = 2.0*np.pi*f

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        alpha_I = tes_params[f'alpha_I_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        L_I = self.compund_params['L_I']

        #T0 = tes_params[f'T_tes_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau = C1 / G1
        tau_I = tau/(1-L_I)  

        tau_el = L/(R_L+R_0*(1+beta_I))      
        #TauAb_y = C2/G2;

        Y0 = (L/tau_el+1j*omega*L)*((C1/tau+G2+1j*omega*C1)*(G2+1j*omega*C2)-G2**2)-L_I*G1*((R_L-R_0)+1j*omega*L)*(G2+1j*omega*C2)
        #Y11 = (C1*(G2/Tau_y-omega.^2*C2)+i.*omega*C1*(2*G2+C2/Tau_y))./Y0;
        Y_int_11 = ((C1/tau+G2+1j*omega*C1)*(G2+1j*omega*C2)-G2**2)/Y0
        Y_ext_11 = ((C1/tau_I+G2+1j*C1*omega)*(G2+1j*C2*omega)-G2**2)/Y0
        Y_int_12 = -L_I*G1/I0*(G2+1j*omega*C2)/Y0
        Y_ext_12 = -L_I*G1/I0*(G2+1j*omega*C2)/Y0
        Y_int_13 = -G2*G1*L_I/I0/Y0 

        return Y_int_11, Y_ext_11, Y_int_12, Y_ext_12, Y_int_13

    def current_noise_solve(self, tes_params_pd, set_index=0):
        ''' Returns the total and various current noises for the compund model.'''

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

        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

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
        '''Calculate the current and temperature of the TES as a function of time.'''

        #not update yet form simple

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        for index in range(len(t_vals)):
            tes_params_pd[f't_{set_index}'].set(value=t_vals[index])
            Ites, Ttes, Tabs = self.tsolve(tes_params_pd, set_index)
            Ites_vals[index] = Ites
            Ttes_vals[index] = Ttes
            Tabs_vals[index] = Tabs

        return Ites_vals, Ttes_vals, Tabs_vals

    def calc_cz(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the impeadence of the TES as a function of frequnecy.'''

        Zt_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            Zt = self.ztes_solve(tes_params_pd, set_index)
            Zt_vals[index] = Zt
  
        return Zt_vals

    def calc_responsivity(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the responsivity of the TES as a function of frequnecy.'''

        sI_vals = np.zeros_like(f_vals, dtype=complex)

        for index in range(len(f_vals)):
            tes_params_pd[f'f_{set_index}'].set(value=f_vals[index])
            sI = self.responsivity_solve(tes_params_pd, set_index)
            sI_vals[index] = sI
  
        return sI_vals

    def calc_noise(self, tes_params_pd, f_vals, set_index=0):
        '''Calculate the inoise msources of the TES as a function of frequnecy.'''

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

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        out = np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1], I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I+G2/C1)*X[1] + G2/C1*X[2], G2/C2*X[1] - G2/C2*X[2] ])

        return out

    def calc_pulse_numint(self, tes_params_pd, t_vals, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations.'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        deltaT = tes_params[f'deltaT_{set_index}']

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        X0 = np.array([0.0, 0.0, deltaT])
        X, infodict = integrate.odeint(self.tsolve_numint, X0, t_vals, args=(tes_params_pd, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals,Tabs_vals = X.T

        return Ites_vals, Ttes_vals, Tabs_vals

    def tsolve_numint_Tpulse(self, X, t, tes_params_pd, T2_in_f, set_index):
        ''' Solves the Complex TES for I and T using numerical integration of
        non-linear TES equations and provides the capabilty to add heat to the 
        absorber as function of time by passing an funcion (possibly interpolated)'''

        T2_in = T2_in_f(t)

        tes_params = tes_params_pd.valuesdict()

        self.calc_derived_params(tes_params_pd, set_index)

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        return np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1] ,
                            I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I+G2/C1)*X[1] + G2/C1*X[2],
                            G2/C2*X[1] - G2/C2*X[2] + T2_in])

    def calc_pulse_numint_Tpulse(self, tes_params_pd, t_vals, T2_in_f, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations and
        provides the capabilty to add heat to the absorber as funmction of time by passing 
        a funcion (possibly interpolated)'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        deltaT = tes_params[f'deltaT_{set_index}']

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        X0 = np.array([0.0, 0.0, 0.0])
        X, infodict = integrate.odeint(self.tsolve_numint_Tpulse, X0, t_vals, args=(tes_params_pd, T2_in_f, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals,Tabs_vals = X.T

        return Ites_vals, Ttes_vals, Tabs_vals

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
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        return np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1]+I_in,
                            I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I+G2/C1)*X[1] + G2/C1*X[2],
                            G2/C2*X[1] - G2/C2*X[2]])

    def calc_pulse_numint_Ipulse(self, tes_params_pd, t_vals, I_in_f, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations and
        provides the capabilty to add heat to the absorber as funmction of time by passing 
        a funcion (possibly interpolated)'''

        tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        deltaT = tes_params[f'deltaT_{set_index}']

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        X0 = np.array([0.0, 0.0, 0.0])
        X, infodict = integrate.odeint(self.tsolve_numint_Ipulse, X0, t_vals, args=(tes_params_pd, I_in_f, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals,Tabs_vals = X.T

        return Ites_vals, Ttes_vals, Tabs_vals


    def tsolve_numint_Tb_pulse(self, X, t, tes_params_pd, Tb_f, set_index):
        ''' Solves the Complex TES for I and T using numerical integration of
        non-linear TES equations and provides the capabilty to add heat to the 
        absorber as function of time by passing an funcion (possibly interpolated)'''

        tes_params_pd[f'T_bath_{set_index}'].set(value=Tb_f(t))
        
        self.calc_derived_params(tes_params_pd, set_index)
        tes_params = tes_params_pd.valuesdict()

        L = self.tes.get_param_unit_scale(tes_params_pd,'L')*tes_params[f'L_{set_index}']
        C1 = self.tes.get_param_unit_scale(tes_params_pd,'C_tes')*tes_params[f'C_tes_{set_index}']
        G1 = self.tes.get_param_unit_scale(tes_params_pd,'G_tes_bath')*tes_params[f'G_tes_bath_{set_index}']
        C2 = self.tes.get_param_unit_scale(tes_params_pd,'C_abs')*tes_params[f'C_abs_{set_index}']
        G2 = self.tes.get_param_unit_scale(tes_params_pd,'G_abs_tes')*tes_params[f'G_abs_tes_{set_index}']
        R_0 = tes_params[f'R_0_{set_index}']
        R_L = tes_params[f'R_L_{set_index}']
        beta_I = tes_params[f'beta_I_{set_index}']
        deltaT = tes_params[f'deltaT_{set_index}']
        I0 = self.compund_params['I0']
        L_I = self.compund_params['L_I']

        tau_el = L/(R_L+R_0*(1+beta_I))
        tau_I = (C1/G1)/(1-L_I)

        return np.array([ -1./tau_el*X[0]-L_I*G1/(I0*L)*X[1] ,
                            I0*R_0*(2+beta_I)/C1*X[0] - (1./tau_I+G2/C1)*X[1] + G2/C1*X[2],
                            G2/C2*X[1] - G2/C2*X[2]])

    def calc_pulse_numint_Tb_pulse(self, tes_params_pd, t_vals, T2_in_f, set_index=0, ):
        '''Calculate the current and temperature of the TES as a function of time by
        numerically integrating the linearized coupled differntail equations and
        provides the capabilty to add heat to the absorber as funmction of time by passing 
        a funcion (possibly interpolated)'''

        #tes_params = tes_params_pd.valuesdict()
        self.calc_derived_params(tes_params_pd, set_index)

        Ites_vals = np.zeros_like(t_vals)
        Ttes_vals = np.zeros_like(t_vals)
        Tabs_vals = np.zeros_like(t_vals)

        X0 = np.array([0.0, 0.0, 0.0])
        X, infodict = integrate.odeint(self.tsolve_numint_Tb_pulse, X0, t_vals, args=(tes_params_pd, T2_in_f, set_index), full_output=True)
        print(infodict['message'])
        Ites_vals, Ttes_vals,Tabs_vals = X.T

        return Ites_vals, Ttes_vals, Tabs_vals

