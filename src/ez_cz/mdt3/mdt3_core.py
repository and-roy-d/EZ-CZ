"""
MDT Microcal Design Tools
Current Version Created on Jan 17, 2020
Original Version Created on Nov 14, 2009

MDT3 is written to modernize the code by switching to python3 and using newer modules for
things like fitting. It is a work in progress!
"""

from lmfit import Parameters

kB = 1.3806503e-23
ec = 1.60217646e-19

class MDT3_Core(object):
    """Class that defines MDT3 glue.

    The futrue of MDT.

    Note:
        Just a skeleton

    Attributes:
        L (float): The inductance in series with the TES.
        C_tes (float): The heat capcity of the TES.

    """
    def __init__(self):
        """Setup all of the paramters in the data strcuture. This setup has mostly become
         irrelvant since the switch to the Paramters structure from lmfit."""
                
        print('WARNING: The units have been changed for the inductance (nH) and all')
        print('heat capacities (pJ/K) and thermal condcutances (nW/K)! See readme for details.')

        self.resistance_model = None
        self.thermal_model = None

        self.t = 0.001
        self.t_des = 'current time step value'
        self.f = 1.0
        self.f_des = 'current frequqency step value'
        self.t0 = 0.0
        self.t0_des = 'need to check this maybe for numerical solve'

        self.dI0t0 = 0.0
        self.dT1t0 = 0.0
        self.dT2t0 = 0.0
        self.dT3t0 = 0.0        

        self.L = 1.0
        self.L_uscale = 1.0e-9
        self.L_des = 'inductance (nH) in series with the TES in the Thevenin equivlent circuit'


        self.C_tes = 3.25
        self.C_tes_uscale = 1.0e-12
        self.C_tes_des = 'heat capacitance (pJ/K) of the TES in all models'

        self.C_abs = 13.2
        self.C_abs_uscale = 1.0e-12
        self.C_abs_des = 'heat capacitence (pJ/K) of the absorber or "other body" in \
            compund and dangling models'

        self.C_dang = 3.25
        self.C_dang_uscale = 1.0e-12
        self.C_dang_des = 'heat capacitence (pJ/K) of the dangling body in three \
            body models.'
        self.C_int = 5.0
        self.C_int_uscale = 1.0e-12
        self.C_int_des = 'heat capacitence (pJ/K) of the intervening body in intervening model'

        self.G_tes_bath = 3.2
        self.G_tes_bath_uscale = 1.0e-9
        self.G_tes_bath_des = 'thermal conductance (nW/K) between the TES and the heat bath'

        self.G_abs_tes = 18.0
        self.G_abs_tes_uscale = 1.0e-9
        self.G_abs_tes_des = 'thermal conductance (nW/K) between the TES and the other thermal \
            body in compund and dangling models'

        self.G_tes_int = 10.0
        self.G_tes_int_uscale = 1.0e-9
        self.G_tes_int_des = 'thermal conductance (nW/K) between the TES and the intervening\
            thermal body in the intervening body model'

        self.G_int_dang = 10.0
        self.G_int_dang_uscale = 1.0e-9
        self.G_int_dang_des = 'thermal conductance (nW/K) between the intervening hrat capavity\
            and a dangling heat capcity in a 3-body model. Defined here as at T_tes and scaled to\
            to T_int in the methods.'

        self.R_0 = 1.8e-3
        self.R_0_des = 'the TES resistance in the Thevenin equivlent circut'

        self.R_0g = 1.8e-3
        self.R_0g_des = 'the TES resistance in the Thevenin equivlent circut of the guard TES'

        self.R_L = 2.14e-4
        self.R_L_des = 'the load or "shunt" resistance in the Thevenin equivlent circut'

        self.alpha_I = 42.6
        self.alpha_I_des = 'the lorgrathmic partial derivative of temperature with respect \
            to time at constant current'

        self.alpha_Ig = 42.6
        self.alpha_Ig_des = 'the lorgrathmic partial derivative of temperature with respect \
            to time at constant current for the gurad TES'

        self.beta_I = 1.78
        self.beta_I_des = 'the lorgrathmic partial derivative of currente with respect \
            to time at constanttemperature'
        
        self.beta_Ig = 1.78
        self.beta_Ig_des = 'the lorgrathmic partial derivative of currente with respect \
            to time at constanttemperature for the guard TES'

        self.T_tes = 0.127
        self.T_tes_des = 'temperature of the TES in units of Kelvin'

        self.T_tesg = 0.127
        self.T_tesg_des = 'temperature of the TES in units of Kelvin for the guard TES'

        self.T_int = 0.120
        self.T_int_des = 'temperature of the interveing heat capacity in units of Kelvin'

        self.T_bath = 0.100
        self.T_bath_des = 'temperature of the heat bath in units of Kelvin'

        self.initE = 1.0
        self.initE_des = 'inital energy of the pulse in eV, will be calcuted from \
            deltaT if initE_use is False.'

        self.initE_use = True
        self.initE_use_des = 'if True the intial change in temperature is taken from \
            the initE paramter, if False the pulse energy is calcuted from the inital \
            change in temperature'

        self.deltaT = self.initE*ec/self.C_tes
        self.deltaT_des = 'inital change in temperature of the pulse in K, will be \
            calcuted from initE if initE_use is True.'

        self.n_mem = 3.32
        self.n_mem_des = 'CHECK THIS. the exponent on the power law thermal behavoir of the membrane'

        self.n_int1 = 3.0
        self.n_int1_des = 'The exponent on the power law thermal behavoir beteen TES and interveing C.'

        self.n_int2 = 3.0
        self.n_int2_des = 'The exponent on the power law thermal behavoir beteen interveing C and bath.'

        self.n_dang = 3.0
        self.n_dang_des = 'The exponent on the power law thermal behavoir beteen interveing C and the dangling C.'

        self.k_int_ratio = 0.5
        self.k_int_ratio_des = 'The ratio of the TES to intervening k and the interveing to bath k.'
  
        self.M = 0.0
        self.M_des = 'the paramter for scaling unexplained noise that has the same form\
            as the TES Johnson noise'

        self.xi = 0.0
        self.xi_des = 'another paramter for scaling unexplained noise. Leave this at zero \
            unless you are familiar with its use'

        self.fcorr = 1.0
        self.fcorr_des = 'controls the range of correlation or anticorrelation of the thermal\
            flucuation noise. It should be set to 1.0 for neergy conservsation.'

        self.squid_noise = 2.0e-11
        self.squid_noise_des = 'the readout noise in amps per root hertz'

        self.background_noise = 0
        self.background_noise_des = 'the noise from stray photons hitting the TES in \
        watts per root hertz. CHECH THIS'

        self.abs_background_noise = 0
        self.abs_background_noise_des = 'the noise from stray photons hitting the absrobers in \
        watts per root hertz. CHECH THIS'

        self.int_background_power = 0
        self.int_background_power_des = 'the noise from stray photons hitting the absrobers in \
        watts per root hertz. CHECH THIS'

    def makeDefaultParamsDict(self, num_sets):

        tes_params = Parameters()

        for index in range(num_sets):
            tes_params.add(f't_{index}', value=self.t, min=0, vary=False)
            tes_params[f't_{index}'].user_data = {'hint': self.t_des}

            tes_params.add(f'f_{index}', value=self.f, min=0, vary=False)
            tes_params[f'f_{index}'].user_data = {'hint': self.f_des}

            tes_params.add(f't0_{index}', value=self.t0, min=0, vary=False)
            tes_params[f't0_{index}'].user_data = {'hint': self.t0_des}

            tes_params.add(f'L_{index}', value=self.L, min=0, vary=False)
            tes_params[f'L_{index}'].user_data = {'hint': self.L_des,'uscale': self.L_uscale}

            tes_params.add(f'C_tes_{index}', value=self.C_tes, min=0, vary=False)
            tes_params[f'C_tes_{index}'].user_data = {'hint': self.C_tes_des,'uscale': self.C_tes_uscale}

            tes_params.add(f'C_abs_{index}', value=self.C_abs, min=0, vary=False)
            tes_params[f'C_abs_{index}'].user_data = {'hint': self.C_abs_des,'uscale': self.C_abs_uscale}

            tes_params.add(f'C_int_{index}', value=self.C_int, min=0, vary=False)
            tes_params[f'C_int_{index}'].user_data = {'hint': self.C_int_des,'uscale': self.C_int_uscale}

            tes_params.add(f'G_tes_bath_{index}', value=self.G_tes_bath, min=0, vary=False)
            tes_params[f'G_tes_bath_{index}'].user_data = {'hint': self.G_tes_bath_des,'uscale': self.G_tes_bath_uscale}

            tes_params.add(f'G_abs_tes_{index}', value=self.G_abs_tes, min=0, vary=False)
            tes_params[f'G_abs_tes_{index}'].user_data = {'hint': self.G_abs_tes_des,'uscale': self.G_abs_tes_uscale}

            tes_params.add(f'R_0_{index}', value=self.R_0, min=0, vary=False)
            tes_params[f'R_0_{index}'].user_data = {'hint': self.R_0_des}

            tes_params.add(f'R_L_{index}', value=self.R_L, min=0, vary=False)
            tes_params[f'R_L_{index}'].user_data = {'hint': self.R_L_des}

            tes_params.add(f'alpha_I_{index}', value=self.alpha_I, min=0, vary=False)
            tes_params[f'alpha_I_{index}'].user_data = {'hint': self.alpha_I_des}

            tes_params.add(f'beta_I_{index}', value=self.beta_I, min=0, vary=False)
            tes_params[f'beta_I_{index}'].user_data = {'hint': self.beta_I_des}

            tes_params.add(f'T_tes_{index}', value=self.T_tes, min=0, vary=False)
            tes_params[f'T_tes_{index}'].user_data = {'hint': self.T_tes_des}

            tes_params.add(f'T_bath_{index}', value=self.T_bath, min=0, vary=False)
            tes_params[f'T_bath_{index}'].user_data = {'hint': self.T_bath_des}

            tes_params.add(f'initE_{index}', value=self.initE, min=0, vary=False)
            tes_params[f'initE_{index}'].user_data = {'hint': self.initE_des}

            tes_params.add(f'initE_use_{index}', value=self.initE_use, min=0, vary=False)
            tes_params[f'initE_use_{index}'].user_data = {'hint': self.initE_use_des}

            tes_params.add(f'deltaT_{index}', value=self.deltaT, min=0, vary=False)
            tes_params[f'deltaT_{index}'].user_data = {'hint': self.deltaT_des}

            tes_params.add(f'n_mem_{index}', value=self.n_mem, min=0, vary=False)
            tes_params[f'n_mem_{index}'].user_data = {'hint': self.n_mem_des}

            tes_params.add(f'M_{index}', value=self.M, min=0, vary=False)
            tes_params[f'M_{index}'].user_data = {'hint': self.M_des}

            tes_params.add(f'xi_{index}', value=self.xi, min=0, vary=False)
            tes_params[f'xi_{index}'].user_data = {'hint': self.xi_des}

            tes_params.add(f'fcorr_{index}', value=self.fcorr, min=0, vary=False)
            tes_params[f'fcorr_{index}'].user_data = {'hint': self.fcorr_des}

            tes_params.add(f'squid_noise_{index}', value=self.squid_noise, min=0, vary=False)
            tes_params[f'squid_noise_{index}'].user_data = {'hint': self.squid_noise_des}

            tes_params.add(f'background_noise_{index}', value=self.background_noise, min=0, vary=False)
            tes_params[f'background_noise_{index}'].user_data = {'hint': self.background_noise_des}

            tes_params.add(f'abs_background_noise_{index}', value=self.abs_background_noise, min=0, vary=False)
            tes_params[f'abs_background_noise_{index}'].user_data = {'hint': self.abs_background_noise_des}

        return tes_params

    def makeDefaultSimpleParamsDict(self, num_sets):

        tes_params = Parameters()

        for index in range(num_sets):
            tes_params.add(f't_{index}', value=self.t, min=0, vary=False)
            tes_params[f't_{index}'].user_data = {'hint': self.t_des}

            tes_params.add(f'f_{index}', value=self.f, min=0, vary=False)
            tes_params[f'f_{index}'].user_data = {'hint': self.f_des}

            tes_params.add(f't0_{index}', value=self.t0, min=0, vary=False)
            tes_params[f't0_{index}'].user_data = {'hint': self.t0_des}

            tes_params.add(f'L_{index}', value=self.L, min=0, vary=False)
            tes_params[f'L_{index}'].user_data = {'hint': self.L_des,'uscale': self.L_uscale}

            tes_params.add(f'C_tes_{index}', value=self.C_tes, min=0, vary=False)
            tes_params[f'C_tes_{index}'].user_data = {'hint': self.C_tes_des,'uscale': self.C_tes_uscale}

            tes_params.add(f'G_tes_bath_{index}', value=self.G_tes_bath, min=0, vary=False)
            tes_params[f'G_tes_bath_{index}'].user_data = {'hint': self.G_tes_bath_des,'uscale': self.G_tes_bath_uscale}

            tes_params.add(f'R_0_{index}', value=self.R_0, min=0, vary=False)
            tes_params[f'R_0_{index}'].user_data = {'hint': self.R_0_des}

            tes_params.add(f'R_L_{index}', value=self.R_L, min=0, vary=False)
            tes_params[f'R_L_{index}'].user_data = {'hint': self.R_L_des}

            tes_params.add(f'alpha_I_{index}', value=self.alpha_I, min=0, vary=False)
            tes_params[f'alpha_I_{index}'].user_data = {'hint': self.alpha_I_des}

            tes_params.add(f'beta_I_{index}', value=self.beta_I, min=0, vary=False)
            tes_params[f'beta_I_{index}'].user_data = {'hint': self.beta_I_des}

            tes_params.add(f'T_tes_{index}', value=self.T_tes, min=0, vary=False)
            tes_params[f'T_tes_{index}'].user_data = {'hint': self.T_tes_des}

            tes_params.add(f'T_bath_{index}', value=self.T_bath, min=0, vary=False)
            tes_params[f'T_bath_{index}'].user_data = {'hint': self.T_bath_des}

            tes_params.add(f'initE_{index}', value=self.initE, min=0, vary=False)
            tes_params[f'initE_{index}'].user_data = {'hint': self.initE_des}

            tes_params.add(f'initE_use_{index}', value=self.initE_use, min=0, vary=False)
            tes_params[f'initE_use_{index}'].user_data = {'hint': self.initE_use_des}

            tes_params.add(f'deltaT_{index}', value=self.deltaT, min=0, vary=False)
            tes_params[f'deltaT_{index}'].user_data = {'hint': self.deltaT_des}

            tes_params.add(f'n_mem_{index}', value=self.n_mem, min=0, vary=False)
            tes_params[f'n_mem_{index}'].user_data = {'hint': self.n_mem_des}

            tes_params.add(f'M_{index}', value=self.M, min=0, vary=False)
            tes_params[f'M_{index}'].user_data = {'hint': self.M_des}

            tes_params.add(f'xi_{index}', value=self.xi, min=0, vary=False)
            tes_params[f'xi_{index}'].user_data = {'hint': self.xi_des}

            tes_params.add(f'fcorr_{index}', value=self.fcorr, min=0, vary=False)
            tes_params[f'fcorr_{index}'].user_data = {'hint': self.fcorr_des}

            tes_params.add(f'squid_noise_{index}', value=self.squid_noise, min=0, vary=False)
            tes_params[f'squid_noise_{index}'].user_data = {'hint': self.squid_noise_des}

            tes_params.add(f'background_noise_{index}', value=self.background_noise, min=0, vary=False)
            tes_params[f'background_noise_{index}'].user_data = {'hint': self.background_noise_des}

        return tes_params

    def makeDefaultInterParamsDict(self, num_sets):

        tes_params = Parameters()

        for index in range(num_sets):
            tes_params.add(f't_{index}', value=self.t, min=0, vary=False)
            tes_params[f't_{index}'].user_data = {'hint': self.t_des}

            tes_params.add(f'f_{index}', value=self.f, min=0, vary=False)
            tes_params[f'f_{index}'].user_data = {'hint': self.f_des}

            tes_params.add(f't0_{index}', value=self.t0, min=0, vary=False)
            tes_params[f't0_{index}'].user_data = {'hint': self.t0_des}

            tes_params.add(f'L_{index}', value=self.L, min=0, vary=False)
            tes_params[f'L_{index}'].user_data = {'hint': self.L_des,'uscale': self.L_uscale}

            tes_params.add(f'C_tes_{index}', value=self.C_tes, min=0, vary=False)
            tes_params[f'C_tes_{index}'].user_data = {'hint': self.C_tes_des,'uscale': self.C_tes_uscale}

            tes_params.add(f'C_abs_{index}', value=self.C_abs, min=0, vary=False)
            tes_params[f'C_abs_{index}'].user_data = {'hint': self.C_abs_des,'uscale': self.C_abs_uscale}

            tes_params.add(f'C_int_{index}', value=self.C_int, min=0, vary=False)
            tes_params[f'C_int_{index}'].user_data = {'hint': self.C_int_des,'uscale': self.C_int_uscale}

            tes_params.add(f'G_tes_bath_{index}', value=self.G_tes_bath, min=0, vary=False)
            tes_params[f'G_tes_bath_{index}'].user_data = {'hint': self.G_tes_bath_des,'uscale': self.G_tes_bath_uscale}

            tes_params.add(f'G_abs_tes_{index}', value=self.G_abs_tes, min=0, vary=False)
            tes_params[f'G_abs_tes_{index}'].user_data = {'hint': self.G_abs_tes_des,'uscale': self.G_abs_tes_uscale}

            tes_params.add(f'G_tes_int_{index}', value=self.G_tes_int, min=0, vary=False)
            tes_params[f'G_tes_int_{index}'].user_data = {'hint': self.G_tes_int_des,'uscale': self.G_tes_int_uscale}

            tes_params.add(f'R_0_{index}', value=self.R_0, min=0, vary=False)
            tes_params[f'R_0_{index}'].user_data = {'hint': self.R_0_des}

            tes_params.add(f'R_L_{index}', value=self.R_L, min=0, vary=False)
            tes_params[f'R_L_{index}'].user_data = {'hint': self.R_L_des}

            tes_params.add(f'alpha_I_{index}', value=self.alpha_I, min=0, vary=False)
            tes_params[f'alpha_I_{index}'].user_data = {'hint': self.alpha_I_des}

            tes_params.add(f'beta_I_{index}', value=self.beta_I, min=0, vary=False)
            tes_params[f'beta_I_{index}'].user_data = {'hint': self.beta_I_des}

            tes_params.add(f'T_tes_{index}', value=self.T_tes, min=0, vary=False)
            tes_params[f'T_tes_{index}'].user_data = {'hint': self.T_tes_des}

            tes_params.add(f'T_int_{index}', value=self.T_int, min=0, vary=False)
            tes_params[f'T_int_{index}'].user_data = {'hint': self.T_int_des}

            tes_params.add(f'T_bath_{index}', value=self.T_bath, min=0, vary=False)
            tes_params[f'T_bath_{index}'].user_data = {'hint': self.T_bath_des}

            tes_params.add(f'initE_{index}', value=False, min=0, vary=False)
            tes_params[f'initE_{index}'].user_data = {'hint': self.initE_des}

            tes_params.add(f'initE_use_{index}', value=self.initE_use, min=0, vary=False)
            tes_params[f'initE_use_{index}'].user_data = {'hint': self.initE_use_des}

            tes_params.add(f'deltaT_{index}', value=self.deltaT, min=0, vary=False)
            tes_params[f'deltaT_{index}'].user_data = {'hint': self.deltaT_des}

            tes_params.add(f'n_mem_{index}', value=self.n_mem, min=0, vary=False)
            tes_params[f'n_mem_{index}'].user_data = {'hint': self.n_mem_des}

            tes_params.add(f'n_int1_{index}', value=self.n_int1, min=0, vary=False)
            tes_params[f'n_int1_{index}'].user_data = {'hint': self.n_int1_des}

            tes_params.add(f'n_int2_{index}', value=self.n_int2, min=0, vary=False)
            tes_params[f'n_int2_{index}'].user_data = {'hint': self.n_int2_des}

            tes_params.add(f'k_int_ratio_{index}', value=self.k_int_ratio, min=0, vary=False)
            tes_params[f'k_int_ratio_{index}'].user_data = {'hint': self.k_int_ratio_des}

            tes_params.add(f'M_{index}', value=self.M, min=0, vary=False)
            tes_params[f'M_{index}'].user_data = {'hint': self.M_des}

            tes_params.add(f'xi_{index}', value=self.xi, min=0, vary=False)
            tes_params[f'xi_{index}'].user_data = {'hint': self.xi_des}

            tes_params.add(f'fcorr_{index}', value=self.fcorr, min=0, vary=False)
            tes_params[f'fcorr_{index}'].user_data = {'hint': self.fcorr_des}

            tes_params.add(f'squid_noise_{index}', value=self.squid_noise, min=0, vary=False)
            tes_params[f'squid_noise_{index}'].user_data = {'hint': self.squid_noise_des}

            tes_params.add(f'background_noise_{index}', value=self.background_noise, min=0, vary=False)
            tes_params[f'background_noise_{index}'].user_data = {'hint': self.background_noise_des}

            tes_params.add(f'abs_background_noise_{index}', value=self.abs_background_noise, min=0, vary=False)
            tes_params[f'abs_background_noise_{index}'].user_data = {'hint': self.abs_background_noise_des}

            tes_params.add(f'int_background_power_{index}', value=self.int_background_power, min=0, vary=False)
            tes_params[f'int_background_power_{index}'].user_data = {'hint': self.int_background_power_des}

        return tes_params

    def makeDefaultInterDangleParamsDict(self, num_sets):

        tes_params = Parameters()

        for index in range(num_sets):
            tes_params.add(f't_{index}', value=self.t, min=0, vary=False)
            tes_params[f't_{index}'].user_data = {'hint': self.t_des}

            tes_params.add(f'f_{index}', value=self.f, min=0, vary=False)
            tes_params[f'f_{index}'].user_data = {'hint': self.f_des}

            tes_params.add(f't0_{index}', value=self.t0, min=0, vary=False)
            tes_params[f't0_{index}'].user_data = {'hint': self.t0_des}

            tes_params.add(f'L_{index}', value=self.L, min=0, vary=False)
            tes_params[f'L_{index}'].user_data = {'hint': self.L_des,'uscale': self.L_uscale}

            tes_params.add(f'C_tes_{index}', value=self.C_tes, min=0, vary=False)
            tes_params[f'C_tes_{index}'].user_data = {'hint': self.C_tes_des,'uscale': self.C_tes_uscale}

            tes_params.add(f'C_int_{index}', value=self.C_int, min=0, vary=False)
            tes_params[f'C_int_{index}'].user_data = {'hint': self.C_int_des,'uscale': self.C_int_uscale}

            tes_params.add(f'C_dang_{index}', value=self.C_dang, min=0, vary=False)
            tes_params[f'C_dang_{index}'].user_data = {'hint': self.C_dang_des,'uscale': self.C_dang_uscale}

            tes_params.add(f'G_tes_bath_{index}', value=self.G_tes_bath, min=0, vary=False)
            tes_params[f'G_tes_bath_{index}'].user_data = {'hint': self.G_tes_bath_des,'uscale': self.G_tes_bath_uscale}

            tes_params.add(f'G_tes_int_{index}', value=self.G_tes_int, min=0, vary=False)
            tes_params[f'G_tes_int_{index}'].user_data = {'hint': self.G_tes_int_des,'uscale': self.G_tes_int_uscale}

            tes_params.add(f'G_int_dang_{index}', value=self.G_int_dang, min=0, vary=False)
            tes_params[f'G_int_dang_{index}'].user_data = {'hint': self.G_int_dang_des,'uscale': self.G_int_dang_uscale}

            tes_params.add(f'R_0_{index}', value=self.R_0, min=0, vary=False)
            tes_params[f'R_0_{index}'].user_data = {'hint': self.R_0_des}

            tes_params.add(f'R_0g_{index}', value=self.R_0, min=0, vary=False)
            tes_params[f'R_0g_{index}'].user_data = {'hint': self.R_0_des}

            tes_params.add(f'R_L_{index}', value=self.R_L, min=0, vary=False)
            tes_params[f'R_L_{index}'].user_data = {'hint': self.R_L_des}

            tes_params.add(f'alpha_I_{index}', value=self.alpha_I, min=0, vary=False)
            tes_params[f'alpha_I_{index}'].user_data = {'hint': self.alpha_I_des}

            tes_params.add(f'alpha_Ig_{index}', value=self.alpha_I, min=0, vary=False)
            tes_params[f'alpha_Ig_{index}'].user_data = {'hint': self.alpha_I_des}

            tes_params.add(f'beta_I_{index}', value=self.beta_I, min=0, vary=False)
            tes_params[f'beta_I_{index}'].user_data = {'hint': self.beta_I_des}

            tes_params.add(f'beta_Ig_{index}', value=self.beta_I, min=0, vary=False)
            tes_params[f'beta_Ig_{index}'].user_data = {'hint': self.beta_I_des}

            tes_params.add(f'T_tes_{index}', value=self.T_tes, min=0, vary=False)
            tes_params[f'T_tes_{index}'].user_data = {'hint': self.T_tes_des}

            tes_params.add(f'T_tesg_{index}', value=self.T_tes, min=0, vary=False)
            tes_params[f'T_tesg_{index}'].user_data = {'hint': self.T_tes_des}

            tes_params.add(f'T_int_{index}', value=self.T_int, min=0, vary=False)
            tes_params[f'T_int_{index}'].user_data = {'hint': self.T_int_des}

            tes_params.add(f'T_bath_{index}', value=self.T_bath, min=0, vary=False)
            tes_params[f'T_bath_{index}'].user_data = {'hint': self.T_bath_des}

            tes_params.add(f'initE_{index}', value=False, min=0, vary=False)
            tes_params[f'initE_{index}'].user_data = {'hint': self.initE_des}

            tes_params.add(f'initE_use_{index}', value=self.initE_use, min=0, vary=False)
            tes_params[f'initE_use_{index}'].user_data = {'hint': self.initE_use_des}

            tes_params.add(f'deltaT_{index}', value=self.deltaT, min=0, vary=False)
            tes_params[f'deltaT_{index}'].user_data = {'hint': self.deltaT_des}

            tes_params.add(f'n_mem_{index}', value=self.n_mem, min=0, vary=False)
            tes_params[f'n_mem_{index}'].user_data = {'hint': self.n_mem_des}

            tes_params.add(f'n_int1_{index}', value=self.n_int1, min=0, vary=False)
            tes_params[f'n_int1_{index}'].user_data = {'hint': self.n_int1_des}

            tes_params.add(f'n_int2_{index}', value=self.n_int2, min=0, vary=False)
            tes_params[f'n_int2_{index}'].user_data = {'hint': self.n_int2_des}

            tes_params.add(f'n_dang_{index}', value=self.n_dang, min=0, vary=False)
            tes_params[f'n_dang_{index}'].user_data = {'hint': self.n_dang_des}

            tes_params.add(f'k_int_ratio_{index}', value=self.k_int_ratio, min=0, vary=False)
            tes_params[f'k_int_ratio_{index}'].user_data = {'hint': self.k_int_ratio_des}

            tes_params.add(f'M_{index}', value=self.M, min=0, vary=False)
            tes_params[f'M_{index}'].user_data = {'hint': self.M_des}

            tes_params.add(f'xi_{index}', value=self.xi, min=0, vary=False)
            tes_params[f'xi_{index}'].user_data = {'hint': self.xi_des}

            tes_params.add(f'fcorr_{index}', value=self.fcorr, min=0, vary=False)
            tes_params[f'fcorr_{index}'].user_data = {'hint': self.fcorr_des}

            tes_params.add(f'squid_noise_{index}', value=self.squid_noise, min=0, vary=False)
            tes_params[f'squid_noise_{index}'].user_data = {'hint': self.squid_noise_des}

            tes_params.add(f'background_noise_{index}', value=self.background_noise, min=0, vary=False)
            tes_params[f'background_noise_{index}'].user_data = {'hint': self.background_noise_des}

            tes_params.add(f'abs_background_noise_{index}', value=self.abs_background_noise, min=0, vary=False)
            tes_params[f'abs_background_noise_{index}'].user_data = {'hint': self.abs_background_noise_des}

            tes_params.add(f'int_background_power_{index}', value=self.int_background_power, min=0, vary=False)
            tes_params[f'int_background_power_{index}'].user_data = {'hint': self.int_background_power_des}

        return tes_params


    def save_tes_params(self, tes_params, file_name):
        """"Save the TES params to file. This just wraps the method to do this built 
        into lmfit."""

        with open(file_name,'w',encoding = 'utf-8') as f:
            tes_params.dump(f)

    def load_tes_params(self, file_name):
        """Load the TES params from a file. This just wraps the method to do this built  
        into lmfit."""

        tes_params = Parameters()

        #with open(file_name,'r',encoding = 'utf-8') as f:
        with open(file_name,'r') as f:
            tes_params.load(f)

        return tes_params

    def get_param_hint(self, tes_params, param_string, set_index=0):
        """Reutrn the paramter hint for a given parameter."""

        hint = tes_params[f'{param_string}_{set_index}'].user_data['hint']

        return hint

    def get_param_unit_scale(self, tes_params, param_string, set_index=0):
        """Reutrn the paramter unit scale for a given parameter."""

        unit_scale = tes_params[f'{param_string}_{set_index}'].user_data['uscale']

        return unit_scale


    def test(self):
        """Testing docstring linking. """

        pass