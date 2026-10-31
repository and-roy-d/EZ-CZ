"""
Two-Fluid TES class of MDT Microcal Design Tools

This class is written to integrate the two-fluid TES calculations into the MDT platform.

Notes:
    Last Updated: Apr 10, 2020
    Current Version Created on Mar 23, 2020
    Original Version Created based on mdt_tf.py Apr 26, 2013
    @author: Douglas Bennett

"""

import numpy as np
import cmath
import math
import scipy.optimize

kB = 1.3806503e-23
ec = 1.60217646e-19

class TES_TwoFluid(object):
    """Class for methods of the Two-Fluid Transition model.

    This class organizes all the methods to calculated IVs in the the 
    two-fluid model. Will eventually be incorperated into MDT3.

    Note:
        A numner of paramters are still beinng tweaked

    Attributes:
        n_mem (float): The exponent for the temeprature dependen cance of
            of the memenbrane `mdt3.mdt3_core.MDT3_Core.test`.
        tes (`mdt3.mdt3_core.MDT3_Core.__init__`): The funnctionlity from the MDT3 class
        T_bath (float): The bath temperature of the TES.
        k_thermal (float): The thermal condcutivity constant for heat flow though membrane.

    """

    def __init__(self, tes):
                
        self.tes = tes

        #thermal parameters
        self.T_bath = None
        self.k_thermal = None #will need to be derived in standard DMT form
        self.n_mem = None
        self.P_background = None

        #derived parmeters for stand input parameters
        self.k = None
        self.I0 = None
        self.PJ0 = None
        self.L_I = None

        #state variables
        self.Ites = None
        self.Ttes = None

        #sc paramters
        self.ic_model = 'GL'
        self.cr_model = 'fixed'
        self.parallel_Rn = True
        self.Tc = None
        self.Ic0 = None
        self.Rn = None
        self.cr = None
        self.ci = None

        self.Tc_L = None
        self.Tc_N = None
        self.length = None
        self.xi_N0 = None
        self.xi_N_max = None
        self.Tc_wl = None
        self.Ic_wl = None
        self.L_xi0 = None
        self.T_gl_wl_trans = None
        self.I_sw = None

        #circuit paramters
        self.R_sh = None
        self.R_par = None
        self.R_bank = None

        self.ver = '1.0.1'

    def normal_cohernece_length(self, T_tes):
        """Calculates the normal coherence length at a given TES temperature. Ic calc

        Note:
            The normal coherence lengths for the Ic calc and the proximety calculation
            have not been reconconciled and are currently different paramters.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The coherence length of the material in the normal state.

        """

        #L_xi = self.L_xi0/(np.absolute(T_tes/self.Tc_wl-1.0))**.5
        L_xi = self.L_xi0*(self.Tc_L/T_tes)**.5*(self.Tc_wl/(T_tes-self.Tc_wl))**.5

        return L_xi


    def current_critical(self, T_tes):
        """Calculates critical current based on model choice. .

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The supercondcuting critical current

        """

        if self.ic_model == 'GL': 
            if T_tes >= self.Tc:
                Ic = 0.0
            else:
                Ic = self.Ic0*(1.0-T_tes/self.Tc)**(1.5)
        elif self.ic_model == 'WL':
            if T_tes <= self.Tc_wl:
                Ic = 0.0 #need an Ic max value here
            else:
                L_xi = self.normal_cohernece_length(T_tes)
                Ic = self.Ic0_wl/L_xi*np.exp(-1.0*self.length/L_xi)
        elif self.ic_model == 'GL+WL':
            #Combined Ginzburg-Landau and weak-link model
            if self.T_gl_wl_trans is None:
                print('No temperature for the GL to WL trnaisistion is defined.')
                raise Exception('No T_gl_wl_trans defined.')
            if T_tes < self.T_gl_wl_trans:
                Ic = self.Ic0*(1.0-T_tes/self.Tc)**(1.5)
            else:
                L_xi = self.normal_cohernece_length(T_tes)
                Ic = self.Ic0_wl/L_xi*np.exp(-1.0*self.length/L_xi)
        else:
            print(self.ic_model)
            raise Exception('Invalid Ic model choice')

        return Ic

    def calc_cr(self, T_tes):
        """Calculates the constant current two-fluid model parameter known as c_r.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The constant current two-fluid model parameter

        """

        if T_tes < self.Tc:
            cr_T = self.cr*(1.0-T_tes/self.Tc)**(-.25)
        else:
            cr_T = self.cr*10000
        return cr_T

    def calc_xi_N(self, T_tes):
        """Calculates the normal coherence length at a given TES temperature. prox calc

        Note:
            The normal coherence lengths for the Ic calc and the proximety calculation
            have not been reconconciled and are currently different paramters.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The coherence length of the material in the normal state.

        """

        #epsilon_N = (self.xi_N0/(T_tes-self.Tc_N))**.5
        if T_tes < self.Tc_N:
            xi_N = self.xi_N_max
        else:
            #xi_N = self.xi_N0*(1.0/(T_tes-self.Tc_N))**.5
            xi_N = self.xi_N0*(self.Tc_N/(T_tes-self.Tc_N))**.5
            if  xi_N > self.xi_N_max:
                xi_N = self.xi_N_max

        return xi_N

    def calc_cp(self, T_tes):
        """Calculates the new two-fluid model paramter for the proximety effect known as c_p.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The two-fluid model parameter for the proximety effect

        """

        # if T_tes <= self.Tc_N:
        #     cp = 1.0
        # else:
        #     xi_N = self.calc_xi_N(T_tes)
        #     cp = 2.0*xi_N/self.length
        #     if cp > 1.0:
        #         cp = 1.0

        xi_N = self.calc_xi_N(T_tes)
        cp = 2.0*xi_N/self.length
        #cp = .14/(T_tes)**.5
        if cp > 1.0:
            cp = 1.0        

        return cp

    def voltage_tes(self, T_tes, I_tes):
        """The voltage across the the TES in the two-fluid model for a given temperature and current.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature
            I_tes: The current through the TES

        Returns:
            The votlage across the TES.

        """

        Ic = self.current_critical(T_tes)
        
        if self.cr_model == 'fixed':
            cr_T = self.cr
        elif self.cr_model == 'weakT':
            cr_T = self.calc_cr(T_tes)
        else:
            raise Exception('Invalid cr model')
        
        cp = self.calc_cp(T_tes)

        if (I_tes < self.ci*Ic) and (T_tes < self.Tc):
            V_tes = 1.0e-15
        else:
            #V_tes = cr_T*self.Rn*(I_tes-self.ci*Ic)-cp*self.Rn*I_tes
            V_tes = cr_T*self.Rn*(I_tes-self.ci*Ic)*(1-cp)

        return V_tes

    def dynamic_reistance_tes(self, T_tes, I_tes):
        """The voltage across the the TES in the two-fluid model for a given temperature and current.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature
            I_tes: The current through the TES

        Returns:
            The votlage across the TES.

        """

        Ic = self.current_critical(T_tes)
        
        if self.cr_model == 'fixed':
            cr_T = self.cr
        elif self.cr_model == 'weakT':
            cr_T = self.calc_cr(T_tes)
        else:
            raise Exception('Invalid cr model')
        
        cp = self.calc_cp(T_tes)

        if (I_tes < self.ci*Ic) and (T_tes < self.Tc):
            Rd_tes = 1.0e-15
        else:
            #V_tes = cr_T*self.Rn*(I_tes-self.ci*Ic)-cp*self.Rn*I_tes
            Rd_tes = cr_T*self.Rn*(1-cp)

        return Rd_tes

    def vdiff_parallel_rbank(self, I_tes_fraction, I_branch_total, T_tes):
        """Calculates the difference of the voltage across bank and the voltage across the TES branch. 
        
        Should be zero when the correct current fraction is input. This method is written to be used by 
        a solver routine. The banks are assumed to be norml metal that add a parralell currnet path.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_tes_fraction: The fraction of the total current that goes though the TES
            I_branch_total: The total current though the branch of the cirucit, as oppes to the shunt
            T_tes: The TES temperature

        Returns:
            The votlage difference between the TES and the norma l metal bank

        """
        
        I_tes = I_tes_fraction*I_branch_total
        I_rbank = I_branch_total-I_tes
        
        V_tes = self.voltage_tes(T_tes,I_tes)
        Vdiff = self.R_bank * I_rbank - V_tes
        
        return Vdiff

    def current_split_parallel_rbank(self, I_branch_total, T_tes):
        """Solves for voltgae balance across the two branches and returns the current though the 
        TES branch and shunt branch and the voltage drop across both. 
        
         The banks are assumed to be norml metal that add a parralell currnet path.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_branch_total: The total current though the branch of the cirucit, as oppes to the shunt
            T_tes: The TES temperature

        Returns:
            A tuple with the following elements
            I_tes: The current through the TES
            I_rbank: The current through the banks
            V_tes: The voltage across the TES, which should be the same as the voltage across the bank

        """

        I_tes_frac_init_guess = 0.9
        I_fraction = scipy.optimize.fsolve(self.vdiff_parallel_rbank,I_tes_frac_init_guess, args=(I_branch_total, T_tes))
        I_tes = I_fraction[0] * I_branch_total
        I_rbank = I_branch_total - I_tes
        
        V_tes = self.voltage_tes(T_tes, I_tes)
        
        return I_tes, I_rbank, V_tes

    def vdiff_parallel_rsh(self, I_fraction, I_total, T_tes):
        """Calculates the difference of the voltage across the shunt and the voltage across the TES branch. 
        
        Should be zero when the correct current fraction is input. This method is written to be used by 
        a solver routine. The shunt (Rsh) is the resistance in parralel with the TES brnach.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_fraction: The fraction of the total current that goes though the TES branch
            I_total: The total bias current before the split to TES and shunt
            T_tes: The TES temperature

        Returns:
            The votlage difference between the TES and the shunt resistance

        """
        
        I_tes = I_fraction*I_total
        I_rsh = I_total-I_tes
        
        V_tes = self.voltage_tes(T_tes,I_tes)
        Vdiff = self.R_sh*I_rsh - V_tes - self.R_par*I_tes
        
        return Vdiff

    def vdiff_parallel_rsh_wrbank(self, I_fraction, I_total, T_tes):
        """Calculates the difference of the voltage across the shunt and the voltage across the TES/bank branch. 
        
        Should be zero when the correct current fraction is input. This method is written to be used by 
        a solver routine. The shunt (Rsh) is the resistance in parralel with the TES/bank brnach.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_fraction: The fraction of the total current that goes though the TES branch
            I_total: The total bias current before the split to TES and shunt
            T_tes: The TES temperature

        Returns:
            The votlage difference between the TES and the shunt resistance
            
        """
        
        I_tes_branch = I_fraction*I_total
        I_rsh = I_total-I_tes_branch
        
        I_tes, I_rbank, V_tes = self.current_split_parallel_rbank(I_tes_branch, T_tes)

        V_tes = self.voltage_tes(T_tes,I_tes)
        Vdiff = self.R_sh*I_rsh - V_tes - self.R_par*I_tes_branch
        
        return Vdiff

    def current_split_parallel_rsh(self, I_total, T_tes):
        """Solves for voltgae balance across the two branches and returns the current though the 
        TES branch and shunt branch and the voltage drop across both. 
        
        The shunt (Rsh) is the resistance in parralel with the TES brnach.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt
            T_tes: The TES temperature

        Returns:
            A tuple with the following elements
            I_tes: The current through the TES
            I_rsh: The current through the shunt
            V_tes: The voltage across the TES, which should be the same as the voltage across the shunt

        """

        I_frac_init_guess = 0.50
        I_fraction = scipy.optimize.fsolve(self.vdiff_parallel_rsh,I_frac_init_guess, args=(I_total, T_tes))
        I_tes = I_fraction[0] * I_total
        I_rsh = I_total-I_tes
        
        V_tes = self.voltage_tes(T_tes, I_tes)
        
        return I_tes, I_rsh, V_tes

    def current_split_parallel_rsh_wrbank(self, I_total, T_tes):
        """Solves for voltgae balance across the two branches and returns the current though the 
        TES/bank branch and shunt branch and the voltage drop across both. 
        
        The shunt (Rsh) is the resistance in parralel with the TES/bank brnach.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt
            T_tes: The TES temperature

        Returns:
            A tuple with the following elements
            I_tes: The current through the TES/bannk
            I_rsh: The current through the shunt
            V_tes: The voltage across the TES/bankn, which should be the same as the voltage across the shunt

        """

        I_frac_init_guess = 0.98
        I_fraction = scipy.optimize.fsolve(self.vdiff_parallel_rsh_wrbank,I_frac_init_guess, args=(I_total, T_tes))
        I_tes_branch = I_fraction[0] * I_total
        I_rsh = I_total-I_tes_branch
        
        I_tes, I_rbank, V_tes = self.current_split_parallel_rbank(I_tes_branch, T_tes)
        
        return I_tes, I_rbank, I_rsh, V_tes

    def joulePower_parallel(self, I_total, T_tes):
        """Calculates the joule power across TES from the Joule heating in the parallel circuit configuration.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature
            I_total: The total bias current before the split to TES and shunt

        Returns:
            The joule power.

        """
        
        I_tes, I_rsh, V_tes = self.current_split_parallel_rsh(I_total, T_tes)
        P_joule = I_tes * V_tes

        return P_joule

    def joulePower_parallel_wrbank(self, I_total, T_tes):
        """Calculates the joule power across TES/bank from the Joule heating in the parallel circuit configuration.

        Sames as joulePower_parallel but takes into consideration the heating in the banks seperatly.

        Assusmes the relevant twofluid class paramters were properly set. 

        Args:
            T_tes: The TES temperature
            I_total: The total bias current before the split to TES and shunt

        Returns:
            The joule power.

        """
        
        I_tes, I_rbank, I_rsh, V_tes = self.current_split_parallel_rsh_wrbank(I_total, T_tes)
        P_joule = I_tes * V_tes + I_rbank**2 * self.R_bank

        return P_joule

    def thermalConductancePower(self,T_tes):
        """Calculates the thermal power leaving the device at a given TES and bath temperature.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature

        Returns:
            The power leaving the TES through the thermal conductance.

        """
        
        P_thermal = self.k_thermal * (T_tes**(self.n_mem) - self.T_bath**self.n_mem)

        return P_thermal

    def pdif_T_parallel(self, T_tes, I_total):
        """Calculates the power difference in the TES for the parralel configuration between the jole power created 
        and the power leaving through the thermal conductecne.

        Should be zero when the correct current fraction is input. This method is written to be used by 
        a solver routine. 

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature
            I_total: The total bias current before the split to TES and shunt

        Returns:
            The difference between the joule power created and thr thermal coundctance power leaving.

        """
        
        P_joule = self.joulePower_parallel(I_total, T_tes)
        P_therm = self.thermalConductancePower(T_tes)
        P_dif = P_joule + self.P_background - P_therm
        return P_dif

    def pdif_T_parallel_wrbank(self, T_tes, I_total):
        """Calculates the power difference in the TES/bank for the parralel configuration between the jole power created 
        and the power leaving through the thermal conductecne.

        Should be zero when the correct current fraction is input. This method is written to be used by 
        a solver routine. 

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            T_tes: The TES temperature
            I_total: The total bias current before the split to TES and shunt

        Returns:
            The difference between the joule power created and thr thermal coundctance power leaving.

        """
        
        P_joule = self.joulePower_parallel_wrbank(I_total, T_tes)
        P_therm = self.thermalConductancePower(T_tes)
        P_dif = P_joule + self.P_background - P_therm
        return P_dif

    def solve_T_TES(self, I_total):
        """Solves for the equalibrium temperature at a given total current in the two-fluid model.
        
        Assumes TES is in parralel with the shunt.  Parasitic resistance is assumed 
        to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt

        Returns:
            TES temperature

        """
 
        T_guess = self.Tc-(self.Tc-self.T_bath)/100.0 #was /100 for the field studies
        T_tes = scipy.optimize.fsolve(self.pdif_T_parallel,T_guess, args=(I_total))
        return T_tes[0]

    def solve_T_TES_wrbank(self, I_total):
        """Solves for the equalibrium temperature at a given total current in the two-fluid model.
        
        Assumes TES is in parallel with the banks and both are in pararlell with the shunt.
        Parasitic resistance is assumed to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt

        Returns:
            TES temperature

        """
 
        T_guess = self.Tc-(self.Tc-self.T_bath)/40.0 #was /100 for the field studies
        T_tes = scipy.optimize.fsolve(self.pdif_T_parallel_wrbank,T_guess, args=(I_total))
        return T_tes[0]

    def solve_iv(self, I_total):
        """Solves for the equalibrium state variables at a given total current in the two-fluid model.
        
        Assumes TES is in parralel with the shunt.  Parasitic resistance is assumed 
        to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt

        Returns:
            A tuple with the following elements
            I_tes: The current through the TES
            T_tes: The temperature of the TES
            V_tes: The voltage across the TES, which should be the same as the voltage across the shunt
            I_rsh: The current though the shunt

        """
        
        T_tes = self.solve_T_TES(I_total)
        I_tes, I_rsh, V_tes = self.current_split_parallel_rsh(I_total, T_tes)
        return I_tes, T_tes, V_tes, I_rsh

    def solve_iv_wrbank(self, I_total):
        """Solves for the equalibrium state variables at a given total current in the two-fluid model.
        
        Assumes TES is in parallel with the banks and both are in pararlell with the shunt.
        Parasitic resistance is assumed to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total: The total bias current before the split to TES and shunt

        Returns:
            A tuple with the following elements
            I_tes: The current through the TES
            T_tes: The temperature of the TES
            V_tes: The voltage across the TES, which should be the same as the voltage across the shunt
            I_rban: The current through the banks
            I_rsh: The current though the shunt

        """
        
        T_tes = self.solve_T_TES_wrbank(I_total)
        I_tes, I_rbank, I_rsh, V_tes = self.current_split_parallel_rsh_wrbank(I_total, T_tes)
        return I_tes, T_tes, V_tes, I_rbank, I_rsh

    def solve_iv_loop(self, I_total_array, calc_alpha_beta = False):
        """Solves for the equalibrium state variables for an array of currents in the two-fluid model.
        
        Step through an array of total currents and calculate the equlibrium IV paramters.

        Assumes TES is in parralel with the shunt.  Parasitic resistance is assumed 
        to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total_array: Array of total bias current before the split to TES and shunt
            calc_alpha_beta: For future use (optional)

        Returns:
            A 2-d array with the following zero axis elemetns.
            I_total_array
            T_tes_array
            V_tes_array

        """
        
        #Create the arrays to store the results
        I_tes_array = np.zeros_like(I_total_array)
        T_tes_array = np.zeros_like(I_total_array)
        V_tes_array = np.zeros_like(I_total_array)
        alpha_I_array = np.zeros_like(I_total_array)
        beta_I_array = np.zeros_like(I_total_array)

        for index, I_total in enumerate(I_total_array):
            I_tes, T_tes, V_tes, I_rsh = self.solve_iv(I_total)
            I_tes_array[index] = I_tes
            T_tes_array[index] = T_tes
            V_tes_array[index] = V_tes

        iv_reults = np.vstack((I_tes_array, T_tes_array, V_tes_array))

        return iv_reults

    def solve_iv_loop_wrbank(self, I_total_array, calc_alpha_beta = False):
        """Solves for the equalibrium state variables for an array of currents in the two-fluid model.
        
        Step through an array of total currents and calculate the equlibrium IV paramters.

        Assumes TES is in parallel with the banks and both are in pararlell with the shunt.
        Parasitic resistance is assumed to be in series with TES.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            I_total_array: Array of total bias current before the split to TES and shunt
            calc_alpha_beta: For future use (optional)

        Returns:
            A 2-d array with the following zero axis elemetns.
            I_total_array
            T_tes_array
            V_tes_array

        """
        
        #Create the arrays to store the results
        I_tes_array = np.zeros_like(I_total_array)
        T_tes_array = np.zeros_like(I_total_array)
        V_tes_array = np.zeros_like(I_total_array)
        I_rbank_array = np.zeros_like(I_total_array)
        alpha_I_array = np.zeros_like(I_total_array)
        beta_I_array = np.zeros_like(I_total_array)

        for index, I_total in enumerate(I_total_array):
            I_tes, T_tes, V_tes, I_rbank, I_rsh = self.solve_iv_wrbank(I_total)
            I_tes_array[index] = I_tes
            T_tes_array[index] = T_tes
            V_tes_array[index] = V_tes
            I_rbank_array[index] = I_rbank

        iv_reults = np.vstack((I_tes_array, T_tes_array, V_tes_array, I_rbank_array))

        return iv_reults

    #Methods to help get good hybrid Ic parameters

    def negative_critical_wl(self, T_tes):
        """Calculates the negative of the weak-link critcal current.

        Assusmes the relevant twofluid class paramters were properly set.
        This method is written to be used by a solver routine.

        Args:
            T_tes: The TES temperature

        Returns:
            The weak-link critical current times -1

        """

        self.ic_model = 'WL'
        Ic_wl_neg = -1.0*self.current_critical(T_tes)
        return Ic_wl_neg

    def diff_GLandWL(self, T_tes):
        """Calculates the difference between the G-L and weak-link critcal currents.

        Assusmes the relevant twofluid class paramters were properly set.
        This method is written to be used by a solver routine.

        Args:
            T_tes: The TES temperature

        Returns:
            Difference between G-L and weak-link critcal currents.

        """

        self.ic_model = 'GL'
        Ic_GL = self.current_critical(T_tes)
        self.ic_model = 'WL'
        Ic_wl = self.current_critical(T_tes)
        Ic_diff = Ic_GL-Ic_wl
        return Ic_diff

    def calcTtrans(self, ic_tol=1.0e-8, verbose=False):
        """Solves for the temperature to switch between the G-L and weak link critica currents.

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            ic_tol: tolerance for checking for intersection, default 1.0e-8
            verbose: comntrols verbose reporting, default False

        Returns:
            A tuple with the following elements
            The temeprature to switch between critical current models
            The difference of the models at the switch point

        """

        if self.Tc_wl > self.Tc:
            raise Exception('Invalid Tcs')
        T_Ic_wl_min = scipy.optimize.fmin(self.negative_critical_wl, self.Tc_wl+.0001,disp=0)
        if verbose is True:
            print('T where Ic_wl is max',T_Ic_wl_min[0])
        if T_Ic_wl_min[0] > self.Tc_wl:
            T_Ic_wl_lowlimit = T_Ic_wl_min[0]
        else:
            raise Exception('Problem finding valid IC_wl max.')
        T_inter_guess = self.Tc-(self.Tc-T_Ic_wl_lowlimit)/5.0
        T_inter_high = scipy.optimize.fsolve(self.diff_GLandWL,T_inter_guess)
        Ic_diff_at_inter = self.diff_GLandWL(T_inter_high)
        #print('Ic_diff_at_inter: {}'.format(Ic_diff_at_inter))
        if abs(Ic_diff_at_inter) > ic_tol:
            print('Warning: no intersection likely. Ic_diff: {}'.format(Ic_diff_at_inter))
        if verbose is True:
            print('Interesection point near Tc', T_inter_high)
        T_inter_guess = (self.Tc-T_Ic_wl_lowlimit)/1000.0+T_Ic_wl_lowlimit
        T_inter_low = scipy.optimize.fsolve(self.diff_GLandWL,T_inter_guess)
        if verbose is True:
            print('Interesection point near Tc_wl', T_inter_low)
        #Checks on these values
        if (T_inter_high < T_Ic_wl_lowlimit):
            print('Ttrans: {} is less than T_Ic_wl_lowlimit: {}'.format(T_inter_high, T_Ic_wl_lowlimit))
        if (T_inter_high > self.Tc):
            print('Warning: Ttrans: {} is greater than T_C: {}'.format(T_inter_high, self.Tc))
        if (T_inter_high > T_Ic_wl_lowlimit) and (T_inter_high < self.Tc):
            self.ic_model = 'GL'
            Ic_GL = self.current_critical(T_inter_high-.00001)
            self.ic_model = 'WL'
            Ic_wl = self.current_critical(T_inter_high-.00001)
            if Ic_GL > Ic_wl:
                self.T_gl_wl_trans = T_inter_high[0]
                if verbose is True:
                    print('Valid Tc_inter_high')
                    print('Setting model switch temperature to ', self.T_gl_wl_trans)
            else:
                self.T_gl_wl_trans = T_inter_high[0]
                if verbose is True:
                    print('Warning: Below T_trans Ic_wl; {} is greater than Ic_gl: {}'.format(Ic_wl,Ic_GL))
                    print('Ic_GL:', Ic_GL, 'Ic_wl:', Ic_wl)
        else:
            print('Warning: Ttrans outside reasonable bounds. T_inter_high is: {}'.format(T_inter_high))
            #print('Setting T_trans to Tc')
            #self.T_gl_wl_trans = self.Tc
            self.T_gl_wl_trans = T_inter_high[0]
        if (T_inter_low > T_Ic_wl_lowlimit) and (T_inter_low < self.Tc):
            if T_inter_low < (T_inter_high-0.00001):
                if verbose is True:
                    print('Valid Tc_inter_low and different than Tc_inter_high')
        self.ic_model = 'GL+WL' #to avoid confusion set ic_model to hybtid model
        return self.T_gl_wl_trans, Ic_diff_at_inter

    def diff_IcandIsw(self, Ic0_wl_guess, ic_tol=1.0e-8, verbose=False):
        """Calculates the difference between the critcal currents at the model transistion and the target.

        I_sw is the target current and Ic0_wl is adjusted until the switch over
        temperature occurs below the taget current.

        Assusmes the relevant twofluid class paramters were properly set.
        This method is written to be used by a solver routine.

        Args:
            Ic0_wl_guess: A rough starting guess for Ic0_wl

        Returns:
            Difference between the current at the switch point and the target current.

        """

        self.Ic0_wl = Ic0_wl_guess
        Ttrans, Ic_diff = self.calcTtrans(ic_tol=ic_tol, verbose=verbose)
        self.ic_model = 'GL+WL'
        Ic_at_trans = self.current_critical(Ttrans)
        Idiff = Ic_at_trans - self.I_sw
        return Idiff

    def calcIc0wl(self, Ic0_wl_guess):
        """Solves for a zero temeprature weak-link critcal current parmaters that gives
        a transition between the G-L and weak-link models that at the target current (I_sw).

        Assusmes the relevant twofluid class paramters were properly set.

        Args:
            Ic0_wl_guess: A rough starting guess for the zero temperature weak-link critcal current

        Returns:
            The zero temeprature weak link critical current that gives give a transition between
            the G-L and weak link models at the target current.

        """
        Ic0_wl = scipy.optimize.fsolve(self.diff_IcandIsw,Ic0_wl_guess)
        return Ic0_wl[0]