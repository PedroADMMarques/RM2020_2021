# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:56:57 2021

@author: pedro
"""

"inverSe Method for nOisy nOn-isoThermal slosHIng Experiments (SMOOTHIE)"

#%% Packages

import time # Check process run times
import numpy as np # Multiple reasons
from scipy.integrate import odeint # Integrate system of ODEs
from scipy.optimize import minimize # Minimize function (optimizer)
from sklearn.model_selection import train_test_split # Split train/test data

#%% Fluid property functions

# Obtain value from the coefficients in a polynomial fit
def value_from_coeffs(coeff, x_i):
    """
    Obtain function value from the coefficients in a polynomial fit at x-value
    *x_i*.
    
    **Inputs**
        - *coeff*: array of coefficients for polynomial fit
        - *x_i*: value at which the polynomial fit is computed
    **Outputs:**
        - *y_i*: result of the polynomial fit at *x_i*
    """
    # Initialize result with zero
    y_i = 0
    # Loop through the coefficients to build the result of the fit
    for i in range(0,len(coeff)):
        # Sum a_n*x^n for all the coefficients n
        y_i = coeff[i]*x_i**(len(coeff)-1-i) + y_i
    # Return value from the fitted polynomial
    return y_i

# Obtain data point from the polynomial fit of data
def value_from_fit(x_data, y_data, n_pol, x_i):
    """
    Obtain data point from the polynomial fit of data. The real data is fitted
    with a polynomial of n_pol order, and the property is evaluated at temperature
    (or other condition) *x_i*.
    
    **Inputs:**
        - *x_data*: data in the x-axis
        - *y_data*: data in the y-axis
        - *n_pol*: order of the polynomial
        - *x_i*: value at which the polynomial fit is computed
    **Outputs:**
        Result of the polynomial fit at *x_i*
    """
    # Generate coefficients from polynomial fit
    coeff = np.polyfit(x=x_data,y=y_data,deg=n_pol)
    # Return the value from the fitted polynomial
    return value_from_coeffs(coeff, x_i)

# Obtain fluid property value (from coefficients or constant value)
def property_value(y,x_i):
    """
    Obtain fluid property value (from coefficients or constant value) for a
    given temperature (or other condition) *x_i*.
    
    **Inputs:**
        - *y*: Can be either a scalar or an array. If it is a scalar, then y corres-
               ponds to the actual property of the fluid. If it is an array, the it
               corresponds to the array of coefficients of the polynomial fit as a
               function of *x_i*
        - *x_i*: data in the x-axis for the polynomial fit (temperature [K])
    **Outputs:**
        Fluid property *y* at temperature *x_i*
    """
    # If the input is a scalar, simply return it. This is used because there is
    # no temperature information for some HFE7200 properties
    if np.isscalar(y) == True: return y
    # If the input is an array, calculate the property based on the poly coeffs
    else: return value_from_coeffs(y, x_i)

# Clausius-Clapeyron equation
def clausius_clapeyron_p(R_v,dh,p_sat_ref,T_sat_ref,T_sat):
    """
    Clausius-Clapeyron equation that relates saturation temperature to saturation
    pressure in a vapor phase given reference conditions *T_sat_ref* & *p_sat_ref*.
    
    **Inputs:**
        - *R_v*: ideal gas constant for the vapor phase [J/kgK]
        - *dh*: latent heat of vaporization/evaporation [J/kg]
        - *p_sat_ref*: saturation pressure at reference conditions [Pa]
        - *T_sat_ref*: saturation temperature at reference conditions [K]
        - *T_sat*: saturation temperature [K]
    **Outputs:**
        Saturation pressure for saturation temperature T_sat [Pa]
    """
    return p_sat_ref*np.exp( (dh/R_v) * (1/T_sat_ref - 1/T_sat) )

# Compute mass and volume of the vapor/liquid from the temp. & pressure data
def get_mass_and_volumes(t,Tl,Tg,pg,inert,fluid,slosh,ma_0,ml_0):
    """
    Compute mass and volume of the vapor/liquid from the input temperature and 
    pressure data. This is possible if initial inert gas and liquid mass are
    known. In addition, the ideal gas law is used as an assumption in the
    intermediate calculations.
    
    **Inputs:**
        - *t*: time array information [s]
        - *Tl*: liquid temperature [K]
        - *Tg*: ullage temperature [K]
        - *pg*: ullage pressure [Pa]
        - *R_a*: inert gas ideal gas constant [J/kgK]
        - *R_v*: vapor ideal gas constant [J/kgK]
        - *fluid*: object that contains all liquid and vapor properties
        - *inert*: object that contains all inert gas properties
        - *slosh*: object that contains all sloshing properties 
        - *ma_0*: inert gas mass [kg]
        - *ml_0*: liquid mass [kg]
    **Outputs:**
        - *mv*: vapor mass over time [kg]
        - *ml*: liquid mass over time [kg]
        - *Vl*: liquid volume over time [m3]
        - *Vg*: vapor volume over time [m3]
    """
    # Liquid density [kg/m3]
    rho_l = np.zeros(len(t))
    # Ideal gas costants for the vapor and inert gas [J/kg*K]
    R_a = inert.R_a
    R_v = fluid.R_v
    # Compute liquid density for each time-step based on temperature [kg/m3]
    for i in range(len(t)): rho_l[i] = fluid.get_liq_density(Tl[i])
    # Vapor pressure - ideal gas mixture [Pa]
    pv = pg - ma_0*R_a*Tg/slosh.V_g
    # Vapor mass - ideal gas law [kg]
    mv = pv*slosh.V_g/(R_v*Tg)
    # Liquid mass [kg]
    ml = np.zeros(len(t)); ml[0] = ml_0
    # Compute liquid mass for each time-step based on evaporation/condensation
    for i in range(len(t)-1): ml[i+1] = ml[i] - (mv[i+1] - mv[i]) 
    # Liquid volume [m3]
    Vl = ml/rho_l
    # Ullage volume [m3]
    Vg = slosh.V - Vl
    # Return ullage/liquid mass & volume
    return mv, ml, Vl, Vg

#%% Classes
    
# Fluid (composed by liquid + vapor)
class Fluid:
    """
    Initialize fluid properties for the vapor and liquid phases. This class is
    initialized by inserting the name of the working fluid (i.e. H2, N2 or 
    HFE7200) and importing its properties from from a separate Python script.
    The H2 and N2 properties were obtained from the NIST (National Institute of
    Standards and Technology). The HFE7200 properties are more tricky to find,
    so they were derived from three distinct sources:
        - 3M Novec 7200 Engineered Fluid Product Information
        - Pramod Warrier and Amyn S. Teja [2011]
        - Rausch et. al [2015]
    **Constant fluid properties:**
        - *M_v*: Molar mass of the vapor species [kg/ml]
        - *R_v*: Ideal gas constant of the vapor [J/kgK]
        - *gamma*: Ratio of specific heats [-]
        - *T_sat_ref*: Reference saturation temperature [K]
        - *p_sat_ref*: Reference saturation pressure [Pa]
    **Temperature-dependent properties:**
        - *rho_l*: Polynomial coefficients for liquid density
        - *k_l*: Polynomial coefficients for liquid thermal conductivity
        - *cv_l*: Polynomial coefficients for liquid specific heat at constant volume
        - *mu_l*: Polynomial coefficients for liquid dynamic viscosity
        - *sigma*: Polynomial coefficients for the surface tension
        - *k_v*: Polynomial coefficients for the vapor thermal conductivity
        - *cv_v*: Polynomial coefficients for the vapor specific heat at constant volume
        - *mu_v*: Polynomial coefficients for the vapor dynamic viscosity
        - *dh*: Polynomial coefficients for the  latent heat for vaporization/condensation
    The properties were obtained in saturation conditions for different temperatures.
    Then, they were fitted with a high order polynomial and the coefficients are
    stored in each respective variable.
    """
    def __init__(self,name):
        print('Initializing fluid properties')
        
        # Initialization of the fluid properties at reference conditions
        self.name = name
        
        # Import fluid properties from database
        if name == 'H2':   import Properties.h2_properties as data
        elif name == 'N2': import Properties.n2_properties as data
        elif name == 'HFE7200': import Properties.hfe7200_properties as data
        
        # Molar mass for the vapor species [kg/mol]
        self.M_v = data.M_v
        # Ratio of specific heats [-]
        self.gamma = data.gamma
        # Reference saturation temperature [K]
        self.T_sat_ref = data.T_sat_ref
        
        # Ideal gas constant [J/kg.K]
        self.R_v = 8.3144626/self.M_v
        
        # Maximum and minimum allowable temperatures [K]
        self.T_max = data.T_max
        self.T_min = data.T_min
        
        # NIST Fluid procedure
        if name == 'H2' or name == 'N2':
            # Reference saturation pressure [Pa]
            self.p_sat_ref = value_from_fit(x_data=data.T_v,
                                            y_data=data.p_v,
                                            n_pol=8,
                                            x_i=self.T_sat_ref)
            # Coefficients of the polynomials for the liquid properties
            self.rho_l = np.polyfit(x=data.T_l,y=data.rho_l, deg=8) # [kg/m3]
            self.k_l   = np.polyfit(x=data.T_l,y=data.k_l,   deg=8) # [W/mK]
            self.cv_l  = np.polyfit(x=data.T_l,y=data.cv_l,  deg=8) # [J/kgK]
            self.mu_l  = np.polyfit(x=data.T_l,y=data.mu_l,  deg=8) # [Pa*s]
            self.sigma = np.polyfit(x=data.T_l,y=data.sigma, deg=8) # [N/m]
            # Coeficcients of the polynomials for the gas properties
            self.k_v   = np.polyfit(x=data.T_v,y=data.k_v,  deg=8) # [W/mK]
            self.cv_v  = np.polyfit(x=data.T_v,y=data.cv_v, deg=8) # [J/kgK]
            self.mu_v  = np.polyfit(x=data.T_v,y=data.mu_v, deg=8) # [Pa*s]
            # Latent heat of vaporization [J/kg]
            self.dh    = np.polyfit(x=data.T_v,
                                    y=data.hg-data.hf,deg=8)
        # HFE Fluid procedure
        elif name == 'HFE7200':
            # Reference saturation pressure [Pa]
            self.p_sat_ref = data.p_sat_ref
            # Coefficients of the polynomials for the liquid properties
            self.rho_l = np.polyfit(x=data.T_rausch,  y=data.rho_l, deg=4) # [kg/m3]
            self.k_l   = np.polyfit(x=data.T_warrier, y=data.k_l,   deg=4) # [W/mK]
            self.cv_l  = data.cv_l                                         # [J/kgK]
            self.mu_l  = np.polyfit(x=data.T_rausch,  y=data.mu_l,  deg=4) # [Pa*s]
            self.sigma = np.polyfit(x=data.T_rausch,  y=data.sigma, deg=4) # [N/m]
            # Coeficcients of the polynomials for the gas properties
            self.k_v   = data.k_v                                       # [W/mK]
            self.cv_v  = data.cv_v                                      # [J/kgK]
            self.mu_v  = np.polyfit(x=data.T_rausch,y=data.mu_v, deg=8) # [Pa*s]
            # Latent heat of vaporization [J/kg]
            self.dh    = data.dh
    
    'Get liquid properties for temperature Tl'
    def get_liq_properties(self,Tl):
        """
        Compute liquid properties at a given temperature Tl. The input temperature
        is bounded based on the maximum and minimum temperatures presented in the
        fluid property database.
        
        **Inputs:**
            - *Tl:* liquid temperature [K]
        **Outputs:**
            - Bounded liquid temperature [K]
            - Liquid density [kg/m3]
            - Liquid thermal conductivity [W/mK]
            - Liquid specific heat at constant volume [J/kgK]
            - Liquid dynamic viscosity [Pa.s]
            - Surface tension [N/m]
        """
        # Impose bounds on the temperature
        if Tl > self.T_max:   Tl = self.T_max
        elif Tl < self.T_min: Tl = self.T_min
        # Return the values of the properties based on the temperature and fits
        return Tl,property_value(self.rho_l, Tl),\
               property_value(self.k_l,   Tl),\
               property_value(self.cv_l,  Tl),\
               property_value(self.mu_l,  Tl),\
               property_value(self.sigma, Tl)
    
    'Get vapor properties for temperature Tg'                       
    def get_vap_properties(self,Tg):
        """
        Compute vapor properties at a given temperature Tg. The input temperature
        is bounded based on the maximum and minimum temperatures presented in the
        fluid property database.
        
        **Inputs:**
            - *Tg:* ullage temperature [K]
        **Outputs:**
            - Bounded ullage temperature [K]
            - Ideal gas constant for the vapor [J/kgK]
            - Reference saturation temperature [K]
            - Reference saturation pressure [Pa]
            - Vapor thermal conductivity [W/mK]
            - Vapor specific heat at constant volume [J/kgK]
            - Vapor dynamic viscosity [Pa.s]
            - Latent heat of vaporization/condensation [J/kg]
        """
        # Impose bounds on the temperature
        if Tg > self.T_max:   Tg = self.T_max
        elif Tg < self.T_min: Tg = self.T_min
        # Return the values of the properties based on the temperature and fits
        return Tg, self.R_v, \
               self.T_sat_ref, self.p_sat_ref,\
               property_value(self.k_v, Tg),\
               property_value(self.cv_v, Tg),\
               property_value(self.mu_v, Tg),\
               property_value(self.dh,   Tg)
    
    'Get liquid density for temperature Ti'
    def get_liq_density(self,Ti):
        """
        Compute liquid density at temperature Tl. The input temperature
        is bounded based on the maximum and minimum temperatures presented in the
        fluid property database.
        
        **Inputs:**
            - *Tl:* liquid temperature [K]
        **Outputs:**
            - Liquid density [kg/m3]
        """
        # Impose bounds on the temperature
        if Ti > self.T_max:   Ti = self.T_max
        elif Ti < self.T_min: Ti = self.T_min
        # Return the values of the properties based on the temperature and fits
        return property_value(self.rho_l, Ti)
    
# Inert gas (used to pressurized the ullage)
class Inert:
    """
    Initialize fluid properties for the inert gas phase. This class is
    initialized by inserting the name of the gas (i.e. He or Air) and importing
    its properties from from a separate Python script. The properties were
    obtained from the NIST (National Institute of Standards and Technology).
    
    **Constant fluid properties:**
        - *R_a*: Ideal gas constant of the inert gas [J/kgK]
        - *gamma*: Ratio of specific heats [-]
    **Temperature-dependent properties:**
        - *k*: Polynomial coefficients for the inert gas thermal conductivity
        - *cv*: Polynomial coefficients for the inert gas specific heat at constant volume
        - *mu*: Polynomial coefficients for the inert gas dynamic viscosity
    The properties were obtained in saturation conditions for different temperatures.
    Then, they were fitted with a high order polynomial and the coefficients are
    stored in each respective variable.
    """
    def __init__(self,name):
        print('Initializing inert gas properties')
        
        # Initialization of the fluid properties at reference conditions
        self.name = name
        
        # Import fluid properties from database
        if name == 'He':    import Properties.he_properties as data
        elif name == 'Air': import Properties.air_properties as data
        
        # Maximum and minimum allowable temperatures [K]
        self.T_max = data.T_max
        self.T_min = data.T_min
        
        # Ideal gas constant [J/kg.K]
        self.R_a   = data.R
        # Ratio of specific heats [-]
        self.gamma = data.gamma
        # Thermal conductivity [W/mK]
        self.k  = np.polyfit(x=data.T, y=data.k,  deg=4)
        # Specific heat at constant volume [J/kgK]
        self.cv = np.polyfit(x=data.T, y=data.cv, deg=4)
        # Dynmic viscosity [Pa*s]
        self.mu = np.polyfit(x=data.T, y=data.mu, deg=4)
    
    'Get inert gas properties for temperature Tg'
    def get_gas_properties(self,Tg):
        """
        Compute inert gas properties at temperature Tg. The input temperature
        is bounded based on the maximum and minimum temperatures presented in the
        fluid property database.
        
        **Inputs:**
            - *Tg:* ullage temperature [K]
        **Outputs:**
            - Bounded ullage temperature [K]
            - Ideal gas constant for the inert gas [J/kgK]
            - Reference saturation temperature [K]
            - Reference saturation pressure [Pa]
            - Inert gas thermal conductivity [W/mK]
            - Inert gas specific heat at constant volume [J/kgK]
            - Inert gas dynamic viscosity [Pa.s]
        """
        # Impose bounds on the temperature
        if Tg > self.T_max:   Tg = self.T_max
        elif Tg < self.T_min: Tg = self.T_min
        # Return the values of the properties based on the temperature and fits
        return Tg, self.R_a, property_value(self.k, Tg),\
               property_value(self.cv, Tg),\
               property_value(self.mu, Tg)

# Sloshing tank
class Slosh:
    """
    Class that contains the sloshing cell dimensions and excitation parameters.
    
    **Inputs:**
        - *R*: cell radius [m]
        - *H*: total cell height [m]
        - *k_H*: cell fill ratio (liquid height over total height) [-]
        - *k_w*: non-dimensional excitation frequency (*f*/*f11*) [-]
        - *k_a*: non-dimensional excitation amplitude (*A0*/*R*) [-]
    **Calculated variables:**
        - *h*: liquid fill height [m]
        - *V*: total cell volume [m3]
        - *V_l*: liquid volume [m3]
        - *V_g*: ullage volume [m3]
        - *S_i*: cross-sectional area [m2]
        - *w11*: natural frequency [rad/s]
        - *f11*: natural frequency [Hz]
        - *Omega*: excitation frequency [rad/s]
        - *f*: excitation frequency [Hz]
    """
    def __init__(self,R,H,k_h,k_w,k_a,g=9.8):
        # Sloshing cell radius converted from [mm] to [m]
        self.R = R*1e-3
        # Sloshing cell height converted from [mm] to [m]
        self.H = H*1e-3
        # Fill ratio of the liquid over the total height [-]
        self.k_h = k_h
        # Non-dimensional excitation frequency [-]
        self.k_w = k_w
        # Non-dimensional excitation amplitude [-]
        self.k_a = k_a
        # Liquid fill height [m]
        self.h = self.k_h*self.H
        # Excitation amplitude [m]
        self.A0 = self.k_a*self.R
        # Natural frequency [rad/s]
        self.w11 = np.sqrt((g*1.841/self.R)*np.tanh(1.841*self.h/self.R))
        # Natural frequency [Hz]
        self.f11 = self.w11/(2*np.pi)
        # Excitation frequency [rad/s]
        self.Omega = self.k_w*self.w11
        # Ecitation frequency [Hz]
        self.f  = self.Omega/(2*np.pi)
        # Sloshing cell volume [m3]
        self.V  = np.pi*self.R*self.R*self.H
        # Liquid volume [m3]
        self.V_l = (self.h/self.H)*self.V
        # Gas volume [m3]
        self.V_g = self.V - self.V_l
        # Cross-section area [m2]
        self.Si = np.pi*self.R*self.R
        
    'Return sloshing cell dimensions'
    def get_sloshing_cell_dimensions(self):
        """
        Return sloshing cell dimensions:
            - *R*: cell radius [m]
            - *H*: total cell height [m]
            - *h*: liquid height [m]
            - *V*: total cell volume [m3]
            - *Si*: cross-sectional area [m2]
        """
        return self.R, self.H, self.h, self.V, self.Si
    
    'Return sloshing cell parameters for 0D model'
    def get_sloshing_cell_params_0d(self):
        """
        Return sloshing cell parameters required by the 0D model:
            - *V_l*: liquid volume [m3]
            - *V_g*: ullage volume [m3]
            - *Si*: cross-sectional area [m2]
        """
        return self.V_l, self.V_g, self.Si
    
    'Return excitation parameters'
    def get_excitation_conditions(self):
        """
        Return sloshing excitation conditions:
            - *A0*: excitation amplitude [m]
            - *f11*: natural frequency [Hz]
            - *f*: excitation frequency [Hz]
        """
        return self.A0, self.f11, self.f
    
# Define class of Inputs (variables OBTAINED DIRECTLY from the input data)
class Inputs:
    """
    Class that groups all the input variables together.
    
    **Inputs:**
        - *t*: time-array [s]
        - *Tl*: liquid temperature [K]
        - *Tg*: ullage temperature [K]
        - *pg*: ullage pressure [Pa]
    """
    def __init__(self,t,Tl,Tg,pg):
        self.t  = t  # Time [s]
        self.Tl = Tl # Liq temperature [K]
        self.Tg = Tg # Gas temperature [K]
        self.pg = pg # Gas pressure [Pa]

# Define class of Derived Inputs (variables COMPUTED from the input data)
class Derived_Inputs:
    """
    Class that groups all the variables derived from the input temperature and
    pressure data.
    
    **Inputs:**
        - *t*: time-array [s]
        - *Tl*: liquid temperature [K]
        - *Tg*: ullage temperature [K]
        - *pg*: ullage pressure [Pa]
        - *fluid*: object that contains all liquid and vapor properties
        - *inert*: object that contains all inert gas properties
        - *slosh*: object that contains all sloshing properties 
        - *ma_0*: initial mass of inert gas [kg]
        - *ml_0*: initial mass of liquid [kg]
    **Calculated variables:**
        - *mv*: vapor mass evolution over time [kg]
        - *ml*: liquid mass evolution over time [kg]
        - *Vl*: liquid volume over time [m3]
        - *Vg*: ullage volume over time [m3]
    """
    def __init__(self,t,Tl,Tg,pg,inert,fluid,slosh,ma_0,ml_0):
        self.mv, self.ml,self.Vl, self.Vg = \
        get_mass_and_volumes(t,Tl,Tg,pg,inert,fluid,slosh,ma_0,ml_0)

#%% 0D model (ODE system)

# 0D model for the temperature and pressure evolution
def model_0d(x,t,m_a,h_iL,h_m,V_l,V_g,fluid,inert,slosh):
    """
    0D model that predicts the evolution of the thermodynamic system inside of
    a closed reservoir when this is submitted to a sloshing excitation. The model
    is composed by two regions: the liquid and the ullage. The latter is a mixture
    between the vapor and the inert gas. The system is assumed to be closed with
    adiabatic walls.
    
    The model receives inputs regarding the initial pressure and temperature
    conditions as well as the heat and mass transfer coefficients between the
    gas and liquid. More detailed information is available in Technical Note
    TN5000-10-05 from the VKI cryogenics team.
    
    **Inputs:**
        - *x*: initial conditions for the system of ODEs
        - *t*: time interval [s]
        - *m_a*: mass of inert gas [kg]
        - *h_iL*: heat transfer coefficient [W/m2K]
        - *h_m*: mass transfer coefficient [m/s]
        - *V_l*: liquid volume [m3]
        - *V_g*: ullage volume [m3]
        - *fluid*: object that contains all liquid and vapor properties
        - *inert*: object that contains all inert gas properties
        - *slosh*: object that contains all sloshing properties 
    **Outputs:**
        - *dmvdt*: temporal rate of change of vapor mass [kg/s]
        - *dTgdt*: temporal rate of change of ullage temperature [K/s]
        - *dTldt*: temporal rate of change of liquid temperature [K/s]
        - *dpgdt*: temporal rate of change of ullage pressure [Pa/s]
    """
    
    ### Initial conditions
    m_v = x[0]; Tg = x[1]; Tl = x[2]; p_g = x[3]
    
    ### Import required properties/parameters
    
    # Liquid properties at temperature Tl
    _, rho_l, k_l, cv_l, mu_l, sigma = fluid.get_liq_properties(Tl)
    # Vapor properties at temperature Tg
    _, R_v, T_sat_ref, p_sat_ref, k_v, cv_v, mu_v, dh = fluid.get_vap_properties(Tg)
    # Inert gas properties at temperature Tg
    _, R_a, k_a, cv_a, mu_a     = inert.get_gas_properties(Tg)
    
    # Sloshing cell dimensions required by the 0D model
    _, _, S_i = slosh.get_sloshing_cell_params_0d()
    
    ### Calculate required variables
    
    # Liquid mass at current time-step [kg]
    m_l = rho_l*V_l
    
    # Vapor density at current time-step [kg/m3]
    rho_v = m_v/V_g
    # Inert gas density at current time-step [kg/m3]
    rho_a = m_a/V_g
    
    # Thermal effusivity of the dry-air
    b_a = np.sqrt(rho_a*cv_a*k_a)
    # Thermal effusivity of the vapor
    b_v = np.sqrt(rho_v*cv_v*k_v)
    # Thermal effusitivity of the ullage (mass-averaged)
    b_g = (m_a*b_a + m_v*b_v)/(m_a + m_v)
    # Thermal effusivity of the liquid
    b_l = np.sqrt(rho_l*cv_l*k_l)
    
    # Interface temperature (semi-infinite body assumption) [K]
    Ti = (b_g*Tg + b_l*Tl)/(b_g+b_l)
    
    # Liquid density in saturation conditions at current time-step [kg/m3]
    rho_l_sat = fluid.get_liq_density(Ti)
    
    # Vapor phase pressure [Pa]
    p_v = p_g - rho_a*R_a*Tg
    # Vapor phase saturation pressure[Pa]
    p_v_sat = clausius_clapeyron_p(R_v,dh,p_sat_ref,T_sat_ref,Ti)
    
    ### 0D Model Equations:
    
    # 1. Mass at transfer at the interface
    dmvdt = h_m*S_i*( p_v_sat/(R_v*Ti) - p_v/(R_v*Tg) )
    
    # 2. Liquid internal energy balance
    dTldt = ((h_iL*S_i)/(m_l*cv_l))*(Ti - Tl) - (dmvdt/(m_l*cv_l))*( cv_l*np.abs(Ti-Tl) + p_v/rho_l_sat )
    
    # 3. Ullage internal energy balance
    dTgdt = - ((h_iL*S_i)/(m_a*cv_a+m_v*cv_v))*(Ti - Tl) + (dmvdt/(m_a*cv_a+m_v*cv_v))*( cv_v*np.abs(Ti-Tg) + np.abs(dh) + p_v/rho_l_sat)
    
    # 4. Ullage pressure evolution
    dpgdt = (1/V_g)*(m_a*R_a+m_v*R_v)*dTgdt + (R_v*Tg/V_g)*dmvdt
    
    return [dmvdt, dTgdt, dTldt, dpgdt] 

#%% Define cost function

def cost_function(X,t,inputs,derived_inputs,fluid,inert,slosh,ma_0,FOL_IN):
    """
    Cost function that drives the optimization problem. The parameter we want to
    minimize is the overall relative error between the input pressure/temperature
    data and the predictions given by the 0D model.
    The cost function is evaluated several times with different values of the
    heat & mass transfer coefficients as inputs in order to obtain the minimum
    error in the prediction.
    
    **Inputs:**
        - *X*: initial estimate for the heat & mass transfer coefficients [W/m2K, m/s]
        - *t*: time array [s]
        - *inputs*: input data (liquid temp, ullage temp, ullage pressure)
        - *derived_inputs*: variables computed from input data (vapor/liquid mass & volumes)
        - *fluid*: object that contains all liquid and vapor properties
        - *inert*: object that contains all inert gas properties
        - *slosh*: object that contains all sloshing properties 
        - *ma_0*: inert gas mass [kg]
        - *FOL_IN*: input folder location
    **Outputs**:
        - *err*: relative error between real data and model predictions
    """
    
    # Function inputs
    h_iL = X[0] # Heat transfer coefficient
    h_m  = X[1] # Mass transfer coefficient
    
    ### Declare variable arrays solved by the ODE system
    
    # Initial conditions obtained from the real data
    Tg = np.zeros(len(t)); Tg[0] = inputs.Tg[0] # [K]
    Tl = np.zeros(len(t)); Tl[0] = inputs.Tl[0] # [K]
    pg = np.zeros(len(t)); pg[0] = inputs.pg[0] # [pg]
    mv = np.zeros(len(t)); mv[0] = derived_inputs.mv[0] # [kg]
    # Declare array for the liquid mass evolution
    ml = np.zeros(len(t)); ml[0] = derived_inputs.ml[0] # [kg]
    # Declare array for ullage and liquid volume evolution
    Vg = np.zeros(len(t)); Vg[0] = derived_inputs.Vg[0] # [m3]\
    Vl = np.zeros(len(t)); Vl[0] = derived_inputs.Vl[0] # [m3]
    
    print('Solving 0D model %s, %s for %s' %(fluid.name,inert.name,FOL_IN))
    print('-> Heat transfer coeff.: %f' %h_iL)
    print('-> Mass transfer coeff.: %f' %h_m)
    
    ### Run 0D model
    
    # Start timer for 0D model solver
    start_time = time.time()
    
    # Solve 0D model with current estimate of the heat & mass transfer coeffs.
    for i in range(len(t)-1):
        # Time-step
        ts = [t[i],t[i+1]]
        
        # Initial solution for the ODE
        x0 = [mv[i],Tg[i],Tl[i],pg[i]]
        
        # Solve system of ODEs for the current time-step
        x = odeint(model_0d, x0, ts, args=(ma_0,h_iL,h_m,Vl[i],Vg[i],fluid,inert,slosh))
        
        # Update vapor mass [kg]
        mv[i+1] = x[1,0]
        # Update ullage temperature [K]
        Tg[i+1] = x[1,1]
        # Update liquid temperature [K]
        Tl[i+1] = x[1,2]
        # Update ullage pressure [Pa]
        pg[i+1] = x[1,3]
        
        # Update liquid mass [kg]
        ml[i+1] = ml[i] - (mv[i+1] - mv[i])
        # Update liquid volume [m3]
        Vl[i+1] = ml[i+1]/fluid.get_liq_density(Tl[i+1])
        # Update ullage volume [m3]
        Vg[i+1] = slosh.V - Vl[i+1]
    
    ### Compute relative error between model predictions and input data
    err_Tl = np.linalg.norm((Tl - inputs.Tl)/inputs.Tl)
    err_Tg = np.linalg.norm((Tg - inputs.Tg)/inputs.Tg)
    err_pg = np.linalg.norm((pg - inputs.pg)/inputs.pg)
    
    # Root-mean square sum of the temperature/pressure relative errors
    err = np.sqrt(err_Tl**2 + err_Tg**2 + err_pg**2)
    print('-> Overall error: %f' %err)
    print('-> Elapsed time time: %f' %(time.time() - start_time))
    return err

#%% Inverse method
def inverse_method(n_trials,p_test,t,Tg,Tl,pg,fluid,inert,slosh,ma_0,ml_0,X_0,optimizer_method,FOL_IN):
    """
    Main function that is used to apply the inverse method. The goal is to
    compute the heat & mass transfer coefficients *h_heat* and *h_mass* which
    generate the temperature and pressure evolution observed in the input data.
    In order to handle noise in the data, a bootstrapping approach is taken.
    
    This function generates a distribution/population for the heat and mass
    transfer coefficients by comparing the input temperature & pressure evolution
    with predictions given by the 0D model.
    
    **Inputs:**
        - *n_trials*: number of optimization loops for bootstrapping
        - *p_test*: ratio of testing data to total data
        - *t*: input time array [s]
        - *Tg*: input ullage temperature [K]
        - *Tl*: input liquid temperature [K]
        - *pg*: input ullage pressure [Pa]
        - *fluid*: object that contains all liquid and vapor properties
        - *inert*: object that contains all inert gas properties
        - *slosh*: object that contains all sloshing properties 
        - *ma_0*: initial inert gas mass [kg]
        - *ml_0*: initial liquid mass [kg]
        - *X_0*: initial estimate for heat & mass transfer coefficients [W/m2K,m/s]
        - *optimizer_method*: method for optimizer function (uses scipy.minimize())
        - *FOL_IN*: location where input data is stored
    **Outputs**:
        - *h_heat*: heat transfer coefficient distribution [W/m2K]
        - *h_mass*: mass transfer coefficient distribution [m/s]
    """
    # Initialize heat transfer coefficient population
    h_heat = np.zeros(n_trials)
    # Initialzie mass transfer coefficient population
    h_mass = np.zeros(n_trials)
    
    # Apply inverse method "n_trials" times
    for j in range(n_trials):
        
        print('Split training and testing data')
        
        # Split arrays to have 70% training and 30% validation data
        t_train, _,\
        Tg_train,_,\
        Tl_train,_,\
        pg_train,_ \
        = train_test_split(t,Tg,Tl,pg,test_size=p_test)
        
        # Unsorted training data
        TRAIN_DATA = np.vstack((t_train,Tg_train,Tl_train,pg_train)).transpose()
        # Sorted training data based on time
        TRAIN_DATA = TRAIN_DATA[TRAIN_DATA[:,0].argsort()]
    
        # Split sorted train and testing arrays
        t_train  = TRAIN_DATA[:,0] # [s]
        Tg_train = TRAIN_DATA[:,1] # [K]
        Tl_train = TRAIN_DATA[:,2] # [K]
        pg_train = TRAIN_DATA[:,3] # [Pa]
    
        # Assign inputs & derived inputs that are used in the cost function
        inputs         = Inputs(t_train,Tl_train,Tg_train,pg_train)
        derived_inputs = Derived_Inputs(t_train,Tl_train,Tg_train,
                                        pg_train,inert,fluid,slosh,
                                        ma_0,ml_0)
        
        ### Optimization
        print('%s Method' %(optimizer_method))
        optimizer_time = time.time()
        # Optimization function (built-in from scipy)
        res = minimize(cost_function, # cost function
                       X_0, # initial condition
                       method = optimizer_method,
                       args   = (t_train,inputs,derived_inputs,fluid,inert,slosh,ma_0,FOL_IN),
                       options= {'ftol': 1e-6, 'disp': True})
        # Print results and total elapsed time
        print('%s Result: h_heat = %f & h_m = %f' %(optimizer_method,res.x[0],res.x[1]))
        print('%s elapsed time time: %f' %(optimizer_method, time.time() - optimizer_time))
        # Store the computed coefficients
        h_heat[j] = res.x[0] # [W/m2K]
        h_mass[j] = res.x[1] # [m/s]
    return h_heat, h_mass


#%% End