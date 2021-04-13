"""
HFE7200 LIQUID AND VAPOUR PROPERTIES
------------------------------------------------------------------------------
SOURRCES:
------------------------------------------------------------------------------
- Pramod Warrier and Amyn S. Teja (2011), Viscosity, and Thermal Conductivity
of Mixtures of 1-Ethoxy-1,1,2,2,3,3,4,4,4-nonafluorobutane (HFE 7200) with
Methanol and 1-Ethoxybutane

- Rausch et. al (2015), Density, Surface Tension, and Kinematic Viscosity of
Hydrofluoroethers HFE-7000, HFE-7100, HFE-7200, HFE-7300, and HFE-7500

- 3M Novec 7200 Engineered Fluid Product Information
------------------------------------------------------------------------------
"""

import numpy as np

#%% Classes

#%% Rausch 2015 data

# Temperature (K) values considered by Rausch
T_rausch = np.array([273.15,283.15,293.15,303.15,313.15,323.15,333.15,343.15,353.15,363.15,373.15])

# Liquid density (kg/m3)
rho_l = np.array([1478.07,1456.13,1433.91,1411.31,1388.26,1364.68,1340.47,1315.57,1289.87,1263.31,1235.79])

# Vapour density (kg/m3)
rho_v = np.array([0.61,0.96,1.47,2.18,3.15,4.47,6.22,8.5,11.44,15.19,19.94])

# Vapour dynamic viscosity (Pa*s)
mu_v = np.array([9.33,9.68,10.03,10.4,10.77,11.15,11.53,11.91,12.3,12.68,13.05])*10**(-6)

# Liquid kinematic viscosity (m2/s)
nu_l = np.array([0.6631,0.5563,0.4824,0.4292,0.3822,0.3429,0.3125,0.2826,0.2564,0.2382,0.2225])*10**(-6)

# Liquid dynamic viscosity (Pa*s)
mu_l = nu_l * rho_l

# Surface tension (N/m)
sigma = np.array([16.03,14.9,14.01,13.21,12.33,11.42,10.56,9.67,8.84,7.99,7.19])*10**(-3)

#%% Warrier 2011 data

# Temperature values considered by Warrier
T_warrier = np.array([278.8,300.6,314.1,328.3,344.1])

# Liquid thermal conductivity (W/mK)
k_l = np.array([0.0712,0.0644,0.0616,0.059,0.0563])

#%% 3M Novec HFE7200 Product information data

# Enthalpy of vaporization
dh = 30 * 4186.8 # J/kg

# Liquid specific heat capacity (Cp = Cv = C)
cv_l = 0.29 * 4186.8 # J/kgK

# Saturation temperature at reference conditions
T_sat_ref = 298.15 # K

# Vapour pressure at reference conditions (T = 278.15 K)
p_sat_ref = 109 * 133.322 # Pa

#%% Additional information

# Molar mass of HFE7200
M_v = 264e-03 # kg/mol

# Ratio of specific heats
gamma = 1.24

# Vapor thermal conductivity
k_v   = 0.0112 # W/mK

# Vapor specific heat at constant volume
cv_v  = 856.3  # J/kgK (HFE7000)

#%% Maximum and mininum temperatures
T_max = np.max(T_rausch)
T_min = np.min(T_rausch)