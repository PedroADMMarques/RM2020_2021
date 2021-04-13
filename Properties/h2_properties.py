import numpy as np
import pandas as pd

#%% Fluid choice and desired pressure

folder = 'Properties/NIST/'
name = "H2.txt"

# NIST Properties column
    # 00 - Temperature (K)
    # 01 - Pressure (MPa)
    # 02 - LIQUID Density
    # 03 - LIQUID Specific volume
    # 04 - LIQUID Internal Energy
    # 05 - LIQUID Enthalpy
    # 06 - LIQUID Entropy 
    # 07 - LIQUID Cv
    # 08 - LIQUID Cp
    # 09 - LIQUID Sound speed
    # 10 - LIQUID Joule-Thomson
    # 11 - LIQUID Dynamic viscosity
    # 12 - LIQUID Thermal conductivity
    # 13 - LIQUID Surface tension
    # 14 - GAS Density
    # 15 - GAS Specific volume
    # 16 - GAS Internal energy
    # 17 - GAS Enthalpy
    # 18 - GAS Entropy 
    # 19 - GAS Cv
    # 20 - GAS Cp
    # 21 - GAS Sound speed
    # 22 - GAS Joule-Thomson
    # 23 - GAS Dynamic viscosity
    # 24 - GAS Thermal conductivity

#%% Import file properties into data frame
data_frame = pd.read_csv(folder+name, sep='	',dtype=np.float64)

# Liquid properties (NIST)
T_l        = data_frame.values[:,0]        # Saturation temperature (K)
p_l        = data_frame.values[:,1]*1e6    # Saturation pressure (Pa)
rho_l      = data_frame.values[:,2]        # kg/m3
hf         = data_frame.values[:,5]*1e3    # J/kg
cv_l       = data_frame.values[:,7]*1e3    # J/kgK
cp_l       = data_frame.values[:,8]*1e3    # J/kgK
mu_l       = data_frame.values[:,11]       # Pa*s
k_l        = data_frame.values[:,12]       # W/mK
sigma      = data_frame.values[:,13]       # N/m

# Gas properties (NIST)
T_v           = data_frame.values[:,0]        # Saturation temperature (K)
p_v           = data_frame.values[:,1]*1e6    # Saturation pressure (Pa)
rho_v         = data_frame.values[:,14]       # kg/m3
hg            = data_frame.values[:,17]*1e3   # J/kg
cv_v          = data_frame.values[:,19]*1e3   # J/kgK
cp_v          = data_frame.values[:,20]*1e3   # J/kgK
mu_v          = data_frame.values[:,23]       # Pa*s
k_v           = data_frame.values[:,24]       # W/mK

#%% Assign additional constants to the fluid
# Vapor molar mass
M_v = 2e-03 # kg/mol
# Ratio of specific heats
gamma = 1.4
# Reference saturation temperature
T_sat_ref = 18 # K

#%% Maximum and mininum temperatures
T_max = np.max(T_l)
T_min = np.min(T_l)