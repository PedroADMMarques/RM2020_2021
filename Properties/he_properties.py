import numpy as np
import pandas as pd

#%% Fluid choice and desired pressure

folder = 'Properties/NIST/'
name = "He.txt"

# NIST Properties column
    # 00 - Temperature (K)
    # 01 - Pressure (MPa)
    # 02 - GAS Density
    # 03 - GAS Specific volume
    # 04 - GAS Internal Energy
    # 05 - GAS Enthalpy
    # 06 - GAS Entropy 
    # 07 - GAS Cv
    # 08 - GAS Cp
    # 09 - GAS Sound speed
    # 10 - GAS Joule-Thomson
    # 11 - GAS Dynamic viscosity
    # 12 - GAS Thermal conductivity
    # 13 - GAS Phase

#%% Import file properties into data frame
data_frame = pd.read_csv(folder+name, sep='	',dtype=np.float64)

# Gas properties (NIST)
T        = data_frame.values[:,0]        # Saturation temperature (K)
p        = data_frame.values[:,1]*1e6    # Saturation pressure (MPa)
rho      = data_frame.values[:,2]        # kg/m3
hf       = data_frame.values[:,5]*1e3    # J/kg
cv       = data_frame.values[:,7]*1e3    # J/kgK
cp       = data_frame.values[:,8]*1e3    # J/kgK
mu       = data_frame.values[:,11]       # Pa*s
k        = data_frame.values[:,12]       # W/mK

#%% Additional properties

R = 8.3144626/(4e-03) # J/kg.K 
gamma = 1.66

#%% Maximum and mininum temperatures
T_max = np.max(T)
T_min = np.min(T)