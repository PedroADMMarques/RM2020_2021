import numpy as np
import pandas as pd

#%% Fluid choice and desired pressure

class N2:
    # Molar fraction in air
    x = 0.78084
    pass
class O2:
    # Molar fraction in air
    x = 0.20946
    pass
class Ar:
    # Molar fraction in air
    x = 0.00934
    pass

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

### N2
data_frame = pd.read_csv('Properties/NIST/GN2.txt', sep='	',dtype=np.float64)
N2.T        = data_frame.values[:,0]        # Saturation temperature (K)
N2.p        = data_frame.values[:,1]*1e6    # Saturation pressure (MPa)
N2.rho      = data_frame.values[:,2]        # kg/m3
N2.hf       = data_frame.values[:,5]*1e3    # J/kg
N2.cv       = data_frame.values[:,7]*1e3    # J/kgK
N2.cp       = data_frame.values[:,8]*1e3    # J/kgK
N2.mu       = data_frame.values[:,11]       # Pa*s
N2.k        = data_frame.values[:,12]       # W/mK

### O2
data_frame = pd.read_csv('Properties/NIST/O2.txt', sep='	',dtype=np.float64)
O2.T        = data_frame.values[:,0]        # Saturation temperature (K)
O2.p        = data_frame.values[:,1]*1e6    # Saturation pressure (MPa)
O2.rho      = data_frame.values[:,2]        # kg/m3
O2.hf       = data_frame.values[:,5]*1e3    # J/kg
O2.cv       = data_frame.values[:,7]*1e3    # J/kgK
O2.cp       = data_frame.values[:,8]*1e3    # J/kgK
O2.mu       = data_frame.values[:,11]       # Pa*s
O2.k        = data_frame.values[:,12]       # W/mK

### Ar
data_frame = pd.read_csv('Properties/NIST/Ar.txt', sep='	',dtype=np.float64)
Ar.T        = data_frame.values[:,0]        # Saturation temperature (K)
Ar.p        = data_frame.values[:,1]*1e6    # Saturation pressure (MPa)
Ar.rho      = data_frame.values[:,2]        # kg/m3
Ar.hf       = data_frame.values[:,5]*1e3    # J/kg
Ar.cv       = data_frame.values[:,7]*1e3    # J/kgK
Ar.cp       = data_frame.values[:,8]*1e3    # J/kgK
Ar.mu       = data_frame.values[:,11]       # Pa*s
Ar.k        = data_frame.values[:,12]       # W/mK

# Same temperature array for all test cases
T   = N2.T 

## Air properties from average composition of air
rho = N2.x*N2.rho + O2.x*O2.rho + Ar.x*Ar.rho
cv  = N2.x*N2.cv  + O2.x*O2.cv  + Ar.x*Ar.cv
cp  = N2.x*N2.cp  + O2.x*O2.cp  + Ar.x*Ar.cp
mu  = N2.x*N2.mu  + O2.x*O2.mu  + Ar.x*Ar.mu
k   = N2.x*N2.k   + O2.x*O2.k   + Ar.x*Ar.k

#%% Additional properties

R = 8.3144626/(28.97e-03) # J/kg.K 
gamma = 1.4

#%% Maximum and mininum temperatures
T_max = np.max(T)
T_min = np.min(T)