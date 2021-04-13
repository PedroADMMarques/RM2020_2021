# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:49:15 2021

@author: Pedro Marques
"""

"inverSe Method for nOisy nOn-isoThermal slosHIng Experiments (SMOOTHIE)"

#%% Packages

import numpy as np # Multiple reasons
import pandas as pd # Dataframe management for .csv & .txt files
import seaborn as sns # Plot smooth distribution
import matplotlib.pyplot as plt # Plots
from matplotlib import gridspec # More control over subplot layout

# Import Custom Classes 
from smoothie import Fluid,Inert,Slosh
# Import Inverse Method function
from smoothie import inverse_method

#%% User INPUTS

print('Importing fluid properties')

'Fluid and sloshing settings'
# Fluids: 'H2', 'N2', 'HFE7200'
fluid = Fluid(name='N2')
# Inert gas: 'Air', 'He' 
inert = Inert(name='He')
# Sloshing cell dimensions & excitations
# R   - radius [mm]
# H   - height [mm]
# k_h - liquid fill ratio [-]
# k_w - non-dimensional sloshing excitation [-]
# k_a - non-dimensional sloshing amplitude  [-]
slosh = Slosh(R=40,H=104,k_h=0.7,k_w=0.8,k_a=0.06)

'Input/Output data settings'
# Folder where the input data is located
FOL_IN  = 'Input_Cryo_Data'
# File that contains input data
FILE_IN = 'model_data.txt'
# Output folder location
FOL_OUT  = 'Output_Cryo_Data'
# Save output data? [True/False]
save_OUT = False

'Optimization settings'
# Number of optimizations we will perform
n_trials = 20
# Percentage of validation/test data (default 30% validation and 70% training)
p_test = 0.3
# Initial optimization conditions [heat-transfer coeff, mass-transfer coeff.]
X_0 = [50,1e-4]
# Select optimization method
optimizer_method = 'Nelder-Mead'

#%% Import data from 0D simulation

print('Importing temperature and pressure data')

# Read simulation output file
df = pd.read_csv(format('%s/%s' %(FOL_IN,FILE_IN)))

# Time-array [s]
t  = np.array(df['Time [s]'])
# Ullage temperature [K]
Tg_clean = np.array(df['T_g [K]'])
# Liquid temperature [K]
Tl_clean = np.array(df['T_l [K]'])
# Interface temperature [K]
Ti_clean = np.array(df['T_i [K]'])
# Ullage pressure [Pa]
pg_clean = np.array(df['p_g [Pa]'])

# Initial liquid mass [kg]
ml_0 = np.array(df['ml [kg]'])[0]
# Initial inert gas mass [kg]
ma_0 = np.array(df['ma [kg]'])[0]

#%% Add random noise to the input data

# Add Gaussian noise with mean 0 and variance 1
Tl = Tl_clean + 0.3*np.random.randn(len(t))
Ti = Ti_clean + 0.3*np.random.randn(len(t))
Tg = Tg_clean + 0.3*np.random.randn(len(t))
pg = pg_clean + 50*np.random.randn(len(t))

# Create output folder if it does not exist
if save_OUT == True:
    import os
    if not os.path.exists(FOL_OUT): os.mkdir(FOL_OUT)

#%% Plot clean and  noisy data for comparison

print('Plot temperature and pressure evolution')

# Blue for cryogenic case
if fluid.name == 'N2': color = 'C0'
# Red for non-cryogenic case
elif fluid.name == 'HFE7200': color = 'C3'

plt.figure(figsize=(8,5))
gs=gridspec.GridSpec(3, 2, width_ratios=[1, 1])

# Ullage temperature
plt.subplot(gs[0,0])
plt.plot(t,Tg,linestyle='',marker='.',color=color,label='Noisy data')
plt.plot(t,Tg_clean,linestyle='dashed',marker='',color='black',label='Clean data')
plt.title('Ullage temperature')
plt.ylabel('Temperature [K]')
plt.xlabel('Time [s]')
plt.xlim(t[0],t[-1])
plt.grid()
plt.legend()
plt.tight_layout()

# Interface temperature
plt.subplot(gs[1,0])
plt.plot(t,Ti,linestyle='',marker='.',color=color,label='Noisy data')
plt.plot(t,Ti_clean,linestyle='dashed',marker='',color='black',label='Clean data')
plt.title('Interface temperature')
plt.ylabel('Temperature [K]')
plt.xlabel('Time [s]')
plt.xlim(t[0],t[-1])
plt.legend()
plt.grid()
plt.tight_layout()

# Liquid temperature
plt.subplot(gs[2,0])
plt.plot(t,Tl,linestyle='',marker='.',color=color,label='Noisy data')
plt.plot(t,Tl_clean,linestyle='dashed',marker='',color='black',label='Clean data')
plt.title('Liquid temperature')
plt.ylabel('Temperature [K]')
plt.xlabel('Time [s]')
plt.xlim(t[0],t[-1])
plt.legend()
plt.grid()
plt.tight_layout()

# Ullage pressure
plt.subplot(gs[:,1])
plt.plot(t,pg/1e5,linestyle='',marker='.',color=color,label='Noisy data')
plt.plot(t,pg_clean/1e5,linestyle='dashed',marker='',color='black',label='Clean data')
plt.title('Ullage pressure')
plt.ylabel('Pressure [bar]')
plt.xlabel('Time [s]')
plt.xlim(t[0],t[-1])
plt.legend()
plt.grid()
plt.tight_layout()
if save_OUT == True: plt.savefig('%s/noisy_data.pdf' %(FOL_OUT))

#%% Inverse method for heat & mass transfer coeffs from temperature & pressure

h_heat, h_mass = inverse_method(n_trials, # No. of trials for bootstrapping
                                p_test, # Ratio of testing data [default: 30%]
                                t,  # Input data time array [s]
                                Tg, # Noisy ullage temperature [K]
                                Tl, # Noisy liquid temperature [K]
                                pg, # Noisy ullage pressure [Pa]
                                fluid, # Fluid properties class (liq. & vapour)
                                inert, # Inert gas properties class
                                slosh, # Sloshing cell & excitation properties class
                                ma_0,ml_0, # Initial inert gas & liquid masss [kg]
                                X_0, # Initial condition for heat & mass transf. coeffs
                                optimizer_method, # Optimizer method
                                FOL_IN # Input folder
                                )

#%% Compute statistics

# Uncertainty (95% confidence assuming normal distribution)
h_heat_unc = 1.96*np.std(h_heat); h_mass_unc = 1.96*np.std(h_mass)

print('h_heat = %e +- %e' %(np.mean(h_heat, dtype=np.float64),h_heat_unc))
print('h_mass = %e +- %e' %(np.mean(h_mass, dtype=np.float64),h_mass_unc))

# Normalized variables
h_heat_norm = (h_heat - np.mean(h_heat, dtype=np.float64))/np.std(h_heat)
h_mass_norm = (h_mass - np.mean(h_mass, dtype=np.float64))/np.std(h_mass)
# Create dataset matrix
X = np.vstack((h_heat_norm,h_mass_norm))
# Sample mean estimator
X_mean = np.sum(X,axis=1)/n_trials
# Covariance matrix estimator
X_cov = np.cov(X)

#%% Plot normalized statistics

### Normalized histograms
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title('Normalized histograms')
plt.hist(h_heat_norm,label=r'$z_1=\frac{h_{heat}-\mu_{h_{heat}}}{\sigma_{h_{heat}}}$',alpha=0.5)
plt.hist(h_mass_norm,label=r'$z_2=\frac{h_{mass}-\mu_{h_{mass}}}{\sigma_{h_{mass}}}$',alpha=0.5)
plt.xlabel(r'$z_1,z_2$')
plt.ylabel(r'$p(z_1),p(z_2)$')
plt.legend()
plt.tight_layout()

# Required for Multivariate Gaussian
from scipy.stats import multivariate_normal

# Create great to plot Multivariate Gaussian
X, Y = np.meshgrid(np.linspace(-3,3,300), np.linspace(-3,3,300))
POS  = np.dstack((X, Y))
RV   = multivariate_normal(X_mean,X_cov)

### Multivariate gaussian distribution
plt.subplot(122)
plt.title('Normalized Multivariate Gaussian Distribution')
plt.contourf(X,Y,RV.pdf(POS),extend='both')
plt.xlabel(r'$z_1$') # Normalized h_heat
plt.ylabel(r'$z_2$') # Normalized h_mass
plt.colorbar()
plt.xlim(left=-3,right=3)
plt.ylim(bottom=-3,top=3)
plt.tight_layout()
if save_OUT == True: plt.savefig('%s/stats_normalized.pdf' %(FOL_OUT))

#%% Plot statistics

plt.figure()
# Histogram for heat transfer coefficient distribution [W/m2K]
plt.subplot(211)
sns.distplot(h_heat,
             hist=True,
             kde=True,
             color=color,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':2,'shade': True})
plt.xlabel(r'Heat transfer coefficient [W/m$^2$K]')
plt.ylabel('Frequency')
plt.title(r'Probabiliy density function for $h_{heat}$')
plt.yticks([]) 
plt.tight_layout()
plt.subplot(212)
# Histogram for mass transfer coefficient distribution [m/s]
sns.distplot(h_mass,
             hist=True,
             kde=True,
             color=color,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':2,'shade': True})
plt.xlabel(r'Mass transfer coefficient [m/s]')
plt.ylabel('Frequency')
plt.title(r'Probabiliy density function for $h_{mass}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.yticks([]) 
plt.tight_layout()
if save_OUT == True: plt.savefig('%s/stats.pdf' %(FOL_OUT))

#%% Save population
if save_OUT == True: 
    # Header name for the .txt file
    header_name='h_heat[W/m2K],h_mass[m/s]'
    data = np.zeros((n_trials,2))
    data[:,0] = h_heat
    data[:,1] = h_mass
    np.savetxt(format('%s/coeffs_population.txt' %FOL_OUT),
                   data,
                   delimiter=",",
                   comments='',
                   header=header_name)