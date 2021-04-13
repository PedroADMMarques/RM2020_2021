.. TSC Exam documentation master file, created by
   sphinx-quickstart on Tue Apr 13 00:56:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TSC Exam April 2021: Pedro Marques
==================================

This documentation describes the SMOOTHIE (inverSe Method for nOisy nOn-isoThermal slosHIng Experiments) module that was developed in order to compute heat and mass transfer coefficients from non-isothermal sloshing data. 

The direct application of this work is related to the sloshing of cryogenic propellants in the upper stages of modern spacecraft. Before take-off, the propellant tanks are pressurized, leading to a thermal stratification to set in the gas and liquid regions. As a consequence of sloshing, thermal mixing is observed between both fluid phases. This generates thermal gradients near the interface which can trigger evaporation or condensation effects. The balance of the aforementioned effects during the propelled flight phase of modern spacecraft results in undesirable pressure fluctuations, which can cause structural instabilities and thrust oscillations.

The current version of this code relies on synthetically generated time-resolved input data for the temperature and pressure evolution. This data was obtained from a 0D model developed in Technical Note TN5000-10-05 of the VKI cryogenics team, with known values for the heat and mass transfer coefficients. Thus, the purpose of this code is to verify if the inverse method works in predicting these coefficients from simply the analysing the input data, and what is the associated uncertainty when noise is added to the signals.

What's included?
----------------

The folder structure of this code is shown below:

| **TSC_Exam_2021**
| |_ Input_Cryo_Data
| |____ 0D_data.pdf
| |____ coeffs.txt
| |____ model_data.txt
| |_ Output_Cryo_Data
| |____ coeffs_population.txt
| |____ noisy_data.pdf
| |____ stats.pdf
| |____ stats_normalized.pdf
| |_ Properties
| |____ NIST
| |________ Ar.txt
| |________ GN2.txt
| |________ H2.txt
| |________ He.txt
| |________ N2.txt
| |________ O2.txt
| |____ air_properties.py
| |____ h2_properties.py
| |____ he_properties.py
| |____ hfe7200_properties.py
| |____ n2_properties.py
| |_ inverse_trial_single_core.py
| |_ README.txt
| |_ smoothie.py

The *Input_Cryo_Data* directory contains three files:

* **0D_data.pdf**: plots for the input temperature and pressure data (synthetic data)
* **coeffs.txt**: values of the heat and mass transfer coefficients used to generate the input synthetic data
* **model_data.txt**: text file which contains the time-resolved input data (total pressure, liquid temperature, interface temperature, ullage temperature, initial liquid mass and initial inert gas mass)

The *Output_Cryo_Data* directory contains four files which are generated after running the *inverse_trial_single_core.py* script. These files are pre-included in the git repository since the code can take several hours to run. The content of these files is now described:

* **coeffs_population.txt**: population of heat and mass transfer coefficients produced by the *inverse_trial_single_core.py* code
* **noisy_data.pdf**: since the input data is synthetically generated, random noise is added as a form of pre-processing. This file shows the noisy data considered for the computation of the heat & mass transfer coefficients.
* **stats.pdf**: plots showing the histograms and probability density functions of the heat and mass transfer coefficients estimated by the inverse method
* **stats_normalized.pdf**: plots showing the normalized histograms and probability density functions of the heat and mass transfer coefficients, alongside the Multivariate Gaussian

The *Properties* directory contains one subdirectory named *NIST* and five additional python scripts. The *NIST* subdirectory contains text files with material properties for different fluids (i.e. Argon, Helium, Nitrogen, Oxygen) that were downloaded from the NIST database. The Python scripts in the *Properties* directory are used to compute the fluid properties required while running the *inverse_trial_single_core.py* code. Separate files were created for each fluid in order to allow for greater flexibility in accounting for different sources of information, or different approaches (e.g. N2, H2 and He are all obtained from the NIST database, whereas the HFE7200 properties are obtained from different sources).

The *inverse_trial_single_core.py* script is the main file that should be executed in order to determine the heat and mass transfer coefficients from the input data. The main outline of this script is explained step-by-step in the **Tutorial** section of the documentation.

Finally, *smoothie.py* is the module that contains all the specific functions and classes required for the application of the inverse method to the non-isothermal sloshing problem. The full description of this module and its members is included in the **SMOOTHIE Module** section.

Requirements:
-------------

The following packages are required to run the code successfully:

* numpy
* time
* matplotlib
* scipy
* sklearn
* seaborn

Contents:
---------

.. toctree::
   :maxdepth: 2

   tutorial
   project
   code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
