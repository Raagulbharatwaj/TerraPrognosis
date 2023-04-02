import numpy as np
from EnergyBalanceModels.ZeroDimensionalEBMs import LayeredZeroDimEBM

#troposphere
troposphere_albedos = np.random.uniform(0.06, 0.12, 100)
troposphere_emissivities = np.random.uniform(0.85, 0.95, 100)

# Stratosphere
stratosphere_albedos = np.random.uniform(0.25, 0.35, 100)
stratosphere_emissivities = np.random.uniform(0.85, 0.95, 100)

# Mesosphere
mesosphere_albedos = np.random.uniform(0.4, 0.5, 100)
mesosphere_emissivities = np.random.uniform(0.85, 0.95, 100)

# Thermosphere
thermosphere_albedos = np.random.uniform(0.6, 0.7, 100)
thermosphere_emissivities = np.random.uniform(0.85, 0.95,100)

albedos = np.stack((troposphere_albedos, stratosphere_albedos, mesosphere_albedos, thermosphere_albedos), axis=0)
emissivities = np.stack((troposphere_emissivities, stratosphere_emissivities, mesosphere_emissivities, thermosphere_emissivities), axis=0)

# Initialize model
curr_temp = 288.0  # K
num_layers = 4
layer_thickness = 1.0  # m

# Create the model object
model = LayeredZeroDimEBM(curr_temp, num_layers, layer_thickness)

temperature, delta_t = model.run_model(albedos, emissivities)

# Plot the results
model.plot_results(temperature, delta_t)
