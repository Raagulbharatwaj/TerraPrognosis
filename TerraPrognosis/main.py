import numpy as np
from EnergyBalanceModels.OneDimensionalEBMs import Simple1DimEBM

layers = 5
years = 1
initial_temp = [273.15] * layers
albedos = [0.3, 0.5, 0.6, 0.5, 0.3]
model = Simple1DimEBM(layers, years,albedos)
model.run_simulation()
model.plot_results()








