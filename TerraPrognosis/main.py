from EnergyBalanceModels.ZeroDimensionalEBMs import  LayeredZeroDimEBM
layer_thickness = 5.0 # Thickness of each layer in meters
num_layers = 4 # Number of atmospheric layers

# Emissivity and albedo of each layer
emissivity_layer = [0.6, 0.7, 0.8, 0.9]
albedo_layer = [0.3, 0.2, 0.1, 0.05]

ebm = LayeredZeroDimEBM(curr_temp=288.0, num_layers=num_layers, layer_thickness=layer_thickness,
                                    emissivity_layer=emissivity_layer, albedo_layer=albedo_layer)

years = 10
albedo_initial = 0.3
emissivity_initial = 0.7
emissivity_rate = 0.001

temperature, delta_t, emissivity = ebm.run_model(years, albedo_initial, emissivity_initial, emissivity_rate)

ebm.plot_results(temperature, delta_t, emissivity)













