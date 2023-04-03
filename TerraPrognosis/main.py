import numpy as np
from EnergyBalanceModels.OneDimensionalEBMs import Simple1DimEBM

layers = 5
years = 1
initial_temp = [273.15] * layers
albedos = [0.4, 0.6, 0.7,0.8,0.86]
time_zones = [0, 3, 6, 9, 12]
model = Simple1DimEBM(layers, years,albedos,time_zones)


model.run_simulation()
model.plot_results()



'''
    def outgoing_longwave_radiation(self,year,day):
        index = (year-1)*365 + day
        outgoing_longwave_radiations = []
        sigma = 5.67e-8
        emissivity = 0.61
        for i in range(self.layers):
            outgoing_longwave_radiation = sigma * emissivity * np.power(self.temps[i,index-1],4)
            outgoing_longwave_radiations.append(outgoing_longwave_radiation)
        return outgoing_longwave_radiations'''