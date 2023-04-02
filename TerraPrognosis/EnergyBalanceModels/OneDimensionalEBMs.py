import numpy as np
import matplotlib.pyplot as plt

class Simple1DimEBM:
    def __init__(self,layers,years,albedos):
        self.layers   = layers
        self.years    = years
        self.bands    = np.linspace(-90, 90, layers)
        self.temps    = np.zeros((layers,years*365+1))
        self.temp     = np.zeros(layers)
        self.albedos  = albedos
    
    def local_solar_time(self,day):
        band_width = 180/self.layers
        n = day
        B = (n-1)*360/365
        E = 229.18 * (0.000075 + 0.001868 * np.cos(np.radians(B)) - 0.032077 * np.sin(np.radians(B)) - 0.014615 * np.cos(np.radians(2 * B)) - 0.040849 * np.sin(np.radians(2 * B)))
        local_solar_times = []
        for i in range(self.layers):
            standard_meridian = -180 + (i + 0.5) * band_width
            local_solar_time = 12 + (4 * (0 - standard_meridian) - E) / 60
            local_solar_times.append(local_solar_time)
        return local_solar_times
    
    def solar_declination(self,day):
        angle  = np.deg2rad(360/365 * (day - 81))
        decliniations = []
        for band in self.bands:
            declination = np.arcsin(np.sin(np.deg2rad(band))*np.sin(angle))
            decliniations.append(np.rad2deg(declination))
        return decliniations

    def hour_angle(self,local_solar_times):
        hour_angles = []
        for local_solar_time in local_solar_times:
            hour_angle = 15 * (local_solar_time - 12)
            hour_angles.append(hour_angle)
        return hour_angles
    
    def solar_zenith_angle(self,day):
        local_solar_times = self.local_solar_time(day)
        decliniations     = self.solar_declination(day)
        hour_angles       = self.hour_angle(local_solar_times)
        zenith_angles     = []
        for i in range(self.layers):
            zenith_angle = np.arccos(np.sin(np.deg2rad(decliniations[i]))*np.sin(np.deg2rad(self.bands[i])) + np.cos(np.deg2rad(decliniations[i]))*np.cos(np.deg2rad(self.bands[i]))*np.cos(np.deg2rad(hour_angles[i])))
            zenith_angles.append(np.rad2deg(zenith_angle))
        return zenith_angles
    
    def solar_insolation(self,day):
        zenith_angles = self.solar_zenith_angle(day)
        solar_insolation = []
        for zenith_angle in zenith_angles:
            solar_insolation.append(1367.0 * np.cos(np.deg2rad(zenith_angle)))
        return solar_insolation
    
    def outgoing_longwave_radiation(self,year,day):
        index = (year-1)*365 + day
        outgoing_longwave_radiations = []
        sigma = 5.67e-8
        emissivity = 0.61
        for i in range(self.layers):
            outgoing_longwave_radiation = sigma * emissivity * np.power(self.temps[i,index-1],4)
            outgoing_longwave_radiations.append(outgoing_longwave_radiation)
        return outgoing_longwave_radiations

    def incoming_solar_radiation(self,year,day):
        incoming_solar_radiations = []
        solar_insolations = self.solar_insolation(day)
        for i in range(self.layers):
            incoming_solar_radiation = solar_insolations[i] * (1 - self.albedos[i])
            incoming_solar_radiations.append(incoming_solar_radiation)
        return incoming_solar_radiations
    
    def run_simulation(self):
        for year in range(1, self.years+1):
            for day in range(1, 366):
                incoming_radiation = self.incoming_solar_radiation(year, day)
                for i in range(self.layers):
                    self.temp[i] = np.float64((incoming_radiation[i]*(0.61)*(5.67e-8))**(1/4))
                self.temps[:,(year-1)*365+day] = self.temp
        
    def plot_results(self):
        # ploth the temprature for each layer over time on the same plot
        plt.figure(figsize=(10, 5))
        for i in range(self.layers):
            plt.plot(self.temps[i], label='Layer {}'.format(i+1))
        plt.xlabel('Day')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.show()



    

