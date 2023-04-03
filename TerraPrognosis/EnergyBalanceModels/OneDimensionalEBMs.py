import numpy as np
import matplotlib.pyplot as plt

class Simple1DimEBM:
    def __init__(self,layers,years,albedos,time_zones):
        self.layers    = layers
        self.years     = years
        self.bands     = np.linspace(0, 180, layers)
        self.temps     = np.zeros((layers,np.ceil(years*365).astype(np.int64)))
        self.temp      = np.zeros(layers)
        self.time_zone = time_zones
        self.albedos   = albedos
    
    def local_solar_time(self,day):
        band_width = 180/self.layers
        B = (day-81)*360/365
        E = 9.87*np.sin(np.deg2rad(2*B)) - 7.53*np.cos(np.deg2rad(B)) - 1.5*np.sin(np.deg2rad(B))
        local_solar_times = []
        for i in range(self.layers):
            standard_meridian = 0 + (i + 0.5) * band_width
            LSTM = 15*self.time_zone[i]
            time_correction   = 4*(standard_meridian - LSTM) + E
            local_solar_time  = 12 + (time_correction/60)
            local_solar_times.append(local_solar_time)
        return local_solar_times
    
    def solar_declination(self,day):
        angle  = np.deg2rad((360/365) * (day - 81))
        declination = 23.45 * np.sin(angle)
        declinations = [declination for i in range(self.layers)]
        return declinations
        
    def hour_angle(self,local_solar_times):
        hour_angles = []
        for local_solar_time in local_solar_times:
            hour_angle = 15 * (local_solar_time - 6)
            hour_angles.append(hour_angle)
        return hour_angles
    
    def solar_zenith_angle(self,day):
        local_solar_times = self.local_solar_time(day)
        decliniations     = self.solar_declination(day)
        hour_angles       = self.hour_angle(local_solar_times)
        zenith_angles     = []
        for i in range(self.layers):
            if i == len(self.bands)-1:
                lat = self.bands[i] + 90 /2
            else:
                lat = self.bands[i]+self.bands[i+1]/2
            zenith_angle = np.arccos(np.sin(np.deg2rad(decliniations[i])) * np.sin(np.deg2rad(lat)) + np.cos(np.deg2rad(decliniations[i])) * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(hour_angles[i])))
            zenith_angles.append(np.rad2deg(zenith_angle))
        return zenith_angles
    
    def solar_insolation(self,day):
        zenith_angles = self.solar_zenith_angle(day)
        solar_insolation = []
        for zenith_angle in zenith_angles:
            solar_insolation.append(1367.0 / 4 * np.sin(np.deg2rad(zenith_angle)))
        return solar_insolation
    

    def incoming_solar_radiation(self,day):
        incoming_solar_radiations = []
        solar_insolations = self.solar_insolation(day)
        for i in range(self.layers):
            incoming_solar_radiation = solar_insolations[i] * (1 - self.albedos[i])
            incoming_solar_radiations.append(incoming_solar_radiation)
        return incoming_solar_radiations
    
    def run_simulation(self):
        for year in range(1, self.years+1):
            for day in range(1, 366):
                incoming_radiation = self.incoming_solar_radiation(day)
                for i in range(self.layers):
                    self.temp[i] = ((incoming_radiation[i])/(0.61*5.67e-8))**(1/4)
                self.temps[:,(year-1)*365+day-1] = self.temp
        
    def plot_results(self):
        # ploth the temprature for each layer over time on the same plot
        plt.figure(figsize=(10, 5))
        for i in range(self.layers):
            plt.plot(self.temps[i], label='Layer {}'.format(i+1))
        plt.xlabel('Day')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.show()



    

