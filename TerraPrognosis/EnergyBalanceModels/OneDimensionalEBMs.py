import numpy as np
import matplotlib.pyplot as plt

class OneDimEBM:
    def __init__(self, n_bands=10, n_years=100):
        self.sigma = 5.67e-8    # Stefan-Boltzmann constant
        self.solar_constant = 1366    # Solar constant
        self.specific_heat = 4.0e7    # Specific heat of the atmosphere
        self.diffusivity = 0.5556    # Thermal diffusivity of the atmosphere
        self.timestep = 60*60*24    # Time step in seconds
        self.n_bands = n_bands    # Number of latitudinal bands
        self.n_years = n_years    # Number of years to simulate
        
        # Initialize state variables
        self.temperature = np.zeros((n_years+1, n_bands))
        self.albedo = np.zeros((n_years+1, n_bands))
        self.emissivity = np.zeros((n_years+1, n_bands))
        self.insolation = np.zeros((n_years+1, n_bands))
        self.time = np.arange(n_years+1)
        
        # Set initial conditions
        self.temperature[0,:] = 288    # Surface temperature in K
        self.albedo[0,:] = 0.3    # Albedo
        self.emissivity[0,:] = 0.9    # Emissivity
        
    def solar_declination_angle(self, day):
        return np.radians(23.45) * np.sin(np.radians(360/365.25*(day-81)))
    
    def hour_angle(self, time):
        return np.radians(360/24*(time-12))
    
    def calculate_insolation(self, day, time, lat):
        declination = self.solar_declination_angle(day)
        angle_of_incidence = np.arcsin(np.sin(np.radians(lat))*np.sin(declination) + np.cos(np.radians(lat))*np.cos(declination)*np.cos(self.hour_angle(time)))
        return self.solar_constant * np.sin(angle_of_incidence)
    
    def step_forward(self, day, lat):
        # Calculate insolation
        time = np.arange(0, 24, 24/self.n_bands)
        for i in range(self.n_bands):
            self.insolation[day,i] = self.calculate_insolation(day, time[i], lat[i])
            
        # Calculate net energy flux
        flux_in = self.insolation[day,:] * (1 - self.albedo[day,:])
        flux_out = self.emissivity[day,:] * self.sigma * self.temperature[day,:]**4
        net_flux = flux_in - flux_out
        
        # Update temperature
        delta_temp = self.timestep / (self.specific_heat * self.diffusivity) * (net_flux[1:] - net_flux[:-1])
        self.temperature[day+1,1:-1] = self.temperature[day,1:-1] + delta_temp
        
        # Update albedo and emissivity
        self.albedo[day+1,:] = np.where(self.temperature[day,:] < 273.15, 0.7, 0.3)
        self.emissivity[day+1,:] = np.where(self.temperature[day,:] < 273.15, 0.95, 0.9)
        
        # Set boundary conditions
        self.temperature[day+1,0] = self.temperature[day,0] + self.timestep * (self.insolation[day,0] * (1 - self.albedo[day,0]) - self.emissivity[day,0] * self.sigma * self.temperature[day,0]**4) / (self.specific_heat * self.diffusivity)
        self.temperature[day+1,-1] = self.temperature[day,-1] + self.timestep * (self.insolation[day,-1] * (1 - self.albedo[day,-1]) - self.emissivity[day,-1] * self.sigma * self.temperature[day,-1]**4) / (self.specific_heat * self.diffusivity)
    
    def run_simulation(self):
        for i in range(self.n_years):
            for j in range(self.n_bands):
                self.step_forward(i, j)

    def plot_results(self):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].plot(self.time, self.temperature[:,0], label='Pole')
        ax[0].plot(self.time, self.temperature[:,-1], label='Equator')
        ax[0].set_xlabel('Time (years)')
        ax[0].set_ylabel('Temperature (K)')
        ax[0].legend()
        
        ax[1].plot(self.time, self.albedo[:,0], label='Pole')
        ax[1].plot(self.time, self.albedo[:,-1], label='Equator')
        ax[1].set_xlabel('Time (years)')
        ax[1].set_ylabel('Albedo')
        ax[1].legend()
        
        ax[2].plot(self.time, self.emissivity[:,0], label='Pole')
        ax[2].plot(self.time, self.emissivity[:,-1], label='Equator')
        ax[2].set_xlabel('Time (years)')
        ax[2].set_ylabel('Emissivity')
        ax[2].legend()
        
        plt.show()

