import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SimpleZeroDimEBM:
    def __init__(self)->None:
        self.albedo           = np.float64(0.3)
        self.emissivity       = np.float64(0.61)
        self.solar_constant   = np.float64(1361.0)
        self.stefan_boltzmann = np.float64(5.67e-8)
        print(f"{'Simple Zero-Dimensional EBM':^50}")
        print(f"{'-'*50}")
        print(f"Energy balance equation: Incoming solar radiation = Outgoing longwave radiation")
        print(f"{'(1 - alpha) S / 4 = (sigma) ε T^4':^50}")
        print(f"where:")
        print(f"{'alpha':<10} is the Earth's albedo, or reflectivity ({self.albedo})")
        print(f"{'S':<10} is the solar constant, or the amount of solar radiation received by the Earth's surface ({self.solar_constant} W/m^2)")
        print(f"{'sigma':<10} is the Stefan-Boltzmann constant, which relates the temperature of an object to the amount of radiation it emits ({self.stefan_boltzmann} W/m^2K^4)")
        print(f"{'ε':<10} is the emissivity of the Earth's atmosphere, or its ability to emit longwave radiation ({self.emissivity})")
        print(f"{'T':<10} is the average temperature of the Earth's surface")
        print(f"{'Solving for T, we get:':^50}")
        print(f"T = [ (1 - alpha) S / 4(sigma)ε ]^0.25\n")
        print(f"{'-'*50}")

    def incoming_solar_radiation(self)->np.float64:
        return np.float64(((1 - self.albedo) * self.solar_constant)/4)
    
    def estimate_temperature(self)->np.float64:
        return np.float64(((self.incoming_solar_radiation() / (self.stefan_boltzmann * self.emissivity))**(1/4)))
    
    def outgoing_longwave_radiation(self)->np.float64:
        return np.float64(self.stefan_boltzmann * self.emissivity * (self.estimate_temperature()**4))
    
    def visualize(self)->None:
        # create plot for albedo vs incoming solar radiation
        fig = plt.figure(figsize=(24,24))
        # create subplot for incoming solar radiation vs albedo
        ax1 = fig.add_subplot(2, 2, 1)
        albedo = np.linspace(0, 1, 100)
        incoming_solar_radiation = ((1-albedo) * self.solar_constant / 4).astype(np.float64)
        sns.lineplot(x=albedo, y=incoming_solar_radiation, ax=ax1)
        ax1.set_xlabel("Albedo")
        ax1.set_ylabel("ISR(W/m^2)")
        ax1.set_title("Incoming Solar Radiation vs Albedo")

        # create subplot for emissivity vs outgoing longwave radiation
        ax2 = fig.add_subplot(2, 2, 2)
        emissivity = np.linspace(0, 1, 100)
        outgoing_longwave_radiation = (self.stefan_boltzmann * emissivity * ((((1-albedo) * self.solar_constant / (4 * self.stefan_boltzmann * self.emissivity))**(1/4))**4)).astype(np.float64)
        sns.lineplot(x=emissivity, y=outgoing_longwave_radiation, ax=ax2)
        ax2.set_xlabel("Emissivity")
        ax2.set_ylabel("OLR(W/m^2)")
        ax2.set_title("Outgoing Longwave Radiation vs Emissivity")

        # create subplot for albedo vs temperature
        ax3 = fig.add_subplot(2, 2, 3)
        temperature = (((1-albedo) * self.solar_constant / (4 * self.stefan_boltzmann * self.emissivity))**(1/4)).astype(np.float64)
        sns.lineplot(x=albedo, y=temperature, ax=ax3)
        ax3.set_xlabel("Albedo")
        ax3.set_ylabel("Temperature (K)")
        ax3.set_title("Temperature vs Albedo")

        # create subplot for emissivity vs temperature
        ax4 = fig.add_subplot(2,2,4)
        temperature = ((self.solar_constant / (4 * self.stefan_boltzmann * emissivity))**(1/4)).astype(np.float64)
        sns.lineplot(x=emissivity, y=temperature, ax=ax4)
        ax4.set_xlabel("Emissivity")
        ax4.set_ylabel("Temperature (K)")
        ax4.set_title("Temperature vs Emissivity")

        # adjust spacing between subplots
        plt.subplots_adjust(wspace=1.5,hspace=0.5)

        # display plot
        plt.show()



class NonLayeredZeroDimEBM:
    def __init__(self):
        self.sigma = 5.67e-8
        self.transmissivity = 0.6114139923607016
        self.Q = 341.3 
        self.albedo = 0.3
        self.C = 4.0e8
    def OutgoingLongwaveRadiation(self,T_surface, tau=0.6114139923607016):
        return self.sigma * tau * T_surface**4
    
    def AbsorbedShortwaveRadiation(self,albedo=0.3):
        return self.Q*(1-albedo)
    
    def step_forward(self,T,alb,emmi):
        dt = 60*60*24
        return T + dt / self.C * ( self.AbsorbedShortwaveRadiation(alb) - self.OutgoingLongwaveRadiation(T, emmi))
    
    def run(self,albedos,emmisivites):
        timesteps = len(albedos)
        Tsteps = np.zeros(timesteps+1)
        Years  = np.zeros(timesteps+1)
        Tsteps[0] = 288.0
        for i in range(timesteps):
            Tsteps[i+1] = self.step_forward(Tsteps[i],albedos[i],emmisivites[i])
            Years[i+1] = Years[i] + 1
        Tsteps = np.delete(Tsteps,0)
        Years  = np.delete(Years,0)
        return (Tsteps,Years)
    

    def plot_results(self, temp_list, time_list, emissivity_list, albedo_list):
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        sns.lineplot(x=time_list, y=temp_list, ax=ax1)
        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Temperature (K)")
        ax1.set_title("Temperature vs Time")
        ax2 = fig.add_subplot(2, 2, 2)
        sns.lineplot(x=time_list, y=emissivity_list, ax=ax2)
        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("Emissivity")
        ax2.set_title("Emissivity vs Time")
        ax3 = fig.add_subplot(2, 2, 3)
        sns.lineplot(x=time_list, y=albedo_list, ax=ax3)
        ax3.set_xlabel("Time (years)")
        ax3.set_ylabel("Albedo")
        ax3.set_title("Albedo vs Time")
        plt.subplots_adjust(wspace=1.5,hspace=0.5)
        plt.show()


class LayeredZeroDimEBM:
    def __init__(self, curr_temp, num_layers, layer_thickness, emissivity_layer, albedo_layer):
        self.stefan_boltzmann = 5.67e-8
        self.transmissivity = 0.6114139923607016
        self.Q = 341.3 
        self.albedo = 0.3
        self.C = 4.0e8
        self.curr_temp = np.float64(curr_temp) # Current temperature of the Earth (K)
        self.num_layers = num_layers # Number of atmospheric layers
        self.layer_thickness = np.float64(layer_thickness) # Thickness of each layer (m)
        self.emissivity_layer = np.array(emissivity_layer) # Emissivity of each atmospheric layer
        self.albedo_layer = np.array(albedo_layer) # Albedo of each atmospheric layer
        print(f"{'Zero Dimensional Layered EBM':^50}")
        print(f"{'-'*50}")
        print(f"This model simulates changes in temperature over time based on changes in Earth's albedo and emissivity, as well as other parameters. The energy balance equation used is:")
        print(f"{'(1 - albedo) S / 4 = (sigma) epsilon_l(t) T_l^4 + delta_Q':^50}")
        print(f"where:")
        print(f"{'albedo':<15} is the Earth's albedo, or reflectivity")
        print(f"{'S':<15} is the solar constant, or the amount of solar radiation received by the Earth's surface ({self.Q} W/m^2)")
        print(f"{'sigma':<15} is the Stefan-Boltzmann constant, which relates the temperature of an object to the amount of radiation it emits ({self.stefan_boltzmann} W/m^2K^4)")
        print(f"{'epsilon_l(t)':<15} is the emissivity of each atmospheric layer l, which changes over time due to the continuous emission of greenhouse gases")
        print(f"{'T_l':<15} is the temperature of each atmospheric layer l at a given time")
        print(f"{'delta_Q':<15} is the change in heat energy over time, calculated as incoming solar radiation minus outgoing longwave radiation")
        print(f"\nTo simulate changes in temperature over time, the following steps are taken for each time step:")
        print(f"1. Compute the emissivity of each atmospheric layer at the current time (epsilon_l(t)) based on the initial emissivity and a constant rate of change (k)")
        print(f"2. Compute the change in heat energy (delta_Q) based on the current albedo, temperature, and emissivity of each layer")
        print(f"3. Compute the change in temperature (delta_T) based on the change in heat energy and the heat capacity and mass of the Earth")
        print(f"4. Compute the new temperature based on the old temperature and the change in temperature")
        print(f"5. Repeat for the desired number of years")
        print(f"\nNote: The initial emissivity and rate of change can be specified when creating an instance of the ClimateModel class.")
        print(f"{'-'*50}")
    
    def compute_delta_q(self, albedo, emissivity):
        """Compute the change in heat energy (delta_Q) for each atmospheric layer at the current time step."""
        solar_radiation = (1 - albedo) * self.Q
        lw_radiation = emissivity * self.stefan_boltzmann * self.curr_temp ** 4
        delta_q = solar_radiation - lw_radiation
        return delta_q

    def compute_delta_t(self, delta_q):
        """Compute the change in temperature (delta_T) for the current time step."""
        dt = 60*60*24
        delta_t = delta_q * dt / self.C
        return delta_t

    def update_temperature(self, delta_t):
        """Update the temperature for the current time step."""
        self.curr_temp += delta_t
        return self.curr_temp

    def run_model(self, years, albedo_initial, emissivity_initial, emissivity_rate):
        """Run the zero-dimensional layered energy balance model for the specified number of years."""
        time_steps_per_year = 12 # simulate changes each month
        num_time_steps = years * time_steps_per_year
        albedo = np.full(self.num_layers, albedo_initial)
        emissivity = np.full((self.num_layers, num_time_steps+1), emissivity_initial)
        emissivity[:,0] = self.emissivity_layer
        delta_t = np.zeros((self.num_layers, num_time_steps+1))
        temperature = np.zeros((self.num_layers, num_time_steps+1))
        temperature[:,0] = self.curr_temp
        
        for t in range(num_time_steps):
            # Compute the emissivity of each atmospheric layer at the current time
            emissivity[:,t+1] = emissivity[:,t] + emissivity_rate
            
            # Compute the change in heat energy (delta_Q) for each atmospheric layer at the current time step
            delta_q = np.zeros(self.num_layers)
            for l in range(self.num_layers):
                delta_q[l] = self.compute_delta_q(albedo[l], emissivity[l,t+1])
            
            # Compute the change in temperature (delta_T) for each atmospheric layer at the current time step
            for l in range(self.num_layers):
                delta_t[l,t+1] = self.compute_delta_t(delta_q[l])
            
            # Update the temperature for each atmospheric layer at the current time step
            for l in range(self.num_layers):
                temperature[l,t+1] = self.update_temperature(delta_t[l,t+1])
        
        return temperature, delta_t, emissivity
    
    def plot_results(self, temperature, delta_t, emissivity):
        """Plot the results of the model."""
        time_steps_per_year = 12
        years = temperature.shape[1] // time_steps_per_year 
        time = np.arange(0, years+1)
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(time, temperature[0,:years+1], label="Troposphere")
        ax[0].plot(time, temperature[1,:years+1], label="Stratosphere")
        ax[0].plot(time, temperature[2,:years+1], label="Mesosphere")
        ax[0].plot(time, temperature[3,:years+1], label="Thermosphere")
        ax[0].set_xlabel("Time (years)")
        ax[0].set_ylabel("Temperature (K)")
        ax[0].set_title("Temperature of Each Atmospheric Layer")
        ax[0].legend()
        ax[1].plot(time, delta_t[0,:years+1], label="Troposphere")
        ax[1].plot(time, delta_t[1,:years+1], label="Stratosphere")
        ax[1].plot(time, delta_t[2,:years+1], label="Mesosphere")
        ax[1].plot(time, delta_t[3,:years+1], label="Thermosphere")
        ax[1].set_xlabel("Time (years)")
        ax[1].set_ylabel("Temperature Change (K)")
        ax[1].set_title("Change in Temperature of Each Atmospheric Layer")
        ax[1].legend()
        plt.show()


