import numpy as np
import jax.numpy as jnp


class Plant:
    def get_output(self):
        # TODO: Implement the logic to calculate and return the output Y
        pass

    def timestep(self, control_signal):
        # TODO: Implement the logic to control the plant using the control signal U
        pass


class BathTubPlant(Plant):
    def __init__(self, area, drain_area, noise_range, water_level):
        self.area = area
        self.drain_area = drain_area
        self.noise_range = noise_range
        self.water_level = (
            water_level  # + np.random.uniform(self.noise_range[0],self.noise_range[1])
        )

    def get_output(self):
        # could also use derivative of water level
        return self.water_level

    def timestep(self, control_signal):
        if self.water_level < 0:
            print("Water level is negative, resetting to 0")
            self.water_level = 0
        velocity = jnp.sqrt(2 * 9.81 * self.water_level)
        disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])

        self.water_level += (
            control_signal + disturbance - self.drain_area * velocity
        ) / self.area


class CournotPlant(Plant):
    def __init__(self, max_price, marginal_cost, initial_q1_q2, noise_range):
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        self.q1 = initial_q1_q2[0]
        self.q2 = initial_q1_q2[1]
        self.noise_range = noise_range

    def price(self):
        return self.max_price - self.q1 - self.q2

    def get_output(self):
        return self.q1 * (self.price() - self.marginal_cost)

    def timestep(self, control_signal):
        def bound(x):
            return jnp.minimum(jnp.maximum(x, 0), 1)

        self.q1 = bound(self.q1 + control_signal)
        self.q2 = bound(
            self.q2 + np.random.uniform(self.noise_range[0], self.noise_range[1])
        )


class PopulationDynamicsPlant(Plant):
    def __init__(self, initial_population, birth_rate, death_rate, carrying_capacity, noise_range):
        self.population = initial_population
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.carrying_capacity = carrying_capacity
        self.noise_range = noise_range

    def timestep(self, control_signal):
        crowding_factor = (1-self.population / self.carrying_capacity)
        disturbance = np.random.uniform(self.noise_range[0], self.noise_range[1])
        growth = (self.birth_rate*crowding_factor - self.death_rate)*self.population
        self.population = jnp.maximum(0,self.population + disturbance + growth + control_signal)

    def get_output(self):
        return self.population