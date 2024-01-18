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
    def __init__(self, area, drain_area, noise_range,water_level):
        self.area = area
        self.drain_area = drain_area
        self.noise_range = noise_range
        self.water_level = water_level

    def get_output(self):
        # could also use derivative of water level
        return self.water_level

    def timestep(self, control_signal):
        velocity = jnp.sqrt(2*9.81*self.water_level)
        disturbance = np.random.uniform(self.noise_range[0],self.noise_range[1])
        self.water_level += control_signal + disturbance -self.drain_area*velocity
    
class CournotPlant(Plant):
    def __init__(self, max_price, marginal_cost, q1,q2, noise_range):
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        self.q1 = q1
        self.q2 = q2

    def price(self):
        return self.max_price - self.q1 - self.q2

    def get_output(self):
        return self.q1 * (self.price()-self.marginal_cost)

    def timestep(self, control_signal):
        self.q1 += control_signal
        self.q2 += np.random.uniform(self.noise_range[0],self.noise_range[1])


