class Controller:

    def __init__(self):
        self.error_history = []

    def set_parameters(self, parameters):
        # Set the parameters of the controller
        # ...
        pass

    def calculate_control_signal(self, error) -> float:
        # Calculate control signal U based on the error
        # ...
        pass

class PIDController(Controller):
    def __init__(self, kp: float, ki: float, kd: float):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_parameters(self, parameters):
        self.kp = parameters[0]
        self.ki = parameters[1]
        self.kd = parameters[2]

    def calculate_control_signal(self, error):
        self.error_history.append(error)
        delta_error = error - self.error_history[-2] if len(self.error_history) > 1 else 0
        return error * self.kp + delta_error * self.kd + sum(self.error_history) * self.ki

class NeuralNetController(Controller):
    def __init__(self, parameters, activation_functions):
        super().__init__()
        self.parameters = parameters
        self.activation_functions = activation_functions

    def set_parameters(self, parameters):
        self.parameters = parameters

    def calculate_control_signal(self, error):
        self.error_history.append(error)
        raise NotImplementedError("Implement this method")
