class Controller:

    def __init__(self):
        self.error_history = []

    def calculate_control_signal(self, parameters, error):
        # Calculate control signal U based on the error
        # ...
        pass

class PIDController(Controller):

    def calculate_control_signal(self, parameters, error):
        kp, kd, ki = parameters
        self.error_history.append(error)
        delta_error = error - self.error_history[-2] if len(self.error_history) > 1 else 0
        return error * parameters[0] + delta_error * parameters[1] + sum(self.error_history) * parameters[2]

class NeuralNetController(Controller):

    def calculate_control_signal(self, parameters, error, activation_functions):
        self.error_history.append(error)
        raise NotImplementedError("Implement this method")
