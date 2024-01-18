class Controller:
    def calculate_control_signal(self, error):
        # Calculate control signal U based on the error
        # ...
        pass

class PIDController(Controller):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_history = []

    def calculate_control_signal(self, error):
        self.error_history.append(error)
        delta_error = error - self.error_history[-2] if len(self.error_history) > 1 else 0
        return error * self.kp + delta_error * self.kd + sum(self.error_history) * self.ki

class NeuralNetController(Controller):
    def __init__(self, num_layers, neurons_per_layer, activation_functions, weight_range, bias_range):
        self.num_layers = num_layers
        if(len(neurons_per_layer) != num_layers):
            raise ValueError("Number of layers and neurons per layer must be equal")
        if(len(activation_functions) != num_layers):
            raise ValueError("Number of layers and activation functions must be equal")
        self.neurons_per_layer = neurons_per_layer
        self.activation_functions = activation_functions
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.weights = []
        self.biases = []
        

    def calculate_control_signal(self, error):
        # Calculate control signal U based on the error
        # ...
        pass
