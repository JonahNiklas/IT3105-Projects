import jax.numpy as jnp
class Controller:

    def __init__(self):
        self.error_history = []

    def calculate_control_signal(self, parameters, error):
        # Calculate control signal U based on the error
        # ...
        pass

class PIDController(Controller):

    def calculate_control_signal(self, parameters, error):
        self.error_history.append(error)
        delta_error = error - self.error_history[-2] if len(self.error_history) > 1 else 0
        input = jnp.array([error, delta_error, sum(self.error_history)])

        return parameters.dot(input)

class NeuralNetController(Controller):
  
    def __init__(self, activation_functions):
        super().__init__()
        self.activation_functions = activation_functions

    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))
      
    def tanh(self, x):
        return jnp.tanh(x)
      
    def relu(self, x):
        return jnp.maximum(0, x)

    def calculate_control_signal(self, parameters, error):
        self.error_history.append(error)
        delta_error = error - self.error_history[-2] if len(self.error_history) > 1 else 0

        if isinstance(self.activation_functions, str):
            self.activation_functions = [self.activation_functions] * len(parameters)
        elif len(self.activation_functions) != len(parameters):
            raise ValueError("Invalid number of activation functions in config")
        
        activation = jnp.array([error, delta_error, sum(self.error_history)]).T
        
        for activation_function, (weight, bias) in zip(self.activation_functions, parameters):
            if activation_function == "sigmoid":
                activation = self.sigmoid(activation.dot(weight) + bias)
                assert activation.shape[0] == 1
            elif activation_function == "tanh":
                activation = self.tanh(activation.dot(weight) + bias)
            elif activation_function == "relu":
                activation = self.relu(activation.dot(weight) + bias)
            else:
                print(activation_function)
                raise ValueError("Invalid activation function in config")
        
        assert activation.shape == (1, 1)
        activation = activation[0][0]
        assert activation.shape == ()
        return activation
