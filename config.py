PLANT = "bathub" # "bathtub" | "cournot" | "population"
CONTROLLER = "neural_net" # "pid" | "neural_net
NEURAL_NETWORK = {
    "num_layers": 3,
    "neurons_per_hidden_layer": [3, 3],
    "activation_functions": ["relu","relu"], # "sigmoid" | "tanh" | "relu"|"linear"
    "output_activation_function": "linear", 
    "weight_range": [-1, 1],
    "bias_range": [-0.5, 0.5]
}
TRAINING_EPOCHS = 20
SIMULATION_TIMESTEPS = 50
LEARNING_RATE = 0.01
NOISE_RANGE = [-0.01, 0.01]
CROSS_SECTIONAL_AREA = {
    "bathtub": 100,
    "drain": 1
}
INITIAL_WATER_HEIGHT = 100
COURNOT_COMPETITION = {
    "max_price": 2,
    "marginal_cost": 0.1,
    "target_profit": 2
}
POPULATION = {
    "initial_population": 100,
    "target_population": 100,
    "birth_rate": 0.05,
    "death_rate": 0.02,
    "carrying_capacity": 150
}