PLANT = "bathtub" # "bathtub" | "cournot"
CONTROLLER = "neural_net" # "pid" | "neural_net
NEURAL_NETWORK = {
    "num_layers": 3,
    "neurons_per_layer": [5, 2, 1],
    "activation_functions": "sigmoid", # "sigmoid" | "tanh" | "relu"
    "weight_range": [-1, 1],
    "bias_range": [-0.5, 0.5]
}
TRAINING_EPOCHS = 20
SIMULATION_TIMESTEPS = 100
LEARNING_RATE = 0.01
NOISE_RANGE = [-0.1, 0.1]
CROSS_SECTIONAL_AREA = {
    "bathtub": 100,
    "drain": 1
}
INITIAL_WATER_HEIGHT = 100
COURNOT_COMPETITION = {
    "max_price": 1,
    "marginal_cost": 0.1,
    "target_profit": 1
}
