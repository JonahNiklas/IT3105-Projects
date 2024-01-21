PLANT = "bathtub"
CONTROLLER = "pid" # "pid" | "neural_net
NEURAL_NETWORK = {
    "num_layers": 3,
    "neurons_per_layer": [10, 20, 5],
    "activation_functions": "sigmoid", # "sigmoid" | "tanh" | "relu"
    "weight_range": [-1, 1],
    "bias_range": [-0.5, 0.5]
}
TRAINING_EPOCHS = 20
SIMULATION_TIMESTEPS = 100
LEARNING_RATE = 0.001
NOISE_RANGE = [-0.1, 0.1]
CROSS_SECTIONAL_AREA = {
    "bathtub": 100,
    "drain": 1
}
INITIAL_WATER_HEIGHT = 100
COURNOT_COMPETITION = {
    "max_price": 100,
    "marginal_cost": 5,
    "target_profit": 100
}
