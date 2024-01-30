VERBOSE = False

PLANT = "population" # "bathtub" | "cournot" | "population"
CONTROLLER = "pid" # "pid" | "neural_net
PID_PARAM_RANGE = [0, 0.5]
NEURAL_NETWORK = {
    "neurons_per_hidden_layer": [3,3],
    "activation_functions": ["relu","relu"], # "sigmoid" | "tanh" | "relu"|"linear"
    "output_activation_function": "linear", 
    "weight_range": [-1, 1],
    "bias_range": [-0.5, 0.5]
}
TRAINING_EPOCHS = 100
SIMULATION_TIMESTEPS = 20
LEARNING_RATE = 0.01
NOISE_RANGE = [-0.01, 0.01]

BATHTUB = {
    "cross_sectional_area": 100,
    "drain_area": 1,
    "initial_water_height": 100,
    "target_water_height": 100,
}
COURNOT_COMPETITION = {
    "max_price": 4,
    "marginal_cost": 0.1,
    "target_profit": 2,
    "initial_q1": 0.45,
    "initial_q2": 0.5,
}
POPULATION = {
    "initial_population": 100,
    "target_population": 100,
    "birth_rate": 0.05,
    "death_rate": 0.02,
    "carrying_capacity": 150
}