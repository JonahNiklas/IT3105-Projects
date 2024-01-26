from project1.config import (
    BATHTUB,
    CONTROLLER,
    COURNOT_COMPETITION,    
    NEURAL_NETWORK,
    NOISE_RANGE,
    PID_PARAM_RANGE,
    PLANT,
    POPULATION,
)
import numpy as np

from project1.controllers import NeuralNetController, PIDController
from project1.plants import BathTubPlant, CournotPlant, PopulationDynamicsPlant


def get_params():
    if CONTROLLER == "pid":
        params = np.random.uniform(PID_PARAM_RANGE[0], PID_PARAM_RANGE[1], 3)

    elif CONTROLLER == "neural_net":
        hidden_layers = NEURAL_NETWORK["neurons_per_hidden_layer"]
        params = []
        sender = 3
        for receiver in hidden_layers+[1]:
            weights = np.random.uniform(
                NEURAL_NETWORK["weight_range"][0],
                NEURAL_NETWORK["weight_range"][1],
                (sender, receiver),
            )
            biases = np.random.uniform(
                NEURAL_NETWORK["bias_range"][0],
                NEURAL_NETWORK["bias_range"][1],
                (1, receiver),
            )
            sender = receiver
            params.append((weights, biases))
    else:
        raise ValueError("Invalid controller type in config")
    return params


def get_controller():
    if CONTROLLER == "pid":
        controller = PIDController()

    elif CONTROLLER == "neural_net":
        controller = NeuralNetController(
            activation_functions=NEURAL_NETWORK["activation_functions"],
            output_activation_function=NEURAL_NETWORK["output_activation_function"],
        )
    else:
        raise ValueError("Invalid controller type in config")

    return controller


def get_plant():
    if PLANT == "bathtub":
        plant = BathTubPlant(
            area=BATHTUB["cross_sectional_area"],
            drain_area=BATHTUB["drain_area"],
            noise_range=NOISE_RANGE,
            water_level=BATHTUB["initial_water_height"],
        )
        target = BATHTUB["target_water_height"]
    elif PLANT == "cournot":
        plant = CournotPlant(
            max_price=COURNOT_COMPETITION["max_price"],
            marginal_cost=COURNOT_COMPETITION["marginal_cost"],
            initial_q1=COURNOT_COMPETITION["initial_q1"],
            initial_q2=COURNOT_COMPETITION["initial_q2"],
            noise_range=NOISE_RANGE
        )
        target = COURNOT_COMPETITION["target_profit"]
    elif PLANT == "population":
        plant = PopulationDynamicsPlant(
            initial_population=POPULATION["initial_population"],
            birth_rate=POPULATION["birth_rate"],
            death_rate=POPULATION["death_rate"],
            carrying_capacity=POPULATION["carrying_capacity"],
            noise_range=NOISE_RANGE
        )
        target = POPULATION["target_population"]
    else:
        raise ValueError("Invalid plant type in config")

    return plant, target