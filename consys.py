import json
from controllers import Controller, PIDController, NeuralNetController
from plants import BathTubPlant, CournotPlant
import matplotlib.pyplot as plt
import jax
import numpy as np
import jax.numpy as jnp
from config import (
    CONTROLLER,
    COURNOT_COMPETITION,
    CROSS_SECTIONAL_AREA,
    INITIAL_WATER_HEIGHT,
    LEARNING_RATE,
    NEURAL_NETWORK,
    NOISE_RANGE,
    PLANT,
    SIMULATION_TIMESTEPS,
    TRAINING_EPOCHS,
)


def run_one_timestep(params, plant, controller: Controller, target):
    error = jnp.abs(target - plant.get_output())
    control_signal = controller.calculate_control_signal(params, error)
    plant.timestep(control_signal)
    return plant.get_output(), control_signal


def run_one_epoch(params):
    controller = get_controller()
    plant, target = get_plant()
    output = []
    control_signal = []
    for _ in range(SIMULATION_TIMESTEPS):
        output_t, control_signal_t = run_one_timestep(params, plant, controller, target)
        output.append(output_t)
        control_signal.append(control_signal_t)

    mse = jnp.mean((target - jnp.array(output)) ** 2)

    return mse


def get_params():
    if CONTROLLER == "pid":
        params = np.random.uniform(0, 1, 3)

    elif CONTROLLER == "neural_net":
        layers = NEURAL_NETWORK
        params = []
        sender = layers[0]
        for receiver in layers[1:]:
            weights = np.random.uniform(
                NEURAL_NETWORK["weightRange"][0],
                NEURAL_NETWORK["weightRange"][1],
                (sender, receiver),
            )
            biases = np.random.uniform(
                NEURAL_NETWORK["biasRange"][0],
                NEURAL_NETWORK["biasRange"][1],
                (1, receiver),
            )
            sender = receiver
            params.append([weights, biases])
    else:
        raise ValueError("Invalid controller type in config")

    return params


def get_controller():
    if CONTROLLER == "pid":
        controller = PIDController()

    elif CONTROLLER == "neural_net":
        controller = NeuralNetController(
            activation_functions=NEURAL_NETWORK["activationFunctions"]
        )
    else:
        raise ValueError("Invalid controller type in config")

    return controller


def get_plant():
    if PLANT == "bathtub":
        plant = BathTubPlant(
            area=CROSS_SECTIONAL_AREA["bathtub"],
            drain_area=CROSS_SECTIONAL_AREA["drain"],
            noise_range=NOISE_RANGE,
            water_level=INITIAL_WATER_HEIGHT,
        )
        target = INITIAL_WATER_HEIGHT
    elif PLANT == "cournot":
        plant = CournotPlant(
            max_price=COURNOT_COMPETITION["max_price"],
            marginal_cost=COURNOT_COMPETITION["marginal_cost"],
            # q1=INI,
            # q2=config["initialQ2"],
            # noise_range=config["noise_range"],
        )
        target = COURNOT_COMPETITION["target_profit"]
    # elif config["plant"] == "plant3":
    # plant = plant3()
    else:
        raise ValueError("Invalid plant type in config")

    return plant, target


def main():
    errors = []
    params = get_params()
    for i in range(TRAINING_EPOCHS):
        print(f"Epoch {i}")
        mse, gradients = jax.value_and_grad(run_one_epoch)(params)
        errors.append(mse)
        params = params - gradients * LEARNING_RATE

        # Update contrrol signal
    plt.plot(errors)
    plt.show()


if __name__ == "__main__":
    main()
