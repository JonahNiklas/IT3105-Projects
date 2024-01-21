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
    error = target - plant.get_output()
    control_signal = controller.calculate_control_signal(params, error)
    plant.timestep(control_signal)
    return error


def run_one_epoch(params):
    controller = get_controller()
    plant, target = get_plant()
    errors = []
    for _ in range(SIMULATION_TIMESTEPS):
        error = run_one_timestep(params, plant, controller, target)
        errors.append(error)

    mse = jnp.mean(jnp.square(jnp.array(errors)))
    return mse
def get_params():
    if CONTROLLER == "pid":
        params = np.random.uniform(0, 0.1, 3)

    elif CONTROLLER == "neural_net":
        layers = NEURAL_NETWORK["neurons_per_layer"]
        params = []
        sender = 3
        for receiver in layers:
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
            activation_functions=NEURAL_NETWORK["activation_functions"]
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
            q1=np.random.uniform(0, 1, 1),
            q2=np.random.uniform(0, 1, 1),
            noise_range=NOISE_RANGE
        )
        target = COURNOT_COMPETITION["target_profit"]
    # elif config["plant"] == "plant3":
    # plant = plant3()
    else:
        raise ValueError("Invalid plant type in config")

    return plant, target


def main():
    mse_epochs = []
    params = get_params()
    for i in range(TRAINING_EPOCHS):
        print(f"Epoch {i+1}")
        print(f"Params: {params}")
        mse, gradients = jax.value_and_grad(run_one_epoch)(params)
        mse_item = mse.item()
        print(f"MSE: {mse_item}")
        print(f"Gradients: {gradients}")
        mse_epochs.append(mse_item)
        # assert params.shape == gradients.shape
        params = params - gradients * LEARNING_RATE
        print("====================================")
        # Update control signal
        plt.plot(mse_epochs)
        plt.pause(0.001)  # Pause for a short duration to update the plot
    plt.show()


if __name__ == "__main__":
  main()
