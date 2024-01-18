import json
from controllers import Controller, PIDController, NeuralNetController
from plants import BathTubPlant, CournotPlant
import matplotlib.pyplot as plt
import jax
import numpy as np
import jax.numpy as jnp

def run_one_timestep(plant, controller: Controller, target):
    error = target - plant.get_output()
    control_signal = controller.calculate_control_signal(error)
    plant.timestep(control_signal)
    return plant.get_output(), control_signal


def run_one_epoch(params, plant, controller, target, num_timesteps):
    controller.set_parameters(params)
    output = []
    control_signal = []
    for _ in range(num_timesteps):
        output_t, control_signal_t = run_one_timestep(plant, controller, target)
        output.append(output_t)
        control_signal.append(control_signal_t)
    # mse = sum([(target - y) ** 2 for y in output]) / len(output)
    mse = jnp.mean((target - jnp.array(output)) ** 2)

    return mse


def load_config():
    with open("pivotal_parameters.json") as file:
        config = json.load(file)

    if config["controller"] == "pid":
        params = np.random.uniform(-0.1, 0.1, 3)
        controller = PIDController(params[0], params[1], params[2])

    elif config["controller"] == "neural_net":
        neural_net_config = config["neuralNetwork"]
        layers = neural_net_config["neuronsPerLayer"]
        params = []
        sender = layers[0]
        for receiver in layers[1:]:
            weights = np.random.uniform(
                neural_net_config["weightRange"][0],
                neural_net_config["weightRange"][1],
                (sender, receiver),
            )
            biases = np.random.uniform(
                neural_net_config["biasRange"][0],
                neural_net_config["biasRange"][1],
                (1, receiver),
            )
            sender = receiver
            params.append([weights, biases])
        controller = NeuralNetController(
            params, activation_functions=neural_net_config["activationFunctions"]
        )
    else:
        raise ValueError("Invalid controller type in config")

    if config["plant"] == "bathtub":
        plant = BathTubPlant(
            area=config["crossSectionalArea"]["bathtub"],
            drain_area=config["crossSectionalArea"]["drain"],
            noise_range=config["noiseRange"],
            water_level=config["initialWaterHeight"],
        )
        target = config["initialWaterHeight"]
    elif config["plant"] == "cournot":
        plant = CournotPlant(
            max_price=config["cournotCompetetion"]["maxPrice"],
            marginal_cost=config["cournotCompetition"]["marginalCost"],
            q1=config["initialQ1"],
            q2=config["initialQ2"],
            noise_range=config["noiseRange"],
        )
        target = config["targetProfit"]
    # elif config["plant"] == "plant3":
    # plant = plant3()
    else:
        raise ValueError("Invalid plant type in config")

    return (
        params,
        target,
        plant,
        controller,
        config["trainingEpochs"],
        config["simulationTimesteps"],
        config["learningRate"],
    )


def main():
    (
        params,
        target,
        plant,
        controller,
        training_epochs,
        simulation_timesteps,
        learning_rate,
    ) = load_config()
    errors = []
    for _ in range(training_epochs):
        mse, gradients = jax.value_and_grad(run_one_epoch)(params, plant, controller, target, simulation_timesteps)
        errors.append(mse)
        params = params - gradients * learning_rate

        # Update contrrol signal
    plt.plot(errors)
    plt.show()


if __name__ == "__main__":
    main()
