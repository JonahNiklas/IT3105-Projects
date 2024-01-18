import json
from controllers import PIDController, NeuralNetController
from plants import BathTubPlant, CournotPlant
import matplotlib.pyplot as plt

def run_one_timestep(plant, controller, target):
    error = target - plant.get_output()
    control_signal = controller.get_control_signal(error)
    plant.timestep(control_signal)
    return plant.get_output(), control_signal

def run_one_epoch(plant, controller, target, num_timesteps):
    output = []
    control_signal = []
    for _ in range(num_timesteps):
        output_t, control_signal_t = run_one_timestep(plant, controller, target)
        output.append(output_t)
        control_signal.append(control_signal_t)
    mse = sum([(target - y)**2 for y in output])/len(output)
    
    return output, control_signal, mse

def load_config():
    with open("pivotal_parameters.json") as file:
        config = json.load(file)
    
    if config["controller"] == "pid":
        controller = PIDController()
    elif config["controller"] == "neural_net":
        neuralNetConfig = config["neuralNetwork"]
        controller = NeuralNetController(num_layers=neuralNetConfig["numLayers"],neurons_per_layer=neuralNetConfig["neuronsPerLayer"],activation_functions=neuralNetConfig["activationFunctions"],weight_range=neuralNetConfig["weightRange"],bias_range=neuralNetConfig["biasRange"])
    else:
        raise ValueError("Invalid controller type in config")

    if config["plant"] == "bathtub":
        plant = BathTubPlant(area=config["crossSectionArea"]["bathtub"],drain_area=config["crossSectionArea"]["drain"],noise_range=config["noiseRange"],water_level=config["initialWaterHeight"])
        target = config["initialWaterHeight"]
    elif config["plant"] == "cournot":
        plant = CournotPlant(max_price=config["cournotCompetetion"]["maxPrice"],marginal_cost=config["cournotCompetition"]["marginalCost"],q1=config["initialQ1"],q2=config["initialQ2"],noise_range=config["noiseRange"])
        target = config["targetProfit"]
    # elif config["plant"] == "plant3":
        # plant = plant3()
    else:
        raise ValueError("Invalid plant type in config")
    
    return target, plant, controller, config["training_epochs"], config["simulationTimesteps"]

def main():
    target, plant, controller, training_epochs, simulation_timesteps = load_config()
    mse= []
    for _ in range(training_epochs):
        output, control_signal, mse = run_one_epoch(plant, controller, target, simulation_timesteps)
        mse.append(mse)
    plt.plot(mse)
    plt.show()

if __name__ == "__main__":
    main()
    