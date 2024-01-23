import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from config import (LEARNING_RATE,SIMULATION_TIMESTEPS, TRAINING_EPOCHS)
from controllers import Controller
from helper import get_controller, get_params, get_plant


def run_one_timestep(params, plant, controller: Controller, target):
    error = target - plant.get_output()
    control_signal = controller.calculate_control_signal(params, error)
    plant.timestep(control_signal)
    return error, control_signal


def run_one_epoch(params):
    controller = get_controller()
    plant, target = get_plant()
    errors = []
    control_signals = []
    for _ in range(SIMULATION_TIMESTEPS):
        error,control_signal = run_one_timestep(params, plant, controller, target)
        errors.append(error)
        control_signals.append(control_signal)

    mse = jnp.mean(jnp.square(jnp.array(errors)))
    return mse

def main():
  mse_epochs = []
  params = get_params()
  plot_params = [params]
  for i in range(TRAINING_EPOCHS):
    mse, gradients = jax.value_and_grad(run_one_epoch)(params)
    mse_item = mse.item()
    print(f"Epoch {i+1}")
    print(f"Params: {params}")
    print(f"MSE: {mse_item}")
    print(f"Gradients: {gradients}")
    mse_epochs.append(mse_item)
    
    if isinstance(params, list):
        params = [[p - g * LEARNING_RATE for p, g in zip(param_layer, grad_layer)] for param_layer, grad_layer in zip(params, gradients)]
    else:
        params = params - gradients * LEARNING_RATE
        plot_params.append(params)

    
    print("====================================")
  
  if len(plot_params) > 1:
      plt.plot(plot_params)
      plt.legend(["Kp", "Ki", "Kd"])
      plt.xlabel("Epoch")
      plt.ylabel("Value")
      plt.show()
  plt.plot(mse_epochs)
  plt.show()
if __name__ == "__main__":
  main()