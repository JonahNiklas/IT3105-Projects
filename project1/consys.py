import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from project1.config import (VERBOSE,LEARNING_RATE,SIMULATION_TIMESTEPS, TRAINING_EPOCHS)
from project1.controllers import Controller
from project1.helper import get_controller, get_params, get_plant


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

def main():
  mse_epochs = []
  params = get_params()
  plot_params = [params]
  grad_func = jax.value_and_grad(run_one_epoch)
  grad_func_jit = jax.jit(grad_func)
  for i in range(TRAINING_EPOCHS):
    mse, gradients = grad_func_jit(params)

    mse_epochs.append(mse.item())
    if isinstance(params, list):
        params = [[p - g * LEARNING_RATE for p, g in zip(param_layer, grad_layer)] for param_layer, grad_layer in zip(params, gradients)]
    else:
        params = params - gradients * LEARNING_RATE
        plot_params.append(params)

    if(VERBOSE):
        print(f"Epoch {i+1}")
        print(f"Params: {params}")
        print(f"MSE: {mse_epochs[-1]}")
        print(f"Gradients: {gradients}")
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