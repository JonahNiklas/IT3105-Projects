# Experimentation

## Experiment 1 - Simple feedforward neural network:

Just checking that our implementation is able to learn. Hyperparameter:

| Parameter                | Value        |
| ------------------------ | ------------ |
| EXPERIMENT_NAME          | feedforward1 |
| VERBOSE                  | False        |
| VISUALIZE                | False        |
| BOARD_SIZE               | 11           |
| NUMBER_OF_EPISODES       | 500          |
| NUMBER_OF_SIMULATIONS    | 50           |
| ANET_LEARNING_RATE       | 0.01         |
| ANET_NUM_HIDDEN_LAYERS   | 7            |
| ANET_NUM_HIDDEN_NODES    | 100          |
| ANET_ACTIVATION_FUNCTION | relu         |
| ANET_OPTIMIZER           | adam         |
| ANET_M                   | 10           |
| ANET_BATCH_SIZE          | 128          |

This will be running for 24-hours on a 4-CPU Intel Cascade Lake instance

### Experiment 1 Results:

Checking that the model learned something by plotting the results of making the 10 (ANET_M) different saves of the model compete against each other.

![feedforward1](figs/feedforward1_topp_results.png)

We unfortunately see a negative correlation between training time and performance. There is likely a bug in the training code. Also, the board is so large (11) that the model might find it hard to learn.
