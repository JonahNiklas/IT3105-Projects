# Experimentation

## Experiment 1 - Simple feedforward neural network:

Just checking that our implementation is able to learn. Hyperparameter:

| Variable Name             | Value |
|---------------------------|-------|
| BOARD_SIZE                | 11    |
| NUMBER_OF_EPISODES        | 1000  |
| NUMBER_OF_SIMULATIONS     | 100   |
| ANET_LEARNING_RATE        | 0.01  |
| ANET_NUM_HIDDEN_LAYERS    | 7     |
| ANET_NUM_HIDDEN_NODES     | 100   |
| ANET_ACTIVATION_FUNCTION  | relu  |
| ANET_OPTIMIZER            | adam  |
| ANET_M                    | 10    |
| ANET_BATCH_SIZE           | 128   |

Checking that the model learned something by plotting the results of making the 10 (ANET_M) different saves of the model compete against each other.

### Experiment 1 Results:

