EXPERIMENT_NAME = "small-board-long-training"
VERBOSE = False
VISUALIZE = False
BOARD_SIZE = 5
NUMBER_OF_EPISODES = 10_000  # actual games
NUMBER_OF_SIMULATIONS = 100  # MCTS simulations per move

# ANET
ANET_LEARNING_RATE = 0.01
ANET_NUM_HIDDEN_LAYERS = 7
ANET_NUM_HIDDEN_NODES = 100
ANET_ACTIVATION_FUNCTION = 'relu'  # linear, sigmoid, tanh, relu
ANET_OPTIMIZER = 'adam'  # adagrad, sgd, adam, rmsprop
ANET_M = 10  # number of ANETs to be cached
ANET_BATCH_SIZE = 128
ANET_d = 8 # number of conv layers
ANET_W = 128 # number of conv filters per layer

# TOPP
TOPP_NUM_GAMES_BETWEEN_ANY_TWO_PLAYERS = 100
TOPP_EPSILON = 0
TOPP_VISUALIZE = True

# MCTS
MCTS_ROLLOUT_EPSILON = 0.20
MCTS_C = 1
