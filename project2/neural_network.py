import torch
from globals import (
    BOARD_SIZE,
    ANET_LEARNING_RATE,
    ANET_NUM_HIDDEN_LAYERS,
    ANET_NUM_HIDDEN_NODES,
    ANET_ACTIVATION_FUNCTION,
    ANET_OPTIMIZER,
    ANET_M,
    ANET_BATCH_SIZE,
)


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_stack = torch.nn.Sequential()
        for i in range(ANET_NUM_HIDDEN_LAYERS):
            if i == 0:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(BOARD_SIZE * BOARD_SIZE, ANET_NUM_HIDDEN_NODES),
                )
            elif i == ANET_NUM_HIDDEN_LAYERS - 1:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(ANET_NUM_HIDDEN_NODES, BOARD_SIZE * BOARD_SIZE),
                )
            else:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(ANET_NUM_HIDDEN_NODES, ANET_NUM_HIDDEN_NODES),
                )
            if ANET_ACTIVATION_FUNCTION == "linear":
                self.linear_stack.add_module("activation_function_{}".format(i), torch.nn.Identity())
            elif ANET_ACTIVATION_FUNCTION == "sigmoid":
                self.linear_stack.add_module("activation_function_{}".format(i), torch.nn.Sigmoid())
            elif ANET_ACTIVATION_FUNCTION == "tanh":
                self.linear_stack.add_module("activation_function_{}".format(i), torch.nn.Tanh())
            elif ANET_ACTIVATION_FUNCTION == "relu":
                self.linear_stack.add_module("activation_function_{}".format(i), torch.nn.ReLU())
            else:
                raise ValueError("Invalid activation function")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, BOARD_SIZE, BOARD_SIZE)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        logits = self.softmax(logits)
        assert logits.shape == (batch_size, BOARD_SIZE * BOARD_SIZE)
        logits = logits.view(batch_size, BOARD_SIZE, BOARD_SIZE)
        return logits
    

