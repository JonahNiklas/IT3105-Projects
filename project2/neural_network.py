import torch
from project2.globals import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

class NeuralNetwork(torch.nn.Module):   
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def train_one_batch(self, RBUF):
        dataloader = torch.utils.data.DataLoader(
            RBUF, batch_size=ANET_BATCH_SIZE, shuffle=True
        )
        if ANET_OPTIMIZER == "adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=ANET_LEARNING_RATE)
        elif ANET_OPTIMIZER == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=ANET_LEARNING_RATE)
        elif ANET_OPTIMIZER == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=ANET_LEARNING_RATE)
        elif ANET_OPTIMIZER == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=ANET_LEARNING_RATE)
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = self(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

    def save(self, i, path="saved_networks"):
        import os

        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), f"{path}/{EXPERIMENT_NAME}_anet_{i}.pt")

class FeedForwardNetwork(NeuralNetwork):
    def __init__(self, board_size=BOARD_SIZE):
        super(FeedForwardNetwork, self).__init__()
        self.board_size = board_size
        self.flatten = torch.nn.Flatten()
        self.linear_stack = torch.nn.Sequential()
        for i in range(ANET_NUM_HIDDEN_LAYERS):
            if i == 0:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(board_size * board_size, ANET_NUM_HIDDEN_NODES),
                )
            elif i == ANET_NUM_HIDDEN_LAYERS - 1:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(ANET_NUM_HIDDEN_NODES, board_size * board_size),
                )
            else:
                self.linear_stack.add_module(
                    "hidden_layer_{}".format(i),
                    torch.nn.Linear(ANET_NUM_HIDDEN_NODES, ANET_NUM_HIDDEN_NODES),
                )
            if ANET_ACTIVATION_FUNCTION == "linear":
                self.linear_stack.add_module(
                    "activation_function_{}".format(i), torch.nn.Identity()
                )
            elif ANET_ACTIVATION_FUNCTION == "sigmoid":
                self.linear_stack.add_module(
                    "activation_function_{}".format(i), torch.nn.Sigmoid()
                )
            elif ANET_ACTIVATION_FUNCTION == "tanh":
                self.linear_stack.add_module(
                    "activation_function_{}".format(i), torch.nn.Tanh()
                )
            elif ANET_ACTIVATION_FUNCTION == "relu":
                self.linear_stack.add_module(
                    "activation_function_{}".format(i), torch.nn.ReLU()
                )
            else:
                raise ValueError("Invalid activation function")
        self.softmax = torch.nn.Softmax(dim=1)
        self.to(device)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.board_size, self.board_size)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        logits = self.softmax(logits)
        assert logits.shape == (batch_size, self.board_size * self.board_size)
        logits = logits.view(batch_size, self.board_size, self.board_size)
        return logits
    
class ConvNetwork(NeuralNetwork):
    def __init__(self, board_size=BOARD_SIZE):
        super(ConvNetwork, self).__init__()
        self.board_size = board_size
        self.conv_stack = torch.nn.Sequential()
        self.conv_stack.add_module(
            "conv_1",
            torch.nn.Conv2d(3, ANET_W, kernel_size=5, padding=0),
        )
        for i in range(ANET_d - 1):
            self.conv_stack.add_module(
                f"conv_{i+2}",
                torch.nn.Conv2d(ANET_W, ANET_W, kernel_size=3, padding=1),
            )
        #this layer should account for positional bias
        self.conv_stack.add_module(
            "conv_final",
            torch.nn.Conv2d(ANET_W, 1, kernel_size=1, padding=0),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, 3, self.board_size + 4, self.board_size + 4) # board padding
        logits = self.conv_stack(x)
        assert logits.shape == (batch_size, 1, self.board_size, self.board_size)
        logits = logits.view(batch_size, self.board_size * self.board_size)
        logits = self.softmax(logits)
        logits = logits.view(batch_size, self.board_size, self.board_size)
        return logits

