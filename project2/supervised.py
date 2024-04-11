from project2.game import HexGame
from project2.neural_network import ConvNetwork, NeuralNetwork
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load scoredPositionsFull.npz
data = np.load("project2/scoredPositionsFull.npz")
# Get a overview of the data
print(data.files)  # ['positions', 'scores']
# Get the data
positions = data["positions"]
training_examples = positions.shape[0]
print(f"Number of training examples: {training_examples}")
positions = positions[:, 0:3, :, :]
# positions = positions[:, 2:-2, 2:-2]
assert positions.shape == (training_examples, 3, 17, 17)
scores = data["scores"]
assert scores.shape == (training_examples, 13, 13)

# Visualize a random training example
example = np.random.randint(training_examples)
p1, p2, empty = positions[example]
p2 = p2 * -1
board_state = p1 + p2
game = HexGame(size=17, board_state=board_state, last_move=None)
# fig, ax = plt.subplots(1)
# plt.ion()
# fig.show()
# fig.canvas.draw()
# game.visualize_board(ax)

# Softmax the scores
scores = np.exp(scores) / np.exp(scores).sum(axis=(1, 2))[:, None, None]
assert scores.shape == (training_examples, 13, 13)

assert np.isclose(scores[0].sum(), 1.0), f"Sum of scores is not one: {scores[0].sum()}"

net = ConvNetwork(board_size=13)
print(net)

def train(net: torch.nn.Module):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    X = torch.tensor(positions, dtype=torch.float32)
    Y = torch.tensor(scores, dtype=torch.float32)
    assert len(X) == len(Y) == training_examples
    dataloader = torch.utils.data.DataLoader(
        list(zip(X, Y)), batch_size=128, shuffle=True
    )
    for epoch in range(10):
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch} loss: {np.mean(epoch_losses)}")


train(net)
net.save(0, path="project2/saved_networks")