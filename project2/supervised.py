from project2.neural_network import NeuralNetwork
import numpy as np
import torch
from tqdm import tqdm

# Load scoredPositionsFull.npz
data = np.load("project2/scoredPositionsFull.npz")
# Get a overview of the data
print(data.files)  # ['positions', 'scores']
# Get the data
positions = data["positions"]
training_examples = positions.shape[0]
print(f"Number of training examples: {training_examples}")
positions_p1 = positions[:, 0, :, :]
positions_p2 = positions[:, 1, :, :]
positions_p1 = np.where(positions_p1 == True, 1, 0)
positions_p2 = np.where(positions_p2 == True, -1, 0)
assert positions_p1.shape == positions_p2.shape == (training_examples, 17, 17)
positions = positions_p1 + positions_p2
assert positions.shape == (training_examples, 17, 17)
positions = positions[:, 2:-2, 2:-2]
assert positions.shape == (training_examples, 13, 13)
scores = data["scores"]
assert scores.shape == (training_examples, 13, 13)
assert positions.shape == scores.shape

# Softmax the scores
scores = np.exp(scores) / np.exp(scores).sum(axis=(1, 2))[:, None, None]
assert scores.shape == (training_examples, 13, 13)

assert np.isclose(scores[0].sum(), 1.0), f"Sum of scores is not one: {scores[0].sum()}"

net = NeuralNetwork(board_size=13)
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
