from project2.game import HexGame
from project2.neural_network import ConvNetwork, NeuralNetwork
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load scoredPositionsFull.npz
data = np.load("project2/scoredPositionsFull.npz")

# Get the data
positions = data["positions"]
training_examples = positions.shape[0]
print(f"Number of training examples: {training_examples}")

p1 = positions[:, 0, :, :]
p2 = positions[:, 1, :, :]
p1, p2 = p1.astype(float), p2.astype(float)
p1_example, p2_example = p1[23657], p2[23657]

empty = np.ones_like(p1) - np.where(p1 + p2 >= 1, 1, 0)
positions = np.stack([p1, p2, empty], axis=1)

assert positions.shape == (training_examples, 3, 17, 17)
scores = data["scores"]
assert scores.shape == (training_examples, 13, 13)

# Softmax the scores
scores = np.exp(scores) / np.exp(scores).sum(axis=(1, 2))[:, None, None]
assert scores.shape == (training_examples, 13, 13)

assert np.isclose(
    scores[0].sum(), 1.0), f"Sum of scores is not one: {scores[0].sum()}"

# Assert that p1 always plays from top <-> bottom
assert np.all(p1[:, 0, :] == 1), "p1 does not always play from top to bottom"
assert np.all(p1[:, -1, :] == 1), "p1 does not always play from top to bottom"
assert np.all(p2[:, :, 0] == 1), "p2 does not always play from left to right"
assert np.all(p2[:, :, -1] == 1), "p2 does not always play from left to right"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ConvNetwork(board_size=13).to(device)
print(net)


def accuracy(pred, target):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    return (pred.argmax(dim=1) == target.argmax(dim=1)).float().mean().item()


def train(net: torch.nn.Module):
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    X = torch.tensor(positions, dtype=torch.float32).to(device)
    Y = torch.tensor(scores, dtype=torch.float32).to(device)
    assert len(X) == len(Y)
    dataloader = torch.utils.data.DataLoader(
        list(zip(X, Y)), batch_size=128, shuffle=True
    )
    for epoch in range(15):
        epoch_losses = []
        epoch_accuracies = []
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy(y_pred, y))
        print(f"Epoch {epoch} loss: {np.mean(epoch_losses)}")
        print(f"Epoch {epoch} accuracy: {np.mean(epoch_accuracies)}")


# net.load_state_dict(torch.load(
#     "saved_networks/supervised_1.pt", map_location=device))
train(net)
net.save(0, dir="saved_networks", name="supervised")
