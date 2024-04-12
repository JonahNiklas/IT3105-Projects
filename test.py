import torch
import numpy as np

x = torch.tensor([-1, -1, 0.5, -1, -1])
print(torch.nn.functional.softmax(x, dim=0))
# scores = np.exp(scores) / np.exp(scores).sum(axis=(1, 2))[:, None, None]
print(np.exp(x) / np.exp(x).sum())