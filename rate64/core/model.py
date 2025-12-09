# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    """
    Input: 782-dim encoded board position
    Output: scalar in [-1, 1] indicating predicted game outcome
    """

    def __init__(self, input_dim=782, hidden_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # prediction in [-1, 1]
        return x
