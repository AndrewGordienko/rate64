import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_DIM = 4672


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style dual-head network.
    Input:
        782-d encoded board (your encode_board output)
    Outputs:
        - policy logits (size POLICY_DIM = 4672)
        - value scalar in [-1, 1]
    """

    def __init__(self, input_dim=782, hidden_dim=512):
        super().__init__()

        # ----- Shared trunk -----
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # ----- Policy head -----
        self.policy_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_out = nn.Linear(hidden_dim, POLICY_DIM)

        # ----- Value head -----
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.value_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Policy
        p = F.relu(self.policy_fc1(x))
        p = self.policy_out(p)  # logits (softmax done in MCTS)

        # Value
        v = F.relu(self.value_fc1(x))
        v = torch.tanh(self.value_out(v))  # [-1, 1]

        return p, v

    # -------------------------------------------------------
    # Move → Policy index (must match dataset_sl.py exactly)
    # -------------------------------------------------------
    def move_to_index(self, move):
        """
        Provides MCTS with the correct index into the policy output
        for the given chess.Move object.
        """
        from rate64.util._dataset.build_dataset_sl import move_to_index as _move_to_index
        return _move_to_index(move)
