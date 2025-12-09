# train_alphazero.py

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from rate64.core.model import AlphaZeroNet

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_on_dataset(dataset_path, old_model_path=None, save_path="policy_value_net_new.pt"):
    print("Loading dataset:", dataset_path)
    data = torch.load(dataset_path)

    X = data["inputs"]
    Pi = data["policy"]
    V = data["value"]

    print("Shapes:")
    print("  X:", X.shape)
    print("  π:", Pi.shape)
    print("  V:", V.shape)

    dataset = TensorDataset(X, Pi, V)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ------------------------------------------------------------
    # LOAD OLD MODEL OR CREATE A NEW ONE
    # ------------------------------------------------------------
    model = AlphaZeroNet().to(DEVICE)

    if old_model_path is not None:
        print(f"Loading base model from {old_model_path}")
        model.load_state_dict(torch.load(old_model_path, map_location=DEVICE))

    # Make a COPY to train (AlphaZero style)
    new_model = AlphaZeroNet().to(DEVICE)
    new_model.load_state_dict(model.state_dict())  # copy weights

    optimizer = torch.optim.Adam(new_model.parameters(), lr=LR)

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        new_model.train()
        total_loss = 0.0

        for xb, pi_b, v_b in loader:
            xb = xb.to(DEVICE)
            pi_b = pi_b.to(DEVICE)
            v_b = v_b.to(DEVICE)

            # Forward pass
            policy_logits, value_pred = new_model(xb)

            # Policy loss (cross entropy)
            target_idx = pi_b.argmax(dim=1)
            policy_loss = F.cross_entropy(policy_logits, target_idx)

            # Value loss (MSE)
            value_loss = F.mse_loss(value_pred, v_b)

            # Combined loss (AlphaZero)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f}")

    # ------------------------------------------------------------
    # SAVE TRAINED MODEL
    # ------------------------------------------------------------
    torch.save(new_model.state_dict(), save_path)
    print(f"Saved NEW model to: {save_path}")

    return new_model



