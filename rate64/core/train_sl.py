import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from rate64.core.model import AlphaZeroNet


DATASET_PATH = "dataset_sl.pt"
BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("Loading dataset:", DATASET_PATH)
    data = torch.load(DATASET_PATH)

    X = data["inputs"]
    Pi = data["policy"]
    V = data["value"]

    print("Shapes:")
    print("  X:", X.shape)
    print("  π:", Pi.shape)
    print("  V:", V.shape)

    dataset = TensorDataset(X, Pi, V)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AlphaZeroNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for xb, pi_b, v_b in loader:
            xb = xb.to(DEVICE)
            pi_b = pi_b.to(DEVICE)
            v_b = v_b.to(DEVICE)

            # Forward pass
            policy_logits, value_pred = model(xb)

            # Policy loss (cross entropy)
            # Convert one-hot to class index
            target_idx = pi_b.argmax(dim=1)
            policy_loss = F.cross_entropy(policy_logits, target_idx)

            # Value loss (MSE)
            value_loss = F.mse_loss(value_pred, v_b)

            # Combined AlphaZero loss
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg = total_loss / len(dataset)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg:.5f}")

    torch.save(model.state_dict(), "policy_value_net.pt")
    print("Saved model to policy_value_net.pt")


if __name__ == "__main__":
    main()
