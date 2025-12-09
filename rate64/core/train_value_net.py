import torch
from torch.utils.data import TensorDataset, DataLoader
from model import ValueNet

DATASET_PATH = "dataset.pt"
BATCH_SIZE = 512
EPOCHS = 100
LR = 1e-3

def main():
    print("Loading dataset...")
    data = torch.load(DATASET_PATH)

    X = data["inputs"]
    y = data["labels"]

    print("Dataset shapes:", X.shape, y.shape)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ValueNet().to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "value_net.pt")
    print("Saved model to value_net.pt")

if __name__ == "__main__":
    main()
