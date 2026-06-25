import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn import MiniCNN

EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
MODEL_PATH = "../artifacts/model.pth"


def get_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("../data", train=True,  download=True, transform=transform)
    val_ds   = datasets.MNIST("../data", train=False, download=True, transform=transform)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=BATCH_SIZE),
    )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model     = MiniCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    train_loader, val_loader = get_loaders()

    for epoch in range(EPOCHS):
        # ── Training loop ──────────────────────────────────────
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── Validation ─────────────────────────────────────────
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                correct += (model(images).argmax(dim=1) == labels).sum().item()

        acc     = correct / len(val_loader.dataset) * 100
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

    os.makedirs("../artifacts", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✓ Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()