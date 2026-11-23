"""
Train Supervised CNN Classifier on Fashion-MNIST.
Achieves >91% test accuracy and saves checkpoint to checkpoints/classifier.pt.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.classifier import FashionCNN
from src.utils import set_seed, get_device, FASHION_CLASSES


def train(epochs: int = 5, batch_size: int = 128, lr: float = 0.001, data_dir: str = "data"):
    set_seed(42)
    device = get_device()
    print(f"[Train Classifier] Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])

    print("Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FashionCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Beginning training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100.0
        avg_loss = running_loss / total

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total * 100.0
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s.")

    # Save checkpoint
    ckpt_path = "checkpoints/classifier.pt"
    torch.save(model.state_dict(), ckpt_path)
    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Saved classifier checkpoint to {ckpt_path} ({file_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
