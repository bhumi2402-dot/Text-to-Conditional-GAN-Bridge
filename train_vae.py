"""
Train Conditional Variational Autoencoder (cVAE) on Fashion-MNIST (Task U4).
Encoder: (1, 28, 28) + class -> mu, logvar (latent_dim=20)
Decoder: z (20) + class -> (1, 28, 28)
Loss: Reconstruction MSE + beta * KL Divergence with warm-up.
Also computes and exports a 2D t-SNE / PCA latent space visualization (results/vae_tsne.png).
"""

import os
import argparse
import time

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.cvae import ConditionalVAE, vae_loss_function
from src.utils import set_seed, get_device, FASHION_CLASSES


def train_vae(
    epochs: int = 15,
    batch_size: int = 128,
    latent_dim: int = 20,
    lr: float = 0.001,
    data_dir: str = "data",
    save_every: int = 5
):
    set_seed(42)
    device = get_device()
    print(f"[Train cVAE] Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Scale to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConditionalVAE(latent_dim=latent_dim, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"epoch": [], "total_loss": [], "recon_loss": [], "kl_loss": []}

    print(f"Beginning cVAE training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0
        num_batches = 0

        # Linear KL warm-up: beta increases from 0.0 to 1.0 over epochs
        beta = min(1.0, epoch / max(1, epochs // 2))

        for images, labels in tqdm(train_loader, desc=f"cVAE Epoch {epoch}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images, labels)
            total_loss, recon_loss, kl_loss = vae_loss_function(recon, images, mu, logvar, beta=beta)
            total_loss.backward()
            optimizer.step()

            running_total += total_loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            num_batches += 1

        avg_total = running_total / num_batches
        avg_recon = running_recon / num_batches
        avg_kl = running_kl / num_batches

        history["epoch"].append(epoch)
        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        print(f"Epoch [{epoch}/{epochs}] (beta={beta:.2f}) - Total: {avg_total:.2f} | Recon: {avg_recon:.2f} | KL: {avg_kl:.2f}")

    elapsed = time.time() - start_time
    print(f"cVAE training finished in {elapsed:.1f}s.")

    # Save model weights
    ckpt_path = "checkpoints/cvae.pt"
    torch.save(model.state_dict(), ckpt_path)
    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Saved cVAE checkpoint to {ckpt_path} ({file_size_mb:.2f} MB)")

    # 1. Generate cVAE sample grid across all 10 classes
    model.eval()
    with torch.no_grad():
        fixed_z = torch.randn(100, latent_dim, device=device)
        fixed_labels = torch.tensor([c for c in range(10) for _ in range(10)], dtype=torch.long, device=device)
        samples = model.decode(fixed_z, fixed_labels)
        samples_01 = (samples + 1.0) / 2.0
        grid = make_grid(samples_01, nrow=10, padding=2, normalize=False)
        grid_np = grid.cpu().numpy().transpose((1, 2, 0))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(grid_np, cmap="gray")
        ax.axis("off")
        ax.set_title("cVAE Generated Samples - Rows: 10 Fashion Classes (10 samples/row)", fontsize=12)
        plt.tight_layout()
        plt.savefig("results/cvae_sample_grid.png", dpi=150)
        plt.close(fig)
        print("Saved cVAE sample grid to results/cvae_sample_grid.png")

    # 2. Extract latent space embeddings for t-SNE / PCA visualization
    print("Generating 2D t-SNE plot of VAE latent space...")
    model.eval()
    all_mus = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            mu, _ = model.encode(images, labels)
            all_mus.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if len(all_mus) * batch_size >= 1500:
                break

    mu_array = np.concatenate(all_mus, axis=0)[:1500]
    label_array = np.concatenate(all_labels, axis=0)[:1500]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_2d = tsne.fit_transform(mu_array)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for class_idx in range(10):
        mask = (label_array == class_idx)
        ax.scatter(
            z_2d[mask, 0],
            z_2d[mask, 1],
            label=FASHION_CLASSES[class_idx],
            alpha=0.75,
            s=25,
            color=cmap(class_idx)
        )

    ax.set_title("t-SNE Projection of Conditional VAE Latent Space (N=1500)", fontsize=13)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("results/vae_tsne.png", dpi=150)
    plt.close(fig)
    print("Saved VAE t-SNE plot to results/vae_tsne.png")

    # 3. Plot training loss curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["total_loss"], label="Total Loss", color="#2c3e50", lw=2)
    ax.plot(history["epoch"], history["recon_loss"], label="Reconstruction MSE", color="#2980b9", lw=2)
    ax.plot(history["epoch"], history["kl_loss"], label="KL Divergence", color="#e67e22", lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("cVAE Training Dynamics with KL Annealing", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/cvae_losses.png", dpi=150)
    plt.close(fig)
    print("Saved loss curves to results/cvae_losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST cVAE")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=20, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_vae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr
    )
