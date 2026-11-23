"""
Quick script to generate results/vae_tsne.png using trained checkpoints/cvae.pt.
"""

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.manifold import TSNE

from src.cvae import load_cvae
from src.utils import set_seed, get_device, FASHION_CLASSES


def generate_tsne_plot(data_dir: str = "data", output_path: str = "results/vae_tsne.png"):
    set_seed(42)
    device = get_device()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading trained cVAE from checkpoints/cvae.pt...")
    model = load_cvae("checkpoints/cvae.pt", device=device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("Extracting latent space embeddings for 1500 test samples...")
    all_mus = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            mu, _ = model.encode(images, labels)
            all_mus.append(mu.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if len(all_mus) * 128 >= 1500:
                break

    mu_array = np.concatenate(all_mus, axis=0)[:1500]
    label_array = np.concatenate(all_labels, axis=0)[:1500]

    print("Fitting t-SNE (2D)...")
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
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Successfully saved VAE t-SNE plot to {output_path}")


if __name__ == "__main__":
    generate_tsne_plot()
