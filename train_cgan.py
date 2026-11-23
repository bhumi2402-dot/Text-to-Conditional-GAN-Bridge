"""
Train Conditional GAN (cGAN) on Fashion-MNIST (Task U2).
Generator: z (100) + label (10) -> (1, 28, 28) with Tanh
Discriminator: (1, 28, 28) + label -> real/fake logit
Uses Adam (lr=2e-4, betas=(0.5, 0.999)), one-sided label smoothing,
and periodic sample grid generation across all 10 classes.
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from src.cgan import ConditionalGenerator, ConditionalDiscriminator
from src.utils import set_seed, get_device, FASHION_CLASSES


def train_cgan(
    epochs: int = 30,
    batch_size: int = 128,
    latent_dim: int = 100,
    lr: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    data_dir: str = "data",
    save_every: int = 5
):
    set_seed(42)
    device = get_device()
    print(f"[Train cGAN] Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Fashion-MNIST normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize networks
    netG = ConditionalGenerator(latent_dim=latent_dim, num_classes=10).to(device)
    netD = ConditionalDiscriminator(num_classes=10).to(device)

    # Weight initialization (standard DCGAN practice)
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Fixed noise for tracking generation quality: 10 classes x 10 samples = 100 images
    fixed_noise = torch.randn(100, latent_dim, device=device)
    fixed_labels = torch.tensor([c for c in range(10) for _ in range(10)], dtype=torch.long, device=device)

    history = {"epoch": [], "d_loss": [], "g_loss": []}

    print(f"Starting cGAN training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        netD.train()
        netG.train()
        running_d_loss = 0.0
        running_g_loss = 0.0
        num_batches = 0

        for real_imgs, real_labels in tqdm(dataloader, desc=f"cGAN Epoch {epoch}/{epochs}"):
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Labels for Discriminator: one-sided label smoothing (real: 0.9, fake: 0.0)
            label_real = torch.full((b_size, 1), 0.9, device=device)
            label_fake = torch.full((b_size, 1), 0.0, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizerD.zero_grad()

            # Real images
            output_real = netD(real_imgs, real_labels)
            loss_d_real = criterion(output_real, label_real)

            # Fake images
            z = torch.randn(b_size, latent_dim, device=device)
            fake_imgs = netG(z, real_labels)
            output_fake = netD(fake_imgs.detach(), real_labels)
            loss_d_fake = criterion(output_fake, label_fake)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizerG.zero_grad()
            # Generator wants Discriminator to think fake images are real (target: 1.0)
            output_g = netD(fake_imgs, real_labels)
            loss_g = criterion(output_g, torch.ones_like(output_g))
            loss_g.backward()
            optimizerG.step()

            running_d_loss += loss_d.item()
            running_g_loss += loss_g.item()
            num_batches += 1

        avg_d_loss = running_d_loss / num_batches
        avg_g_loss = running_g_loss / num_batches
        history["epoch"].append(epoch)
        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)

        print(f"Epoch [{epoch}/{epochs}] - Loss D: {avg_d_loss:.4f} | Loss G: {avg_g_loss:.4f}")

        # Save sample grid and checkpoints periodically or on last epoch
        if epoch % save_every == 0 or epoch == epochs:
            netG.eval()
            with torch.no_grad():
                gen_samples = netG(fixed_noise, fixed_labels)
                # Rescale from [-1, 1] to [0, 1]
                gen_samples = (gen_samples + 1.0) / 2.0
                grid = make_grid(gen_samples, nrow=10, padding=2, normalize=False)
                grid_np = grid.cpu().numpy().transpose((1, 2, 0))

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(grid_np, cmap="gray")
                ax.axis("off")
                ax.set_title(f"cGAN Samples (Epoch {epoch}) - Rows: 10 Fashion Classes", fontsize=12)
                plt.tight_layout()
                plt.savefig(f"results/cgan_sample_grid_epoch_{epoch}.png", dpi=150)
                plt.savefig("results/cgan_sample_grid.png", dpi=150)
                plt.close(fig)

            # Save checkpoint after this epoch
            torch.save(netG.state_dict(), "checkpoints/cgan_generator.pt")
            torch.save(netD.state_dict(), "checkpoints/cgan_discriminator.pt")
            
            # Plot Loss Curve
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(history["epoch"], history["d_loss"], label="Discriminator Loss (D)", color="#e74c3c", lw=2)
            ax.plot(history["epoch"], history["g_loss"], label="Generator Loss (G)", color="#2980b9", lw=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_title("cGAN Minimax Training Loss Dynamics (Fashion-MNIST)", fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig("results/cgan_losses.png", dpi=150)
            plt.close(fig)
            print(f"Saved checkpoint and sample grid at epoch {epoch}")

    elapsed = time.time() - start_time
    print(f"cGAN training finished in {elapsed:.1f}s.")

    # Save Generator checkpoint
    gen_ckpt = "checkpoints/cgan_generator.pt"
    torch.save(netG.state_dict(), gen_ckpt)
    ckpt_size_mb = os.path.getsize(gen_ckpt) / (1024 * 1024)
    print(f"Saved Generator checkpoint to {gen_ckpt} ({ckpt_size_mb:.2f} MB)")

    # Save Discriminator checkpoint
    disc_ckpt = "checkpoints/cgan_discriminator.pt"
    torch.save(netD.state_dict(), disc_ckpt)

    # Plot Loss Curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["epoch"], history["d_loss"], label="Discriminator Loss (D)", color="#e74c3c", lw=2)
    ax.plot(history["epoch"], history["g_loss"], label="Generator Loss (G)", color="#2980b9", lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("cGAN Minimax Training Loss Dynamics (Fashion-MNIST)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/cgan_losses.png", dpi=150)
    plt.close(fig)
    print("Saved loss curves to results/cgan_losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST cGAN")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of latent noise z")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=5, help="Save grid every N epochs")
    args = parser.parse_args()

    train_cgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        save_every=args.save_every
    )
