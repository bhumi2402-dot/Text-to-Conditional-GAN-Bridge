"""
Conditional Variational Autoencoder (cVAE) for Fashion-MNIST.
Includes Encoder, Decoder, Reparameterization, Loss function, and Sampling.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConditionalVAE(nn.Module):
    """
    Conditional VAE: encodes (image + class) to latent Gaussian distribution (mu, logvar),
    and decodes (latent sample + class) back to image (1, 28, 28) in range [-1, 1].
    """
    def __init__(self, latent_dim: int = 20, num_classes: int = 10, class_embed_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        # Encoder class embedding: projected to a spatial 1x28x28 channel
        self.enc_label_emb = nn.Embedding(num_classes, 28 * 28)

        # Encoder convolutional stack: (2, 28, 28) -> (64, 7, 7)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder class embedding
        self.dec_label_emb = nn.Embedding(num_classes, class_embed_dim)

        # Decoder projection
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + class_embed_dim, 64 * 7 * 7),
            nn.BatchNorm1d(64 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder convolutional upsampling stack: (64, 7, 7) -> (1, 28, 28)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  # -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, bias=True),  # -> 28x28
            nn.Tanh()  # Output range [-1, 1]
        )

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, 28, 28), y: (B,)
        y_spatial = self.enc_label_emb(y).view(-1, 1, 28, 28)
        concat_input = torch.cat([x, y_spatial], dim=1)  # (B, 2, 28, 28)
        h = self.enc_conv(concat_input)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), y: (B,)
        y_embed = self.dec_label_emb(y)  # (B, class_embed_dim)
        z_cond = torch.cat([z, y_embed], dim=1)  # (B, latent_dim + class_embed_dim)
        h = self.dec_fc(z_cond)
        h_spatial = h.view(-1, 64, 7, 7)
        recon = self.dec_conv(h_spatial)
        return recon

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes VAE loss = Reconstruction MSE Loss + beta * KL Divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div


def load_cvae(
    checkpoint_path: str,
    device: torch.device,
    latent_dim: int = 20
) -> ConditionalVAE:
    """Loads trained Conditional VAE weights from disk."""
    model = ConditionalVAE(latent_dim=latent_dim).to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return model


@torch.no_grad()
def sample_cvae(
    model: ConditionalVAE,
    class_idx: int,
    count: int = 4,
    device: Optional[torch.device] = None,
    latent_dim: int = 20,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Samples `count` images from N(0, I) prior for `class_idx`.
    Returns tensor of shape (count, 1, 28, 28) in range [-1, 1].
    """
    dev = device or next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(count, latent_dim, device=dev)
    labels = torch.full((count,), class_idx, dtype=torch.long, device=dev)
    imgs = model.decode(z, labels)
    return imgs
