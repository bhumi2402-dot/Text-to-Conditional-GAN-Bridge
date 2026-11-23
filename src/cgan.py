"""
Conditional Generative Adversarial Network (cGAN) for Fashion-MNIST.
Includes Generator, Discriminator, and sampling utilities.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator: maps noise z in R^100 + class label y in {0..9}
    to a synthetic Fashion-MNIST image (1, 28, 28) in range [-1, 1].
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, embed_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        # Initial linear projection: (latent_dim + embed_dim) -> 128 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Upsampling convolutional stack: (7x7) -> (14x14) -> (28x28)
        self.conv_blocks = nn.Sequential(
            # Block 1: 7x7 -> 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 14x14 -> 28x28
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Output block: 32 -> 1 channel with Tanh activation
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim), labels: (B,)
        c = self.label_emb(labels)  # (B, embed_dim)
        x = torch.cat([z, c], dim=1)  # (B, latent_dim + embed_dim)
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7)
        img = self.conv_blocks(x)  # (B, 1, 28, 28)
        return img


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator: maps image (1, 28, 28) + class label y
    to real/fake logit score.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Spatial class embedding: maps class index to a 1x28x28 channel
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)

        # Convolutional feature extractor on 2 channels (image + class channel)
        self.net = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)  # Raw logit
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # img: (B, 1, 28, 28), labels: (B,)
        c = self.label_embedding(labels).view(-1, 1, 28, 28)
        x = torch.cat([img, c], dim=1)  # (B, 2, 28, 28)
        logit = self.net(x)  # (B, 1)
        return logit


def load_cgan_generator(
    checkpoint_path: str,
    device: torch.device,
    latent_dim: int = 100
) -> ConditionalGenerator:
    """Loads trained Conditional Generator weights from disk."""
    netG = ConditionalGenerator(latent_dim=latent_dim).to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        netG.load_state_dict(state_dict)
        netG.eval()
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return netG


@torch.no_grad()
def sample_cgan(
    generator: ConditionalGenerator,
    class_idx: int,
    count: int = 4,
    device: Optional[torch.device] = None,
    latent_dim: int = 100,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generates `count` synthetic samples of `class_idx`.
    Returns tensor of shape (count, 1, 28, 28) in range [-1, 1].
    """
    dev = device or next(generator.parameters()).device
    generator.eval()
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(count, latent_dim, device=dev)
    labels = torch.full((count,), class_idx, dtype=torch.long, device=dev)
    imgs = generator(z, labels)
    return imgs
