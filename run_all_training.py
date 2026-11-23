"""
Master One-Command Training Pipeline for Text-to-Conditional-GAN Bridge.
Trains:
1. Supervised Fashion-MNIST CNN Classifier (checkpoints/classifier.pt)
2. Conditional Variational Autoencoder (checkpoints/cvae.pt + results/vae_tsne.png)
3. Conditional Generative Adversarial Network (checkpoints/cgan_generator.pt)
"""

import os
import sys
import time
import argparse

from train_classifier import train as train_classifier_fn
from train_vae import train_vae as train_vae_fn
from train_cgan import train_cgan as train_cgan_fn
from docs.generate_lifecycle import generate_diagram


def run_pipeline(smoke_test: bool = False):
    total_start = time.time()
    print("=" * 70)
    print("🚀 TEXT-TO-CONDITIONAL-GAN BRIDGE: COMPLETE TRAINING PIPELINE")
    print(f"   Mode: {'SMOKE TEST (Quick 2-epoch run)' if smoke_test else 'FULL CA SUBMISSION TRAINING'}")
    print("=" * 70)

    # 1. Lifecycle diagram
    print("\n[Stage 1/4] Generating Architecture & Lifecycle Diagrams...")
    generate_diagram("docs/lifecycle.png")

    # 2. Supervised CNN Classifier
    clf_epochs = 2 if smoke_test else 5
    print(f"\n[Stage 2/4] Training Supervised CNN Judge ({clf_epochs} epochs)...")
    train_classifier_fn(epochs=clf_epochs, batch_size=128)

    # 3. Conditional VAE
    vae_epochs = 2 if smoke_test else 10
    print(f"\n[Stage 3/4] Training Conditional VAE ({vae_epochs} epochs)...")
    train_vae_fn(epochs=vae_epochs, batch_size=128, latent_dim=20)

    # 4. Conditional GAN
    cgan_epochs = 3 if smoke_test else 25
    print(f"\n[Stage 4/4] Training Conditional GAN ({cgan_epochs} epochs)...")
    train_cgan_fn(epochs=cgan_epochs, batch_size=128, latent_dim=100, save_every=5)

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"🎉 ALL MODELS SUCCESSFULLY TRAINED & CHECKPOINTED in {total_time:.1f}s ({total_time/60:.1f} mins)!")
    print("   Checkpoints saved in: ./checkpoints/")
    print("   Sample grids saved in: ./results/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Training Script")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test with reduced epochs")
    args = parser.parse_args()

    run_pipeline(smoke_test=args.smoke_test)
