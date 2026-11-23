"""
Unit tests for model architectures, tensor shapes, and guardrails.
"""

import torch
import pytest
from src.cgan import ConditionalGenerator, ConditionalDiscriminator
from src.cvae import ConditionalVAE, vae_loss_function
from src.classifier import FashionCNN
from src.guardrails import sanitize_input, validate_guardrails, RateLimiter


def test_cgan_shapes():
    netG = ConditionalGenerator(latent_dim=100, num_classes=10)
    netD = ConditionalDiscriminator(num_classes=10)

    z = torch.randn(4, 100)
    labels = torch.tensor([0, 3, 7, 9], dtype=torch.long)

    # Generator test
    fake_imgs = netG(z, labels)
    assert fake_imgs.shape == (4, 1, 28, 28)
    assert fake_imgs.min() >= -1.0
    assert fake_imgs.max() <= 1.0

    # Discriminator test
    logits = netD(fake_imgs, labels)
    assert logits.shape == (4, 1)


def test_cvae_shapes_and_loss():
    cvae = ConditionalVAE(latent_dim=20, num_classes=10)
    imgs = torch.randn(4, 1, 28, 28).clamp(-1.0, 1.0)
    labels = torch.tensor([1, 4, 6, 8], dtype=torch.long)

    recon, mu, logvar = cvae(imgs, labels)
    assert recon.shape == (4, 1, 28, 28)
    assert mu.shape == (4, 20)
    assert logvar.shape == (4, 20)

    total_loss, recon_loss, kl_loss = vae_loss_function(recon, imgs, mu, logvar, beta=1.0)
    assert total_loss.item() > 0.0
    assert recon_loss.item() > 0.0


def test_classifier_shapes():
    clf = FashionCNN(num_classes=10)
    imgs = torch.randn(4, 1, 28, 28)
    logits = clf(imgs)
    assert logits.shape == (4, 10)


def test_guardrails_sanitizer():
    # Valid input
    ok, text, err = sanitize_input("show me four ankle boots")
    assert ok is True
    assert text == "show me four ankle boots"
    assert err == ""

    # Shell characters rejection
    ok_bad, _, err_bad = sanitize_input("four boots; rm -rf /")
    assert ok_bad is False
    assert "disallowed" in err_bad.lower()

    # Empty rejection
    ok_empty, _, err_empty = sanitize_input("   ")
    assert ok_empty is False

    # Overlength rejection
    ok_long, _, err_long = sanitize_input("a" * 200)
    assert ok_long is False


def test_guardrails_validation():
    ok1, _ = validate_guardrails("Ankle boot", 4)
    assert ok1 is True

    ok_bad_class, _ = validate_guardrails("Spaceship", 4)
    assert ok_bad_class is False

    ok_bad_count, _ = validate_guardrails("Dress", 50)
    assert ok_bad_count is False


def test_rate_limiter():
    limiter = RateLimiter(max_requests=3, window_seconds=2)
    assert limiter.is_allowed("user1")[0] is True
    assert limiter.is_allowed("user1")[0] is True
    assert limiter.is_allowed("user1")[0] is True
    # 4th request in window should be blocked
    allowed, msg = limiter.is_allowed("user1")
    assert allowed is False
    assert "Rate limit" in msg
