"""
Utility functions, constants, and reproducibility helpers for the
Text-to-Conditional-GAN Bridge project.
"""

import os
import random
import numpy as np
import torch

# Fashion-MNIST 10 official class labels in index order
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Mapping from class name to integer label
CLASS_TO_IDX = {name: idx for idx, name in enumerate(FASHION_CLASSES)}

# Common synonyms and plural normalizations for clothing items
SYNONYM_MAP = {
    # T-shirt/top
    "t-shirt": "T-shirt/top",
    "tshirt": "T-shirt/top",
    "t shirt": "T-shirt/top",
    "tee": "T-shirt/top",
    "top": "T-shirt/top",
    "tees": "T-shirt/top",
    "t-shirts": "T-shirt/top",
    "tshirts": "T-shirt/top",
    "tank top": "T-shirt/top",
    # Trouser
    "trouser": "Trouser",
    "trousers": "Trouser",
    "pants": "Trouser",
    "pant": "Trouser",
    "jeans": "Trouser",
    "slacks": "Trouser",
    "chinos": "Trouser",
    "sweatpants": "Trouser",
    "bottoms": "Trouser",
    # Pullover
    "pullover": "Pullover",
    "pullovers": "Pullover",
    "sweater": "Pullover",
    "sweaters": "Pullover",
    "jumper": "Pullover",
    "jumpers": "Pullover",
    "hoodie": "Pullover",
    "hoodies": "Pullover",
    "sweatshirt": "Pullover",
    "sweatshirts": "Pullover",
    # Dress
    "dress": "Dress",
    "dresses": "Dress",
    "gown": "Dress",
    "gowns": "Dress",
    "frock": "Dress",
    "frocks": "Dress",
    # Coat
    "coat": "Coat",
    "coats": "Coat",
    "jacket": "Coat",
    "jackets": "Coat",
    "blazer": "Coat",
    "blazers": "Coat",
    "overcoat": "Coat",
    "parka": "Coat",
    "trench": "Coat",
    # Sandal
    "sandal": "Sandal",
    "sandals": "Sandal",
    "flip-flop": "Sandal",
    "flip-flops": "Sandal",
    "flip flops": "Sandal",
    "slides": "Sandal",
    "slide": "Sandal",
    "open toe shoes": "Sandal",
    # Shirt
    "shirt": "Shirt",
    "shirts": "Shirt",
    "button-down": "Shirt",
    "button down": "Shirt",
    "button-up": "Shirt",
    "collared shirt": "Shirt",
    "dress shirt": "Shirt",
    "blouse": "Shirt",
    "blouses": "Shirt",
    # Sneaker
    "sneaker": "Sneaker",
    "sneakers": "Sneaker",
    "trainer": "Sneaker",
    "trainers": "Sneaker",
    "running shoe": "Sneaker",
    "running shoes": "Sneaker",
    "tennis shoe": "Sneaker",
    "tennis shoes": "Sneaker",
    "kicks": "Sneaker",
    "athletic shoes": "Sneaker",
    # Bag
    "bag": "Bag",
    "bags": "Bag",
    "handbag": "Bag",
    "handbags": "Bag",
    "purse": "Bag",
    "purses": "Bag",
    "tote": "Bag",
    "totes": "Bag",
    "backpack": "Bag",
    "backpacks": "Bag",
    "clutch": "Bag",
    # Ankle boot
    "ankle boot": "Ankle boot",
    "ankle boots": "Ankle boot",
    "boot": "Ankle boot",
    "boots": "Ankle boot",
    "bootie": "Ankle boot",
    "booties": "Ankle boot",
    "chelsea boot": "Ankle boot",
    "chelsea boots": "Ankle boot",
}

# Number words mapping to integers
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "a": 1, "an": 1, "single": 1, "couple": 2, "pair": 2,
    "few": 3, "several": 4,
}


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return torch device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
