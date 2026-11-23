"""
BERT-Family Semantic Class Matcher (Task U3).
Uses SentenceTransformer (all-MiniLM-L6-v2) to map user input text
to the closest Fashion-MNIST category via cosine similarity in embedding space.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.utils import FASHION_CLASSES

_SENTENCE_MODEL = None

CLASS_SEMANTIC_ANCHORS = [
    "a t-shirt, top, tee, casual short sleeve shirt",
    "a pair of trousers, pants, jeans, slacks, chinos",
    "a pullover, sweater, jumper, hoodie, knitwear",
    "a dress, gown, frock, one-piece garment",
    "a coat, jacket, blazer, overcoat, parka, outerwear",
    "a pair of sandals, flip-flops, slides, open-toe footwear",
    "a button-down collared shirt, dress shirt, formal blouse",
    "a pair of athletic sneakers, trainers, running shoes, gym kicks",
    "a handbag, bag, purse, tote bag, shoulder bag, backpack",
    "a pair of ankle boots, chelsea boots, leather booties, footwear boots",
]


def get_sentence_model():
    """Lazily loads and caches the SentenceTransformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL

    # If offline mode or sandbox, avoid HTTP retries
    if os.getenv("OFFLINE_MODE", "0") == "1":
        return None

    try:
        from sentence_transformers import SentenceTransformer
        model_name = "all-MiniLM-L6-v2"
        _SENTENCE_MODEL = SentenceTransformer(model_name, local_files_only=True)
        return _SENTENCE_MODEL
    except Exception:
        return None


def match_class_bert(text: str) -> Dict[str, Any]:
    """
    Finds closest Fashion-MNIST class using SentenceTransformer embeddings.
    Returns:
      {
        "matched_class": str,
        "class_idx": int,
        "similarity": float,
        "all_scores": Dict[str, float],
        "method": str
      }
    """
    model = get_sentence_model()

    if model is None:
        # High quality offline fallback using normalized token overlap
        # Token stem overlap matching
        stemmed_query = {w.rstrip("s").lower() for w in text.replace(",", " ").split()}
        best_idx = 0
        best_score = -1.0
        scores = {}
        for idx, anchor in enumerate(CLASS_SEMANTIC_ANCHORS):
            anchor_stems = {w.rstrip("s").lower() for w in anchor.replace(",", " ").split()}
            intersection = len(stemmed_query.intersection(anchor_stems))
            score = intersection / (len(stemmed_query) + 1e-5)
            scores[FASHION_CLASSES[idx]] = score
            if score > best_score:
                best_score = score
                best_idx = idx

        # Default to highest if zero
        if best_score == 0.0:
            best_idx = 9  # default
            best_score = 0.5

        return {
            "matched_class": FASHION_CLASSES[best_idx],
            "class_idx": best_idx,
            "similarity": float(best_score),
            "all_scores": scores,
            "method": "lexical_semantic_fallback"
        }

    # Encode query and class semantic anchors
    query_emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    anchor_embs = model.encode(CLASS_SEMANTIC_ANCHORS, convert_to_tensor=True, normalize_embeddings=True)

    # Cosine similarities
    cos_sims = torch.matmul(anchor_embs, query_emb).cpu().numpy()
    best_idx = int(np.argmax(cos_sims))
    best_sim = float(cos_sims[best_idx])

    all_scores = {FASHION_CLASSES[i]: float(cos_sims[i]) for i in range(len(FASHION_CLASSES))}

    return {
        "matched_class": FASHION_CLASSES[best_idx],
        "class_idx": best_idx,
        "similarity": best_sim,
        "all_scores": all_scores,
        "method": "all-MiniLM-L6-v2"
    }
