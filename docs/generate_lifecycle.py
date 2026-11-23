"""
Generate lifecycle diagram (data -> train -> evaluate -> deploy -> monitor)
for README and report.
"""

import os
import tempfile

# Force headless Agg backend and local cache directory
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram(output_path: str = "docs/lifecycle.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    stages = [
        {
            "name": "1. Data Pipeline",
            "desc": "Fashion-MNIST (60k 28x28)\n[-1, 1] Normalization\nOne-hot Class Embeddings\n10 Target Categories",
            "color": "#3498db"
        },
        {
            "name": "2. Offline Training",
            "desc": "cGAN: z(100)+y -> Gen/Disc\ncVAE: Enc/Dec + KL Warmup\nCNN Judge: >91% Acc\nLoss & Weight Checkpoints",
            "color": "#9b59b6"
        },
        {
            "name": "3. Multimodal Evaluation",
            "desc": "CLIP ViT-B/32 Cross-Modal\nSentence-BERT Matching\nConfusion Matrices\nQualitative GAN vs VAE",
            "color": "#e67e22"
        },
        {
            "name": "4. Deployment",
            "desc": "Hugging Face Spaces (CPU)\nGradio Interactive App\nPydantic Schema Parser\nLazy Model Caching (<20MB)",
            "color": "#2ecc71"
        },
        {
            "name": "5. Monitoring & Safety",
            "desc": "Sliding Window Rate Limiter\nInput Sanitization & Length Cap\nOOD Fallback Refusal\nModel Cards & License Tracking",
            "color": "#e74c3c"
        }
    ]

    n = len(stages)
    box_w = 2.2
    box_h = 2.4
    spacing = 0.5
    start_x = 0.5
    y = 1.3

    for i, s in enumerate(stages):
        x = start_x + i * (box_w + spacing)

        # Draw main card
        card = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor="white",
            edgecolor=s["color"],
            linewidth=2.5,
            zorder=3
        )
        ax.add_patch(card)

        # Draw header banner
        header = patches.FancyBboxPatch(
            (x, y + box_h - 0.6), box_w, 0.6,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=s["color"],
            edgecolor=s["color"],
            zorder=4
        )
        ax.add_patch(header)

        # Header text
        ax.text(
            x + box_w / 2.0, y + box_h - 0.3,
            s["name"],
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color="white", zorder=5
        )

        # Body text
        ax.text(
            x + box_w / 2.0, y + (box_h - 0.6) / 2.0,
            s["desc"],
            ha="center", va="center",
            fontsize=9.5, color="#2c3e50",
            linespacing=1.4, zorder=5
        )

        # Arrow to next stage
        if i < n - 1:
            arrow_x = x + box_w + 0.08
            arrow_w = spacing - 0.16
            ax.annotate(
                "",
                xy=(arrow_x + arrow_w, y + box_h / 2.0),
                xytext=(arrow_x, y + box_h / 2.0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#7f8c8d",
                    lw=3.0,
                    mutation_scale=20
                ),
                zorder=2
            )

    ax.set_xlim(0, start_x + n * box_w + (n - 1) * spacing + 0.5)
    ax.set_ylim(0.5, 4.2)
    ax.axis("off")

    plt.title(
        "Text-to-Conditional-GAN Bridge: End-to-End Machine Learning Lifecycle",
        fontsize=14, fontweight="bold", pad=20, color="#1a252f"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated lifecycle diagram at {output_path}")

if __name__ == "__main__":
    generate_diagram()
