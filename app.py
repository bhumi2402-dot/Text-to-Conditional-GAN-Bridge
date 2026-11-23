"""
Text-to-Conditional-GAN Bridge: Gradio Web Application.
Connects Natural Language Prompts -> LLM Structured Parsing -> cGAN Synthesis -> CLIP Verification.
Also features cVAE side-by-side comparison, empirical metrics dashboard, and architecture notes.
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = ".mpl_cache"

import json
import time
from typing import List, Tuple, Optional
import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.utils import FASHION_CLASSES, CLASS_TO_IDX, get_device, set_seed
from src.llm_parser import parse_request
from src.cgan import load_cgan_generator, sample_cgan
from src.cvae import load_cvae, sample_cvae
from src.classifier import load_classifier, predict_cnn
from src.clip_verify import verify_with_clip
from src.guardrails import sanitize_input, validate_guardrails, RateLimiter

# Initialize global rate limiter
rate_limiter = RateLimiter(max_requests=25, window_seconds=60)

# Device setup
device = get_device()

# Lazy model references
netG = None
cvae = None
classifier = None


def get_models():
    """Lazily loads model checkpoints."""
    global netG, cvae, classifier
    if netG is None and os.path.exists("checkpoints/cgan_generator.pt"):
        netG = load_cgan_generator("checkpoints/cgan_generator.pt", device=device)
    if cvae is None and os.path.exists("checkpoints/cvae.pt"):
        cvae = load_cvae("checkpoints/cvae.pt", device=device)
    if classifier is None and os.path.exists("checkpoints/classifier.pt"):
        classifier = load_classifier("checkpoints/classifier.pt", device=device)
    return netG, cvae, classifier


def tensor_to_pil_list(imgs: torch.Tensor) -> List[Image.Image]:
    """Converts a batch of (B, 1, 28, 28) tensors in [-1, 1] to a list of PIL Images."""
    imgs_01 = (imgs + 1.0) / 2.0
    imgs_01 = torch.clamp(imgs_01, 0.0, 1.0)
    imgs_np = (imgs_01.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    pil_imgs = [Image.fromarray(img, mode="L").resize((112, 112), Image.Resampling.NEAREST) for img in imgs_np]
    return pil_imgs


def generate_fashion(
    prompt: str,
    compare_vae: bool,
    provider: str,
    prompt_technique: str
):
    """
    Main user-facing pipeline:
    1. Input sanitization & rate limit check
    2. LLM / Fallback structured JSON parsing
    3. cGAN generation
    4. Multimodal CLIP & CNN verification
    5. Optional cVAE comparison synthesis
    """
    # Rate limit check
    allowed, rate_msg = rate_limiter.is_allowed()
    if not allowed:
        return (
            {"error": rate_msg},
            f"⚠️ **Rate Limit Alert**: {rate_msg}",
            [],
            [],
            gr.update(visible=False)
        )

    # Input sanitization
    is_valid, cleaned_prompt, sanitize_err = sanitize_input(prompt)
    if not is_valid:
        return (
            {"error": sanitize_err},
            f"❌ **Input Error**: {sanitize_err}",
            [],
            [],
            gr.update(visible=False)
        )

    # 1. Parse Prompt to JSON
    parse_result = parse_request(cleaned_prompt, provider=provider.lower(), technique=prompt_technique)
    if not parse_result.success:
        return (
            {"error": parse_result.error_message, "method": parse_result.method_used},
            f"🛡️ **Refusal / Parse Warning**: {parse_result.error_message}",
            [],
            [],
            gr.update(visible=False)
        )

    req = parse_result.data
    class_label = req.class_label.value
    count = req.count
    class_idx = CLASS_TO_IDX[class_label]

    # Validate guardrails
    g_valid, g_msg = validate_guardrails(class_label, count)
    if not g_valid:
        return (
            {"error": g_msg},
            f"⚠️ **Guardrail Warning**: {g_msg}",
            [],
            [],
            gr.update(visible=False)
        )

    # Load models
    g_model, v_model, clf = get_models()
    if g_model is None:
        return (
            {"error": "cGAN checkpoint not found. Run training script first."},
            "❌ Model checkpoints missing. Train models first.",
            [],
            [],
            gr.update(visible=False)
        )

    # 2. cGAN Generation
    seed = int(time.time() * 1000) % 100000
    cgan_tensors = sample_cgan(g_model, class_idx, count=count, device=device, seed=seed)
    cgan_pil = tensor_to_pil_list(cgan_tensors)

    # 3. Multimodal CLIP Verification
    clip_verdicts = verify_with_clip(cgan_tensors, class_idx, device=device)

    # 4. Supervised CNN Judge
    cnn_preds, cnn_confs, cnn_names = predict_cnn(clf, cgan_tensors, device=device)

    # Format cGAN Gallery
    cgan_gallery_items = []
    pass_count = 0
    for i in range(count):
        v = clip_verdicts[i]
        is_pass = v["pass"]
        if is_pass:
            pass_count += 1
        clip_str = f"CLIP: {'PASS ✅' if is_pass else 'FAIL ❌'} ({v['confidence']*100:.1f}%)"
        cnn_str = f"CNN: {cnn_names[i]} ({cnn_confs[i]*100:.1f}%)"
        caption = f"#{i+1} {class_label} | {clip_str} | {cnn_str}"
        cgan_gallery_items.append((cgan_pil[i], caption))

    # 5. Optional cVAE Comparison Generation
    cvae_gallery_items = []
    if compare_vae and v_model is not None:
        cvae_tensors = sample_cvae(v_model, class_idx, count=count, device=device, seed=seed)
        cvae_pil = tensor_to_pil_list(cvae_tensors)
        _, cvae_confs, cvae_names = predict_cnn(clf, cvae_tensors, device=device)

        for i in range(count):
            caption = f"cVAE #{i+1} | Oracle CNN: {cvae_names[i]} ({cvae_confs[i]*100:.1f}%)"
            cvae_gallery_items.append((cvae_pil[i], caption))

    json_output = {
        "class_label": class_label,
        "count": count,
        "class_index": class_idx,
        "method_used": parse_result.method_used,
        "prompt_technique": prompt_technique,
        "clip_pass_rate": f"{pass_count}/{count} ({pass_count/count*100:.0f}%)",
        "device": str(device)
    }

    status_markdown = (
        f"✅ **Generation Successful**: Synthesized **{count}** images of category **'{class_label}'**.<br>"
        f"📊 **CLIP Verification**: **{pass_count}/{count}** images passed the semantic cross-modal test. "
        f"Inference performed on `{device}`."
    )

    return (
        json_output,
        status_markdown,
        cgan_gallery_items,
        cvae_gallery_items,
        gr.update(visible=compare_vae)
    )


def compare_models_tab(class_choice: str):
    """Generates side-by-side samples for the dedicated comparison tab."""
    g_model, v_model, clf = get_models()
    if g_model is None or v_model is None or clf is None:
        return [], [], "Model checkpoints not loaded."

    class_idx = CLASS_TO_IDX[class_choice]
    count = 6
    seed = 42

    cgan_t = sample_cgan(g_model, class_idx, count=count, device=device, seed=seed)
    cvae_t = sample_cvae(v_model, class_idx, count=count, device=device, seed=seed)

    cgan_pil = tensor_to_pil_list(cgan_t)
    cvae_pil = tensor_to_pil_list(cvae_t)

    _, cgan_confs, cgan_names = predict_cnn(clf, cgan_t, device=device)
    _, cvae_confs, cvae_names = predict_cnn(clf, cvae_t, device=device)

    cgan_items = [(cgan_pil[i], f"cGAN #{i+1} ({cgan_confs[i]*100:.1f}%)") for i in range(count)]
    cvae_items = [(cvae_pil[i], f"cVAE #{i+1} ({cvae_confs[i]*100:.1f}%)") for i in range(count)]

    analysis_md = (
        f"### Qualitative & Architectural Analysis for **{class_choice}**\n\n"
        f"- **cGAN (Generative Adversarial Network)**: Shows sharper high-frequency edges and distinct contours due to the "
        f"adversarial discriminator penalizing blurry averages ($L_{{BCE}}$). Risk: minor textural artifacts or mode collapse.\n"
        f"- **cVAE (Variational Autoencoder)**: Produces smoother, continuous density samples resulting from optimizing the "
        f"Variational Lower Bound (ELBO) with Gaussian latent prior $\\mathcal{{N}}(0, I)$. Exhibits zero mode collapse but blurrier edges.\n"
        f"- **CNN Oracle Confidence**: cGAN Average = `{cgan_confs.mean()*100:.1f}%` vs. cVAE Average = `{cvae_confs.mean()*100:.1f}%`."
    )

    return cgan_items, cvae_items, analysis_md


# Custom Gradio Theme & Layout (System fonts only to prevent network hang)
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=["Segoe UI", "Helvetica Neue", "sans-serif"]
)

with gr.Blocks(title="Text-to-Conditional-GAN Fashion Bridge") as demo:
    gr.Markdown(
        """
        # 👗 Text-to-Conditional-GAN Bridge
        ### *Multimodal Deep Learning System for Fashion-MNIST Synthesis & Cross-Modal Verification*
        **Workflow**: Natural Language Request $\\to$ LLM Structured JSON $\\to$ Deep cGAN Synthesis $\\to$ Multimodal CLIP & CNN Verification
        """
    )

    with gr.Tabs():
        # TAB 1: INTERACTIVE GENERATOR
        with gr.TabItem("🎨 Interactive Generator"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(
                        label="User Request (Natural Language)",
                        placeholder="e.g. 'show me four ankle boots' or 'a couple of warm hoodies'",
                        value="show me four ankle boots",
                        lines=2
                    )
                    with gr.Row():
                        compare_toggle = gr.Checkbox(label="Compare with cVAE", value=True)
                        provider_dropdown = gr.Dropdown(
                            label="LLM Parser Provider",
                            choices=["Groq", "Rule_Based"],
                            value="Groq"
                        )
                        technique_dropdown = gr.Dropdown(
                            label="Prompt Technique",
                            choices=["few_shot", "zero_shot", "chain_of_thought"],
                            value="few_shot"
                        )
                    generate_btn = gr.Button("✨ Generate Fashion Items", variant="primary", size="lg")

                    status_box = gr.Markdown("Ready for generation.")
                    json_display = gr.JSON(label="Structured Pydantic JSON")

                with gr.Column(scale=5):
                    gr.Markdown("### 🌟 Synthesized cGAN Output (with Multimodal Verifications)")
                    cgan_gallery = gr.Gallery(label="cGAN Images", columns=4, height="auto")

                    with gr.Group(visible=True) as vae_group:
                        gr.Markdown("### 🔬 Conditional VAE Comparison Output")
                        cvae_gallery = gr.Gallery(label="cVAE Images", columns=4, height="auto")

            generate_btn.click(
                fn=generate_fashion,
                inputs=[prompt_input, compare_toggle, provider_dropdown, technique_dropdown],
                outputs=[json_display, status_box, cgan_gallery, cvae_gallery, vae_group]
            )

        # TAB 2: CGAN VS CVAE COMPARISON
        with gr.TabItem("⚖️ cGAN vs. cVAE Comparison"):
            gr.Markdown("### Direct Empirical Comparison across the 10 Fashion-MNIST Classes")
            with gr.Row():
                class_selector = gr.Dropdown(
                    label="Select Category",
                    choices=FASHION_CLASSES,
                    value="Ankle boot"
                )
                run_compare_btn = gr.Button("Compare Generative Models", variant="secondary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### cGAN Generated Samples (Sharp Edges, Adversarial)")
                    comp_cgan_gallery = gr.Gallery(label="cGAN", columns=3)
                with gr.Column():
                    gr.Markdown("#### cVAE Generated Samples (Smooth, Likelihood / ELBO)")
                    comp_cvae_gallery = gr.Gallery(label="cVAE", columns=3)

            comp_analysis = gr.Markdown("")
            run_compare_btn.click(
                fn=compare_models_tab,
                inputs=[class_selector],
                outputs=[comp_cgan_gallery, comp_cvae_gallery, comp_analysis]
            )

        # TAB 3: METRICS & VISUALIZATIONS
        with gr.TabItem("📊 Metrics & Visualizations"):
            gr.Markdown("## Rigorous Empirical Benchmarks & Evaluation Artifacts")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔄 Machine Learning Lifecycle")
                    if os.path.exists("docs/lifecycle.png"):
                        gr.Image("docs/lifecycle.png", label="Lifecycle Diagram")
                    else:
                        gr.Markdown("*Lifecycle diagram generating...*")

                with gr.Column():
                    gr.Markdown("### 🌌 VAE Latent Space Projection (t-SNE)")
                    if os.path.exists("results/vae_tsne.png"):
                        gr.Image("results/vae_tsne.png", label="t-SNE Latent Space")
                    else:
                        gr.Markdown("*t-SNE plot generating...*")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📉 cGAN Confusion Matrix")
                    if os.path.exists("results/confusion_matrix_cgan.png"):
                        gr.Image("results/confusion_matrix_cgan.png", label="cGAN Confusion Matrix")
                with gr.Column():
                    gr.Markdown("### 📉 cVAE Confusion Matrix")
                    if os.path.exists("results/confusion_matrix_cvae.png"):
                        gr.Image("results/confusion_matrix_cvae.png", label="cVAE Confusion Matrix")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 Training Loss Dynamics")
                    if os.path.exists("results/cgan_losses.png"):
                        gr.Image("results/cgan_losses.png", label="cGAN Losses")
                with gr.Column():
                    gr.Markdown("### 📈 cVAE Loss Dynamics (with KL Annealing)")
                    if os.path.exists("results/cvae_losses.png"):
                        gr.Image("results/cvae_losses.png", label="cVAE Losses")

        # TAB 4: ARCHITECTURE & VIVA NOTES
        with gr.TabItem("📖 Architecture & Theory Notes"):
            gr.Markdown(
                """
                ## Continuous Assessment Theoretical Explanations

                ### 1. Generative vs. Discriminative Modeling Framing
                - **Discriminative models** model the conditional probability $P(Y \\mid X)$, mapping inputs directly to category boundaries (e.g., our CNN Classifier predicting `Ankle boot` from pixel arrays). They focus strictly on decision boundaries and discard features not useful for discrimination.
                - **Generative models** capture the joint distribution $P(X, Y) = P(X \\mid Y) P(Y)$ or density $P(X)$. They learn the underlying data distribution enabling synthetic sample generation ($x \\sim P(X \\mid Y)$).
                - **In this project**: The user specifies $Y$ via natural text; the cGAN and cVAE sample $X \\sim P(X \\mid Y)$; and the discriminative CNN/CLIP models estimate $P(Y \\mid X)$ to audit generation fidelity.

                ### 2. Adversarial Minimax Formulation
                The conditional GAN is formulated as a two-player zero-sum game:
                $$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x, y \\sim p_{data}}[\\log D(x, y)] + \\mathbb{E}_{z \\sim p_z, y \\sim p_{label}}[\\log (1 - D(G(z, y), y))]$$
                - **Discriminator ($D$)** acts as an adaptive loss function, learning to distinguish true fashion items from synthetic outputs while conditioning on label $y$.
                - **Generator ($G$)** learns to fool $D$ by synthesizing realistic high-frequency edge textures.
                - Unlike mean-squared error (which averages modes and creates blurry pixels), the adversarial feedback forces $G$ to sample crisp individual data modes.

                ### 3. Transformer Self-Attention in Resolving "four ankle boots"
                In the LLM parsing phase, multi-head self-attention:
                $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
                resolves compositional dependencies:
                - The modifier token `"four"` (quantity) casts strong attention heads to the nominal head `"ankle boots"` (fashion item).
                - Self-attention binds the noun phrase `[four] -> [ankle boots]` rather than assigning the count to a dangling modifier or preceding clause.
                - Subword tokenization breaks `"ankle boots"` into `["ankle", "boots"]`, where multi-head attention aggregates the compound noun into a single semantic embedding that matches the target schema category.

                ### 4. Prompt-Based Learning vs. Fine-Tuning
                - **Prompt-Based Zero/Few-Shot Learning**:
                  - *Mechanism*: Freezes LLM weights and conditions output format via in-context exemplars and strict schema prompts.
                  - *Strengths*: Zero training latency, instant adaptation, handles synonym variations out-of-the-box.
                  - *Best choice here*: Fashion-MNIST query extraction is a low-cardinality semantic parsing task easily solved by in-context few-shot learning without needing parameter updates.
                - **Fine-Tuning (SFT / LoRA)**:
                  - *Mechanism*: Updates model weights on domain-specific $(X, Y)$ pairs.
                  - *When better*: Essential when teaching novel terminology, domain-specific syntax, or deploying ultra-compact edge models (<1B params) where in-context adherence is inconsistent.
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=custom_theme, share=False)
