# Demo Video Script: Text-to-Conditional-GAN Fashion Bridge

**Estimated Duration**: 2 minutes 30 seconds to 3 minutes  
**Format**: Screen recording with voiceover narration  
**Objective**: Demonstrate end-to-end functionality, multimodal verification, cGAN vs. cVAE architectural trade-offs, and responsible AI guardrails for Continuous Assessment viva submission.

---

## Scene Breakdown & Shot List

### Scene 1: Title & System Overview (0:00 – 0:30)
- **Visual**: Screen shows the Gradio web interface running locally or on Hugging Face Spaces. The cursor highlights the title and the four tabs (`Interactive Generator`, `cGAN vs. cVAE Comparison`, `Metrics & Visualizations`, `Architecture & Theory Notes`).
- **Action**: Pan down to the text box containing the default prompt: `"show me four ankle boots"`.
- **Spoken Narration**:
  > *"Welcome to the demonstration of the Text-to-Conditional-GAN Fashion Bridge. In this project, we built an end-to-end multimodal machine learning system that translates unconstrained natural language into conditionally synthesized fashion items using Fashion-MNIST.*
  > *Our architecture combines four key paradigms: first, LLM-based structured semantic parsing; second, deep conditional adversarial synthesis; third, multimodal zero-shot verification using OpenAI's CLIP ViT-B/32; and fourth, a comparative analysis against a Conditional Variational Autoencoder."*

---

### Scene 2: Structured Parsing & Guardrails (0:30 – 1:00)
- **Visual**: Focus on the input card. Type: `"show me four ankle boots"`. Click **"Generate Fashion Items"**.
- **Action**: The JSON display renders:
  ```json
  {
    "class_label": "Ankle boot",
    "count": 4,
    "method_used": "groq_api",
    "prompt_technique": "few_shot"
  }
  ```
- **Action**: Quickly type an edge-case prompt: `"three warm hoodies"` $\to$ parses to `Pullover, 3`. Then type an out-of-scope query: `"show me a cute cat"` $\to$ triggers the safety banner: *"Refusal: 'cat' is not an available Fashion-MNIST category."*
- **Spoken Narration**:
  > *"Under the hood, our parser extracts the target category and quantity into a strictly validated Pydantic schema. It seamlessly normalizes clothing plurals, synonyms—like 'hoodies' to 'Pullover' or 'trainers' to 'Sneaker'—and converts number words.*
  > *Notice our responsible AI guardrail: when an out-of-distribution request like 'show me a cute cat' is submitted, the system refuses gracefully without hallucination or crash, protecting downstream generative compute."*

---

### Scene 3: Conditional GAN Synthesis & Multimodal CLIP Verification (1:00 – 1:40)
- **Visual**: The gallery populates with four synthesized $28 \times 28$ ankle boots.
- **Action**: Mouse hover over each generated image card showing the two badges:
  - `CLIP: PASS ✅ (91.4%)`
  - `CNN: Ankle boot (96.2%)`
- **Spoken Narration**:
  > *"Once validated, the cGAN Generator takes a 100-dimensional Gaussian noise vector conditioned on the class embedding to synthesize crisp single-channel fashion silhouettes.*
  > *To verify output alignment, we pass each generated image to two independent judges:*
  > *First, OpenAI's CLIP ViT-B/32, which upscales the 28-by-28 image to 224-by-224 and scores it against textual prompts like 'a photo of an ankle boot'.*
  > *Second, our supervised Fashion-MNIST CNN oracle with over 91% test accuracy.*
  > *Both models confirm that the generated items strictly match the user's intent."*

---

### Scene 4: cGAN vs. cVAE Comparison (1:40 – 2:15)
- **Visual**: Toggle to the **"⚖️ cGAN vs. cVAE Comparison"** tab. Select `"Sneaker"` from the dropdown and click **"Compare Generative Models"**.
- **Action**: Side-by-side galleries appear: 6 cGAN sneakers on the left, 6 cVAE sneakers on the right.
- **Spoken Narration**:
  > *"Here we examine the fundamental trade-off between GANs and Variational Autoencoders.*
  > *On the left, the conditional GAN uses an adversarial discriminator that penalizes blurry outputs, yielding sharp, high-frequency contours and sole outlines.*
  > *On the right, the conditional VAE optimizes the Evidence Lower Bound under a Gaussian prior. It offers continuous density coverage with zero mode collapse, but because of the pixel-wise mean squared error loss, its outputs are visibly smoother and softer around the perimeter.*
  > *Our oracle CNN confirms that cGAN achieves higher per-class confidence on structural items like footwear."*

---

### Scene 5: Metrics, Latent Space t-SNE & Failure Modes (2:15 – 2:45)
- **Visual**: Click on the **"📊 Metrics & Visualizations"** tab. Scroll through:
  1. The End-to-End Lifecycle diagram (`docs/lifecycle.png`).
  2. The 2D t-SNE projection of the VAE latent space (`results/vae_tsne.png`).
  3. The confusion matrix (`results/confusion_matrix_cgan.png`).
- **Action**: Point cursor to the confusion matrix between `Shirt` and `T-shirt/top`.
- **Spoken Narration**:
  > *"In our empirical evaluation tab, every metric is computed from actual pipeline executions.*
  > *Our 2D t-SNE plot demonstrates clear class separation in the VAE latent manifold.*
  > *Looking at our confusion matrix, we document an authentic failure mode: visual ambiguity between 'Shirt', 'T-shirt/top', and 'Coat'. At 28-by-28 resolution, collar details and buttons are sub-sampled, causing occasional classifier confusion—a documented limitation discussed in our Model Card."*

---

### Scene 6: Conclusion & Viva Wrap-up (2:45 – 3:00)
- **Visual**: Return to the main generator tab or the project README on GitHub.
- **Spoken Narration**:
  > *"In summary, the Text-to-Conditional-GAN Bridge demonstrates complete multimodal integration: robust LLM parsing, adversarial generation, cross-modal verification, and responsible AI guardrails, all packaged in a lightweight, CPU-efficient Gradio app ready for Hugging Face Spaces.*
  > *Thank you for watching."*
