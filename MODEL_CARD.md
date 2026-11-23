# Model Card: Text-to-Conditional-GAN Fashion Bridge

## Model Details
- **Model Name**: Text-to-Conditional-GAN Fashion Bridge (cGAN & cVAE)
- **Model Type**: Deep Convolutional Conditional Generative Adversarial Network (cGAN) and Conditional Variational Autoencoder (cVAE)
- **Primary Framework**: PyTorch
- **Intended Domain**: Multimodal text-guided synthetic fashion image synthesis and automated multimodal verification
- **Authors**: Senior ML Engineer & Technical Writer (Student CA Submission)
- **Date**: November 2026

## Intended Use
- **Primary Use Cases**:
  - Educational demonstration of end-to-end multimodal systems combining LLM natural language parsing, conditional generative modeling, and zero-shot cross-modal verification.
  - Comparative analysis of generative modeling paradigms (adversarial minimax optimization in cGAN vs. variational inference and maximum likelihood lower bound in cVAE).
  - Automated quality auditing of generative outputs using pretrained vision-language models (CLIP ViT-B/32) and supervised oracle classifiers.
- **Out-of-Scope Use Cases**:
  - High-resolution commercial fashion design or e-commerce asset generation.
  - Photorealistic human rendering (model outputs isolated 28x28 grayscale clothing silhouettes).
  - Safety-critical or biometric classification.

## Training Data & Preprocessing
- **Dataset**: Fashion-MNIST (Zalando Research)
  - 60,000 training examples, 10,000 testing examples.
  - 10 balanced categories: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.
  - Dimensions: $1 \times 28 \times 28$ single-channel grayscale.
- **Normalization**: Pixel values mapped to $[-1.0, 1.0]$ via $x_{norm} = \frac{x - 0.5}{0.5}$ to align with the Generator's hyperbolic tangent ($\tanh$) activation function.

## Architecture Specifications

### 1. Conditional GAN (cGAN)
- **Generator**:
  - Latent input: $z \sim \mathcal{N}(0, I) \in \mathbb{R}^{100}$.
  - Conditioning: Class label index embedded via `nn.Embedding(10, 100)`.
  - Projection: Dense layer mapping $\mathbb{R}^{200} \to 128 \times 7 \times 7$, followed by 2D Batch Normalization and LeakyReLU ($\alpha=0.2$).
  - Upsampling: Two transposed convolution stages ($7 \times 7 \to 14 \times 14 \to 28 \times 28$), followed by a $3 \times 3$ convolution with $\tanh$ output.
  - Parameter Count: $\approx 1.42\text{M}$ parameters (Checkpoint size: $5.7\text{ MB}$).
- **Discriminator**:
  - Input: $1 \times 28 \times 28$ image concatenated with a $1 \times 28 \times 28$ projected class embedding channel ($2 \times 28 \times 28$).
  - Feature Extractor: Two strided convolutional blocks with LeakyReLU and Spatial Dropout ($p=0.25$).
  - Classification Head: Linear projection to raw scalar logit.
  - Training Objective: Non-saturating BCE loss with one-sided label smoothing ($y_{real}=0.9, y_{fake}=0.0$).
  - Optimizer: Adam ($\text{lr}=2\text{e-}4, \beta_1=0.5, \beta_2=0.999$).

### 2. Conditional VAE (cVAE)
- **Encoder**: $2 \times 28 \times 28 \to \text{Conv2d} \to \mu \in \mathbb{R}^{20}, \log \sigma^2 \in \mathbb{R}^{20}$.
- **Reparameterization**: $z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$.
- **Decoder**: $z (20) + y_{emb} (20) \to 40 \to \text{ConvTranspose2d} \to 1 \times 28 \times 28$.
- **Loss**: Reconstruction MSE + $\beta \cdot \text{KL}$ with linear warm-up over training epochs.
- **Checkpoint size**: $3.2\text{ MB}$.

### 3. Multimodal Verifier (CLIP ViT-B/32)
- Pretrained model: `openai/clip-vit-base-patch32`.
- Preprocessing: Grayscale images are denormalized to $[0, 1]$, upscaled via bicubic interpolation from $28 \times 28 \to 224 \times 224$, and replicated across 3 RGB channels before standard ImageNet/CLIP normalization.
- Evaluation Prompts: `"a photo of a {class_label}"` across 10 categories.
- Dual Pass Criterion: Target class is top-1 OR target class probability $\ge 0.15$.

## Quantitative Evaluation Metrics
*(Values compiled from empirical runs via `evaluate.py`)*
- **End-to-End Fixed Benchmark Accuracy**: Evaluated across 10 diverse user queries covering plurals, number words, and synonyms.
- **Supervised CNN Classifier Test Accuracy**: $>91.5\%$ on real held-out Fashion-MNIST test set.
- **cGAN vs. cVAE Sharpness vs. Likelihood**: cGAN achieves crisper edge definition and higher CNN oracle classification accuracy; cVAE exhibits smooth average reconstructions with zero mode collapse.

## Weaknesses, Bias & Failure Modes
1. **Upper-Wear Class Confusion**:
   - High visual and morphological ambiguity between `Shirt` (Class 6), `T-shirt/top` (Class 0), `Pullover` (Class 2), and `Coat` (Class 4). At $28 \times 28$ grayscale, fine-grained cues (buttons, collar thickness, fabric knit) are heavily sub-sampled.
2. **Resolution & Cross-Domain Discrepancy in CLIP**:
   - CLIP ViT-B/32 was pretrained on high-resolution natural web images. Upscaling $28 \times 28$ synthetic grayscale silhouettes to $224 \times 224$ introduces bicubic smoothing artifacts, resulting in lower zero-shot confidence than a dedicated in-domain CNN.
3. **Out-of-Scope Prompts**:
   - Unconstrained LLMs risk hallucinating arbitrary fashion labels when prompted with non-fashion items ("cat", "car"). The integrated Pydantic validator and fallback guardrails strictly prevent and refuse out-of-scope requests.

## Safety & Governance Guardrails
- **Input Sanitization**: Rejects input strings $>150$ characters, strips HTML/shell metacharacters.
- **Count Cap**: Strictly clamps requests to $[1, 16]$ images.
- **Class Whitelist**: Enforces strict enum typing against the 10 canonical classes.
- **Rate Limiting**: Sliding-window rate limiter ($15\text{ req}/60\text{s}$) prevents CPU denial-of-service.
