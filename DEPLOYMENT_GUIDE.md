# Step-by-Step Deployment Guide: Hugging Face Spaces

This guide walks through deploying the **"Text-to-Conditional-GAN Bridge"** app to a public, free-tier Hugging Face Space running on CPU.

---

## Prerequisites
1. A free Hugging Face account at [huggingface.co](https://huggingface.co).
2. Git installed on your local machine.
3. Your Groq API key (free tier from [console.groq.com](https://console.groq.com/keys)).

---

## Step 1: Create a New Space on Hugging Face
1. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Enter the Space name: `text-to-cgan-bridge` (or your preferred name).
3. Select the Space SDK: **Gradio**.
4. Select the Space Hardware: **CPU Basic (2 vCPU · 16 GB RAM · Free)**.
5. Set Space visibility to **Public**.
6. Click **Create Space**.

---

## Step 2: Configure Space Secrets (API Key)
> ⚠️ **CRITICAL SECURITY NOTE**: Never commit `.env` or hardcode API keys in `app.py` or `README.md`. Hugging Face Spaces injects secrets securely as environment variables.

1. In your newly created Space, click on the **Settings** tab.
2. Scroll down to the **Variables and secrets** section.
3. Click **New secret**.
4. Set:
   - **Name**: `GROQ_API_KEY`
   - **Value**: `gsk_...` (paste your personal Groq API key)
5. (Optional) If using Gemini API:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: `AIza...`
6. Click **Save**.

---

## Step 3: Clone Your Space Locally and Push Code
In your local terminal, initialize git in the project repository and connect your Hugging Face Space remote:

```bash
cd C:\Users\shahb\.gemini\antigravity\scratch\p14-text-to-cgan-bridge

# Initialize git if not already initialized
git init
git branch -M main

# Add the Hugging Face Space remote (replace <username> with your HF username)
git remote add origin https://huggingface.co/spaces/<username>/text-to-cgan-bridge

# Ensure small checkpoints (<20MB) are tracked
git add app.py requirements.txt README.md MODEL_CARD.md LICENSES.md
git add src/ docs/ results/ checkpoints/

# Commit files
git commit -m "Deploy Text-to-Conditional-GAN Bridge to Hugging Face Spaces"

# Push to Hugging Face
git push -u origin main
```

*(Note: Hugging Face will prompt you for your Hugging Face username and Personal Access Token with write permissions).*

---

## Step 4: Verification and Live App Testing
1. Return to your Space page on Hugging Face.
2. Observe the **Building** status in the top bar. The Space will install dependencies from `requirements.txt` and launch `app.py`.
3. Once the status turns to **Running**, test the interface:
   - Enter: `"show me four ankle boots"`
   - Click: **Generate Fashion Items**
   - Verify that:
     1. Structured JSON output appears: `{"class_label": "Ankle boot", "count": 4}`.
     2. Four $28 \times 28$ synthetic images render in the gallery.
     3. CLIP returns `PASS ✅` with confidence percentages.
     4. Supervised CNN oracle confirms category predictions.
     5. The **"Compare with cVAE"** toggle displays parallel smooth reconstruction samples.

---

## Troubleshooting Common Issues

| Symptom | Root Cause | Solution |
| :--- | :--- | :--- |
| `KeyError: GROQ_API_KEY` | Secret missing in Space settings | Add `GROQ_API_KEY` in Space Settings > Variables & Secrets. The app will automatically fall back to rule-based regex parsing if the key is missing. |
| Space restart or slow startup | CLIP model downloading synchronously | CLIP is loaded lazily and cached via `get_clip_model()`. First inference takes ~4 seconds; subsequent queries take <200ms. |
| Checkpoint file rejection (>50MB) | File exceeds non-LFS size | Our checkpoints are intentionally lightweight ($5.7\text{ MB}$ for cGAN, $3.2\text{ MB}$ for cVAE, $1.6\text{ MB}$ for CNN) and require no Git LFS. |
| Out of memory on free tier | Large mini-batches during generation | Generation batches are capped at 16 images ($16 \times 1 \times 28 \times 28$), using $<20\text{ MB}$ RAM on CPU. |
