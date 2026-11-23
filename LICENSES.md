# Third-Party Licenses and Intellectual Property Notices

This project, **"Text-to-Conditional-GAN Bridge"**, utilizes open-source datasets, models, libraries, and APIs. Below is a comprehensive audit of all licensing terms verified against their official sources:

---

## 1. Fashion-MNIST Dataset
- **Provider**: Zalando Research (Han Xiao, Kashif Rasul, Roland Vollgraf)
- **Official Repository**: [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- **License**: MIT License
- **Terms Summary**:
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

---

## 2. OpenAI CLIP (Contrastive Language-Image Pre-Training)
- **Provider**: OpenAI
- **Model Checkpoint**: `openai/clip-vit-base-patch32` (via Hugging Face)
- **License**: MIT License
- **Terms Summary**:
  The CLIP architecture and published checkpoint weights are released under the MIT License for research and commercial applications, subject to standard copyright attribution.

---

## 3. Sentence-Transformers
- **Provider**: UKP Lab / Nils Reimers / Hugging Face
- **Model Checkpoint**: `all-MiniLM-L6-v2`
- **License**: Apache License 2.0
- **Terms Summary**:
  Permits commercial use, modification, distribution, and patent use, provided that appropriate credit is given and license notices are retained.

---

## 4. PyTorch & Torchvision
- **Provider**: Meta AI / PyTorch Foundation (Linux Foundation)
- **License**: BSD 3-Clause License
- **Terms Summary**:
  Redistribution and use in source and binary forms, with or without modification, are permitted provided that copyright notices and condition disclaimers are preserved.

---

## 5. Gradio Web Interface
- **Provider**: Hugging Face, Inc.
- **License**: Apache License 2.0
- **Terms Summary**:
  Open-source web interface library permitted for commercial and non-commercial deployment.

---

## 6. Large Language Model API Usage
- **Groq API**: Governed by Groq Developer Terms of Service. API access adheres to free-tier rate limits and user data privacy standards (prompts are not retained for training under default developer endpoints).
- **Google Gemini API**: Governed by Google Cloud / Google AI Studio Terms of Service.

---

## 7. Synthetic Output Intellectual Property
All generated clothing images produced by the trained cGAN and cVAE generators are low-resolution ($28 \times 28$) grayscale synthetic pixel arrays sampled from learned latent probability distributions. They do not contain copyrightable human depictions, proprietary trademarks, or personally identifiable information (PII).
