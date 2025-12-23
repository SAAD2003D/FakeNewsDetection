# 🛡️ FusionHCAT: Hybrid Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![HuggingFace](https://img.shields.io/badge/Transformers-RoBERTa-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

**FusionHCAT** (Hybrid Context-Aware Transformer) is a state-of-the-art fake news detection system designed to operate under strict resource constraints (**No External Search/RAG** and **Limited VRAM**). 

Unlike traditional models that rely solely on text or solely on social graphs, this project introduces a **Hybrid Architecture** that fuses:
1.  **Linguistic Stylometry:** Using **RoBERTa-base** to detect sensationalism and clickbait patterns.
2.  **Metadata Profiling:** Using **Embedding Layers** to model speaker credibility and political bias.
3.  **LLM Cascade:** A fallback mechanism that triggers an Expert LLM (**Llama 3 via Groq**) only when the local model is uncertain.

This system was trained on a unified dataset of **57,000+ samples** (combining LIAR and ISOT) and achieves **90% accuracy** on general fake news while maintaining robustness on nuanced political discourse.

---

## 🚀 Key Features

*   **⚡ Resource Efficient:** Optimized for consumer GPUs (4GB VRAM) using **Mixed Precision (FP16)**, **Gradient Checkpointing**, and **Gradient Accumulation**.
*   **🧠 Fusion Architecture:** Concatenates 768-dim text vectors with 112-dim metadata embeddings for context-aware classification.
*   **🌐 Domain Generalization:** Proven Zero-Shot transfer to unseen domains like **COVID-19** and **Technology**.
*   **🤖 Hybrid Cascade:** Automatically routes ambiguous claims to **Llama 3-70B** for expert verification, combining the speed of local inference with the knowledge of Large Language Models.
*   **🛡️ Safety:** Achieved **97.6% True Negative Rate** on legitimate journalism (CNN/DailyMail), ensuring low censorship risk.

---

## 📂 Project Structure

```text
├── liar/               # Baseline experiments on LIAR dataset
├── isot/               # Baseline experiments on ISOT dataset
├── liar_isot/          # MAIN MODEL: Fusion Architecture & Training Scripts
│   ├── dataset_fusion.py   # Handles data unification (LIAR + ISOT)
│   ├── model_fusion.py     # RoBERTa + Metadata Network Definition
│   ├── train_fusion.py     # Training loop with FP16 & Accumulation
│   └── data/               # Training data (train.tsv, True.csv, Fake.csv)
├── llm_system/         # INFERENCE ENGINE: App & Hybrid Logic
│   ├── app.py              # Streamlit User Interface
│   ├── hybrid_llm_predict.py # Logic Gate (Fusion -> Threshold -> LLM)
│   └── llm_engine.py       # Groq API integration
├── fusion_confusion_matrix.png # Visual Results
└── README.md           # Documentation
