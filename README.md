# NLP Architecture Comparison: BERT vs. GPT vs. Text-GAN

## Project Overview

This repository contains the code, evaluation metrics, and inferences for an advanced NLP assignment. The objective is to implement, evaluate, and compare the **operational mechanics**, **performance limits**, and **behavioral differences** of three distinct deep learning architectures:

- 🟦 **BERT** — Language Representation
- 🟩 **GPT** — Language Generation
- 🟥 **Text-GAN** — Adversarial Text Generation

---

## Repository Structure

All data processing, model pipelines, and outputs are consolidated within the primary Jupyter Notebook:

📓 [`GRP7_Activity4_Multi-Variant Text Analysis and Generation.ipynb`](GRP7_Activity4_Multi-Variant%20Text%20Analysis%20and%20Generation.ipynb)

---

## 1. Data Preparation

- **Dataset:** `mteb/amazon_polarity` (Amazon Polarity Reviews)
- **Sampling:** 5,000 training rows and 1,000 testing rows
- **Truncation:** Text sequences were capped at a maximum of **40 tokens** to reduce memory usage, especially during adversarial GAN training.

---

## 2. Model Implementations

### Variant 1: BERT

> **Bidirectional Encoder Representations from Transformers**

| Attribute | Description |
|-----------|-------------|
| **Task** | Text Classification (Binary Review Polarity) |
| **Model** | `bert-base-uncased` |
| **Implementation** | Fine-tuned BERT with a sequence classification head |

### Variant 2: GPT

> **Generative Pre-trained Transformer**

| Attribute | Description |
|-----------|-------------|
| **Task** | Causal Language Modeling / Controlled Text Generation |
| **Model** | `distilgpt2` |
| **Implementation** | Fine-tuned GPT-2 to autoregressively generate colloquial product reviews |

### Variant 3: Text-GAN

> **Generative Adversarial Network for Text**

| Attribute | Description |
|-----------|-------------|
| **Task** | Synthetic Adversarial Text Generation |
| **Architecture** | Lightweight SeqGAN-style continuous approximation |
| **Generator** | LSTM-based network |
| **Discriminator** | 1D-CNN classifier |
| **Stabilization** | Label smoothing (`0.9`), input noise (`0.1`), and 2:1 generator-to-discriminator update ratio |

---

## 3. Performance Evaluation Matrix

Because these architectures serve structurally different purposes, they were evaluated using a dual-metric approach (discriminative and generative).

| Model Variant | Primary Metric (Precision / Recall / F1) | Generative Quality Metric (BLEU / ROUGE / Perplexity) | Training Time / Epoch | Key Observations / Constraints |
|---------------|------------------------------------------|-------------------------------------------------------|-----------------------|--------------------------------|
| **BERT** | **Precision:** `0.8989`<br>**Recall:** `0.9181`<br>**F1:** `0.9084` | N/A (Classification Task) | **39.10s** | Fine-tuning full encoder converges fast and stably; highest F1 of all three variants; most computationally efficient per epoch. |
| **GPT** | N/A (Generative Task) | **Perplexity:** `21.97`<br>**ROUGE-1:** `0.2335`<br>**ROUGE-2:** `0.0671`<br>**ROUGE-L:** `0.1773` | **55.20s** | Causal LM via direct supervised next-token loss; stable, monotonic loss decline; generations remain topically coherent given only a 3-word prompt. |
| **Text-GAN** | **Discriminator Accuracy:** `1.0000` | **BLEU:** `0.0000` | **42.58s** | Lightweight continuous-approximation SeqGAN; despite label smoothing, noise injection, and 2:1 generator updates, the model suffered **discriminator domination / mode collapse** by Epoch 9, after which generator loss saturated and output degenerated into repeated single-token sequences. |

---

## 4. Key Behavioral Inferences

The three variants demonstrate a clear hierarchy of training stability that directly tracks the **directness of their learning signal**:

- **BERT** was trained with a fully supervised classification loss on a fixed two-class target. It converged the fastest and most reliably, reaching a **90.84% F1-score** in roughly two minutes of total training.
- **GPT** was trained with a supervised next-token prediction loss spread across every position in every sequence. It converged more slowly per epoch due to the higher computational cost of token-level loss, but still showed smooth, monotonic improvement and produced coherent, topically plausible generations.
- **Text-GAN** was trained with an **indirect, non-stationary adversarial signal** — the generator's loss landscape constantly shifts as the discriminator itself improves. Despite three dedicated stabilization techniques, this indirect signal proved insufficient to prevent discriminator domination, ultimately producing degenerate output with a **BLEU score of zero**.

---

## How to Run

This project was built and executed in **Google Colab**.

1. Open the `.ipynb` notebook to view training histories and evaluation matrices.
2. To re-run the training pipeline, click the **"Open in Colab"** badge at the top of the notebook.
3. Ensure hardware acceleration is set to **T4 GPU**.

---

> **Note:** All reported metrics were extracted directly from the executed notebook outputs.
