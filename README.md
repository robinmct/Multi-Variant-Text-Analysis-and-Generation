NLP Architecture Comparison: BERT vs. GPT vs. Text-GAN

Project Overview

This repository contains the code, evaluation metrics, and inferences for an advanced NLP assignment. The objective is to implement, evaluate, and compare the operational mechanics, performance limits, and behavioral differences of three distinct deep learning architectures:

Language Representation (BERT)

Language Generation (GPT)

Adversarial Text Generation (Text-GAN)

Repository Structure

All data processing, model pipelines, and outputs are consolidated within the primary Jupyter Notebook: GRP7_Activity4_Multi-Variant Text Analysis and Generation.ipynb.

1. Data Preparation

Dataset: Amazon Polarity Reviews (mteb/amazon_polarity)

Processing: The dataset was sampled to 5,000 training and 1,000 testing rows. To accommodate memory limits, especially for the adversarial GAN training, text sequences were truncated to a maximum of 40 tokens.

2. Model Implementations

Variant 1: BERT (Bidirectional Encoder Representations from Transformers)

Task: Text Classification (Binary Review Polarity)

Implementation: Fine-tuned bert-base-uncased with sequence classification heads.

Variant 2: GPT (Generative Pre-trained Transformer)

Task: Causal Language Modeling / Controlled Text Generation

Implementation: Fine-tuned distilgpt2 to autoregressively generate colloquial product reviews.

Variant 3: Text-GAN (Generative Adversarial Network)

Task: Synthetic Adversarial Text Generation

Implementation: A lightweight SeqGAN/continuous approximation GAN utilizing LSTM generator and 1D-CNN discriminator layers.

3. Performance Evaluation Matrix

Because these architectures serve structurally different purposes, they were evaluated using a dual-metric approach (Discriminative and Generative).

Model Variant

Primary Metric (Precision / Recall / F1)

Generative Quality Metric (BLEU / ROUGE / Perplexity)

Training Time per Epoch

Key Observations / Constraints

BERT

Precision: 0.8989
Recall: 0.9181
F1: 0.9084

N/A (Classification Task)

39.10s

Fine-tuning full encoder converges fast and stably; highest F1 of all three variants; most computationally efficient per epoch.

GPT

N/A (Generative Task)

Perplexity: 21.97
ROUGE-1: 0.2335
ROUGE-2: 0.0671
ROUGE-L: 0.1773

55.20s

Causal LM via direct supervised next-token loss; stable, monotonic loss decline; generations remain topically coherent given only a 3-word prompt.

Text-GAN

Discriminator Acc: 1.0000

BLEU: 0.0000

42.58s

Lightweight continuous-approximation SeqGAN; despite label smoothing, noise injection, and 2:1 generator-to-discriminator updates, the model suffered discriminator domination / mode collapse by Epoch 9, after which generator loss saturated and output degenerated into repeated single-token sequences.

4. Key Behavioral Inferences

The three variants demonstrate a clear hierarchy of training stability that directly tracks the directness of their learning signal. BERT, trained with a fully supervised classification loss on a fixed two-class target, converged the fastest and most reliably, reaching a 90.84% F1-score in roughly two minutes of total training. GPT, trained with a supervised next-token prediction loss spread across every position in every sequence, converged more slowly per-epoch (due to the higher computational cost of token-level loss) but still showed smooth, monotonic improvement and produced coherent, topically plausible generations. The Text-GAN, by contrast, was trained with an indirect, non-stationary adversarial signal — the generator's loss landscape constantly shifts as the discriminator itself improves — and despite three dedicated stabilization techniques, this indirect signal proved insufficient to prevent discriminator domination, ultimately producing degenerate output with a BLEU score of zero.

How to Run

This project was built and executed in Google Colab. To view the training histories and evaluation matrices, simply open the .ipynb file. To re-run the training pipeline, click the "Open in Colab" badge at the top of the notebook and ensure hardware acceleration is set to T4 GPU.
