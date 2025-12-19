# Evaluating LLM-based Data Augmentation for the Functional Classification of *Look*

This repository contains code and experiments for evaluating the impact of
LLM-based data augmentation on the functional classification of the English
verb *look*.

---

## Task Description

The task is a **sentence-level multi-class classification problem**.

Given a sentence containing the verb *look*, the goal is to predict its
functional use. The dataset distinguishes four pragmatically motivated
functions:

- **AS** — Attention Signal  
- **DIR** — Directive  
- **DM** — Discourse Marker  
- **INTJ** — Interjection  

---

## Data Overview

The experiments use two types of data:

- **Real data**: Human-annotated sentences containing *look*
- **Synthetic data**: LLM-generated sentences used to balance minority
  function classes

Only the minority classes (**DIR, DM, INTJ**) were augmented.
The **AS (Attention Signal)** class already contained 287 real samples and was
**not augmented**. Synthetic samples were generated to raise the remaining
classes to the same level (287 samples each).

All evaluation is performed **exclusively on real data** to avoid synthetic
evaluation bias.

---

### Data Organization

All samples (real and synthetic) are stored in a **single data file**.
Each instance is explicitly marked with a flag indicating whether it is
synthetic.

This design allows flexible construction of training configurations
(e.g. real-only, real + synthetic, synthetic-only) through filtering,
while ensuring that synthetic samples can be fully excluded from evaluation.

Each data instance includes at least the following fields:

- `text`: the sentence containing *look*
- `label`: functional category (AS, DIR, DM, INTJ)
- `is_synthetic`: boolean flag indicating whether the instance is synthetic

---

## Experimental Setups

Three training configurations are implemented and compared:

### 1. Real-only Training
- Training data: Real samples only (naturally imbalanced)
- Purpose: Baseline performance

### 2. Real + Synthetic Training
- Training data:
  - Real samples for all classes
  - Synthetic samples **only for DIR, DM, and INTJ**
- Data characteristics: Balanced across all four functions
- Purpose: Main experimental condition evaluating the effect of augmentation

### 3. Synthetic-only Training
- Training data:
  - Synthetic samples for DIR, DM, and INTJ
  - Real samples for AS (no synthetic AS data exists)
- Purpose: Diagnostic analysis of what information is captured by synthetic data

Synthetic-only training is **not** used to support primary claims, but to aid
interpretation and error analysis.

---

## Evaluation Protocol

All models are evaluated exclusively on **real, human-annotated samples**.
Synthetic data is never used for evaluation.

To ensure robustness and reduce variance due to data splits and random
initialization, we use **repeated 2-fold cross-validation**:

- The real dataset is split into **2 folds**
- For each training configuration, training is repeated **5 times**
- Each run uses a different random seed
- Each experiment therefore consists of **10 evaluations** (2 folds × 5 runs)
- Reported results are **averaged across all runs and folds**

Performance is reported using:
- Macro F1-score (primary metric)
- Per-class Precision, Recall, and F1
- Confusion matrices aggregated across runs




