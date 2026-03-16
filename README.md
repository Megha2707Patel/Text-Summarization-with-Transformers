# Metric Misalignment in Abstractive Summarization
### High ROUGE Scores Do Not Imply Factual Reliability

> **Megha Punamchand Patel** — Independent Researcher, 2026
>
> *Manuscript under preparation for arXiv submission (cs.CL)*

---

## Overview

This repository contains the full code for the research paper:

**"Metric Misalignment in Abstractive Summarization: High ROUGE Scores Do Not Imply Factual Reliability"**

The study investigates a fundamental gap in how abstractive summarization systems are evaluated. We fine-tune a BART-base transformer on the CNN/DailyMail dataset and conduct a systematic manual factuality analysis of 200 generated summaries.

**Key finding:** Despite achieving a competitive ROUGE-L score of 0.42, approximately **23% of generated summaries contain factual errors** — including entity hallucinations, temporal distortions, and fabricated causal relationships. Critically, summaries with factual errors achieve ROUGE scores statistically indistinguishable from factually correct ones.

This exposes a fundamental misalignment between widely used evaluation metrics and real-world reliability requirements.

---

## Repository Structure

```
├── bart_summarizer.py      # BART fine-tuning, training loop, ROUGE evaluation
├── factuality_eval.py      # Manual factuality annotation tool (reproduces paper results)
├── results/                # Annotation outputs (created at runtime)
└── outputs/                # Model checkpoints (created at runtime)
```

---

## Key Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.439 |
| ROUGE-2 | 0.206 |
| ROUGE-L | 0.420 |

| Factuality Category | Count | Percentage |
|--------------------|-------|------------|
| Factually consistent | 154 | 77% |
| Minor distortion | 28 | 14% |
| Major hallucination | 18 | 9% |
| **Total with errors** | **46** | **23%** |

**ROUGE-L: factually correct summaries = 0.423 vs error summaries = 0.415**
— a difference that is neither statistically significant nor practically meaningful, confirming that ROUGE cannot distinguish reliable summaries from hallucinated ones.

---

## Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Framework | PyTorch |
| NLP | Hugging Face Transformers |
| Data | CNN/DailyMail via Hugging Face Datasets |
| Evaluation | ROUGE (rouge-score), Manual annotation |
| Experiment Tracking | Weights & Biases (optional) |
| Deployment | Hugging Face Hub (optional) |

---

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac/Linux

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" \
            "evaluate>=0.4" "rouge-score>=0.1.2" numpy \
            sentencepiece huggingface_hub wandb
```

---

## Usage

### Step 1 — Fine-tune BART

```bash
python bart_summarizer.py \
    --do_train \
    --do_eval \
    --model_name facebook/bart-base \
    --dataset_name cnn_dailymail \
    --dataset_config 3.0.0 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 16 \
    --lr 3e-5 \
    --src_max_len 1024 \
    --tgt_max_len 142 \
    --gen_num_beams 4 \
    --out_dir outputs/ \
    --wandb_mode disabled
```

### Step 2 — Run Manual Factuality Evaluation

This tool replicates the human annotation study from the paper. It generates 200 summaries and walks through each one interactively for annotation.

```bash
python factuality_eval.py \
    --model_dir outputs/ \
    --num_samples 200 \
    --seed 42 \
    --out_file results/factuality_results.json
```

Progress auto-saves after every sample. To resume an interrupted session, simply run the same command again.

### Step 3 — View Analysis

Once annotation is complete, reproduce the paper's quantitative results:

```bash
python factuality_eval.py \
    --do_analysis \
    --results_file results/factuality_results.json
```

This prints Table 1, Table 2, the error type distribution, and the ROUGE vs factuality comparison from Section 6.3 of the paper.

---

## Annotation Guide

During factuality evaluation, each generated summary is compared against its **source article** (not the reference summary) and labeled as:

| Label | Meaning |
|-------|---------|
| `1` — factually_consistent | No identifiable errors |
| `2` — minor_distortion | Partial inaccuracies, does not fundamentally misrepresent the source |
| `3` — major_hallucination | Fabricated or directly contradicted claims |

Error types (for labels 2 and 3):

| Code | Error Type | Description |
|------|-----------|-------------|
| `1` | entity_hallucination | Wrong names, organizations, or locations |
| `2` | temporal_distortion | Incorrect dates, sequences, or time references |
| `3` | causal_fabrication | Invented cause-effect relationships |
| `4` | numerical_error | Wrong statistics or figures |
| `5` | contextual_omission | Missing critical qualifying information |

---

## Paper Abstract

> Abstractive text summarization systems are commonly evaluated using lexical overlap metrics such as ROUGE. While these metrics correlate with surface similarity between generated and reference summaries, they may fail to capture factual reliability. In this work, we investigate the relationship between ROUGE scores and factual correctness in neural summarization models. We fine-tune a BART-base transformer on the CNN/DailyMail dataset and conduct manual factuality evaluation on generated summaries. Although the model achieves competitive benchmark performance (ROUGE-L = 0.42), human evaluation reveals that approximately 23% of summaries contain factual inconsistencies, including entity hallucinations, temporal distortions, and fabricated causal relationships. These findings highlight a fundamental misalignment between widely used evaluation metrics and real-world reliability requirements for summarization systems.

---

## Citation

If you use this code or findings in your work, please cite:

```
Patel, M. P. (2026). Metric Misalignment in Abstractive Summarization:
High ROUGE Scores Do Not Imply Factual Reliability.
Manuscript under preparation.
GitHub: https://github.com/Megha2707Patel/Text-Summarization-with-Transformers
```

Once the arXiv preprint is live, the citation will be updated with the arXiv ID.

---

## License

MIT License — free to use, modify, and distribute with attribution.
