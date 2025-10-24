# 📰 Text Summarization with Transformers (BART)

A deep learning project that fine-tunes **Facebook’s BART transformer model** to generate concise summaries of news articles using the **CNN/DailyMail** dataset.  
This project demonstrates the full NLP pipeline — from dataset loading to model training, evaluation (ROUGE), and real-world text summarization.

---

## 🚀 Project Overview

Transformer-based models like **BART** (Bidirectional and Auto-Regressive Transformers) have become state-of-the-art for NLP tasks such as summarization.  
This project uses **PyTorch** and **Hugging Face Transformers** to fine-tune the pretrained `facebook/bart-base` model on a news summarization dataset.

The fine-tuned model is then evaluated using **ROUGE metrics** to measure the quality of generated summaries compared to human-written ones.

---

## 🧠 Key Features
- Fine-tuning BART on **CNN/DailyMail** summarization dataset.  
- Uses **Hugging Face Hub API** to load pretrained models and datasets.  
- **PyTorch** implementation with gradient accumulation for memory efficiency.  
- **Weights & Biases** integration for experiment tracking (optional).  
- **Evaluation using ROUGE metrics** via the `evaluate` library.  
- CPU/GPU compatible and ready to deploy.  

---

## 🧩 Technologies Used

| Category | Tools |
|-----------|-------|
| Language | Python 3.12 |
| Framework | PyTorch |
| NLP | Hugging Face Transformers |
| Data Handling | Datasets, Evaluate |
| Visualization | Weights & Biases (optional) |
| Evaluation | ROUGE score |
| Deployment | Hugging Face Hub |

---

## 📦 Installation

```bash
# 1. Create project folder
mkdir news-summarizer && cd news-summarizer

# 2. Create a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" "evaluate>=0.4" "rouge-score>=0.1.2" sentencepiece huggingface_hub wandb
