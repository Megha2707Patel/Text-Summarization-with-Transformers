# factuality_eval.py
# Manual Factuality Evaluation Tool for Abstractive Summarization
#
# This script supports the human annotation study described in:
# "Metric Misalignment in Abstractive Summarization:
#  High ROUGE Scores Do Not Imply Factual Reliability"
# Megha Punamchand Patel, 2026
#
# Usage:
#   python factuality_eval.py --model_dir outputs/ --num_samples 200 --out_file results/factuality_results.json
#   python factuality_eval.py --model_dir outputs/ --do_analysis --results_file results/factuality_results.json

import os
import json
import argparse
import random
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration
import evaluate


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

# Factuality labels (3-way annotation scheme used in the paper)
FACTUALITY_LABELS = {
    "1": "factually_consistent",
    "2": "minor_distortion",
    "3": "major_hallucination",
}

# Error type taxonomy (5-category scheme used in the paper)
ERROR_TYPES = {
    "1": "entity_hallucination",      # Wrong names, orgs, locations
    "2": "temporal_distortion",       # Wrong dates, sequences, time refs
    "3": "causal_fabrication",        # Invented cause-effect relationships
    "4": "numerical_error",           # Wrong statistics or figures
    "5": "contextual_omission",       # Missing critical qualifying info
    "0": "none",                      # No error (factually consistent)
}

@dataclass
class AnnotationRecord:
    sample_id: int
    source_article: str
    reference_summary: str
    generated_summary: str
    rouge1: float
    rouge2: float
    rougeL: float
    factuality_label: Optional[str] = None       # Set during annotation
    error_types: Optional[List[str]] = None      # Set during annotation
    annotator_notes: Optional[str] = None        # Free-text notes


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def build_argparser():
    p = argparse.ArgumentParser(
        description="Manual factuality evaluation for BART summarization study"
    )
    p.add_argument("--model_dir", type=str, default="outputs",
                   help="Path to fine-tuned model directory")
    p.add_argument("--dataset_name", type=str, default="cnn_dailymail")
    p.add_argument("--dataset_config", type=str, default="3.0.0")
    p.add_argument("--num_samples", type=int, default=200,
                   help="Number of summaries to evaluate (paper uses 200)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sample selection")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gen_max_len", type=int, default=142)
    p.add_argument("--gen_min_len", type=int, default=56)
    p.add_argument("--gen_num_beams", type=int, default=4)
    p.add_argument("--out_file", type=str, default="results/factuality_results.json",
                   help="Where to save/resume annotation results")
    p.add_argument("--do_analysis", action="store_true",
                   help="Run analysis on completed annotations (skip annotation loop)")
    p.add_argument("--results_file", type=str, default="",
                   help="Path to completed annotations JSON for analysis")
    p.add_argument("--article_display_len", type=int, default=1500,
                   help="Max characters of source article to display during annotation")
    return p


# ─────────────────────────────────────────────
# Model loading and generation
# ─────────────────────────────────────────────

def load_model(model_dir: str):
    print(f"\nLoading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer, device


def generate_summaries(args, model, tokenizer, device, articles: List[str]) -> List[str]:
    """Generate summaries for a list of articles using beam search."""
    all_summaries = []
    for i in range(0, len(articles), args.batch_size):
        batch_articles = articles[i:i + args.batch_size]
        inputs = tokenizer(
            batch_articles,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.gen_max_len,
                min_length=args.gen_min_len,
                num_beams=args.gen_num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_summaries.extend([s.strip() for s in summaries])

        if (i // args.batch_size + 1) % 5 == 0:
            print(f"  Generated {min(i + args.batch_size, len(articles))}/{len(articles)} summaries...")

    return all_summaries


def compute_rouge_scores(predictions: List[str], references: List[str]) -> List[dict]:
    """Compute per-sample ROUGE scores."""
    rouge = evaluate.load("rouge")
    per_sample_scores = []
    for pred, ref in zip(predictions, references):
        result = rouge.compute(
            predictions=[pred],
            references=[ref],
            use_stemmer=True
        )
        per_sample_scores.append({
            "rouge1": round(float(result["rouge1"]), 4),
            "rouge2": round(float(result["rouge2"]), 4),
            "rougeL": round(float(result["rougeL"]), 4),
        })
    return per_sample_scores


# ─────────────────────────────────────────────
# Sample preparation
# ─────────────────────────────────────────────

def prepare_samples(args, model, tokenizer, device) -> List[AnnotationRecord]:
    """Load dataset, sample, generate summaries, compute ROUGE."""
    print(f"\nLoading dataset: {args.dataset_name} ({args.dataset_config})")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    test_data = dataset["test"]

    # Reproducible random sample
    random.seed(args.seed)
    indices = random.sample(range(len(test_data)), args.num_samples)
    samples = test_data.select(indices)

    articles = samples["article"]
    references = samples["highlights"]

    print(f"\nGenerating {args.num_samples} summaries...")
    generated = generate_summaries(args, model, tokenizer, device, articles)

    print("\nComputing per-sample ROUGE scores...")
    rouge_scores = compute_rouge_scores(generated, references)

    records = []
    for idx, (article, ref, gen, scores) in enumerate(
        zip(articles, references, generated, rouge_scores)
    ):
        records.append(AnnotationRecord(
            sample_id=idx,
            source_article=article,
            reference_summary=ref,
            generated_summary=gen,
            rouge1=scores["rouge1"],
            rouge2=scores["rouge2"],
            rougeL=scores["rougeL"],
        ))

    return records


# ─────────────────────────────────────────────
# Annotation interface
# ─────────────────────────────────────────────

def display_sample(record: AnnotationRecord, total: int, display_len: int):
    """Print a formatted sample for annotation."""
    print("\n" + "=" * 80)
    print(f"SAMPLE {record.sample_id + 1} of {total}")
    print("=" * 80)

    print("\n[SOURCE ARTICLE]")
    print("-" * 40)
    article_preview = record.source_article[:display_len]
    if len(record.source_article) > display_len:
        article_preview += f"\n... [truncated, full length: {len(record.source_article)} chars]"
    print(article_preview)

    print("\n[REFERENCE SUMMARY]")
    print("-" * 40)
    print(record.reference_summary)

    print("\n[GENERATED SUMMARY]")
    print("-" * 40)
    print(record.generated_summary)

    print(f"\n[ROUGE SCORES]  R-1: {record.rouge1:.4f}  |  R-2: {record.rouge2:.4f}  |  R-L: {record.rougeL:.4f}")


def annotate_sample(record: AnnotationRecord) -> AnnotationRecord:
    """Interactive annotation for a single sample."""

    # Step 1: Factuality label
    print("\n[STEP 1] Assign a factuality label:")
    for key, label in FACTUALITY_LABELS.items():
        print(f"  {key} = {label}")
    print("  s = skip this sample")
    print("  q = quit and save progress")

    while True:
        choice = input("Your choice: ").strip().lower()
        if choice == "q":
            return None  # Signal to quit
        if choice == "s":
            record.factuality_label = "skipped"
            record.error_types = []
            return record
        if choice in FACTUALITY_LABELS:
            record.factuality_label = FACTUALITY_LABELS[choice]
            break
        print("  Invalid input. Please enter 1, 2, 3, s, or q.")

    # Step 2: Error types (only if not factually consistent)
    if record.factuality_label != "factually_consistent":
        print("\n[STEP 2] Select all error types present (comma-separated, e.g. 1,3):")
        for key, etype in ERROR_TYPES.items():
            if key != "0":
                print(f"  {key} = {etype}")

        while True:
            raw = input("Error types: ").strip()
            selected_keys = [k.strip() for k in raw.split(",") if k.strip()]
            if all(k in ERROR_TYPES and k != "0" for k in selected_keys):
                record.error_types = [ERROR_TYPES[k] for k in selected_keys]
                break
            print("  Invalid input. Enter numbers from 1-5, comma-separated.")
    else:
        record.error_types = ["none"]

    # Step 3: Optional notes
    notes = input("\n[STEP 3] Optional notes (press Enter to skip): ").strip()
    record.annotator_notes = notes if notes else None

    return record


def run_annotation(args, records: List[AnnotationRecord]) -> List[AnnotationRecord]:
    """Run the interactive annotation loop with save/resume support."""
    out_path = args.out_file
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume from existing file if present
    completed = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            saved = json.load(f)
        for item in saved:
            completed[item["sample_id"]] = item
        print(f"\nResuming annotation: {len(completed)}/{len(records)} already completed.")

    annotated = []
    for record in records:
        # Skip already annotated
        if record.sample_id in completed:
            annotated.append(AnnotationRecord(**completed[record.sample_id]))
            continue

        display_sample(record, len(records), args.article_display_len)
        result = annotate_sample(record)

        if result is None:
            # User chose to quit — save progress and exit
            print("\nSaving progress and exiting...")
            all_results = [asdict(r) for r in annotated]
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved {len(annotated)} annotations to {out_path}")
            return annotated

        annotated.append(result)

        # Auto-save after every annotation
        all_results = [asdict(r) for r in annotated]
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

        remaining = len(records) - len(annotated)
        print(f"\nSaved. {len(annotated)} done, {remaining} remaining.")

    print(f"\nAnnotation complete. All {len(annotated)} samples saved to {out_path}")
    return annotated


# ─────────────────────────────────────────────
# Analysis (reproduces paper results)
# ─────────────────────────────────────────────

def run_analysis(records: List[AnnotationRecord]):
    """
    Reproduce the quantitative results reported in the paper:
    - Factuality label distribution (Table 2)
    - Error type distribution
    - ROUGE scores by factuality category (Section 6.3)
    - Overall aggregate ROUGE scores (Table 1)
    """
    print("\n" + "=" * 80)
    print("FACTUALITY EVALUATION ANALYSIS")
    print("=" * 80)

    # Filter out skipped samples
    valid = [r for r in records if r.factuality_label and r.factuality_label != "skipped"]
    n = len(valid)
    print(f"\nTotal annotated samples (excluding skipped): {n}")

    # ── Table 2: Factuality label distribution ──
    label_counts = {}
    for r in valid:
        label_counts[r.factuality_label] = label_counts.get(r.factuality_label, 0) + 1

    print("\n── TABLE 2: Factuality Label Distribution ──")
    print(f"{'Category':<30} {'Count':>8} {'Percentage':>12}")
    print("-" * 52)
    for label in ["factually_consistent", "minor_distortion", "major_hallucination"]:
        count = label_counts.get(label, 0)
        pct = 100 * count / n if n > 0 else 0
        print(f"{label:<30} {count:>8} {pct:>11.1f}%")

    error_count = label_counts.get("minor_distortion", 0) + label_counts.get("major_hallucination", 0)
    print(f"{'Total with errors':<30} {error_count:>8} {100*error_count/n:>11.1f}%")

    # ── Error type distribution ──
    error_type_counts = {}
    for r in valid:
        if r.error_types:
            for et in r.error_types:
                if et != "none":
                    error_type_counts[et] = error_type_counts.get(et, 0) + 1

    total_errors = sum(error_type_counts.values())
    if total_errors > 0:
        print("\n── Error Type Distribution ──")
        print(f"{'Error Type':<30} {'Count':>8} {'% of Errors':>14}")
        print("-" * 54)
        for etype in ["entity_hallucination", "temporal_distortion",
                      "causal_fabrication", "numerical_error", "contextual_omission"]:
            count = error_type_counts.get(etype, 0)
            pct = 100 * count / total_errors if total_errors > 0 else 0
            print(f"{etype:<30} {count:>8} {pct:>13.1f}%")

    # ── Section 6.3: ROUGE by factuality category ──
    def avg_rouge(subset, metric):
        vals = [getattr(r, metric) for r in subset if getattr(r, metric) is not None]
        return np.mean(vals) if vals else 0.0

    correct = [r for r in valid if r.factuality_label == "factually_consistent"]
    errors  = [r for r in valid if r.factuality_label in ("minor_distortion", "major_hallucination")]

    print("\n── Section 6.3: ROUGE Scores by Factuality Category ──")
    print(f"{'Category':<30} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10}")
    print("-" * 62)
    print(f"{'Factually consistent':<30} "
          f"{avg_rouge(correct,'rouge1'):>10.4f} "
          f"{avg_rouge(correct,'rouge2'):>10.4f} "
          f"{avg_rouge(correct,'rougeL'):>10.4f}")
    print(f"{'Contains errors':<30} "
          f"{avg_rouge(errors,'rouge1'):>10.4f} "
          f"{avg_rouge(errors,'rouge2'):>10.4f} "
          f"{avg_rouge(errors,'rougeL'):>10.4f}")

    # ── Table 1: Overall ROUGE scores ──
    print("\n── TABLE 1: Overall ROUGE Scores ──")
    print(f"{'Metric':<15} {'Score':>10}")
    print("-" * 27)
    for metric, label in [("rouge1","ROUGE-1"), ("rouge2","ROUGE-2"), ("rougeL","ROUGE-L")]:
        print(f"{label:<15} {avg_rouge(valid, metric):>10.4f}")

    # ── Key finding ──
    diff = abs(avg_rouge(correct, "rougeL") - avg_rouge(errors, "rougeL"))
    print(f"\nKey finding: ROUGE-L difference between factually correct and error "
          f"summaries = {diff:.4f}")
    print("This supports the paper's central claim that ROUGE scores are not "
          "predictive of factual reliability.")

    print("\n" + "=" * 80)
    print("Analysis complete.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = build_argparser().parse_args()

    if args.do_analysis:
        # Analysis-only mode: load saved annotations and print results
        results_path = args.results_file or args.out_file
        if not os.path.exists(results_path):
            print(f"Error: results file not found at {results_path}")
            print("Run annotation first, then use --do_analysis --results_file <path>")
            return
        print(f"Loading annotations from {results_path}")
        with open(results_path, "r") as f:
            saved = json.load(f)
        records = [AnnotationRecord(**item) for item in saved]
        run_analysis(records)
        return

    # Full annotation mode
    print("\n" + "=" * 80)
    print("FACTUALITY EVALUATION TOOL")
    print("Metric Misalignment in Abstractive Summarization")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Model:    {args.model_dir}")
    print(f"  Samples:  {args.num_samples}")
    print(f"  Seed:     {args.seed}")
    print(f"  Output:   {args.out_file}")
    print("\nAnnotation instructions:")
    print("  - Read the source article carefully")
    print("  - Compare the generated summary against the source (not the reference)")
    print("  - Assign a factuality label and identify any error types")
    print("  - Type 'q' at any time to save and quit")
    print("  - Progress is auto-saved after every sample\n")

    input("Press Enter to begin annotation...")

    # Load model and prepare samples
    model, tokenizer, device = load_model(args.model_dir)
    records = prepare_samples(args, model, tokenizer, device)

    # Free GPU memory before interactive loop
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run annotation
    annotated = run_annotation(args, records)

    # Run analysis on completed annotations
    completed = [r for r in annotated if r.factuality_label and r.factuality_label != "skipped"]
    if len(completed) >= 10:
        run_analysis(completed)
    else:
        print(f"\nOnly {len(completed)} samples annotated. Annotate more samples to see analysis.")


if __name__ == "__main__":
    main()
