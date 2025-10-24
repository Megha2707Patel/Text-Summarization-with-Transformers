# bart_summarizer.py
# Covers: Tasks 0–12 (intro -> metrics)

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

import datasets
from datasets import load_dataset
import evaluate

import wandb
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)

# -------------------------
# Task 3: Parameters (CLI)
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Fine-tune BART for summarization")

    # Data / model
    p.add_argument("--model_name", type=str, default="facebook/bart-base")
    p.add_argument("--dataset_name", type=str, default="cnn_dailymail")
    p.add_argument("--dataset_config", type=str, default="3.0.0")
    p.add_argument("--source_col", type=str, default="article")
    p.add_argument("--target_col", type=str, default="highlights")

    # Tokenization
    p.add_argument("--src_max_len", type=int, default=512)
    p.add_argument("--tgt_max_len", type=int, default=128)

    # Training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--fp16", action="store_true", help="Use mixed precision")
    p.add_argument("--max_train_steps", type=int, default=0, help="Override total steps")
    p.add_argument("--save_every", type=int, default=1000, help="steps between checkpoints (0 disables)")
    p.add_argument("--eval_every", type=int, default=2000, help="steps between evaluations (0 disables)")

    # Generation / evaluation
    p.add_argument("--gen_max_len", type=int, default=142)
    p.add_argument("--gen_min_len", type=int, default=56)
    p.add_argument("--gen_num_beams", type=int, default=4)

    # Output / hub
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo_id", type=str, default="", help="e.g. your-username/bart-news-summarizer")
    p.add_argument("--hub_private", action="store_true")

    # W&B
    p.add_argument("--wandb_project", type=str, default="bart-summarization")
    p.add_argument("--wandb_run", type=str, default="run")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online","offline","disabled"])

    # Eval-only / resume
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default="", help="path or hub id to resume/eval")

    return p


# -------------------------------
# Task 4–6: Load & prepare data
# -------------------------------
@dataclass
class SummarizationExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

def prepare_datasets(args, tokenizer):
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    # For custom CSV/JSON, you can switch to load_dataset("csv", data_files=...)

    # map -> tokenize
    def preprocess(batch):
        # Tokenize sources
        model_inputs = tokenizer(
            batch[args.source_col],
            max_length=args.src_max_len,
            truncation=True,
            padding=False,
        )
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch[args.target_col],
                max_length=args.tgt_max_len,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    return tokenized


def build_dataloaders(args, tokenized, tokenizer):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=args.model_name)

    train_loader = DataLoader(
        tokenized["train"], shuffle=True,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=data_collator, pin_memory=True
    )
    val_split = "validation" if "validation" in tokenized else "test"
    eval_loader = DataLoader(
        tokenized[val_split], shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=data_collator, pin_memory=True
    )
    return train_loader, eval_loader, val_split


# --------------------------------------
# Task 7: Load model from Hugging Face
# --------------------------------------
def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint or args.model_name, use_fast=True)
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint or args.model_name)
    return model, tokenizer


# --------------------------------
# Task 8–9: Training function
# --------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    tokenized = prepare_datasets(args, tokenizer)
    train_loader, eval_loader, val_split = build_dataloaders(args, tokenized, tokenizer)

    model.to(device)

    # Total steps
    if args.max_train_steps > 0:
        t_total = args.max_train_steps
    else:
        t_total = math.ceil(len(train_loader) * args.epochs / args.grad_accum)

    # Optim/scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    # AMP
    scaler = GradScaler(enabled=args.fp16)

    # W&B
    if args.wandb_mode != "disabled":
        wandb.init(project=args.wandb_project, name=args.wandb_run, mode=args.wandb_mode)
        wandb.config.update(vars(args))

    step = 0
    global_step = 0
    model.train()

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        for batch in train_loader:
            step += 1
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if args.wandb_mode != "disabled":
                    wandb.log({"train/loss": loss.item() * args.grad_accum,
                               "train/lr": scheduler.get_last_lr()[0],
                               "train/step": global_step})

                # Save checkpoints
                if args.save_every and global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                # Eval during training
                if args.eval_every and global_step % args.eval_every == 0:
                    eval_metrics = evaluate_model(args, model, tokenizer, eval_loader)
                    if args.wandb_mode != "disabled":
                        wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                    print(f"[step {global_step}] Eval:", eval_metrics)

            if global_step >= t_total:
                break

        if global_step >= t_total:
            break

    # Final save
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Optional: push to hub
    if args.push_to_hub and args.hub_repo_id:
        model.push_to_hub(args.hub_repo_id, private=args.hub_private)
        tokenizer.push_to_hub(args.hub_repo_id, private=args.hub_private)

    return model, tokenizer


# ----------------------------------------
# Task 10–12: Evaluation / ROUGE metrics
# ----------------------------------------
def evaluate_model(args, model, tokenizer, eval_loader):
    device = next(model.parameters()).device
    rouge = evaluate.load("rouge")

    model.eval()
    preds, refs = [], []

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.gen_max_len,
                min_length=args.gen_min_len,
                num_beams=args.gen_num_beams,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            preds.extend([p.strip() for p in decoded_preds])
            refs.extend([r.strip() for r in decoded_labels])

    # Compute ROUGE
    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    # Add average generated length for context
    gen_lens = [len(p.split()) for p in preds]
    result["gen_len"] = sum(gen_lens) / len(gen_lens)

    model.train()
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    # Modes
    if args.do_train:
        model, tokenizer = train(args)
    else:
        model, tokenizer = load_model_and_tokenizer(args)

    # Evaluation pass if requested
    if args.do_eval:
        tokenized = prepare_datasets(args, tokenizer)
        _, eval_loader, _ = build_dataloaders(args, tokenized, tokenizer)
        metrics = evaluate_model(args, model, tokenizer, eval_loader)
        print("Final evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
