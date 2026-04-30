#!/usr/bin/env python3
"""
scripts/evaluate_mmlu_prox.py

Evaluate a checkpoint (or base model) on MMLU-ProX.
Multiple choice with 10 options (A-J). Reports accuracy@1 and per-category breakdown.

Usage:
    # Base model
    python3 scripts/evaluate_mmlu_prox.py \
        --eval-file data/eval/mmlu_prox_en_test.jsonl \
        --n-samples 200

    # LoRA checkpoint
    python3 scripts/evaluate_mmlu_prox.py \
        --checkpoint runs/.../checkpoint-1750 \
        --eval-file data/eval/mmlu_prox_en_test.jsonl \
        --n-samples 200
"""

import sys
import re
import json
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM = (
    "You are answering a multiple-choice question. Read the question and options carefully, "
    "reason step by step, then state your final choice in the format: 'The answer is (X).' "
    "where X is the letter A through J."
)


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def make_user_prompt(question: str, options: list[str]) -> str:
    lines = [f"Question: {question}", ""]
    for i, opt in enumerate(options):
        if opt:
            lines.append(f"{LETTERS[i]}: {opt}")
    lines.append("")
    lines.append("Reason step by step, then end with: \"The answer is (X).\"")
    return "\n".join(lines)


# Try strongest format first, then fall back to looser patterns
_PATTERNS = [
    re.compile(r"answer is\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"answer:\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"\\boxed\{\s*([A-J])\s*\}"),
    re.compile(r"\b\(([A-J])\)", re.IGNORECASE),  # last "(X)"
]


def extract_answer(completion: str) -> str | None:
    """Extract single letter A-J from the completion. Returns None if nothing matches."""
    for pat in _PATTERNS:
        matches = pat.findall(completion)
        if matches:
            return matches[-1].upper()
    return None


def evaluate(
    checkpoint_path: str | None,
    eval_file: str,
    model_name: str = "/workspace/models/Qwen3.5-2B",
    n_samples: int = 200,
    max_new_tokens: int = 4096,
    temperature: float = 0.3,
    enable_thinking: bool = True,
    save_results: str | None = None,
    output_file: str | None = None,
):
    is_zeroshot = (not checkpoint_path) or checkpoint_path == "baseline"
    mode_str = "ZERO-SHOT (no LoRA)" if is_zeroshot else f"CHECKPOINT: {checkpoint_path}"
    logger.info(f"{'='*50}")
    logger.info(f"  Mode: {mode_str}")
    logger.info(f"  Thinking: {'ON' if enable_thinking else 'OFF'}")
    logger.info(f"  Eval file: {eval_file}")
    logger.info(f"  Max tokens: {max_new_tokens}  |  Temp: {temperature}")
    logger.info(f"{'='*50}")

    # ── Model ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if checkpoint_path and checkpoint_path != "baseline":
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info(f"Loaded LoRA checkpoint from {checkpoint_path}")
    else:
        model = base_model
        logger.info("Running ZERO-SHOT base model")
    model.eval()

    # ── Problems ──────────────────────────────────────────────────────────
    problems = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if not problems:
        raise RuntimeError(
            f"No problems loaded from {eval_file}. "
            "Did you run data/generate_mmlu_prox.py?"
        )

    random.seed(42)
    if len(problems) > n_samples:
        problems = random.sample(problems, n_samples)
    logger.info(f"Loaded {len(problems)} problems")

    # ── Eval loop ─────────────────────────────────────────────────────────
    results = []
    do_sample = temperature > 0.0
    for i, p in enumerate(problems):
        question = p["question"]
        options  = p["options"]
        gold     = str(p["answer"]).strip().upper()
        category = p.get("category", "")

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": make_user_prompt(question, options)},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(raw_ids, skip_special_tokens=False)
        for tok in [tokenizer.eos_token, tokenizer.pad_token]:
            if tok:
                completion = completion.replace(tok, "")
        completion = completion.strip()

        pred    = extract_answer(completion)
        correct = (pred is not None and pred == gold)

        results.append({
            "correct":           correct,
            "predicted_answer":  pred,
            "gold_answer":       gold,
            "category":          category,
            "completion_length": len(completion.split()),
            "completion":        completion,
            "question_id":       p.get("question_id"),
        })

        marker = "✓" if correct else "✗"
        logger.info(f"  [{i+1}/{len(problems)}] {marker}  pred={pred} gold={gold}  ({category})")
        if (i + 1) % 25 == 0:
            so_far = sum(r["correct"] for r in results)
            logger.info(f"  --> running accuracy: {so_far}/{i+1} = {so_far/(i+1):.3f}")

    # ── Report ────────────────────────────────────────────────────────────
    n         = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / n
    extracted = sum(1 for r in results if r["predicted_answer"] is not None) / n
    mean_len  = sum(r["completion_length"] for r in results) / n

    print("\n" + "="*60)
    print(f"  MMLU-ProX Evaluation Results")
    print("="*60)
    print(f"  Mode:            {mode_str}")
    print(f"  N samples:       {n}")
    print(f"  Accuracy:        {n_correct}/{n} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Extraction rate: {extracted:.4f} ({extracted*100:.1f}%)")
    print(f"  Mean length:     {mean_len:.1f} words")
    print(f"  Random baseline: 10.0% (10 options)")
    print("="*60)

    # Per-category breakdown
    by_cat = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        c = r["category"] or "uncategorized"
        by_cat[c]["n"] += 1
        if r["correct"]:
            by_cat[c]["correct"] += 1

    if by_cat:
        print(f"\n  Per-category accuracy:")
        for cat in sorted(by_cat.keys()):
            stats = by_cat[cat]
            acc = stats["correct"] / stats["n"] if stats["n"] else 0
            print(f"    {cat:30s}  {stats['correct']:>3d}/{stats['n']:<3d} = {acc*100:5.1f}%")
        print("="*60)

    if save_results:
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {save_results}")

    if output_file:
        out = {
            "mode":        "zero-shot" if is_zeroshot else "checkpoint",
            "checkpoint":  checkpoint_path,
            "eval_file":   eval_file,
            "n_samples":   n,
            "accuracy":    accuracy,
            "n_correct":   n_correct,
            "extraction_rate": extracted,
            "mean_length": mean_len,
            "by_category": {c: dict(v) for c, v in by_cat.items()},
            "results":     results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full output to {output_file}")

    return accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None, help="LoRA checkpoint path (omit for base)")
    p.add_argument("--baseline",    action="store_true")
    p.add_argument("--eval-file",   required=True)
    p.add_argument("--model",       default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--n-samples",   type=int, default=200)
    p.add_argument("--max-tokens",  type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-think",    action="store_true")
    p.add_argument("--save-results",default=None)
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    checkpoint = args.checkpoint
    if args.baseline or checkpoint is None:
        checkpoint = None

    evaluate(
        checkpoint_path = checkpoint,
        eval_file       = args.eval_file,
        model_name      = args.model,
        n_samples       = args.n_samples,
        max_new_tokens  = args.max_tokens,
        temperature     = args.temperature,
        enable_thinking = not args.no_think,
        save_results    = args.save_results,
        output_file     = args.output_file,
    )


if __name__ == "__main__":
    main()
