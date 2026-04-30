#!/usr/bin/env python3
"""
scripts/evaluate_aime.py

Evaluate a checkpoint (or base model) on AIME 2026.
Reports accuracy@1 — fraction of problems with correct integer answer.

Usage:
    # Base model
    python3 scripts/evaluate_aime.py \
        --eval-file data/eval/aime26.jsonl \
        --max-tokens 16384

    # LoRA checkpoint
    python3 scripts/evaluate_aime.py \
        --checkpoint runs/.../checkpoint-1750 \
        --eval-file data/eval/aime26.jsonl \
        --max-tokens 16384
"""

import sys
import re
import json
import argparse
import logging
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM = (
    "You are a careful mathematician. Solve the following AIME problem step by step. "
    "Show your reasoning, then put your final integer answer (an integer between 0 and 999) "
    "inside \\boxed{}."
)


def make_user_prompt(question: str) -> str:
    return f"{question}\n\nRemember to put your answer inside \\boxed{{}}."


# Match \boxed{...} and pull the inner content. Handles nested braces shallowly.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_INT_RE   = re.compile(r"-?\d+")


def extract_answer(completion: str) -> str | None:
    """Return the integer string inside the last \\boxed{...}, or None."""
    matches = _BOXED_RE.findall(completion)
    if not matches:
        return None
    last = matches[-1].strip()
    m = _INT_RE.search(last)
    return m.group(0) if m else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return int(pred) == int(gold)
    except ValueError:
        return False


def evaluate(
    checkpoint_path: str | None,
    eval_file: str,
    model_name: str = "/workspace/models/Qwen3.5-2B",
    n_samples: int = 30,
    max_new_tokens: int = 16384,
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

    # ── Load model + checkpoint ───────────────────────────────────────────
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

    # ── Load problems ─────────────────────────────────────────────────────
    problems = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    random.seed(42)
    if len(problems) > n_samples:
        problems = random.sample(problems, n_samples)
    logger.info(f"Loaded {len(problems)} problems")

    # ── Run evaluation ────────────────────────────────────────────────────
    results = []
    for i, p in enumerate(problems):
        question = p["question"]
        gold     = p["answer"]

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": make_user_prompt(question)},
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
                do_sample=True,
                temperature=temperature,
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
        correct = is_correct(pred, gold)

        results.append({
            "correct":           correct,
            "predicted_answer":  pred,
            "gold_answer":       gold,
            "completion_length": len(completion.split()),
            "completion":        completion,
            "question":          question,
        })

        marker = "✓" if correct else "✗"
        logger.info(f"  [{i+1}/{len(problems)}] {marker}  pred={pred} gold={gold}")
        if (i + 1) % 5 == 0:
            so_far = sum(r["correct"] for r in results)
            logger.info(f"  --> running accuracy: {so_far}/{i+1} = {so_far/(i+1):.3f}")

    # ── Report ────────────────────────────────────────────────────────────
    n        = len(results)
    n_correct= sum(r["correct"] for r in results)
    accuracy = n_correct / n
    extracted= sum(1 for r in results if r["predicted_answer"] is not None) / n
    mean_len = sum(r["completion_length"] for r in results) / n

    print("\n" + "="*50)
    print(f"  AIME 2026 Evaluation Results")
    print("="*50)
    print(f"  Mode:           {mode_str}")
    print(f"  N samples:      {n}")
    print(f"  Accuracy:       {n_correct}/{n} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Extraction rate:{extracted:.4f} ({extracted*100:.1f}%)")
    print(f"  Mean length:    {mean_len:.1f} tokens")
    print("="*50)

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
            "results":     results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full output to {output_file}")

    return accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None, help="Path to LoRA checkpoint (omit for base model)")
    p.add_argument("--baseline",    action="store_true", help="Run base model (alias for omitting --checkpoint)")
    p.add_argument("--eval-file",   required=True, help="Path to AIME JSONL")
    p.add_argument("--model",       default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--n-samples",   type=int, default=30)
    p.add_argument("--max-tokens",  type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-think",    action="store_true", help="Disable thinking")
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
