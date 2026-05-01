#!/usr/bin/env python3
"""
scripts/evaluate_math500.py

Evaluate a checkpoint (or base model) on HuggingFaceH4/MATH-500.
Reports:
  - accuracy@1 (exact match after normalization, with sympy fallback for equivalence)
  - per-subject and per-level breakdown
  - extraction rate

Usage:
    python3 scripts/evaluate_math500.py \
        --eval-file data/eval/math500.jsonl \
        --n-samples 100 \
        --max-tokens 4096
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
    "You are a careful mathematician. Solve the following problem step by step. "
    "Show your reasoning, then put your final answer inside \\boxed{}."
)


def make_user_prompt(problem: str) -> str:
    return f"{problem}\n\nRemember to put your final answer inside \\boxed{{}}."


# ---------------------------------------------------------------------------
# Answer extraction: pulls the content of the LAST \boxed{...}, handling
# nested braces (e.g. \boxed{\frac{1}{2}}).
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        # Fallback: \boxed without brace, e.g. "\boxed 5"
        m = re.findall(r"\\boxed\s+([^\s$]+)", text)
        return m[-1].strip() if m else None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth > 0:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out).strip() if out else None


# ---------------------------------------------------------------------------
# Answer normalisation + equivalence check.
# Strategy:
#   1) Normalise both strings (strip whitespace, unify \dfrac/\frac, etc.)
#   2) Exact string match → True
#   3) Try sympy: parse both as expressions, simplify difference → True if zero
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # Unify common LaTeX variants
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("$", "")
    # Strip surrounding \text{...}
    m = re.match(r"^\s*\\text\{(.+)\}\s*$", s)
    if m:
        s = m.group(1)
    # Collapse whitespace
    s = re.sub(r"\s+", "", s)
    # Strip trailing period
    s = s.rstrip(".")
    return s


def _try_sympy_equiv(a: str, b: str) -> bool | None:
    """Return True/False if sympy can decide; None if it can't parse."""
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex
    except Exception:
        return None

    def _parse(x):
        x = x.replace("\\$", "")
        # Try LaTeX parser first
        try:
            return parse_latex(x)
        except Exception:
            pass
        # Fall back to plain sympify
        try:
            return sympify(x)
        except Exception:
            return None

    pa, pb = _parse(a), _parse(b)
    if pa is None or pb is None:
        return None
    try:
        return simplify(pa - pb) == 0
    except Exception:
        return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    np_pred = _normalise(pred)
    np_gold = _normalise(gold)
    if np_pred == np_gold:
        return True
    # sympy fallback for symbolic equivalence
    eq = _try_sympy_equiv(pred, gold)
    return bool(eq)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str | None,
    eval_file: str,
    model_name: str = "/workspace/models/Qwen3.5-2B",
    n_samples: int = 100,
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

    problems = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if not problems:
        raise RuntimeError(f"No problems loaded from {eval_file}")

    random.seed(42)
    if len(problems) > n_samples:
        problems = random.sample(problems, n_samples)
    logger.info(f"Loaded {len(problems)} problems")

    do_sample = temperature > 0.0
    results = []
    for i, p in enumerate(problems):
        problem = p["problem"]
        gold    = p["answer"]
        subject = p.get("subject", "")
        level   = p.get("level", 0)

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": make_user_prompt(problem)},
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

        pred    = extract_boxed(completion)
        correct = is_correct(pred, gold)

        results.append({
            "correct":           correct,
            "predicted_answer":  pred,
            "gold_answer":       gold,
            "subject":           subject,
            "level":             level,
            "completion_length": len(completion.split()),
            "completion":        completion,
            "unique_id":         p.get("unique_id"),
        })

        marker = "✓" if correct else "✗"
        pred_short = (pred[:30] + "…") if pred and len(pred) > 30 else pred
        gold_short = (gold[:30] + "…") if len(gold) > 30 else gold
        logger.info(f"  [{i+1}/{len(problems)}] {marker}  pred={pred_short!r} gold={gold_short!r}  ({subject} L{level})")
        if (i + 1) % 25 == 0:
            so_far = sum(r["correct"] for r in results)
            logger.info(f"  --> running accuracy: {so_far}/{i+1} = {so_far/(i+1):.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────
    n         = len(results)
    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / n
    extracted = sum(1 for r in results if r["predicted_answer"] is not None) / n
    mean_len  = sum(r["completion_length"] for r in results) / n

    print("\n" + "="*60)
    print(f"  MATH-500 Evaluation Results")
    print("="*60)
    print(f"  Mode:            {mode_str}")
    print(f"  N samples:       {n}")
    print(f"  Accuracy:        {n_correct}/{n} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Extraction rate: {extracted:.4f} ({extracted*100:.1f}%)")
    print(f"  Mean length:     {mean_len:.1f} words")
    print("="*60)

    by_sub = defaultdict(lambda: {"n": 0, "correct": 0})
    by_lvl = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        s = r["subject"] or "uncategorized"
        l = r["level"] or 0
        by_sub[s]["n"] += 1; by_sub[s]["correct"] += int(r["correct"])
        by_lvl[l]["n"] += 1; by_lvl[l]["correct"] += int(r["correct"])

    if by_sub:
        print(f"\n  Per-subject accuracy:")
        for k in sorted(by_sub.keys()):
            s = by_sub[k]
            acc = s["correct"] / s["n"] if s["n"] else 0
            print(f"    {k:30s}  {s['correct']:>3d}/{s['n']:<3d} = {acc*100:5.1f}%")

    if by_lvl:
        print(f"\n  Per-level accuracy:")
        for k in sorted(by_lvl.keys()):
            s = by_lvl[k]
            acc = s["correct"] / s["n"] if s["n"] else 0
            print(f"    Level {k}                          {s['correct']:>3d}/{s['n']:<3d} = {acc*100:5.1f}%")
    print("="*60)

    if save_results:
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {save_results}")

    if output_file:
        out = {
            "mode":            "zero-shot" if is_zeroshot else "checkpoint",
            "checkpoint":      checkpoint_path,
            "eval_file":       eval_file,
            "n_samples":       n,
            "accuracy":        accuracy,
            "n_correct":       n_correct,
            "extraction_rate": extracted,
            "mean_length":     mean_len,
            "by_subject":      {k: dict(v) for k, v in by_sub.items()},
            "by_level":        {str(k): dict(v) for k, v in by_lvl.items()},
            "results":         results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full output to {output_file}")

    return accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--baseline",    action="store_true")
    p.add_argument("--eval-file",   required=True)
    p.add_argument("--model",       default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--n-samples",   type=int, default=100)
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
