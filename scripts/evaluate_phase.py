#!/usr/bin/env python3
"""
scripts/evaluate_phase.py

Evaluate a trained checkpoint on a held-out eval set.
Reports solve_rate@1 — the key metric for curriculum advancement.

Usage:
    python3 scripts/evaluate_phase.py \
        --checkpoint runs/trs_rl/checkpoint-500 \
        --eval-file data/eval/phase1.jsonl \
        --phase 1
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from generator.instance import TRSInstance
from agent.prompt import make_chat_messages
from agent.parser import parse_proof_from_output
from training.reward import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    eval_file: str,
    phase: int,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_samples: int = 100,
    max_new_tokens: int = 512,
    temperature: float = 0.8,   # greedy for eval
    save_failures: bool = False,
    failures_output: str = "failures.json",
):
    logger.info(f"Evaluating Phase {phase} checkpoint: {checkpoint_path}")
    logger.info(f"Eval file: {eval_file}")
    logger.info(f"Samples: {n_samples}")

    # ── Load model + checkpoint ───────────────────────────────────────────
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    logger.info("Model loaded.")

    # ── Load eval instances ───────────────────────────────────────────────
    instances = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    # Sample n_samples
    import random
    random.seed(42)
    if len(instances) > n_samples:
        instances = random.sample(instances, n_samples)

    logger.info(f"Loaded {len(instances)} eval instances")

    # ── Run evaluation ────────────────────────────────────────────────────
    results = []
    for i, d in enumerate(instances):
        inst = TRSInstance.from_dict(d)
        msgs = make_chat_messages(inst)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,        # greedy — deterministic
                temperature=0.8,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        reward = compute_reward(completion, inst)
        solved = reward >= 0.99
        results.append({
            "solved": solved,
            "reward": reward,
            "completion_length": len(completion.split()),
            "completion": completion,
            "start": str(inst.start),
            "normal_form": str(inst.normal_form),
        })

        if (i + 1) % 10 == 0:
            so_far = sum(r["solved"] for r in results)
            logger.info(f"  {i+1}/{len(instances)} — solve_rate so far: {so_far/(i+1):.3f}")

    # ── Report ────────────────────────────────────────────────────────────
    n = len(results)
    solve_rate = sum(r["solved"] for r in results) / n
    mean_reward = sum(r["reward"] for r in results) / n
    mean_length = sum(r["completion_length"] for r in results) / n

    print("\n" + "="*50)
    print(f"  Phase {phase} Evaluation Results")
    print("="*50)
    print(f"  Checkpoint:   {checkpoint_path}")
    print(f"  Eval file:    {eval_file}")
    print(f"  N samples:    {n}")
    print(f"  Solve rate:   {solve_rate:.4f} ({solve_rate*100:.1f}%)")
    print(f"  Mean reward:  {mean_reward:.4f}")
    print(f"  Mean length:  {mean_length:.1f} tokens")
    print(f"  Threshold:    0.75 (advance if above)")
    print(f"  Decision:     {'✅ ADVANCE to Phase 2' if solve_rate >= 0.75 else '❌ STAY on Phase 1'}")
    print("="*50)

    if save_failures:
        failures = [r for r in results if not r["solved"]]
        with open(failures_output, "w") as f:
            json.dump(failures, f, indent=2)
        logger.info(f"Saved {len(failures)} failures to {failures_output}")

    return solve_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,  help="Path to LoRA checkpoint")
    p.add_argument("--eval-file",     required=True,  help="Path to eval JSONL file")
    p.add_argument("--phase",         type=int, default=1)
    p.add_argument("--model",         default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-samples",     type=int, default=100)
    p.add_argument("--max-tokens",    type=int, default=512)
    p.add_argument("--save-failures", action="store_true", default=False)
    p.add_argument("--failures-output", default="failures.json")
    args = p.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        eval_file       = args.eval_file,
        phase           = args.phase,
        model_name      = args.model,
        n_samples       = args.n_samples,
        max_new_tokens  = args.max_tokens,
        save_failures   = args.save_failures,
        failures_output = args.failures_output,
    )


if __name__ == "__main__":
    main()