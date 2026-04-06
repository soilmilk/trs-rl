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

from rewritelang import expr_to_str
from generator.instance import TRSInstance
from agent.prompt import make_chat_messages
from agent.parser import parse_proof_from_output, extract_think_block
from training.reward import compute_reward
from training.emergence import emergence_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    eval_file: str,
    phase: int,
    model_name: str = "/workspace/models/Qwen3.5-2B",
    n_samples: int = 100,
    max_new_tokens: int = 512,
    temperature: float = 0.8,   # greedy for eval
    save_failures: bool = False,
    failures_output: str = "failures.json",
    save_results: str = None,
    emergence_output: str = None,
):
    logger.info(f"Evaluating Phase {phase} checkpoint: {checkpoint_path}")
    logger.info(f"Eval file: {eval_file}")
    logger.info(f"Samples: {n_samples}")

    # ── Load model + checkpoint ───────────────────────────────────────────
    logger.info("Loading model...")
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
        logger.info("Running BASELINE (no LoRA checkpoint)")
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
    all_think_texts = []
    all_parsed_proofs = []
    all_instances = []
    for i, d in enumerate(instances):
        inst = TRSInstance.from_dict(d)
        msgs = make_chat_messages(inst)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
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

        # Decode with skip_special_tokens=False to preserve <think> tags
        raw_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(raw_ids, skip_special_tokens=False)
        # Strip EOS/pad tokens but keep <think>...</think>
        for special_tok in [tokenizer.eos_token, tokenizer.pad_token]:
            if special_tok:
                completion = completion.replace(special_tok, "")
        completion = completion.strip()

        # Extract and print think trace
        think_start = completion.find("<think>")
        think_end = completion.find("</think>")
        if think_start != -1 and think_end != -1:
            think_trace = completion[think_start + 7:think_end].strip()
            print(f"\n{'='*60}")
            print(f"Instance {i} | start: {expr_to_str(inst.start)}")
            print(f"{'─'*60}")
            print(f"THINK: {think_trace}")
            print(f"{'='*60}")
        else:
            print(f"\nInstance {i} | No <think> block found")

        # Collect for emergence analysis
        think_text = extract_think_block(completion)
        parsed_proof = parse_proof_from_output(completion, n_rules=len(inst.rules))
        all_think_texts.append(think_text)
        all_parsed_proofs.append(parsed_proof)
        all_instances.append(inst)

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

    # ── Emergence Analysis ─────────────────────────────────────────────
    completions_raw = [r["completion"] for r in results]
    report = emergence_report(
        outputs       = completions_raw,
        parsed_proofs = all_parsed_proofs,
        instances     = all_instances,
        think_texts   = all_think_texts,
    )

    think_lengths = [len(t.split()) for t in all_think_texts if t]
    mean_think_len = sum(think_lengths) / len(think_lengths) if think_lengths else 0.0
    think_present = sum(1 for t in all_think_texts if t)

    print(f"\n{'='*50}")
    print(f"  Emergence Analysis")
    print(f"{'='*50}")
    print(f"  Think blocks found:        {think_present}/{n}")
    print(f"  Mean think length:         {mean_think_len:.1f} words")
    print(f"  LO alignment (mean):       {report['lo_alignment_mean']:.4f}")
    print(f"  Innermost alignment (mean):{report['innermost_alignment_mean']:.4f}")
    print(f"  Reflection rate:           {report['reflection_rate']:.4f} ({report['reflection_rate']*100:.1f}%)")
    print(f"  Reflection count (mean):   {report['reflection_count_mean']:.2f}")
    print(f"  Proofs analysed:           {report['n_proofs_analysed']}")

    if report['rule_frequency']:
        print(f"  Rule frequency distribution:")
        for idx, freq in enumerate(report['rule_frequency']):
            if freq > 0:
                print(f"    RULE {idx+1}: {freq:.3f}")
    print(f"{'='*50}")

    if emergence_output:
        emergence_data = {
            "checkpoint": checkpoint_path,
            "phase": phase,
            "n_samples": n,
            "solve_rate": solve_rate,
            "mean_reward": mean_reward,
            "think_blocks_found": think_present,
            "mean_think_length_words": mean_think_len,
            "lo_alignment_mean": report["lo_alignment_mean"],
            "innermost_alignment_mean": report["innermost_alignment_mean"],
            "reflection_rate": report["reflection_rate"],
            "reflection_count_mean": report["reflection_count_mean"],
            "rule_frequency": report["rule_frequency"],
            "n_proofs_analysed": report["n_proofs_analysed"],
        }
        with open(emergence_output, "w") as f:
            json.dump(emergence_data, f, indent=2)
        logger.info(f"Saved emergence report to {emergence_output}")

    if save_results:
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved {len(results)} results to {save_results}")

    if save_failures:
        failures = [r for r in results if not r["solved"]]
        with open(failures_output, "w") as f:
            json.dump(failures, f, indent=2)
        logger.info(f"Saved {len(failures)} failures to {failures_output}")

    return solve_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default=None,   help="Path to LoRA checkpoint (omit or 'baseline' for zero-shot)")
    p.add_argument("--baseline",      action="store_true", help="Run zero-shot baseline without LoRA")
    p.add_argument("--eval-file",     required=True,  help="Path to eval JSONL file")
    p.add_argument("--phase",         type=int, default=1)
    p.add_argument("--model",         default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--n-samples",     type=int, default=100)
    p.add_argument("--max-tokens",    type=int, default=512)
    p.add_argument("--save-failures", action="store_true", default=False)
    p.add_argument("--failures-output", default="failures.json")
    p.add_argument("--save-results", default=None, help="Save ALL results (solved+unsolved) as JSON")
    p.add_argument("--emergence-output", default=None, help="Save emergence report as JSON")
    args = p.parse_args()

    checkpoint = args.checkpoint
    if args.baseline:
        checkpoint = "baseline"
    elif checkpoint is None:
        p.error("--checkpoint is required unless --baseline is set")

    evaluate(
        checkpoint_path = checkpoint,
        eval_file       = args.eval_file,
        phase           = args.phase,
        model_name      = args.model,
        n_samples       = args.n_samples,
        max_new_tokens  = args.max_tokens,
        save_failures    = args.save_failures,
        failures_output  = args.failures_output,
        save_results     = args.save_results,
        emergence_output = args.emergence_output,
    )


if __name__ == "__main__":
    main()