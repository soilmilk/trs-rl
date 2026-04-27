#!/usr/bin/env python3
"""
scripts/evaluate_api.py

Evaluate any model accessible via OpenAI-compatible API on TRS eval sets.
Works with OpenRouter, Together AI, Fireworks, or any OpenAI-compatible endpoint.

Also supports local Qwen3.5-2B evaluation (--local mode).

Usage:
    # API model (e.g. Qwen3-235B via OpenRouter)
    python3 scripts/evaluate_api.py \
        --provider openrouter \
        --model qwen/qwen3-235b-a22b \
        --eval-file data/eval/phase5.jsonl \
        --n-samples 50 \
        --api-key $OPENROUTER_API_KEY

    # Local model (same as evaluate_phase.py but in one script)
    python3 scripts/evaluate_api.py \
        --local \
        --checkpoint runs/no_think/phase5/trs_rl_phase5_v2/checkpoint-1000 \
        --eval-file data/eval/phase5.jsonl \
        --n-samples 50
"""

import sys
import json
import time
import argparse
import logging
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rewritelang import expr_to_str
from generator.instance import TRSInstance
from agent.prompt import make_chat_messages, SYSTEM, make_prompt
from agent.parser import parse_proof_from_output
from training.reward import compute_reward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API providers
# ---------------------------------------------------------------------------

PROVIDER_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "together":   "https://api.together.xyz/v1",
    "fireworks":  "https://api.fireworks.ai/inference/v1",
    "openai":     "https://api.openai.com/v1",
}


def call_api(
    messages: list[dict],
    model: str,
    base_url: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    max_retries: int = 6,
) -> str:
    """Call an OpenAI-compatible chat completion API."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"API failed after {max_retries} attempts: {e}")
                return ""


# ---------------------------------------------------------------------------
# Local model inference
# ---------------------------------------------------------------------------

def load_local_model(model_name: str, checkpoint: str = None):
    """Load local model + tokenizer, return (model, tokenizer)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

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

    if checkpoint and checkpoint != "baseline":
        model = PeftModel.from_pretrained(base_model, checkpoint)
        logger.info(f"Loaded LoRA checkpoint from {checkpoint}")
    else:
        model = base_model
        logger.info("Running base model (no LoRA)")

    model.eval()
    return model, tokenizer


def call_local(
    messages: list[dict],
    model,
    tokenizer,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    enable_thinking: bool = False,
) -> str:
    """Run inference on local model."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(raw_ids, skip_special_tokens=False)
    for special_tok in [tokenizer.eos_token, tokenizer.pad_token]:
        if special_tok:
            completion = completion.replace(special_tok, "")
    return completion.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    instances: list[dict],
    inference_fn,
    model_label: str,
    n_samples: int = 50,
    save_results: str = None,
):
    """Run evaluation using the provided inference function."""
    random.seed(42)
    if len(instances) > n_samples:
        instances = random.sample(instances, n_samples)

    logger.info(f"Evaluating {len(instances)} instances with {model_label}")

    results = []
    for i, d in enumerate(instances):
        inst = TRSInstance.from_dict(d)
        msgs = make_chat_messages(inst)

        completion = inference_fn(msgs) or ""

        reward = compute_reward(completion, inst)
        solved = reward >= 0.99
        parsed = parse_proof_from_output(completion, n_rules=len(inst.rules))

        results.append({
            "solved": solved,
            "reward": reward,
            "completion_length": len(completion.split()),
            "completion": completion,
            "start": expr_to_str(inst.start),
            "normal_form": expr_to_str(inst.normal_form),
            "n_parsed_steps": len(parsed),
        })

        if (i + 1) % 10 == 0:
            so_far = sum(r["solved"] for r in results)
            logger.info(f"  {i+1}/{len(instances)} — solve_rate: {so_far/(i+1):.3f}")

    # ── Report ────────────────────────────────────────────────────────
    n = len(results)
    solve_rate = sum(r["solved"] for r in results) / n
    mean_reward = sum(r["reward"] for r in results) / n
    mean_length = sum(r["completion_length"] for r in results) / n
    parse_rate = sum(1 for r in results if r["n_parsed_steps"] > 0) / n

    print(f"\n{'='*50}")
    print(f"  Evaluation Results: {model_label}")
    print(f"{'='*50}")
    print(f"  N samples:    {n}")
    print(f"  Solve rate:   {solve_rate:.4f} ({solve_rate*100:.1f}%)")
    print(f"  Mean reward:  {mean_reward:.4f}")
    print(f"  Mean length:  {mean_length:.1f} tokens")
    print(f"  Parse rate:   {parse_rate:.4f} ({parse_rate*100:.1f}%)")
    print(f"{'='*50}")

    if save_results:
        Path(save_results).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "model": model_label,
            "n_samples": n,
            "solve_rate": solve_rate,
            "mean_reward": mean_reward,
            "mean_length": mean_length,
            "parse_rate": parse_rate,
            "results": results,
        }
        with open(save_results, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {save_results}")

    return solve_rate


def main():
    p = argparse.ArgumentParser(description="Evaluate TRS models (API or local)")

    # Mode selection
    p.add_argument("--local", action="store_true", help="Use local model instead of API")

    # API options
    p.add_argument("--provider", default="openrouter",
                   choices=list(PROVIDER_URLS.keys()) + ["custom"],
                   help="API provider")
    p.add_argument("--base-url", default=None, help="Custom API base URL (overrides --provider)")
    p.add_argument("--model", default=None, help="Model name/ID for API")
    p.add_argument("--api-key", default=None, help="API key (or set via env var)")

    # Local options
    p.add_argument("--local-model", default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint path")
    p.add_argument("--no-think", action="store_true", help="Disable thinking for local model")

    # Shared options
    p.add_argument("--eval-file", required=True)
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--save-results", default=None)

    args = p.parse_args()

    # ── Load eval instances ──────────────────────────────────────────
    instances = []
    with open(args.eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    logger.info(f"Loaded {len(instances)} instances from {args.eval_file}")

    # ── Set up inference function ────────────────────────────────────
    if args.local:
        model_obj, tokenizer = load_local_model(args.local_model, args.checkpoint)
        enable_thinking = not args.no_think
        model_label = f"local:{args.local_model}"
        if args.checkpoint:
            model_label += f" + {args.checkpoint}"

        def inference_fn(msgs):
            return call_local(msgs, model_obj, tokenizer,
                              max_tokens=args.max_tokens,
                              temperature=args.temperature,
                              enable_thinking=enable_thinking)
    else:
        if not args.model:
            p.error("--model is required for API mode")

        # Resolve API key
        import os
        api_key = args.api_key
        if not api_key:
            env_keys = {
                "openrouter": "OPENROUTER_API_KEY",
                "together":   "TOGETHER_API_KEY",
                "fireworks":  "FIREWORKS_API_KEY",
                "openai":     "OPENAI_API_KEY",
            }
            env_var = env_keys.get(args.provider, "API_KEY")
            api_key = os.environ.get(env_var)
            if not api_key:
                p.error(f"No API key. Pass --api-key or set ${env_var}")

        base_url = args.base_url or PROVIDER_URLS.get(args.provider)
        if not base_url:
            p.error("--base-url required for custom provider")

        model_label = f"{args.provider}:{args.model}"

        def inference_fn(msgs):
            return call_api(msgs, args.model, base_url, api_key,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature)

    # ── Print config ─────────────────────────────────────────────────
    logger.info(f"{'='*50}")
    logger.info(f"  Model: {model_label}")
    logger.info(f"  Eval file: {args.eval_file}")
    logger.info(f"  N samples: {args.n_samples}")
    logger.info(f"  Max tokens: {args.max_tokens}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"{'='*50}")

    # ── Run evaluation ───────────────────────────────────────────────
    evaluate(
        instances=instances,
        inference_fn=inference_fn,
        model_label=model_label,
        n_samples=args.n_samples,
        save_results=args.save_results,
    )


if __name__ == "__main__":
    main()