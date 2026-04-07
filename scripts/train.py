#!/usr/bin/env python3
"""
scripts/train.py — Main training entry point.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.grpo import train


def main():
    p = argparse.ArgumentParser(description="TRS-RL GRPO Training")
    p.add_argument("--model",               default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--output-dir",          default="runs/qwen3.5")
    p.add_argument("--train-data-dir",      default="data/train")
    p.add_argument("--start-phase",         type=int,   default=1,    help="Which phase data to load (1-5)")
    p.add_argument("--resume-checkpoint",   default=None,             help="Path to LoRA checkpoint to resume from")
    p.add_argument("--max-steps",           type=int,   default=3500)
    p.add_argument("--lr",                  type=float, default=8e-6)
    p.add_argument("--batch-size",          type=int,   default=4)
    p.add_argument("--group-size",          type=int,   default=8)
    p.add_argument("--max-tokens",          type=int,   default=1536)
    p.add_argument("--temperature",         type=float, default=0.7)
    p.add_argument("--kl-coeff",            type=float, default=0.05)
    p.add_argument("--lora-rank",           type=int,   default=16)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--eval-every",          type=int,   default=250)
    p.add_argument("--save-every",          type=int,   default=500)
    p.add_argument("--no-think",            action="store_true", help="Disable thinking in chat template (for no_think training)")
    args = p.parse_args()

    train(
        model_name        = args.model,
        output_dir        = args.output_dir,
        train_data_dir    = args.train_data_dir,
        start_phase       = args.start_phase,
        resume_checkpoint = args.resume_checkpoint,
        max_steps         = args.max_steps,
        learning_rate     = args.lr,
        batch_size        = args.batch_size,
        group_size        = args.group_size,
        max_new_tokens    = args.max_tokens,
        temperature       = args.temperature,
        kl_coeff          = args.kl_coeff,
        lora_rank         = args.lora_rank,
        seed              = args.seed,
        eval_every        = args.eval_every,
        save_every        = args.save_every,
        enable_thinking   = not args.no_think,
    )


if __name__ == "__main__":
    main()