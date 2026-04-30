#!/usr/bin/env python3
"""
data/generate_mmlu_prox.py

Download MMLU-ProX (English) and save as JSONL for evaluation.
Run once before evaluating.

Usage:
    python3 data/generate_mmlu_prox.py
    python3 data/generate_mmlu_prox.py --language en --split test --limit 500
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", default="en", help="Language code (en, zh, fr, ...)")
    p.add_argument("--split",    default="test", choices=["test", "validation"])
    p.add_argument("--limit",    type=int, default=500,
                   help="Max problems to keep (test set is 11.8k). Use 0 for all.")
    p.add_argument("--output",   default=None,
                   help="Output JSONL path (default: data/eval/mmlu_prox_<lang>_<split>.jsonl)")
    args = p.parse_args()

    out_path = args.output or f"data/eval/mmlu_prox_{args.language}_{args.split}.jsonl"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading li-lab/MMLU-ProX [{args.language}/{args.split}]...")
    ds = load_dataset("li-lab/MMLU-ProX", args.language, split=args.split)
    print(f"Loaded {len(ds)} problems")

    n = 0
    with open(out_path, "w") as f:
        for ex in ds:
            options = [ex.get(f"option_{i}", "") for i in range(10)]
            row = {
                "question_id": ex["question_id"],
                "category":    ex.get("category", ""),
                "question":    ex["question"],
                "options":     options,
                "answer":      ex["answer"],          # letter A-J
                "answer_index":ex["answer_index"],   # 0-9
            }
            f.write(json.dumps(row) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} problems to {out_path}")


if __name__ == "__main__":
    main()
