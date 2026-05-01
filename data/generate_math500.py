#!/usr/bin/env python3
"""
data/generate_math500.py

Download HuggingFaceH4/MATH-500 and save as JSONL.

Usage:
    python3 data/generate_math500.py
    python3 data/generate_math500.py --limit 200
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="Max problems (0 = all 500)")
    p.add_argument("--output", default="data/eval/math500.jsonl")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading HuggingFaceH4/MATH-500...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} problems")

    n = 0
    with open(out, "w") as f:
        for ex in ds:
            row = {
                "unique_id": ex["unique_id"],
                "problem":   ex["problem"],
                "answer":    ex["answer"],
                "solution":  ex["solution"],
                "subject":   ex["subject"],
                "level":     ex["level"],
            }
            f.write(json.dumps(row) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} problems to {out}")


if __name__ == "__main__":
    main()
