#!/usr/bin/env python3
"""
data/generate_ifeval.py

Download Google IFEval and save as JSONL.

Usage:
    python3 data/generate_ifeval.py
    python3 data/generate_ifeval.py --limit 200
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="Max problems (0 = all 541)")
    p.add_argument("--output", default="data/eval/ifeval.jsonl")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading google/IFEval...")
    ds = load_dataset("google/IFEval", split="train")
    print(f"Loaded {len(ds)} examples")

    n = 0
    with open(out, "w") as f:
        for ex in ds:
            row = {
                "key":                 ex["key"],
                "prompt":              ex["prompt"],
                "instruction_id_list": ex["instruction_id_list"],
                "kwargs":              ex["kwargs"],
            }
            f.write(json.dumps(row) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"Wrote {n} prompts to {out}")


if __name__ == "__main__":
    main()
