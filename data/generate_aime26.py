#!/usr/bin/env python3
"""
data/generate_aime26.py

Download AIME 2026 problems and save as JSONL for evaluation.
Run once before evaluating.

Usage:
    python3 data/generate_aime26.py
"""

import json
from pathlib import Path

from datasets import load_dataset


def main():
    print("Downloading evalscope/aime26 from HuggingFace...")
    ds = load_dataset("evalscope/aime26", split="test")

    out = Path("data/eval/aime26.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w") as f:
        for ex in ds:
            f.write(json.dumps({
                "question": ex["question"],
                "answer":   str(ex["answer"]).strip(),
            }) + "\n")
            n += 1

    print(f"Wrote {n} problems to {out}")


if __name__ == "__main__":
    main()
