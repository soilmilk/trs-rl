#!/usr/bin/env python3
"""
data/generate_aime26.py

Download AIME 2026 problems and save as JSONL for evaluation.
Tries ModelScope first (where evalscope/aime26 lives), falls back to HF.

Usage:
    pip install modelscope     # if not already installed
    python3 data/generate_aime26.py
"""

import json
from pathlib import Path


def _try_modelscope():
    from modelscope.msdatasets import MsDataset
    print("Downloading evalscope/aime26 from ModelScope...")
    return MsDataset.load("evalscope/aime26", split="test")


def _try_huggingface():
    from datasets import load_dataset
    # HF mirrors of AIME 2026 (try a few known IDs)
    candidates = [
        "math-ai/aime26",
        "yentinglin/aime_2026",
        "MathArena/aime_2026_I",
    ]
    last_err = None
    for cid in candidates:
        try:
            print(f"Trying HuggingFace dataset {cid}...")
            return load_dataset(cid, split="test")
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise last_err


def _normalise(ex: dict) -> dict:
    """Map heterogeneous field names to {question, answer}."""
    q = ex.get("question") or ex.get("problem") or ex.get("prompt")
    a = ex.get("answer") or ex.get("solution") or ex.get("final_answer")
    return {"question": q, "answer": str(a).strip()}


def main():
    try:
        ds = _try_modelscope()
    except Exception as e:
        print(f"ModelScope failed: {e}")
        print("Falling back to HuggingFace...")
        ds = _try_huggingface()

    out = Path("data/eval/aime26.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w") as f:
        for ex in ds:
            row = _normalise(ex)
            if not row["question"] or not row["answer"]:
                print(f"  skipping malformed row: {ex}")
                continue
            f.write(json.dumps(row) + "\n")
            n += 1

    print(f"Wrote {n} problems to {out}")


if __name__ == "__main__":
    main()
