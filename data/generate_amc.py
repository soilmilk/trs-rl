#!/usr/bin/env python3
"""
data/generate_amc.py

Download AMC problems and save as JSONL for evaluation.
Uses AI-MO/aimo-validation-amc (AMC 12 2022/2023, 83 problems with integer answers)
as primary, with fallbacks to other known mirrors.

Usage:
    python3 data/generate_amc.py
"""

import json
from pathlib import Path


def _try_huggingface():
    from datasets import load_dataset
    candidates = [
        "AI-MO/aimo-validation-amc",
        "MathArena/amc_2024",
        "math-ai/amc23",
    ]
    last_err = None
    for cid in candidates:
        try:
            print(f"Trying HuggingFace dataset {cid}...")
            for split in ("train", "test", "validation"):
                try:
                    ds = load_dataset(cid, split=split)
                    print(f"  loaded {cid} split={split} ({len(ds)} rows)")
                    return ds
                except Exception:
                    continue
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise last_err if last_err else RuntimeError("No AMC dataset could be loaded")


def _normalise(ex: dict) -> dict:
    """Map heterogeneous field names to {question, answer}."""
    q = ex.get("question") or ex.get("problem") or ex.get("prompt")

    # Use explicit None checks — answer can legitimately be 0
    a = ex.get("answer")
    if a is None:
        a = ex.get("solution")
    if a is None:
        a = ex.get("final_answer")

    # AMC answers are integers; if stored as float (e.g. 0.0), normalize to int string
    if a is not None:
        try:
            a_float = float(a)
            if a_float.is_integer():
                a = str(int(a_float))
            else:
                a = str(a).strip()
        except (ValueError, TypeError):
            a = str(a).strip()

    return {"question": q, "answer": a}


def main():
    ds = _try_huggingface()

    out = Path("data/eval/amc.jsonl")
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