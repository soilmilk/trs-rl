#!/usr/bin/env python3
"""
data/generate_putnam.py

Download Putnam problems and save as JSONL for evaluation.
Tries Putnam-AXIOM (the cleanest source with answers) first, with fallbacks.

Note: Putnam answers are often free-form expressions (e.g. "n^2+1", "1/2"),
not integers. The evaluator uses string-normalized matching with multiple
heuristics — see scripts/evaluate_putnam.py.

Usage:
    python3 data/generate_putnam.py
"""

import json
from pathlib import Path


def _try_huggingface():
    from datasets import load_dataset
    candidates = [
        "Putnam-AXIOM/putnam-axiom-original",
        "amitayusht/PutnamBench",
        "MathArena/putnam_2024",
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
    raise last_err if last_err else RuntimeError("No Putnam dataset could be loaded")


def _normalise(ex: dict) -> dict:
    """Map heterogeneous field names to {question, answer}."""
    q = (
        ex.get("question")
        or ex.get("problem")
        or ex.get("informal_statement")
        or ex.get("prompt")
    )

    a = ex.get("answer")
    if a is None:
        a = ex.get("solution")
    if a is None:
        a = ex.get("final_answer")
    if a is None:
        a = ex.get("informal_solution")

    if a is not None:
        # If numeric and integer-valued, normalize to int string
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

    out = Path("data/eval/putnam.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_skipped = 0
    with open(out, "w") as f:
        for ex in ds:
            row = _normalise(ex)
            if not row["question"] or not row["answer"]:
                n_skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            n += 1

    print(f"Wrote {n} problems to {out}")
    if n_skipped:
        print(f"  (skipped {n_skipped} rows missing question or answer)")


if __name__ == "__main__":
    main()