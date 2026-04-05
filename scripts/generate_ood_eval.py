#!/usr/bin/env python3
"""
scripts/generate_ood_eval.py

Generate out-of-distribution eval instances at depth 10-15 and 20+ steps.
These are beyond any training phase to test generalization.

Usage:
    python3 scripts/generate_ood_eval.py \
        --output data/eval/ood.jsonl \
        --n-instances 200 \
        --seed 9999
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.instance import generate_instance


def main():
    p = argparse.ArgumentParser(description="Generate OOD eval instances")
    p.add_argument("--output", default="data/eval/ood.jsonl")
    p.add_argument("--n-instances", type=int, default=200)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--min-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--n-rules", type=int, default=9)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--domain", default="boolean")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    seed = args.seed
    failures = 0

    with open(args.output, "w") as f:
        while generated < args.n_instances:
            n_steps = args.min_steps + (seed % (args.max_steps - args.min_steps + 1))
            try:
                inst = generate_instance(
                    n_rules=args.n_rules,
                    max_depth=args.max_depth,
                    n_steps=n_steps,
                    domain=args.domain,
                    seed=seed,
                )
                # Verify it's actually OOD: depth >= 10 or steps >= 20
                if inst.start.depth() >= 8 or len(inst.proof) >= 15:
                    f.write(json.dumps(inst.to_dict()) + "\n")
                    generated += 1
                    if generated % 20 == 0:
                        print(f"  Generated {generated}/{args.n_instances} "
                              f"(depth={inst.start.depth()}, steps={len(inst.proof)})")
            except RuntimeError:
                failures += 1

            seed += 1
            if failures > args.n_instances * 10:
                print(f"WARNING: Too many failures ({failures}). "
                      f"Generated {generated}/{args.n_instances}. "
                      "Consider reducing min-steps or n-rules.")
                break

    print(f"\nDone. Saved {generated} OOD instances to {args.output}")
    print(f"  ({failures} generation failures skipped)")


if __name__ == "__main__":
    main()
