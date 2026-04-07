#!/usr/bin/env python3
"""
scripts/generate_ood_eval.py

Generate 200 out-of-distribution eval instances harder than any training phase.
  max_depth=14, n_steps=28, n_rules=9, domain=boolean, seeds 10000-10199.

Usage:
    python3 scripts/generate_ood_eval.py
    python3 scripts/generate_ood_eval.py --output data/eval/ood.jsonl
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
    p.add_argument("--seed-start", type=int, default=10000,
                   help="First seed (instances use seeds seed_start..seed_start+n_instances-1)")
    p.add_argument("--n-steps", type=int, default=28)
    p.add_argument("--n-rules", type=int, default=9)
    p.add_argument("--max-depth", type=int, default=14)
    p.add_argument("--domain", default="boolean")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    failures = 0
    instances = []

    print(f"Generating {args.n_instances} OOD instances...")
    print(f"  max_depth={args.max_depth}, n_steps={args.n_steps}, "
          f"n_rules={args.n_rules}, domain={args.domain}")
    print(f"  seeds: {args.seed_start}–{args.seed_start + args.n_instances - 1}")

    with open(args.output, "w") as f:
        for i in range(args.n_instances):
            seed = args.seed_start + i
            try:
                inst = generate_instance(
                    n_rules=args.n_rules,
                    max_depth=args.max_depth,
                    n_steps=args.n_steps,
                    domain=args.domain,
                    seed=seed,
                )
                d = inst.to_dict()
                f.write(json.dumps(d) + "\n")
                instances.append(inst)
                generated += 1
                if generated % 20 == 0:
                    print(f"  Generated {generated}/{args.n_instances} "
                          f"(depth={inst.start.depth()}, steps={len(inst.proof)})")
            except RuntimeError as e:
                failures += 1
                print(f"  Seed {seed} failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  OOD Generation Summary")
    print(f"{'='*50}")
    print(f"  Count:      {generated} instances ({failures} failures)")
    print(f"  Output:     {args.output}")

    if instances:
        depths = [inst.start.depth() for inst in instances]
        steps = [len(inst.proof) for inst in instances]
        print(f"  Depth:      mean={sum(depths)/len(depths):.1f}  "
              f"min={min(depths)}  max={max(depths)}")
        print(f"  Steps:      mean={sum(steps)/len(steps):.1f}  "
              f"min={min(steps)}  max={max(steps)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
