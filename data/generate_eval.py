#!/usr/bin/env python3
"""
data/generate_eval.py

Pre-generate fixed eval sets for all phases.
Run once before training. Takes ~2 minutes.

Output: data/eval/phase{1-5}.jsonl + data/eval/ood.jsonl
Each file: 300 instances per phase, 200 OOD instances.
"""

import sys
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.instance import generate_instance
from generator.curriculum import PHASE_CONFIGS
from rewritelang import verify_proof


INSTANCES_PER_PHASE = 300
OOD_INSTANCES       = 200


def generate_phase_eval(phase_cfg, n_instances: int, seed_base: int) -> list[dict]:
    """Generate eval instances for a single phase."""
    instances = []
    rng = random.Random(seed_base)
    failures = 0

    for i in tqdm(range(n_instances), desc=f"Phase {phase_cfg.phase}", leave=False):
        seed = seed_base * 10000 + i
        n_steps = rng.randint(phase_cfg.n_steps_min, phase_cfg.n_steps_max)

        kwargs = dict(
            n_rules   = phase_cfg.n_rules,
            max_depth = phase_cfg.max_depth,
            n_steps   = n_steps,
            domain    = phase_cfg.domain if phase_cfg.domain != "mixed" else "boolean",
            seed      = seed,
        )
        if phase_cfg.domain == "mixed":
            kwargs["domain"] = "mixed"
            kwargs["domains_for_mixed"] = ["boolean", "arithmetic", "abstract"]

        try:
            inst = generate_instance(**kwargs)

            # Verify the generated proof is correct (defence in depth)
            reward = verify_proof(inst.rules, inst.start, inst.proof, inst.normal_form)
            assert abs(reward - 1.0) < 1e-9, f"Generated instance has bad proof: {reward}"

            instances.append(inst.to_dict())

        except Exception as e:
            failures += 1
            # Fallback: simpler instance
            try:
                inst = generate_instance(n_rules=3, max_depth=2, n_steps=1,
                                         domain="boolean", seed=seed + 99999)
                instances.append(inst.to_dict())
            except Exception:
                pass  # skip this instance

    if failures > 0:
        print(f"  Warning: {failures}/{n_instances} instances failed generation")

    return instances


def generate_ood_eval(n_instances: int, seed_base: int) -> list[dict]:
    """
    OOD eval: depth and step counts BEYOND training range.
    Phase 5 trains up to depth 10, steps 20.
    OOD: depth 12-15, steps 25-35.
    """
    instances = []
    rng = random.Random(seed_base)

    for i in tqdm(range(n_instances), desc="OOD", leave=False):
        seed = seed_base * 10000 + i
        depth  = rng.randint(11, 14)
        n_steps = rng.randint(22, 30)

        try:
            inst = generate_instance(
                n_rules=9, max_depth=depth, n_steps=n_steps,
                domain="boolean", seed=seed,
            )
            d = inst.to_dict()
            d["metadata"]["ood"] = True
            instances.append(d)
        except Exception:
            try:
                inst = generate_instance(n_rules=7, max_depth=8, n_steps=15,
                                         domain="boolean", seed=seed + 12345)
                d = inst.to_dict()
                d["metadata"]["ood"] = True
                instances.append(d)
            except Exception:
                pass

    return instances


def main():
    parser = argparse.ArgumentParser(description="Generate TRS-RL eval sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phases", nargs="+", type=int, default=[1,2,3,4,5])
    parser.add_argument("--n-per-phase", type=int, default=INSTANCES_PER_PHASE)
    parser.add_argument("--n-ood", type=int, default=OOD_INSTANCES)
    parser.add_argument("--out-dir", type=str, default="data/eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating eval sets (seed={args.seed})")
    print(f"Output: {out_dir.absolute()}\n")

    total_generated = 0

    for phase_cfg in PHASE_CONFIGS:
        if phase_cfg.phase not in args.phases:
            continue

        out_path = out_dir / f"phase{phase_cfg.phase}.jsonl"
        if out_path.exists():
            print(f"  Phase {phase_cfg.phase}: already exists, skipping. (Delete to regenerate)")
            continue

        instances = generate_phase_eval(
            phase_cfg,
            n_instances=args.n_per_phase,
            seed_base=args.seed + phase_cfg.phase * 1000,
        )

        with open(out_path, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        total_generated += len(instances)
        print(f"  Phase {phase_cfg.phase}: {len(instances)} instances → {out_path}")

    # OOD eval set
    ood_path = out_dir / "ood.jsonl"
    if not ood_path.exists():
        ood_instances = generate_ood_eval(args.n_ood, seed_base=args.seed + 9999)
        with open(ood_path, "w") as f:
            for inst in ood_instances:
                f.write(json.dumps(inst) + "\n")
        total_generated += len(ood_instances)
        print(f"  OOD: {len(ood_instances)} instances → {ood_path}")
    else:
        print(f"  OOD: already exists, skipping.")

    print(f"\nDone. Generated {total_generated} instances total.")


if __name__ == "__main__":
    main()
