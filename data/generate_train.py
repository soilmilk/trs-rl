#!/usr/bin/env python3
"""
data/generate_train.py

Pre-generate training instances locally before SSH-ing to GPU.
Saves per-phase pools to data/train/phase{N}.jsonl.

The training loop loads from these files instead of generating on-the-fly,
saving CPU time on expensive GPU instances.

Usage:
    # Generate default pools (2000 instances per phase)
    python data/generate_train.py

    # Match exact training run: max_steps=8000, batch_size=4
    python data/generate_train.py --max-steps 8000 --batch-size 4

    # Only regenerate specific phases
    python data/generate_train.py --phases 1 2 3
"""

import sys
import json
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from generator.instance import generate_instance
from generator.curriculum import PHASE_CONFIGS
from rewritelang import verify_proof


DEFAULT_MAX_STEPS  = 8000
DEFAULT_BATCH_SIZE = 4


def generate_phase_pool(phase_cfg, n_instances: int, seed_base: int) -> list[dict]:
    rng = random.Random(seed_base)
    instances = []
    failures = 0

    print(f"  Phase {phase_cfg.phase}: generating {n_instances} instances...", flush=True)
    for i in range(n_instances):
        if i > 0 and i % 500 == 0:
            print(f"    {i}/{n_instances}", flush=True)
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

            reward = verify_proof(inst.rules, inst.start, inst.proof, inst.normal_form)
            assert abs(reward - 1.0) < 1e-9, f"Bad proof reward: {reward}"

            instances.append(inst.to_dict())

        except Exception:
            failures += 1
            try:
                inst = generate_instance(n_rules=3, max_depth=2, n_steps=1,
                                         domain="boolean", seed=seed + 99999)
                instances.append(inst.to_dict())
            except Exception:
                pass

    if failures > 0:
        print(f"  Warning: {failures}/{n_instances} instances failed, used fallback")

    return instances


def main():
    parser = argparse.ArgumentParser(description="Pre-generate TRS-RL training data")
    parser.add_argument("--max-steps",  type=int, default=DEFAULT_MAX_STEPS,
                        help="Training steps (used to size per-phase pools)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training batch size per step")
    parser.add_argument("--phases",     nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed",       type=int, default=1337)
    parser.add_argument("--out-dir",    type=str, default="data/train")
    parser.add_argument("--force",      action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Distribute total instances across phases. Each phase gets an equal slice
    # of the full budget so any phase can sustain a full training run.
    n_phases        = len(args.phases)
    total_instances = args.max_steps * args.batch_size
    per_phase       = max(total_instances // n_phases, 100)

    print(f"\nGenerating training pools  (seed={args.seed})")
    print(f"  max_steps={args.max_steps}  batch_size={args.batch_size}")
    print(f"  total={total_instances}  per_phase={per_phase}")
    print(f"  Output: {out_dir.absolute()}\n")

    total_generated = 0

    for phase_cfg in PHASE_CONFIGS:
        if phase_cfg.phase not in args.phases:
            continue

        out_path = out_dir / f"phase{phase_cfg.phase}.jsonl"
        if out_path.exists() and not args.force:
            print(f"  Phase {phase_cfg.phase}: already exists, skipping (--force to overwrite)")
            continue

        instances = generate_phase_pool(
            phase_cfg,
            n_instances = per_phase,
            seed_base   = args.seed + phase_cfg.phase * 1000,
        )

        with open(out_path, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        total_generated += len(instances)
        print(f"  Phase {phase_cfg.phase}: {len(instances)} instances -> {out_path}")

    print(f"\nDone. Generated {total_generated} instances total.")
    print(f"\nTo use during training, pass --train-data {out_dir.absolute()}")


if __name__ == "__main__":
    main()
