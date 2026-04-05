#!/usr/bin/env python3
"""
scripts/analyze_metrics.py

Standalone post-hoc analysis of TRS-RL eval results.
Reads saved eval JSON files + JSONL ground truth and computes
all Section 9 metrics from the design doc.

Usage:
    python3 scripts/analyze_metrics.py \
        --eval-json runs/qwen35/phase1/eval_results.json \
        --ground-truth data/train/phase1.jsonl \
        --phase 1

    # Or batch across phases:
    python3 scripts/analyze_metrics.py \
        --eval-json runs/qwen35/phase1/eval.json runs/qwen35/phase2/eval.json \
        --ground-truth data/train/phase1.jsonl data/train/phase2.jsonl \
        --phase 1 2

    # Save markdown report:
    python3 scripts/analyze_metrics.py \
        --eval-json runs/qwen35/phase1/eval.json \
        --ground-truth data/train/phase1.jsonl \
        --phase 1 \
        --output-md results/report.md \
        --output-json results/metrics.json
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rewritelang import (
    Expr, Rule, parse_expr, expr_to_str, expr_equal,
    match, apply_rule_at, all_positions, find_all_matches,
    verify_proof, verify_proof_detailed, is_normal_form, ParseError,
)
from rewritelang.match import get_subexpr
from generator.instance import TRSInstance
from agent.parser import parse_proof_from_output, extract_think_block
from training.emergence import (
    lo_alignment_score, innermost_alignment_score,
    has_reflection, reflection_count, rule_frequency,
    compute_lo_choice, compute_innermost_choice,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. solve_by_depth — bucket by start expression depth
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_BUCKETS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 999)]
DEPTH_LABELS  = ["1-2", "3-4", "5-6", "7-8", "9+"]

def compute_solve_by_depth(results: List[dict]) -> Dict[str, dict]:
    """Compute solve_rate@1 bucketed by expression depth."""
    buckets = {label: {"solved": 0, "total": 0} for label in DEPTH_LABELS}

    for r in results:
        try:
            expr = parse_expr(r["start"])
            depth = expr.depth()
        except Exception:
            continue

        for (lo, hi), label in zip(DEPTH_BUCKETS, DEPTH_LABELS):
            if lo <= depth <= hi:
                buckets[label]["total"] += 1
                if r["solved"]:
                    buckets[label]["solved"] += 1
                break

    out = {}
    for label in DEPTH_LABELS:
        b = buckets[label]
        out[label] = {
            "solve_rate": b["solved"] / b["total"] if b["total"] > 0 else 0.0,
            "solved": b["solved"],
            "total": b["total"],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. solve_by_steps — bucket by ground truth proof length
# ─────────────────────────────────────────────────────────────────────────────

STEP_BUCKETS = [(1, 4), (5, 8), (9, 15), (16, 999)]
STEP_LABELS  = ["1-4", "5-8", "9-15", "16+"]

def compute_solve_by_steps(
    results: List[dict],
    gt_by_start: Dict[str, dict],
) -> Dict[str, dict]:
    """Compute solve_rate@1 bucketed by ground truth proof length."""
    buckets = {label: {"solved": 0, "total": 0} for label in STEP_LABELS}

    for r in results:
        start_key = r["start"]
        gt = gt_by_start.get(start_key)
        if gt is None:
            continue
        n_steps = len(gt.get("proof", []))
        if n_steps == 0:
            # Try metadata
            n_steps = gt.get("metadata", {}).get("n_steps", 0)

        for (lo, hi), label in zip(STEP_BUCKETS, STEP_LABELS):
            if lo <= n_steps <= hi:
                buckets[label]["total"] += 1
                if r["solved"]:
                    buckets[label]["solved"] += 1
                break

    out = {}
    for label in STEP_LABELS:
        b = buckets[label]
        out[label] = {
            "solve_rate": b["solved"] / b["total"] if b["total"] > 0 else 0.0,
            "solved": b["solved"],
            "total": b["total"],
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. parse_rate — fraction of completions with a valid PROOF block
# ─────────────────────────────────────────────────────────────────────────────

def compute_parse_rate(results: List[dict], gt_instances: List[dict]) -> float:
    """Fraction of completions containing at least one parseable proof step."""
    n_parseable = 0
    for r, gt in zip(results, gt_instances):
        n_rules = len(gt.get("rules", []))
        if n_rules == 0:
            n_rules = 9  # fallback
        steps = parse_proof_from_output(r["completion"], n_rules=n_rules)
        if steps:
            n_parseable += 1
    return n_parseable / len(results) if results else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. partial_step_rate — mean fraction of valid steps per proof
# ─────────────────────────────────────────────────────────────────────────────

def compute_partial_step_rate(results: List[dict], gt_instances: List[dict]) -> float:
    """Mean fraction of valid steps per proof (including unsolved)."""
    rates = []
    for r, gt in zip(results, gt_instances):
        inst = TRSInstance.from_dict(gt)
        steps = parse_proof_from_output(r["completion"], n_rules=len(inst.rules))
        if not steps:
            rates.append(0.0)
            continue
        detail = verify_proof_detailed(inst.rules, inst.start, steps, inst.normal_form)
        total = detail["total_steps"]
        valid = detail["valid_steps"]
        rates.append(valid / total if total > 0 else 0.0)
    return sum(rates) / len(rates) if rates else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. solve_rate@3 estimate from single-run data
# ─────────────────────────────────────────────────────────────────────────────

def estimate_solve_rate_at_3(results: List[dict]) -> float:
    """
    Estimate solve_rate@3 from single-run solve_rate@1.
    If p = P(solve in one try), then P(solve in 3 tries) = 1 - (1-p)^3.
    This is a lower bound — actual solve@3 with temperature sampling may be higher.
    """
    p = sum(r["solved"] for r in results) / len(results) if results else 0.0
    return 1.0 - (1.0 - p) ** 3


# ─────────────────────────────────────────────────────────────────────────────
# 6. LO alignment score — the most important emergence metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_lo_alignment(results: List[dict], gt_instances: List[dict]) -> dict:
    """
    For each proof with at least one valid step, compute what fraction of
    the agent's rule choices match what LO strategy would choose.
    """
    scores = []
    for r, gt in zip(results, gt_instances):
        inst = TRSInstance.from_dict(gt)
        steps = parse_proof_from_output(r["completion"], n_rules=len(inst.rules))
        if not steps:
            continue
        # Filter to only steps with valid expressions
        valid_steps = [(e, idx) for e, idx in steps if e is not None and idx >= 0]
        if not valid_steps:
            continue
        score = lo_alignment_score(valid_steps, inst)
        if score >= 0:
            scores.append(score)

    return {
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "n_proofs": len(scores),
        "scores": scores,
    }


def compute_innermost_alignment(results: List[dict], gt_instances: List[dict]) -> dict:
    """Same as LO but for innermost-first strategy."""
    scores = []
    for r, gt in zip(results, gt_instances):
        inst = TRSInstance.from_dict(gt)
        steps = parse_proof_from_output(r["completion"], n_rules=len(inst.rules))
        if not steps:
            continue
        valid_steps = [(e, idx) for e, idx in steps if e is not None and idx >= 0]
        if not valid_steps:
            continue
        score = innermost_alignment_score(valid_steps, inst)
        if score >= 0:
            scores.append(score)

    return {
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "n_proofs": len(scores),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rule frequency analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_rule_frequency(
    results: List[dict],
    gt_instances: List[dict],
) -> dict:
    """
    Count which rule numbers appear in agent proofs vs ground truth proofs.
    Returns both distributions for comparison.
    """
    agent_counts = Counter()
    agent_total = 0
    gt_counts = Counter()
    gt_total = 0

    n_rules = max((len(gt.get("rules", [])) for gt in gt_instances), default=0)

    for r, gt in zip(results, gt_instances):
        nr = len(gt.get("rules", []))
        # Agent's proof
        steps = parse_proof_from_output(r["completion"], n_rules=nr)
        for _, rule_idx in steps:
            if 0 <= rule_idx < nr:
                agent_counts[rule_idx] += 1
                agent_total += 1

        # Ground truth proof
        for step in gt.get("proof", []):
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                gt_idx = step[1]
                if 0 <= gt_idx < nr:
                    gt_counts[gt_idx] += 1
                    gt_total += 1

    agent_freq = {f"RULE {i+1}": agent_counts[i] / agent_total if agent_total > 0 else 0.0
                  for i in range(n_rules) if agent_counts[i] > 0}
    gt_freq = {f"RULE {i+1}": gt_counts[i] / gt_total if gt_total > 0 else 0.0
               for i in range(n_rules) if gt_counts[i] > 0}

    return {
        "agent_frequency": agent_freq,
        "ground_truth_frequency": gt_freq,
        "n_rules": n_rules,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Think block analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_think_metrics(results: List[dict]) -> dict:
    """Compute think-related metrics from completions."""
    think_texts = [extract_think_block(r["completion"]) for r in results]
    present = sum(1 for t in think_texts if t)
    lengths = [len(t.split()) for t in think_texts if t]
    reflections = [has_reflection(t) for t in think_texts]
    ref_counts = [reflection_count(t) for t in think_texts]

    return {
        "think_blocks_found": present,
        "think_blocks_total": len(results),
        "mean_think_length_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "reflection_rate": sum(reflections) / len(reflections) if reflections else 0.0,
        "reflection_count_mean": sum(ref_counts) / len(ref_counts) if ref_counts else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_json(path: str) -> List[dict]:
    """Load eval results JSON (list of {solved, reward, completion, start, normal_form})."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # If it's a dict with a "results" key
    return data.get("results", [data])


def load_ground_truth_jsonl(path: str) -> List[dict]:
    """Load ground truth instances from JSONL."""
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def build_gt_lookup(gt_instances: List[dict]) -> Dict[str, dict]:
    """Build a lookup from start expression string to ground truth instance."""
    lookup = {}
    for gt in gt_instances:
        start_str = gt.get("start", "")
        lookup[start_str] = gt
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(all_metrics: List[dict]) -> str:
    """Generate a markdown report from metrics across phases."""
    lines = []
    lines.append("# TRS-RL Evaluation Report\n")
    lines.append(f"Generated from {len(all_metrics)} phase evaluation(s).\n")

    for m in all_metrics:
        phase = m["phase"]
        lines.append(f"\n## Phase {phase}\n")
        lines.append(f"**Eval file:** `{m.get('eval_json', 'N/A')}`\n")

        # Core metrics table
        lines.append("### Performance Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| solve_rate@1 | {m['solve_rate']:.4f} ({m['solve_rate']*100:.1f}%) |")
        lines.append(f"| solve_rate@3 (est.) | {m['solve_rate_at_3']:.4f} ({m['solve_rate_at_3']*100:.1f}%) |")
        lines.append(f"| mean_reward | {m['mean_reward']:.4f} |")
        lines.append(f"| parse_rate | {m['parse_rate']:.4f} ({m['parse_rate']*100:.1f}%) |")
        lines.append(f"| partial_step_rate | {m['partial_step_rate']:.4f} |")
        lines.append(f"| N samples | {m['n_samples']} |")
        lines.append("")

        # Solve by depth
        lines.append("### Solve Rate by Expression Depth\n")
        lines.append("| Depth | Solved/Total | Rate |")
        lines.append("|-------|-------------|------|")
        for label in DEPTH_LABELS:
            d = m["solve_by_depth"].get(label, {})
            if d.get("total", 0) > 0:
                lines.append(f"| {label} | {d['solved']}/{d['total']} | {d['solve_rate']:.3f} |")
        lines.append("")

        # Solve by steps
        lines.append("### Solve Rate by Proof Length\n")
        lines.append("| Steps | Solved/Total | Rate |")
        lines.append("|-------|-------------|------|")
        for label in STEP_LABELS:
            d = m["solve_by_steps"].get(label, {})
            if d.get("total", 0) > 0:
                lines.append(f"| {label} | {d['solved']}/{d['total']} | {d['solve_rate']:.3f} |")
        lines.append("")

        # Emergence metrics
        lines.append("### Emergence Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| LO alignment (mean) | {m['lo_alignment']['mean']:.4f} |")
        lines.append(f"| Innermost alignment (mean) | {m['innermost_alignment']['mean']:.4f} |")
        lines.append(f"| LO proofs analysed | {m['lo_alignment']['n_proofs']} |")

        think = m["think_metrics"]
        lines.append(f"| Think blocks found | {think['think_blocks_found']}/{think['think_blocks_total']} |")
        lines.append(f"| Mean think length | {think['mean_think_length_words']:.1f} words |")
        lines.append(f"| Reflection rate | {think['reflection_rate']:.4f} |")
        lines.append(f"| Reflection count (mean) | {think['reflection_count_mean']:.2f} |")
        lines.append("")

        # Rule frequency
        rf = m["rule_frequency"]
        if rf["agent_frequency"]:
            lines.append("### Rule Frequency (Agent vs Ground Truth)\n")
            lines.append("| Rule | Agent | Ground Truth |")
            lines.append("|------|-------|-------------|")
            all_rules = sorted(set(list(rf["agent_frequency"].keys()) + list(rf["ground_truth_frequency"].keys())))
            for rule in all_rules:
                af = rf["agent_frequency"].get(rule, 0.0)
                gf = rf["ground_truth_frequency"].get(rule, 0.0)
                lines.append(f"| {rule} | {af:.3f} | {gf:.3f} |")
            lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def analyze_one_phase(
    eval_json_path: str,
    gt_jsonl_path: str,
    phase: int,
) -> dict:
    """Run all metrics for one phase."""
    results = load_eval_json(eval_json_path)
    gt_all = load_ground_truth_jsonl(gt_jsonl_path)
    gt_lookup = build_gt_lookup(gt_all)

    # Match results to ground truth by start expression
    matched_gt = []
    matched_results = []
    for r in results:
        gt = gt_lookup.get(r["start"])
        if gt is not None:
            matched_gt.append(gt)
            matched_results.append(r)
        else:
            # Try to find by parsing and re-stringifying
            matched_gt.append({"rules": [], "proof": [], "metadata": {}})
            matched_results.append(r)

    n = len(results)
    solve_rate = sum(r["solved"] for r in results) / n if n > 0 else 0.0
    mean_reward = sum(r["reward"] for r in results) / n if n > 0 else 0.0

    print(f"\n  Analyzing Phase {phase}: {n} results, {len(gt_all)} ground truth instances")
    print(f"  Matched to GT: {sum(1 for gt in matched_gt if gt.get('rules'))} / {n}")

    metrics = {
        "phase": phase,
        "eval_json": eval_json_path,
        "n_samples": n,
        "solve_rate": solve_rate,
        "mean_reward": mean_reward,
        "solve_rate_at_3": estimate_solve_rate_at_3(results),
        "solve_by_depth": compute_solve_by_depth(results),
        "solve_by_steps": compute_solve_by_steps(results, gt_lookup),
        "parse_rate": compute_parse_rate(matched_results, matched_gt),
        "partial_step_rate": compute_partial_step_rate(matched_results, matched_gt),
        "lo_alignment": compute_lo_alignment(matched_results, matched_gt),
        "innermost_alignment": compute_innermost_alignment(matched_results, matched_gt),
        "rule_frequency": compute_rule_frequency(matched_results, matched_gt),
        "think_metrics": compute_think_metrics(results),
    }

    return metrics


def main():
    p = argparse.ArgumentParser(
        description="Analyze TRS-RL eval results — all Section 9 metrics"
    )
    p.add_argument("--eval-json", nargs="+", required=True,
                   help="Path(s) to eval result JSON file(s)")
    p.add_argument("--ground-truth", nargs="+", required=True,
                   help="Path(s) to ground truth JSONL file(s)")
    p.add_argument("--phase", nargs="+", type=int, required=True,
                   help="Phase number(s) corresponding to each eval/gt pair")
    p.add_argument("--output-md", default=None,
                   help="Save markdown report to file")
    p.add_argument("--output-json", default=None,
                   help="Save metrics as JSON")
    args = p.parse_args()

    if len(args.eval_json) != len(args.ground_truth) or len(args.eval_json) != len(args.phase):
        print("ERROR: --eval-json, --ground-truth, and --phase must have the same number of arguments")
        sys.exit(1)

    all_metrics = []
    for ej, gt, ph in zip(args.eval_json, args.ground_truth, args.phase):
        metrics = analyze_one_phase(ej, gt, ph)
        all_metrics.append(metrics)

        # Print summary to stdout
        print(f"\n{'='*60}")
        print(f"  Phase {ph} Metrics Summary")
        print(f"{'='*60}")
        print(f"  solve_rate@1:       {metrics['solve_rate']:.4f} ({metrics['solve_rate']*100:.1f}%)")
        print(f"  solve_rate@3 (est): {metrics['solve_rate_at_3']:.4f}")
        print(f"  parse_rate:         {metrics['parse_rate']:.4f}")
        print(f"  partial_step_rate:  {metrics['partial_step_rate']:.4f}")
        print(f"  LO alignment:       {metrics['lo_alignment']['mean']:.4f} ({metrics['lo_alignment']['n_proofs']} proofs)")
        print(f"  IN alignment:       {metrics['innermost_alignment']['mean']:.4f}")

        print(f"  ── Solve by depth ──")
        for label in DEPTH_LABELS:
            d = metrics["solve_by_depth"][label]
            if d["total"] > 0:
                print(f"    {label}: {d['solved']}/{d['total']} = {d['solve_rate']:.3f}")

        print(f"  ── Solve by steps ──")
        for label in STEP_LABELS:
            d = metrics["solve_by_steps"][label]
            if d["total"] > 0:
                print(f"    {label}: {d['solved']}/{d['total']} = {d['solve_rate']:.3f}")

        think = metrics["think_metrics"]
        print(f"  ── Think blocks ──")
        print(f"    Found: {think['think_blocks_found']}/{think['think_blocks_total']}")
        print(f"    Reflection rate: {think['reflection_rate']:.4f}")

        rf = metrics["rule_frequency"]
        if rf["agent_frequency"]:
            print(f"  ── Rule frequency (agent / GT) ──")
            all_rules = sorted(set(list(rf["agent_frequency"].keys()) + list(rf["ground_truth_frequency"].keys())))
            for rule in all_rules:
                af = rf["agent_frequency"].get(rule, 0.0)
                gf = rf["ground_truth_frequency"].get(rule, 0.0)
                print(f"    {rule}: {af:.3f} / {gf:.3f}")
        print(f"{'='*60}")

    # Save outputs
    if args.output_md:
        report = generate_report(all_metrics)
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write(report)
        print(f"\nMarkdown report saved to {args.output_md}")

    if args.output_json:
        # Make JSON-serializable (remove raw score lists)
        for m in all_metrics:
            if "scores" in m.get("lo_alignment", {}):
                del m["lo_alignment"]["scores"]
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"JSON metrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
