"""
training/emergence.py

Emergence analysis — the scientific core of Section 9.2.
These metrics tell you WHETHER the model is discovering reduction strategies,
not just whether it's getting the right answer.

Metrics:
  - LO alignment score (leftmost-outermost strategy discovery)
  - Innermost-first alignment score
  - Reflection detection (backtracking in think traces)
  - Rule frequency analysis
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional
from collections import Counter

from rewritelang import Expr, Rule, all_positions, find_all_matches, apply_rule_at
from generator.instance import TRSInstance


# ---------------------------------------------------------------------------
# Leftmost-Outermost (LO) strategy
# ---------------------------------------------------------------------------

def compute_lo_choice(
    expr: Expr,
    rules: List[Rule],
) -> Optional[Tuple[int, list]]:
    """
    Compute the Leftmost-Outermost reduction choice for expr.
    LO = apply the rule at the leftmost, outermost (shallowest then leftmost) position.

    Returns (rule_idx, position) or None if in normal form.
    """
    # Collect all (rule_idx, position) where rule fires
    all_applicable = []
    for rule_idx, rule in enumerate(rules):
        for pos in all_positions(expr):
            from rewritelang.match import match, get_subexpr
            sub = get_subexpr(expr, pos)
            sigma = match(rule.lhs, sub)
            if sigma is not None:
                all_applicable.append((rule_idx, pos))

    if not all_applicable:
        return None

    # LO order: shallowest first (len(pos)), then leftmost within same depth
    def lo_key(item):
        _, pos = item
        # Position as tuple for lexicographic comparison (leftmost = smallest first element)
        return (len(pos), pos if pos else [])

    all_applicable.sort(key=lambda x: (len(x[1]), x[1]))
    return all_applicable[0]


def lo_alignment_score(
    proof_steps: List[Tuple[Expr, int]],  # (result_expr, rule_idx)
    instance: TRSInstance,
) -> float:
    """
    Fraction of agent's step choices that match what LO reduction would choose.
    If this increases over training → agent is discovering LO spontaneously.

    Returns float in [0, 1], or -1.0 if proof is empty.
    """
    if not proof_steps:
        return -1.0

    current = instance.start
    matches = 0
    total   = 0

    for (result_expr, agent_rule_idx) in proof_steps:
        lo_choice = compute_lo_choice(current, instance.rules)
        if lo_choice is None:
            break  # already in normal form

        lo_rule_idx, lo_pos = lo_choice
        total += 1

        if agent_rule_idx == lo_rule_idx:
            matches += 1

        # Advance current (use agent's claimed result, since that's what the model did)
        current = result_expr

    return matches / total if total > 0 else -1.0


# ---------------------------------------------------------------------------
# Innermost-First (IN) strategy
# ---------------------------------------------------------------------------

def compute_innermost_choice(
    expr: Expr,
    rules: List[Rule],
) -> Optional[Tuple[int, list]]:
    """
    Compute the Innermost reduction choice: deepest applicable position first.
    """
    all_applicable = []
    for rule_idx, rule in enumerate(rules):
        for pos in all_positions(expr):
            from rewritelang.match import match, get_subexpr
            sub = get_subexpr(expr, pos)
            sigma = match(rule.lhs, sub)
            if sigma is not None:
                all_applicable.append((rule_idx, pos))

    if not all_applicable:
        return None

    # Innermost = deepest first
    all_applicable.sort(key=lambda x: (-len(x[1]), x[1]))
    return all_applicable[0]


def innermost_alignment_score(
    proof_steps: List[Tuple[Expr, int]],
    instance: TRSInstance,
) -> float:
    """Same structure as lo_alignment_score but for innermost-first strategy."""
    if not proof_steps:
        return -1.0

    current = instance.start
    matches = 0
    total   = 0

    for (result_expr, agent_rule_idx) in proof_steps:
        in_choice = compute_innermost_choice(current, instance.rules)
        if in_choice is None:
            break

        in_rule_idx, _ = in_choice
        total += 1
        if agent_rule_idx == in_rule_idx:
            matches += 1

        current = result_expr

    return matches / total if total > 0 else -1.0


# ---------------------------------------------------------------------------
# Reflection detection
# ---------------------------------------------------------------------------

REFLECTION_PATTERNS = [
    r"\bwrong\b",
    r"\bdoesn'?t work\b",
    r"\btry instead\b",
    r"\btry again\b",
    r"\bwait\b",
    r"\bactually\b",
    r"\blet me reconsider\b",
    r"\bthat'?s not right\b",
    r"\bstep back\b",
    r"\bdead end\b",
    r"\bbacktrack\b",
    r"\binstead\b",
    r"\bno,?\s+actually\b",
]
_REFLECTION_RE = re.compile("|".join(REFLECTION_PATTERNS), re.IGNORECASE)


def has_reflection(think_text: str) -> bool:
    """
    Does the model's <think> block contain backtracking/self-correction behavior?
    Proxy for whether the model is doing search vs. direct reduction.
    """
    return bool(_REFLECTION_RE.search(think_text))


def reflection_count(think_text: str) -> int:
    """Count number of reflection markers in think text."""
    return len(_REFLECTION_RE.findall(think_text))


# ---------------------------------------------------------------------------
# Rule frequency analysis
# ---------------------------------------------------------------------------

def rule_frequency(
    proofs: List[List[Tuple[Expr, int]]],
    n_rules: int,
) -> List[float]:
    """
    Given a batch of proofs, return normalised frequency of each rule.
    If agent learns to prefer rules that unlock more reductions, this
    will match the frequency in optimal proofs.
    """
    counts = Counter()
    total  = 0
    for proof in proofs:
        for _, rule_idx in proof:
            if 0 <= rule_idx < n_rules:
                counts[rule_idx] += 1
                total += 1

    if total == 0:
        return [0.0] * n_rules

    return [counts[i] / total for i in range(n_rules)]


# ---------------------------------------------------------------------------
# Aggregated emergence report
# ---------------------------------------------------------------------------

def emergence_report(
    outputs: List[str],
    parsed_proofs: List[List[Tuple[Expr, int]]],
    instances: List[TRSInstance],
    think_texts: List[str],
) -> dict:
    """
    Compute all emergence metrics for a batch. Called after each eval.
    """
    lo_scores       = []
    in_scores       = []
    reflection_flags= []
    reflection_cnts = []

    for proof, instance, think in zip(parsed_proofs, instances, think_texts):
        if proof:
            # Filter out steps with None expressions (failed parses)
            valid_proof = [(e, idx) for e, idx in proof if e is not None and idx >= 0]
            if not valid_proof:
                continue
            lo = lo_alignment_score(valid_proof, instance)
            inn= innermost_alignment_score(valid_proof, instance)
            if lo  >= 0: lo_scores.append(lo)
            if inn >= 0: in_scores.append(inn)

        reflection_flags.append(has_reflection(think))
        reflection_cnts.append(reflection_count(think))

    n_rules = max((len(i.rules) for i in instances), default=0)
    all_proofs = [p for p in parsed_proofs if p]

    return {
        "lo_alignment_mean":         sum(lo_scores) / len(lo_scores) if lo_scores else 0.0,
        "innermost_alignment_mean":  sum(in_scores) / len(in_scores) if in_scores else 0.0,
        "reflection_rate":           sum(reflection_flags) / len(reflection_flags) if reflection_flags else 0.0,
        "reflection_count_mean":     sum(reflection_cnts) / len(reflection_cnts) if reflection_cnts else 0.0,
        "rule_frequency":            rule_frequency(all_proofs, n_rules),
        "n_proofs_analysed":         len(all_proofs),
    }
