"""
generator/instance.py

TRSInstance dataclass and generate_instance().

Key insight: we build problems BACKWARDS from the solution.
This guarantees solvability with zero manual annotation.

Algorithm:
  1. Pick a rule system for the domain
  2. Sample a normal form NF (expression where no rule fires)
  3. Build the start expression by reversing n_steps of rewriting:
       - At each reverse step, pick a subterm of current expression
       - Apply a rule BACKWARDS: replace σ(R) with σ(L) for some rule L→R
       - This guarantees start_expr reduces to NF in ≥ n_steps
  4. Return TRSInstance(rules, start, NF, proof_trace)
     -- proof_trace is NEVER shown to the model
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from rewritelang import (
    Expr, Rule, parse_expr, expr_to_str,
    match, substitute, apply_rule_at, all_positions, find_all_matches,
    is_normal_form, get_domain_rules, get_mixed_rules,
    BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES,
)


@dataclass
class TRSInstance:
    rules:       List[Rule]
    start:       Expr
    normal_form: Expr           # ground truth target — never shown to model
    proof:       List[Tuple[Expr, int]]  # (result_after_step, rule_idx_0indexed)
    domain:      str = "boolean"
    metadata:    dict = field(default_factory=dict)  # phase, depth, steps, seed

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for saving eval sets."""
        return {
            "domain":      self.domain,
            "rules":       [{"lhs": expr_to_str(r.lhs), "rhs": expr_to_str(r.rhs), "name": r.name}
                            for r in self.rules],
            "start":       expr_to_str(self.start),
            "normal_form": expr_to_str(self.normal_form),
            "proof":       [(expr_to_str(e), idx) for e, idx in self.proof],
            "metadata":    self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> "TRSInstance":
        rules = [Rule(lhs=parse_expr(r["lhs"]), rhs=parse_expr(r["rhs"]), name=r.get("name",""))
                 for r in d["rules"]]
        start = parse_expr(d["start"])
        nf    = parse_expr(d["normal_form"])
        proof = [(parse_expr(e), idx) for e, idx in d["proof"]]
        return TRSInstance(rules, start, nf, proof, d.get("domain","boolean"), d.get("metadata",{}))


# ---------------------------------------------------------------------------
# Normal form sampling
# ---------------------------------------------------------------------------

def _sample_normal_form(
    rules: List[Rule],
    rng: random.Random,
    max_depth: int = 2,
    domain: str = "boolean",
    max_attempts: int = 500,
) -> Expr:
    """
    Sample a ground expression that is in normal form w.r.t. the rule set.
    Strategy: generate random ground expressions and check is_normal_form.
    For most domains this succeeds quickly — constants are always normal forms.
    """
    # Fast path: single constants are always ground normal forms if rules don't fire on them
    constants = _get_domain_constants(domain)
    rng.shuffle(constants)
    for c in constants:
        e = parse_expr(c)
        if is_normal_form(e, rules):
            return e

    # Slower: try random ground terms of bounded depth
    for _ in range(max_attempts):
        e = _random_ground_expr(rules, rng, depth=rng.randint(0, max_depth), domain=domain)
        if is_normal_form(e, rules):
            return e

    raise RuntimeError(
        f"Could not find a normal form for domain={domain!r} after {max_attempts} attempts. "
        "This likely means all ground terms are reducible — check your rule set."
    )


def _get_domain_constants(domain: str) -> List[str]:
    if domain == "boolean":
        return ["T", "F"]
    elif domain == "arithmetic":
        return ["ZERO"]
    elif domain == "abstract":
        return ["A", "B"]
    else:
        return ["A", "B", "T", "F", "ZERO"]


def _get_domain_symbols(domain: str) -> List[Tuple[str, int]]:
    """Return (symbol_name, arity) pairs for the domain."""
    if domain == "boolean":
        return [("not", 1), ("and", 2), ("or", 2)]
    elif domain == "arithmetic":
        return [("succ", 1), ("add", 2), ("mul", 2), ("pred", 1)]
    elif domain == "abstract":
        return [("f", 1), ("g", 1), ("h", 2)]
    else:
        return [("not", 1), ("and", 2), ("or", 2), ("f", 1)]


def _random_ground_expr(
    rules: List[Rule],
    rng: random.Random,
    depth: int,
    domain: str,
) -> Expr:
    """Generate a random ground (variable-free) expression of given depth."""
    constants = _get_domain_constants(domain)
    symbols = _get_domain_symbols(domain)

    if depth == 0 or not symbols:
        return parse_expr(rng.choice(constants))

    if rng.random() < 0.4:  # bias toward atoms to keep expressions manageable
        return parse_expr(rng.choice(constants))

    sym, arity = rng.choice(symbols)
    children = [_random_ground_expr(rules, rng, depth - 1, domain) for _ in range(arity)]
    return Expr("fun", sym, children)


# ---------------------------------------------------------------------------
# Reverse rewriting — the key algorithmic insight
# ---------------------------------------------------------------------------

def _reverse_rewrite_step(
    expr: Expr,
    rules: List[Rule],
    rng: random.Random,
    domain: str,
) -> Optional[Tuple[Expr, int, list]]:
    """
    Apply ONE reverse rewriting step.
    For rule L → R, we look for occurrences of σ(R) in expr and replace with σ(L).

    This is non-trivial because R may contain variables — we need to find a subexpression
    that LOOKS LIKE R (possibly with some subexpressions filling in variables), then
    build the corresponding σ(L).

    Returns (new_expr, rule_idx, position) or None if no reverse step is possible.
    """
    # Collect all (rule_idx, position, sigma) candidates
    candidates = []

    for rule_idx, rule in enumerate(rules):
        # Find places where rule.rhs matches (in the forward direction)
        # i.e., we look for σ such that σ(rule.rhs) = expr|pos
        # This gives us a "reverse" step: replace σ(rule.rhs) with σ(rule.lhs)
        matches = find_all_matches(expr, rule.rhs)
        for pos, sigma in matches:
            # Verify sigma is ground (all variables bound to ground terms)
            # so the reverse step produces a valid start expression
            if all(v.is_ground() for v in sigma.values()):
                candidates.append((rule_idx, pos, sigma))

    if not candidates:
        return None

    rule_idx, pos, sigma = rng.choice(candidates)
    rule = rules[rule_idx]

    # Build σ(L) — the new (more complex) expression
    new_subexpr = substitute(rule.lhs, sigma)
    new_expr = _replace_at_position(expr, pos, new_subexpr)

    return new_expr, rule_idx, pos


def _replace_at_position(expr: Expr, position: list, replacement: Expr) -> Expr:
    """Replace subtree at position with replacement."""
    if not position:
        return replacement
    new_children = list(expr.children)
    idx = position[0]
    new_children[idx] = _replace_at_position(expr.children[idx], position[1:], replacement)
    return Expr("fun", expr.name, new_children)


def _build_by_reverse_rewriting(
    rules: List[Rule],
    normal_form: Expr,
    n_steps: int,
    rng: random.Random,
    domain: str,
    max_retries: int = 50,
) -> Tuple[Expr, List[Tuple[Expr, int]]]:
    """
    Build a start expression by applying n_steps of reverse rewriting to normal_form.

    Returns (start_expr, forward_proof_trace).
    The forward proof trace is the reversed sequence of reverse steps.

    If we get stuck (no reverse step applicable), we retry from scratch.
    """
    for attempt in range(max_retries):
        current = normal_form.copy()
        reverse_steps = []  # list of (expr_before_reverse_step, rule_idx, position)

        success = True
        for step_i in range(n_steps):
            result = _reverse_rewrite_step(current, rules, rng, domain)
            if result is None:
                success = False
                break
            new_expr, rule_idx, pos = result
            reverse_steps.append((current, rule_idx))  # (what current was before complexifying)
            current = new_expr

        if success and len(reverse_steps) == n_steps:
            start_expr = current
            # Forward proof: reverse the steps
            # reverse_steps[i] = (expr_that_results_from_forward_step_i, rule_idx)
            forward_proof = list(reversed(reverse_steps))
            # forward_proof[i] = (expression AFTER step i+1, rule_idx used)
            return start_expr, forward_proof

    raise RuntimeError(
        f"Could not build a {n_steps}-step instance after {max_retries} attempts. "
        f"Domain={domain!r}, rules={len(rules)}. "
        "Try reducing n_steps or using more rules."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_instance(
    n_rules: int,
    max_depth: int,
    n_steps: int,
    domain: str,
    seed: int,
    domains_for_mixed: Optional[List[str]] = None,
) -> TRSInstance:
    """
    Generate a TRS instance with a KNOWN reduction sequence.

    Args:
        n_rules:           How many rules to include (subset of domain's full set)
        max_depth:         Max depth of the normal form used as seed
        n_steps:           Number of reverse steps = minimum proof length
        domain:            "boolean", "arithmetic", "abstract", or "mixed"
        seed:              RNG seed for reproducibility
        domains_for_mixed: If domain="mixed", which domains to pull rules from

    Returns:
        TRSInstance with (rules, start, normal_form, proof)
        The proof is the GROUND TRUTH and is NEVER shown to the model.
    """
    rng = random.Random(seed)

    # Get rule set
    if domain == "mixed":
        if domains_for_mixed is None:
            raise ValueError("Must specify domains_for_mixed when domain='mixed'")
        rules = get_mixed_rules(domains_for_mixed)
        rules = rules[:n_rules] if n_rules < len(rules) else rules
        nf_domain = domains_for_mixed[0]
    else:
        rules = get_domain_rules(domain, n_rules)
        nf_domain = domain

    # Sample a normal form
    normal_form = _sample_normal_form(rules, rng, max_depth=max_depth, domain=nf_domain)

    # Build start expression by reverse rewriting
    start, proof = _build_by_reverse_rewriting(
        rules, normal_form, n_steps, rng, domain=nf_domain
    )

    return TRSInstance(
        rules=rules,
        start=start,
        normal_form=normal_form,
        proof=proof,
        domain=domain,
        metadata={
            "seed":      seed,
            "n_rules":   n_rules,
            "max_depth": max_depth,
            "n_steps":   n_steps,
            "domain":    domain,
            "start_depth": start.depth(),
            "start_size":  start.size(),
        },
    )
