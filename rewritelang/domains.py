"""
rewritelang/domains.py

Built-in domain rule sets: boolean, arithmetic, abstract.
Each domain is a list of Rule objects.

Start with boolean (Phase 1-3), add arithmetic (Phase 4), abstract (Phase 5).
"""

from __future__ import annotations
from typing import Optional
from .grammar import parse_expr
from .verifier import Rule


def _rule(lhs_str: str, rhs_str: str, name: str = "") -> Rule:
    return Rule(lhs=parse_expr(lhs_str), rhs=parse_expr(rhs_str), name=name)


# ---------------------------------------------------------------------------
# Boolean domain (intuitionistic propositional logic reduction rules)
# ---------------------------------------------------------------------------
BOOLEAN_RULES: list[Rule] = [
    # Order matches design doc Section 3.2 exactly (so RULE 1-7 in the example work)
    _rule("not(not(x))", "x",         name="not_not_elim"),       # rule 1
    _rule("and(T, x)",   "x",         name="and_true_left"),       # rule 2
    _rule("and(F, x)",   "F",         name="and_false_left"),      # rule 3
    _rule("or(T, x)",    "T",         name="or_true_left"),        # rule 4
    _rule("or(F, x)",    "x",         name="or_false_left"),       # rule 5
    _rule("not(T)",      "F",         name="not_true"),            # rule 6
    _rule("not(F)",      "T",         name="not_false"),           # rule 7
    _rule("and(x, T)",   "x",         name="and_true_right"),      # rule 8
    _rule("and(x, F)",   "F",         name="and_false_right"),     # rule 9
]

# ---------------------------------------------------------------------------
# Arithmetic domain (Peano-style natural numbers)
# ---------------------------------------------------------------------------
ARITHMETIC_RULES: list[Rule] = [
    _rule("add(ZERO, x)",      "x",                name="add_zero_left"),   # rule 1
    _rule("add(succ(x), y)",   "succ(add(x, y))",  name="add_succ"),        # rule 2
    _rule("mul(ZERO, x)",      "ZERO",             name="mul_zero"),        # rule 3
    _rule("mul(succ(x), y)",   "add(y, mul(x, y))",name="mul_succ"),        # rule 4
    _rule("pred(succ(x))",     "x",                name="pred_succ"),       # rule 5
]

# ---------------------------------------------------------------------------
# Abstract domain (lambda-calculus flavored combinators)
# ---------------------------------------------------------------------------
ABSTRACT_RULES: list[Rule] = [
    _rule("f(f(x))",    "x",            name="f_involution"),     # rule 1
    _rule("g(f(x))",    "f(g(x))",      name="g_f_commute"),      # rule 2
    _rule("h(x, x)",    "x",            name="h_diagonal"),       # rule 3
    _rule("f(h(x, y))", "h(f(x), f(y))",name="f_h_distribute"),  # rule 4
]

# ---------------------------------------------------------------------------
# Registry: look up rules by domain name and count
# ---------------------------------------------------------------------------
DOMAIN_RULES: dict[str, list[Rule]] = {
    "boolean":    BOOLEAN_RULES,
    "arithmetic": ARITHMETIC_RULES,
    "abstract":   ABSTRACT_RULES,
}


def get_domain_rules(domain: str, n_rules: Optional[int] = None) -> list[Rule]:
    """
    Return rules for a domain, optionally truncated to n_rules.
    If n_rules is None, return all rules for the domain.
    """
    if domain not in DOMAIN_RULES:
        raise ValueError(f"Unknown domain: {domain!r}. Choose from {list(DOMAIN_RULES)}")
    rules = DOMAIN_RULES[domain]
    if n_rules is not None:
        if n_rules > len(rules):
            raise ValueError(
                f"Domain {domain!r} only has {len(rules)} rules, requested {n_rules}"
            )
        return rules[:n_rules]
    return rules


def get_mixed_rules(domains: list[str]) -> list[Rule]:
    """Return concatenated rules from multiple domains (for Phase 5 multi-domain)."""
    result = []
    for d in domains:
        result.extend(DOMAIN_RULES[d])
    return result
