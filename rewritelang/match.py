"""
rewritelang/match.py

match(), substitute(), apply_rule_at(), all_positions()

Axiom 2 — Matching:
  A rule L → R matches E at position p iff
  ∃ substitution σ : {variables} → {expressions} such that σ(L) = E|p

Axiom 3 — Rewriting:
  E' = E with subtree E|p replaced by σ(R)
"""

from __future__ import annotations
from typing import Optional, List, Iterator
from .grammar import Expr


# ---------------------------------------------------------------------------
# Positions
# A position is a list of child indices, e.g. [] = root, [0] = first child,
# [1, 0] = second child's first child.
# ---------------------------------------------------------------------------

def all_positions(expr: Expr) -> List[List[int]]:
    """
    Return all positions in the expression tree, including the root ([]).
    Order: root first, then children left-to-right depth-first.
    """
    result = [[]]
    for i, child in enumerate(expr.children):
        for sub_pos in all_positions(child):
            result.append([i] + sub_pos)
    return result


def get_subexpr(expr: Expr, position: List[int]) -> Expr:
    """Return the subtree at a given position."""
    current = expr
    for idx in position:
        if current.kind != "fun" or idx >= len(current.children):
            raise IndexError(f"Position {position} invalid for expression {expr}")
        current = current.children[idx]
    return current


def replace_at(expr: Expr, position: List[int], replacement: Expr) -> Expr:
    """Return a new expression with the subtree at `position` replaced."""
    if not position:
        return replacement
    if expr.kind != "fun":
        raise IndexError(f"Cannot navigate into non-fun node at position {position}")
    idx = position[0]
    if idx >= len(expr.children):
        raise IndexError(f"Child index {idx} out of range for {expr}")
    new_children = list(expr.children)
    new_children[idx] = replace_at(expr.children[idx], position[1:], replacement)
    return Expr("fun", expr.name, new_children)


# ---------------------------------------------------------------------------
# Matching and substitution
# ---------------------------------------------------------------------------

def match(pattern: Expr, expr: Expr) -> Optional[dict]:
    """
    Try to match `pattern` against `expr`.
    Returns a substitution dict {var_name: Expr} or None if no match.

    Variables in pattern are wildcards — they match any subexpression.
    Constants must match exactly.
    Function symbols must match name and arity.
    """
    if pattern.kind == "var":
        # Variable matches anything
        return {pattern.name: expr}

    if pattern.kind == "const":
        if expr.kind == "const" and expr.name == pattern.name:
            return {}
        return None

    # pattern.kind == "fun"
    if expr.kind != "fun":
        return None
    if expr.name != pattern.name:
        return None
    if len(pattern.children) != len(expr.children):
        return None

    sigma: dict = {}
    for p_child, e_child in zip(pattern.children, expr.children):
        sub = match(p_child, e_child)
        if sub is None:
            return None
        # Check consistency: same variable must match same subexpression
        for var, val in sub.items():
            if var in sigma:
                if sigma[var] != val:
                    return None  # inconsistent binding
            else:
                sigma[var] = val
    return sigma


def substitute(expr: Expr, sigma: dict) -> Expr:
    """
    Apply substitution sigma to expr.
    Replaces every variable v with sigma[v] if v is in sigma,
    leaving unbound variables unchanged.
    """
    if expr.kind == "var":
        return sigma.get(expr.name, expr)
    if expr.kind == "const":
        return expr
    # fun
    new_children = [substitute(c, sigma) for c in expr.children]
    return Expr("fun", expr.name, new_children)


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def apply_rule_at(
    expr: Expr,
    lhs: Expr,
    rhs: Expr,
    position: List[int]
) -> Optional[Expr]:
    """
    Attempt to apply rule lhs → rhs at the given position in expr.
    Returns the rewritten expression, or None if the rule doesn't match.
    """
    subexpr = get_subexpr(expr, position)
    sigma = match(lhs, subexpr)
    if sigma is None:
        return None
    rewritten = substitute(rhs, sigma)
    return replace_at(expr, position, rewritten)


def find_all_matches(
    expr: Expr,
    lhs: Expr
) -> List[tuple]:
    """
    Return all (position, substitution) pairs where lhs matches in expr.
    """
    results = []
    for pos in all_positions(expr):
        subexpr = get_subexpr(expr, pos)
        sigma = match(lhs, subexpr)
        if sigma is not None:
            results.append((pos, sigma))
    return results
