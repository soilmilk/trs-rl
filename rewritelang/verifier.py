"""
rewritelang/verifier.py

verify_proof(), is_normal_form()

Axiom 4 — Normal Form:
  E is in normal form iff no rule applies anywhere in E.

Reward formula (from design doc):
  R_total = 0.2 × (valid_steps / total_steps) + 0.8 × R_final
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .grammar import Expr, expr_equal
from .match import all_positions, apply_rule_at, find_all_matches


@dataclass
class Rule:
    lhs: Expr
    rhs: Expr
    name: str = ""          # optional human-readable label, e.g. "not_not_elim"

    def __repr__(self) -> str:
        from .grammar import expr_to_str
        label = f"[{self.name}] " if self.name else ""
        return f"{label}{expr_to_str(self.lhs)} => {expr_to_str(self.rhs)}"


def is_normal_form(expr: Expr, rules: List[Rule]) -> bool:
    """
    True if no rule in `rules` applies anywhere in `expr`.
    This is the termination check (Axiom 4).
    """
    for rule in rules:
        if find_all_matches(expr, rule.lhs):
            return False
    return True


def verify_proof(
    rules: List[Rule],
    start: Expr,
    proof_steps: List[Tuple[Expr, int]],   # (result_expr_after_step, rule_idx 0-indexed)
    target: Expr,
) -> float:
    """
    Verify a proof trace and return shaped reward in [0.0, 1.0].

    proof_steps: list of (claimed_result_expression, rule_index)
      - rule_index is 0-indexed (rule 1 in the DSL = index 0 here)
      - claimed_result is what the model says the expression looks like AFTER applying the rule

    Verification strategy per step:
      We don't know which position the model applied the rule at,
      so we try ALL positions and accept if any produces the claimed result.
      This is position-agnostic and generous — the model only needs to get
      the expression right, not specify exactly where.

   
    """
    if not proof_steps:
        return 0.0

    current = start
    valid_steps = 0

    for step_idx, (claimed_result, rule_idx) in enumerate(proof_steps):
        # Guard: invalid rule index
        if rule_idx < 0 or rule_idx >= len(rules):
            break

        # Guard: malformed claimed_result (None means parser failed)
        if claimed_result is None:
            break

        rule = rules[rule_idx]
        found = False

        for position in all_positions(current):
            result = apply_rule_at(current, rule.lhs, rule.rhs, position)
            if result is not None and expr_equal(result, claimed_result):
                found = True
                break

        if found:
            valid_steps += 1
            current = claimed_result
        else:
            # First invalid step — stop counting
            # (We don't continue past an invalid step because the state is undefined)
            break

    final_correct = expr_equal(current, target)
    step_reward = valid_steps / len(proof_steps)
    if final_correct:
        final_reward = 1.0
    else:
        d = sum(1 for rule in rules if find_all_matches(current, rule.lhs))
    final_reward = 1.0 / (1.0 + d)

    return 0.4 * step_reward + 0.6 * final_reward


def verify_proof_detailed(
    rules: List[Rule],
    start: Expr,
    proof_steps: List[Tuple[Expr, int]],
    target: Expr,
) -> dict:
    """
    Same as verify_proof but returns a detailed breakdown for debugging/logging.
    """
    if not proof_steps:
        return {
            "reward": 0.0,
            "valid_steps": 0,
            "total_steps": 0,
            "final_correct": False,
            "failure_reason": "empty proof",
            "step_details": [],
        }

    current = start
    valid_steps = 0
    step_details = []
    failure_reason = None

    for step_idx, (claimed_result, rule_idx) in enumerate(proof_steps):
        detail = {
            "step": step_idx + 1,
            "rule_idx": rule_idx,
            "claimed": claimed_result,
            "valid": False,
            "failure": None,
        }

        if rule_idx < 0 or rule_idx >= len(rules):
            detail["failure"] = f"invalid rule index {rule_idx}"
            step_details.append(detail)
            failure_reason = detail["failure"]
            break

        if claimed_result is None:
            detail["failure"] = "malformed expression (parse failed)"
            step_details.append(detail)
            failure_reason = detail["failure"]
            break

        rule = rules[rule_idx]
        found = False
        for position in all_positions(current):
            result = apply_rule_at(current, rule.lhs, rule.rhs, position)
            if result is not None and expr_equal(result, claimed_result):
                found = True
                break

        if found:
            valid_steps += 1
            current = claimed_result
            detail["valid"] = True
        else:
            detail["failure"] = (
                f"rule {rule_idx} ({rule}) cannot produce {claimed_result} from {current}"
            )
            step_details.append(detail)
            failure_reason = detail["failure"]
            break

        step_details.append(detail)

    final_correct = expr_equal(current, target)
    step_reward = valid_steps / len(proof_steps) if proof_steps else 0.0
    
    if final_correct:
        final_reward = 0.0
    else :
        # Penalize based on how many rules could still apply to the final expression
        d = sum(1 for rule in rules if find_all_matches(current, rule.lhs))
        final_reward = 1.0 / (1.0 + d) 

    total_reward = 0.4 * step_reward + 0.6 * final_reward

    return {
        "reward": total_reward,
        "valid_steps": valid_steps,
        "total_steps": len(proof_steps),
        "final_correct": final_correct,
        "final_expr": current,
        "failure_reason": failure_reason,
        "step_details": step_details,
    }
