"""
rewritelang — the formal TRS engine.

Public API:
  from rewritelang import Expr, Rule, parse_expr, expr_to_str, expr_equal
  from rewritelang import match, substitute, apply_rule_at, all_positions
  from rewritelang import verify_proof, verify_proof_detailed, is_normal_form
  from rewritelang import get_domain_rules, BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES
"""

from .grammar import Expr, parse_expr, expr_to_str, expr_equal, ParseError
from .match import match, substitute, apply_rule_at, all_positions, find_all_matches
from .verifier import Rule, verify_proof, verify_proof_detailed, is_normal_form
from .domains import (
    BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES,
    DOMAIN_RULES, get_domain_rules, get_mixed_rules,
)

__all__ = [
    # grammar
    "Expr", "parse_expr", "expr_to_str", "expr_equal", "ParseError",
    # match
    "match", "substitute", "apply_rule_at", "all_positions", "find_all_matches",
    # verifier
    "Rule", "verify_proof", "verify_proof_detailed", "is_normal_form",
    # domains
    "BOOLEAN_RULES", "ARITHMETIC_RULES", "ABSTRACT_RULES",
    "DOMAIN_RULES", "get_domain_rules", "get_mixed_rules",
]
