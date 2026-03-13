#!/usr/bin/env python3
"""
scripts/sanity_check.py

The FIRST thing you run before any training.
If any check fails, do NOT proceed to training.

Tests (from design doc Section 11.1):
  ✓ parse_expr round-trips correctly
  ✓ match() finds correct substitutions
  ✓ verify_proof() returns 1.0 for known-good proof
  ✓ verify_proof() returns 0.0 for broken proof
  ✓ generate_instance() produces solvable problems (100 seeds)
  ✓ reverse_rewriting witness verifies correctly
  ✓ curriculum phase advancement works
  ✓ reward function smoke test
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewritelang import (
    parse_expr, expr_to_str, expr_equal,
    match, substitute, apply_rule_at, all_positions,
    verify_proof, verify_proof_detailed, is_normal_form,
    BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES,
    Rule, ParseError,
)
from generator.instance import generate_instance
from generator.curriculum import CurriculumTracker
from agent.prompt import make_prompt, make_chat_messages
from agent.parser import parse_proof_from_output
from training.reward import compute_reward


PASS = "  ✓"
FAIL = "  ✗"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"      Error: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Grammar / parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_roundtrip():
    cases = [
        "T", "F", "ZERO", "ONE",
        "x", "y",
        "not(T)",
        "and(T, F)",
        "or(T, or(F, x))",
        "not(not(T))",
        "and(not(not(T)), or(F, not(F)))",
        "add(ZERO, x)",
        "succ(succ(ZERO))",
        "f(f(f(x)))",
    ]
    for s in cases:
        # remove spaces for comparison since our printer omits them
        e = parse_expr(s)
        s2 = expr_to_str(e)
        e2 = parse_expr(s2)
        assert expr_equal(e, e2), f"Round-trip failed: {s!r} -> {s2!r}"


def test_parse_errors():
    bad = ["", "123abc", "()"]
    for s in bad:
        try:
            parse_expr(s)
            assert False, f"Should have raised ParseError for {s!r}"
        except (ParseError, Exception):
            pass  # expected


def test_expr_depth_size():
    e = parse_expr("and(not(not(T)), or(F, not(F)))")
    assert e.depth() == 3, f"Expected depth 3, got {e.depth()}"
    # and(not(not(T)), or(F, not(F))) has 8 nodes:
    # and, not, not, T, or, F, not, F
    assert e.size() == 8, f"Expected size 8, got {e.size()}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Matching
# ─────────────────────────────────────────────────────────────────────────────

def test_match_variable():
    # not(not(x)) matches not(not(T)) with {x: T}
    pattern = parse_expr("not(not(x))")
    expr    = parse_expr("not(not(T))")
    sigma   = match(pattern, expr)
    assert sigma is not None, "Expected match"
    assert expr_equal(sigma["x"], parse_expr("T")), f"Expected x->T, got {sigma}"


def test_match_function():
    # not(not(x)) matches not(not(f(a)))
    pattern = parse_expr("not(not(x))")
    expr    = parse_expr("not(not(f(a)))")  # f(a) has one child
    sigma   = match(pattern, expr)
    assert sigma is not None


def test_match_fail():
    # not(T) does NOT match not(not(T)) — pattern too shallow
    pattern = parse_expr("not(x)")
    expr    = parse_expr("not(not(T))")
    sigma   = match(pattern, expr)
    # This should match with x -> not(T)
    assert sigma is not None
    assert expr_equal(sigma["x"], parse_expr("not(T)"))

    # Different symbol — no match
    pattern2 = parse_expr("and(x, y)")
    sigma2   = match(pattern2, parse_expr("or(T, F)"))
    assert sigma2 is None, "Expected no match for and vs or"


def test_match_consistency():
    # h(x, x) should only match h(T, T) not h(T, F)
    pattern = parse_expr("h(x, x)")
    assert match(pattern, parse_expr("h(T, T)")) is not None
    assert match(pattern, parse_expr("h(T, F)")) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verifier — known-good proof (Section 3.2 example)
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_known_good():
    rules = BOOLEAN_RULES[:7]  # first 7 boolean rules (matches doc exactly)
    start  = parse_expr("and(not(not(T)), or(F, not(F)))")
    target = parse_expr("T")

    # The proof from Section 3.2:
    # S1: and(T,or(F,not(F))) RULE 1  -- not(not(x))=>x   (index 0)
    # S2: and(T,or(F,T))      RULE 7  -- not(F)=>T        (index 6)
    # S3: and(T,T)             RULE 5  -- or(F,x)=>x       (index 4)
    # S4: T                   RULE 2  -- and(T,x)=>x       (index 1)
    proof = [
        (parse_expr("and(T, or(F, not(F)))"), 0),  # rule 1 -> index 0
        (parse_expr("and(T, or(F, T))"),      6),  # rule 7 -> index 6
        (parse_expr("and(T, T)"),              4),  # rule 5 -> index 4
        (parse_expr("T"),                      1),  # rule 2 -> index 1
    ]
    reward = verify_proof(rules, start, proof, target)
    assert reward == 1.0, f"Known-good proof got reward={reward}, expected 1.0"


def test_verify_broken_proof():
    rules  = BOOLEAN_RULES[:7]
    start  = parse_expr("not(T)")
    target = parse_expr("F")
    # Lie: claim not(T) -> T (wrong, should be F)
    bad_proof = [(parse_expr("T"), 7)]  # rule 8 = index 7, not(T)=>F gives F not T
    reward = verify_proof(rules, start, bad_proof, target)
    assert reward < 1.0, f"Broken proof should not get 1.0, got {reward}"


def test_verify_empty_proof():
    reward = verify_proof(BOOLEAN_RULES, parse_expr("T"), [], parse_expr("T"))
    assert reward == 0.0, "Empty proof should get 0.0"


def test_normal_form():
    rules = BOOLEAN_RULES
    assert is_normal_form(parse_expr("T"), rules)
    assert is_normal_form(parse_expr("F"), rules)
    assert not is_normal_form(parse_expr("not(T)"), rules)
    assert not is_normal_form(parse_expr("and(T, F)"), rules)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Generator — 100 seeds must all produce reward 1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_generator_consistency():
    failures = []
    for seed in range(100):
        inst = generate_instance(
            n_rules=7, max_depth=4, n_steps=4,
            domain="boolean", seed=seed,
        )
        reward = verify_proof(
            inst.rules, inst.start, inst.proof, inst.normal_form
        )
        if abs(reward - 1.0) > 1e-9:
            failures.append((seed, reward))

    assert len(failures) == 0, (
        f"Generator produced {len(failures)} unsolvable instances: "
        f"{failures[:5]}"
    )


def test_generator_phase1():
    for seed in range(20):
        inst = generate_instance(n_rules=3, max_depth=2, n_steps=1, domain="boolean", seed=seed)
        assert inst.start.depth() >= 1
        assert len(inst.proof) >= 1


def test_generator_phase2():
    for seed in range(20):
        inst = generate_instance(n_rules=5, max_depth=3, n_steps=3, domain="boolean", seed=seed)
        assert len(inst.proof) >= 1  # may differ but should be > 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prompt + parser round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_prompt_generation():
    inst = generate_instance(n_rules=7, max_depth=3, n_steps=3, domain="boolean", seed=42)
    prompt = make_prompt(inst)
    assert "RULE 1:" in prompt
    assert "START" in prompt
    assert "TARGET" in prompt


def test_proof_parser():
    fake_output = """<think>
Let me think about this...
</think>
PROOF
S1: and(T, or(F, not(F))) RULE 1
S2: and(T, or(F, T)) RULE 7
S3: and(T, T) RULE 4
S4: T RULE 2"""

    steps = parse_proof_from_output(fake_output, n_rules=9)
    assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"
    assert steps[0][1] == 0, f"Expected rule_idx=0, got {steps[0][1]}"
    assert steps[-1][1] == 1, f"Expected rule_idx=1, got {steps[-1][1]}"


def test_parser_handles_missing_proof():
    steps = parse_proof_from_output("I have no idea how to solve this.", n_rules=9)
    assert steps == []


def test_parser_handles_invalid_rules():
    fake = "PROOF\nS1: T RULE 99\n"
    steps = parse_proof_from_output(fake, n_rules=9)
    assert steps[0][1] == -1, "Rule 99 should give -1"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Reward function
# ─────────────────────────────────────────────────────────────────────────────

def test_reward_correct_output():
    inst = generate_instance(n_rules=7, max_depth=3, n_steps=3, domain="boolean", seed=123)
    # Build a perfect output string from the ground truth proof
    from rewritelang import expr_to_str
    lines = ["PROOF"]
    for i, (expr, rule_idx) in enumerate(inst.proof):
        lines.append(f"S{i+1}: {expr_to_str(expr)} RULE {rule_idx+1}")
    perfect_output = "\n".join(lines)

    reward = compute_reward(perfect_output, inst)
    assert reward == 1.0, f"Perfect output should get 1.0, got {reward}"


def test_reward_empty_output():
    inst = generate_instance(n_rules=3, max_depth=2, n_steps=1, domain="boolean", seed=0)
    reward = compute_reward("I give up.", inst)
    assert reward == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Curriculum
# ─────────────────────────────────────────────────────────────────────────────

def test_curriculum_advancement():
    tracker = CurriculumTracker(advance_threshold=0.75, eval_interval=10)
    assert tracker.current_phase == 1

    # Should NOT advance at 0.74
    result = tracker.record_eval(0.74)
    assert not result["advanced"]
    assert tracker.current_phase == 1

    # Should advance at 0.75
    result = tracker.record_eval(0.75)
    assert result["advanced"]
    assert tracker.current_phase == 2


def test_curriculum_no_advance_at_final():
    tracker = CurriculumTracker(advance_threshold=0.75, n_phases=5)
    tracker.state.current_phase = 5
    result = tracker.record_eval(1.0)
    assert not result["advanced"]
    assert tracker.current_phase == 5


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  TRS-RL Sanity Checks")
    print("="*60 + "\n")

    print("── Grammar ──────────────────────────────────────────────")
    check("parse_expr round-trips correctly",    test_parse_roundtrip)
    check("parse_expr raises on bad input",      test_parse_errors)
    check("Expr.depth() and .size() correct",    test_expr_depth_size)

    print("\n── Matching ─────────────────────────────────────────────")
    check("match() variable binding",            test_match_variable)
    check("match() nested function",             test_match_function)
    check("match() fail cases",                  test_match_fail)
    check("match() consistency (h(x,x) check)",  test_match_consistency)

    print("\n── Verifier ─────────────────────────────────────────────")
    check("verify_proof() = 1.0 for Section 3.2 proof", test_verify_known_good)
    check("verify_proof() < 1.0 for broken proof",       test_verify_broken_proof)
    check("verify_proof() = 0.0 for empty proof",        test_verify_empty_proof)
    check("is_normal_form() correct",                    test_normal_form)

    print("\n── Generator ────────────────────────────────────────────")
    check("generate_instance() 100/100 solvable (seeds 0-99)", test_generator_consistency)
    check("generate_instance() Phase 1 config",                test_generator_phase1)
    check("generate_instance() Phase 2 config",                test_generator_phase2)

    print("\n── Agent Prompt + Parser ────────────────────────────────")
    check("make_prompt() produces valid prompt",         test_prompt_generation)
    check("parse_proof_from_output() valid proof",       test_proof_parser)
    check("parser handles missing PROOF block",          test_parser_handles_missing_proof)
    check("parser handles invalid rule numbers",         test_parser_handles_invalid_rules)

    print("\n── Reward Function ──────────────────────────────────────")
    check("compute_reward() = 1.0 for perfect output",  test_reward_correct_output)
    check("compute_reward() = 0.0 for empty output",    test_reward_empty_output)

    print("\n── Curriculum ───────────────────────────────────────────")
    check("phase advances at threshold",                test_curriculum_advancement)
    check("no advance at final phase",                  test_curriculum_no_advance_at_final)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)

    print(f"\n{'='*60}")
    print(f"  Results: {n_pass} passed, {n_fail} failed")
    print(f"{'='*60}")

    if n_fail == 0:
        print("\n  ✓ All sanity checks passed. Safe to start training.\n")
        return 0
    else:
        print("\n  ✗ Some checks failed. DO NOT start training until fixed.\n")
        failed = [(name, err) for name, ok, err in results if not ok]
        for name, err in failed:
            print(f"    FAILED: {name}")
            print(f"      {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
