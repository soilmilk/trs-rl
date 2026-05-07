# TRS-RL: Term Rewriting via Reinforcement Learning

Code accompanying the paper *TRS-RL: Learning Symbolic Term Rewriting from First Principles via Reinforcement Learning*.

A small language model (Qwen3.5-2B + LoRA) is trained with GRPO on a five-phase curriculum of term rewriting problems. The reward is computed by a deterministic verifier; no expert demonstrations, proof traces, or human annotations are used at training time.

---

## Anonymity notice (review period)

This is an anonymized snapshot for double-blind review. Operational scaffolding (AWS launch scripts, Dockerfile, original setup scripts) has been removed. A non-anonymized release with full deployment tooling will accompany the camera-ready version.

---

## Quick start (single H100, local)

```bash
# 1. Install
git clone <this repo> trs-rl
cd trs-rl
pip install -e ".[dev]"

# 2. Sanity check (must pass before training)
python3 scripts/sanity_check.py

# 3. Generate the held-out evaluation sets
python3 data/generate_eval.py --seed 42

# 4. Generate the training instance pools
python3 data/generate_train.py

# 5. Train Phase 1 (proof of life — should reach >75% solve rate in <250 steps)
python3 scripts/train.py \
    --model Qwen/Qwen3.5-2B-Instruct \
    --start-phase 1 \
    --max-steps 1500 \
    --max-tokens 1024 \
    --temperature 0.7 \
    --kl-coeff 0.05 \
    --output-dir runs/phase1
```

For a full curriculum run (Phases 1–5, both no-think and think tracks), see `Reproduction Recipe` below.

---

## What the system does

Term rewriting systems (TRS) are defined by a finite set of rewrite rules `L → R`. Given a starting expression, the goal is to find an ordered sequence of rule applications that reduces it to *normal form* (no rule applies anywhere). The model emits proofs in this format:

```
PROOF
S1: <expression after step 1> RULE <rule number>
...
SN: <normal form> RULE <rule number>
```

The verifier checks (i) that every step is a valid single rule application and (ii) that the final expression matches the target normal form, then computes the shaped reward (Equation 1 in the paper):

```
R_total = 0.4 × (valid_steps / n)                         # step validity
        + 0.4 × 1[final == target]                         # outcome correctness
        + 0.2 × 1 / (1 + |rules matching final state|)     # proximity
```

All three terms are bounded in [0, 1] and the total is in [0, 1].

---

## Reproduction recipe

The full paper-result reproduction is a five-phase no-think track plus a two-phase think track resuming from the Phase 3 no-think checkpoint. Hyperparameters per phase are listed in Appendix C of the paper; the explicit `train.py` invocations are below.

### No-think track (Phases 1–5)

```bash
# Phase 1
python3 scripts/train.py --start-phase 1 \
    --max-steps 3500 --max-tokens 1024 --temperature 0.7 --kl-coeff 0.05 \
    --output-dir runs/no_think/phase1

# Phase 2 (resume from Phase 1)
python3 scripts/train.py --start-phase 2 \
    --resume-checkpoint runs/no_think/phase1/checkpoint-1250 \
    --max-steps 3500 --max-tokens 1024 --temperature 0.7 --lr 5e-6 --kl-coeff 0.05 \
    --output-dir runs/no_think/phase2

# Phase 3
python3 scripts/train.py --start-phase 3 \
    --resume-checkpoint runs/no_think/phase2/checkpoint-1250 \
    --max-steps 2000 --max-tokens 768 --temperature 0.9 --kl-coeff 0.04 \
    --output-dir runs/no_think/phase3

# Phase 4 / Phase 5: same pattern, max-tokens 1024, temp 0.9, kl 0.04
```

### Think track (Phases 4–5, resumed from no-think Phase 3)

```bash
python3 scripts/train.py --start-phase 4 \
    --resume-checkpoint runs/no_think/phase3/checkpoint-1250 \
    --max-steps 2000 --max-tokens 3072 --temperature 0.9 --kl-coeff 0.04 \
    --output-dir runs/think/phase4

python3 scripts/train.py --start-phase 5 \
    --resume-checkpoint runs/think/phase4/checkpoint-1750 \
    --max-steps 2000 --max-tokens 3072 --temperature 0.9 --kl-coeff 0.04 \
    --output-dir runs/think/phase5
```

The headline result (Phase 5 think, 80.0% solve rate) corresponds to `runs/think/phase5/checkpoint-1750` evaluated at temperature 0.3.

### Evaluation

```bash
# In-domain solve rate (Phase 5 eval set)
python3 scripts/evaluate_phase.py \
    --checkpoint runs/think/phase5/checkpoint-1750 \
    --phase 5 --temperature 0.3

# Cross-domain (paper Section 5.6)
python3 scripts/evaluate_math500.py    --checkpoint runs/think/phase5/checkpoint-1750 --max-tokens 8192
python3 scripts/evaluate_mmlu_prox.py  --checkpoint runs/think/phase5/checkpoint-1750 --max-tokens 2048
python3 scripts/evaluate_ifeval.py     --checkpoint runs/think/phase5/checkpoint-1750 --max-tokens 2048
```

Each evaluator writes a `_full.json` with per-instance results, and a summary JSON with aggregate metrics.

---

## Project structure

```
trs-rl/
├── rewritelang/        Pure-Python TRS engine (no ML)
│   ├── grammar.py      Expr datatype, parse_expr, expr_to_str
│   ├── match.py        match(), substitute(), apply_rule_at()
│   ├── verifier.py     verify_proof(), is_normal_form(), reward
│   └── domains.py      BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES
│
├── generator/
│   ├── instance.py     Reverse-rewriting instance generator
│   └── curriculum.py   CurriculumTracker (auto-advance @ 0.75 solve rate)
│
├── agent/
│   ├── prompt.py       Chat-template prompt construction
│   └── parser.py       Robust output parser for PROOF blocks
│
├── training/
│   ├── reward.py       compute_reward() — wraps the verifier
│   ├── grpo.py         GRPO trainer (HuggingFace TRL)
│   └── emergence.py    LO/IN alignment, reflection detection
│
├── data/
│   ├── generate_eval.py     Generate held-out eval sets
│   ├── generate_train.py    Generate training pools
│   ├── generate_math500.py  Cross-domain eval data prep
│   ├── generate_mmlu_prox.py
│   ├── generate_ifeval.py
│   ├── eval/                Output: phase{1-5}.jsonl, ood.jsonl, math500.jsonl, ...
│   └── train/               Output: phase{1-5}.jsonl
│
├── scripts/
│   ├── sanity_check.py      Run before any training
│   ├── train.py             Training entry point
│   ├── evaluate_phase.py    In-domain solve rate
│   ├── evaluate_math500.py  Cross-domain (math)
│   ├── evaluate_mmlu_prox.py
│   ├── evaluate_ifeval.py
│   └── analyze_metrics.py   LO alignment, thought-dropout statistics
│
├── pyproject.toml
└── requirements.txt
```

---

## Curriculum

Phases advance automatically when `solve_rate@1 ≥ 0.75` on the 200-instance held-out eval set. Rule counts are fixed per phase; depth and step count are sampled within the listed ranges.

| Phase | Rules | Depth | Steps  | Domain                   |
|-------|-------|-------|--------|--------------------------|
| 1     | 3     | ≤2    | 1      | Boolean                  |
| 2     | 5     | ≤3    | 2–4    | Boolean                  |
| 3     | 7     | ≤5    | 4–8    | Boolean                  |
| 4     | 9     | ≤8    | 8–15   | Boolean + Arithmetic     |
| 5     | 12    | ≤10   | 10–20  | Boolean + Arithmetic + Abstract (combinators) |

Rule sets are listed in Appendix B of the paper.

---

## Instance generation: backward construction

Training and evaluation instances are generated by *reverse rewriting*:

1. Sample a normal form `NF` (an expression on which no rule fires).
2. Apply rules **in reverse** for `n_steps` to construct a start expression.
3. Return `(rules, start, NF, witness_proof)`.

This guarantees solvability by construction. The witness proof is held by the generator — the model never sees it during training. See `generator/instance.py`.

---

## Reward function

The reward is computed by `rewritelang/verifier.py:verify_proof` (the same code used at evaluation time):

```python
R_total = 0.4 * step_reward + 0.4 * final_reward + 0.2 * proximity
```

where:

- `step_reward = valid_steps / n` — fraction of intermediate rewrite steps that are valid single rule applications under the rule set.
- `final_reward = 1 if final_expression == target else 0` — outcome correctness.
- `proximity = 1 / (1 + |applicable rules at final state|)` — partial credit for arriving at an expression close to a normal form.

Section 7.1 of the paper compares this formulation to a proximity-only variant (which causes reward hacking).

---

## Hardware

Training was performed on 8× NVIDIA H100 80GB GPUs. A single H100 is sufficient for Phase 1–3 reproduction; Phases 4–5 think mode benefit from multi-GPU due to the longer 3072-token completion budget.

Approximate wall-clock per phase (8× H100): no-think Phases 1–5 ≈ 6h each; think Phases 4–5 ≈ 12h each. Total full-curriculum run ≈ 2 days.

---

## Dependencies

Pinned in `requirements.txt` and `pyproject.toml`. Core stack: `torch >= 2.2`, `transformers >= 4.40`, `peft >= 0.10`, `trl >= 0.8`, `vllm >= 0.4` (for fast rollouts). Tested on Python 3.10 with CUDA 12.1.

---

## License

MIT (see `LICENSE`). External assets used: Qwen3.5-2B-Instruct (Qwen license), MATH500, MMLU-Pro, IFEval (research-permitted), `lm-evaluation-harness` (MIT).
