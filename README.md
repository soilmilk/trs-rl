# TRS-RL: Term Rewriting Agent via Reinforcement Learning

RL agent (fine-tuned LLM) that learns to reduce symbolic expressions to normal form using only a binary reward signal — no expert demonstrations, no human annotations.

---

## Quick Start

### If you have a `.pem` file and an EC2 IP address

```bash
# 1. Make the setup script executable
chmod +x aws_setup.sh

# 2. Run it — this does everything
./aws_setup.sh /path/to/your-key.pem YOUR_EC2_IP
```

That script will:
- Fix your SSH key permissions (AWS requires 600)
- Upload the codebase to `~/trs-rl` on your instance
- Install all dependencies
- Run sanity checks
- Start eval set generation in the background

Then SSH in and start training:
```bash
ssh -i /path/to/your-key.pem ubuntu@YOUR_EC2_IP
cd ~/trs-rl
python3 scripts/train.py --model Qwen/Qwen2.5-1.5B-Instruct
```

---

## Manual Setup (if you prefer)

```bash
# On your AWS instance
git clone <your-repo> ~/trs-rl
cd ~/trs-rl
pip install -e ".[dev]"

# MUST pass before training
python3 scripts/sanity_check.py

# Generate fixed eval sets (~2 min)
python3 data/generate_eval.py --seed 42

# Run training
python3 scripts/train.py
```

---

## SSH Cheat Sheet

```bash
# Basic SSH
ssh -i /path/to/key.pem ubuntu@YOUR_IP

# Copy files to instance
scp -i /path/to/key.pem localfile.py ubuntu@YOUR_IP:~/trs-rl/

# Sync entire project
rsync -avz -e "ssh -i /path/to/key.pem" ./ ubuntu@YOUR_IP:~/trs-rl/

# Run training in background (survives SSH disconnect)
nohup python3 scripts/train.py > runs/train.log 2>&1 &

# Watch training logs live
tail -f runs/train.log

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi   # refresh every second
```

---

## Project Structure

```
trs-rl/
├── rewritelang/        # The formal TRS engine (no ML, pure logic)
│   ├── grammar.py      # Expr, parse_expr, expr_to_str
│   ├── match.py        # match(), substitute(), apply_rule_at()
│   ├── verifier.py     # verify_proof(), is_normal_form(), Rule
│   └── domains.py      # BOOLEAN_RULES, ARITHMETIC_RULES, ABSTRACT_RULES
│
├── generator/          # Automated instance generation (no human annotation needed)
│   ├── instance.py     # generate_instance() — reverse-rewriting construction
│   └── curriculum.py   # CurriculumTracker — automated phase advancement
│
├── agent/              # Model interface
│   ├── prompt.py       # make_prompt(), SYSTEM prompt
│   └── parser.py       # parse_proof_from_output() — robust to malformed output
│
├── training/           # RL training
│   ├── reward.py       # compute_reward() — TRS verifier as reward oracle
│   ├── grpo.py         # GRPO training loop (HuggingFace TRL + vLLM)
│   └── emergence.py    # LO alignment, reflection detection, rule frequency
│
├── data/
│   ├── generate_eval.py    # Pre-generate eval sets (run once)
│   └── eval/               # phase1-5.jsonl + ood.jsonl
│
├── scripts/
│   ├── sanity_check.py     # Run this first. Always.
│   └── train.py            # Training entry point
│
├── aws_setup.sh            # One-command AWS setup
└── pyproject.toml
```

---

## Training Config

Default config targets a single H100 (80GB). Key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| model | Qwen2.5-1.5B-Instruct | Can upgrade to 3B/7B |
| lora_rank | 16 | Increase to 32 for more capacity |
| group_size | 8 | GRPO rollouts per problem |
| max_new_tokens | 768 | TRS proofs need more room than SAT |
| temperature | 0.9 | Higher than SAT — more exploration |
| learning_rate | 8e-6 | Conservative for sequential task |
| max_steps | 8000 | ~2-3 days total across all phases |

---

## Curriculum (Fully Automated)

Phases advance automatically when `solve_rate@1 >= 0.75` on the eval set.

| Phase | Rules | Depth | Steps | Domain |
|-------|-------|-------|-------|--------|
| 1 | 3 | 1-2 | 1 | boolean |
| 2 | 5 | 2-3 | 2-4 | boolean |
| 3 | 7 | 3-5 | 4-8 | boolean |
| 4 | 9 | 5-8 | 8-15 | boolean+arithmetic |
| 5 | 12 | 6-10 | 10-20 | mixed |

---

## What Makes This Work

**The key insight:** We generate training problems *backwards* from the solution.

1. Sample a normal form `NF` (expression where no rule fires)
2. Apply rules in reverse `n_steps` times to build a `start` expression
3. The `start → NF` proof is guaranteed to exist — we built it

This means:
- Zero manual annotation
- Infinite training data
- Every problem is solvable by construction
- The model never sees the proof trace

---

## Reward Signal

```
R_total = 0.4 × (valid_steps / total_steps) + 0.6 × R_final
```

- **60% weight on reaching normal form** — the actual goal
- **40% weight on step validity** — partial credit for getting individual steps right
- This prevents the agent from learning to game the format

---

## Success Criteria

The experiment succeeds if ANY of these hold:
1. LO alignment score increases monotonically with training
2. Reflection behavior appears and correlates with harder problems  
3. OOD solve rate > zero-shot baseline
4. Think trace length grows with proof complexity

If none hold → also publishable (strong negative result).
