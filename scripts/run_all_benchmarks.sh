#!/bin/bash
# scripts/run_all_benchmarks.sh
#
# Runs all math benchmarks (AIME, AMC, Putnam) on:
#   1) Base model (no LoRA)
#   2) Phase 5 checkpoint with thinking ON
#   3) Phase 5 checkpoint with thinking OFF
#
# Saves results as JSON under results/.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh /path/to/phase5/checkpoint
#
# If no checkpoint is provided, only the base model is run.

set -e  # exit on first error

CHECKPOINT="${1:-}"
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "==========================================="
echo "  TRS-RL Math Benchmark Suite"
echo "==========================================="
echo "  Checkpoint: ${CHECKPOINT:-<none, base only>}"
echo "  Results:    $RESULTS_DIR"
echo "==========================================="

# ── Make sure eval datasets exist ─────────────────────────────────────────
for gen in generate_aime26 generate_amc generate_putnam; do
    out_file="data/eval/${gen#generate_}.jsonl"
    out_file="${out_file/aime26/aime26}"  # keep aime26 name
    if [ ! -f "$out_file" ]; then
        echo ">>> Generating $out_file ..."
        python3 "data/${gen}.py"
    fi
done

# ── Helper ────────────────────────────────────────────────────────────────
run_one() {
    local script=$1       # scripts/evaluate_xxx.py
    local eval_file=$2    # data/eval/xxx.jsonl
    local tag=$3          # aime / amc / putnam
    local mode=$4         # base / phase5_think_on / phase5_think_off
    local extra_args=$5   # checkpoint + --no-think flags

    local out_file="${RESULTS_DIR}/${tag}_${mode}.json"

    echo ""
    echo "==========================================="
    echo "  Running $tag  |  $mode"
    echo "  -> $out_file"
    echo "==========================================="

    python3 "$script" \
        --eval-file "$eval_file" \
        --output-file "$out_file" \
        $extra_args
}

# ── Base model (no checkpoint) ────────────────────────────────────────────
echo ""
echo ">>> BASE MODEL RUNS"

run_one scripts/evaluate_aime.py   data/eval/aime26.jsonl aime   base ""
run_one scripts/evaluate_amc.py    data/eval/amc.jsonl    amc    base ""
run_one scripts/evaluate_putnam.py data/eval/putnam.jsonl putnam base ""

# ── Phase 5 checkpoint runs (only if checkpoint provided) ─────────────────
if [ -n "$CHECKPOINT" ]; then
    echo ""
    echo ">>> PHASE 5 CHECKPOINT RUNS (thinking ON)"

    run_one scripts/evaluate_aime.py   data/eval/aime26.jsonl aime   phase5_think_on "--checkpoint $CHECKPOINT"
    run_one scripts/evaluate_amc.py    data/eval/amc.jsonl    amc    phase5_think_on "--checkpoint $CHECKPOINT"
    run_one scripts/evaluate_putnam.py data/eval/putnam.jsonl putnam phase5_think_on "--checkpoint $CHECKPOINT"

    echo ""
    echo ">>> PHASE 5 CHECKPOINT RUNS (thinking OFF)"

    run_one scripts/evaluate_aime.py   data/eval/aime26.jsonl aime   phase5_think_off "--checkpoint $CHECKPOINT --no-think"
    run_one scripts/evaluate_amc.py    data/eval/amc.jsonl    amc    phase5_think_off "--checkpoint $CHECKPOINT --no-think"
    run_one scripts/evaluate_putnam.py data/eval/putnam.jsonl putnam phase5_think_off "--checkpoint $CHECKPOINT --no-think"
else
    echo ""
    echo ">>> Skipping checkpoint runs (no --checkpoint provided)"
fi

echo ""
echo "==========================================="
echo "  DONE. All results in $RESULTS_DIR"
echo "==========================================="
ls -la "$RESULTS_DIR"