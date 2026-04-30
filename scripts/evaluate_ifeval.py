#!/usr/bin/env python3
"""
scripts/evaluate_ifeval.py

Evaluate a checkpoint (or base model) on Google IFEval.
IFEval uses programmatic constraint verification — each prompt has 1+ instructions
that the response must satisfy (word count, no commas, JSON format, etc.).

Reports:
  - Prompt-level strict accuracy: fraction of prompts where ALL instructions satisfied
  - Instruction-level strict accuracy: fraction of all instructions satisfied
  - Per-instruction-type breakdown

Coverage: implements the most common ~15 instruction types from IFEval. Less common
types (e.g. some language-specific checks) are marked SKIPPED and excluded from
denominators.

Usage:
    python3 scripts/evaluate_ifeval.py \
        --eval-file data/eval/ifeval.jsonl \
        --n-samples 200
"""

import sys
import re
import json
import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verifiers — one per instruction id. Each takes (response: str, kwargs: dict)
# and returns True/False/None (None = unsupported, excluded from stats).
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def _paragraph_count(text: str) -> int:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\*\*\*", text) if p.strip()]
    return len(paras)


def _check_relation(value: int, relation: str, threshold: int) -> bool:
    if relation in ("at least", "atleast", "at_least"):
        return value >= threshold
    if relation in ("less than", "lessthan", "less_than"):
        return value < threshold
    if relation in ("at most", "atmost", "at_most"):
        return value <= threshold
    if relation in ("more than", "morethan", "more_than"):
        return value > threshold
    if relation == "equal to" or relation == "exactly":
        return value == threshold
    return False


def v_no_comma(r, kw):                  return "," not in r
def v_lowercase(r, kw):                 return r == r.lower()
def v_uppercase(r, kw):                 return r == r.upper()


def v_capital_word_frequency(r, kw):
    cap = sum(1 for w in r.split() if w.isupper() and len(w) > 1)
    rel = kw.get("capital_relation") or kw.get("relation")
    n   = kw.get("capital_frequency") or kw.get("num_words") or kw.get("frequency") or 0
    return _check_relation(cap, rel, n) if rel else cap >= 1


def v_num_words(r, kw):
    rel = kw.get("relation", "at least")
    n   = kw.get("num_words", 0) or 0
    return _check_relation(_word_count(r), rel, n)


def v_num_sentences(r, kw):
    rel = kw.get("relation", "at least")
    n   = kw.get("num_sentences", 0) or 0
    return _check_relation(_sentence_count(r), rel, n)


def v_num_paragraphs(r, kw):
    n = kw.get("num_paragraphs", 0) or 0
    return _paragraph_count(r) == n


def v_nth_paragraph_first_word(r, kw):
    n         = kw.get("nth_paragraph", 1) or 1
    first_w   = (kw.get("first_word") or "").lower()
    paras     = [p.strip() for p in re.split(r"\n\s*\n|\*\*\*", r) if p.strip()]
    if n <= 0 or n > len(paras):
        return False
    fw = paras[n - 1].split()
    return bool(fw) and fw[0].lower().strip(".,!?;:") == first_w


def v_keyword_existence(r, kw):
    keywords = kw.get("keywords") or []
    rl = r.lower()
    return all(k.lower() in rl for k in keywords)


def v_keyword_frequency(r, kw):
    keyword = kw.get("keyword") or ""
    rel     = kw.get("relation", "at least")
    n       = kw.get("frequency", 1) or 1
    if not keyword:
        return False
    count = len(re.findall(r"\b" + re.escape(keyword.lower()) + r"\b", r.lower()))
    return _check_relation(count, rel, n)


def v_forbidden_words(r, kw):
    forbidden = kw.get("forbidden_words") or []
    rl = r.lower()
    return all(re.search(r"\b" + re.escape(w.lower()) + r"\b", rl) is None for w in forbidden)


def v_letter_frequency(r, kw):
    letter = (kw.get("letter") or "").lower()
    rel    = kw.get("let_relation") or kw.get("relation", "at least")
    n      = kw.get("let_frequency") or kw.get("frequency", 0) or 0
    if not letter:
        return False
    count = r.lower().count(letter)
    return _check_relation(count, rel, n)


def v_json_format(r, kw):
    # Strip code fences if any
    s = re.sub(r"```(?:json)?", "", r).strip().strip("`").strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def v_number_bullets(r, kw):
    n = kw.get("num_bullets", 0) or 0
    bullets = re.findall(r"^\s*[\*\-]\s+", r, flags=re.MULTILINE)
    return len(bullets) == n


def v_number_highlights(r, kw):
    n = kw.get("num_highlights", 0) or 0
    highlights = re.findall(r"\*[^*\n]+\*|_[^_\n]+_", r)
    return len(highlights) >= n


def v_multiple_sections(r, kw):
    n      = kw.get("num_sections", 0) or 0
    spliter= kw.get("section_spliter") or "Section"
    sections = re.findall(rf"{re.escape(spliter)}\s*\d+", r, flags=re.IGNORECASE)
    return len(sections) >= n


def v_title(r, kw):
    return bool(re.search(r"<<[^>]+>>", r))


def v_constrained_response(r, kw):
    valid = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
    return any(v in r for v in valid)


def v_two_responses(r, kw):
    return "******" in r and len(r.split("******")) >= 2


def v_repeat_prompt(r, kw):
    prompt = kw.get("prompt_to_repeat") or ""
    return bool(prompt) and r.strip().startswith(prompt.strip())


def v_postscript(r, kw):
    marker = (kw.get("postscript_marker") or "P.S.").strip()
    return marker in r


def v_number_placeholders(r, kw):
    n = kw.get("num_placeholders", 0) or 0
    return len(re.findall(r"\[[^\]]+\]", r)) >= n


def v_end_checker(r, kw):
    end = (kw.get("end_phrase") or "").strip()
    return bool(end) and r.strip().endswith(end)


def v_quotation(r, kw):
    s = r.strip()
    return len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"))


def v_response_language(r, kw):
    # Coarse heuristic: only handle English (most common in IFEval).
    lang = (kw.get("language") or "").lower()
    if lang in ("english", "en"):
        # English response: should contain mostly ASCII letters
        ascii_letters = sum(1 for c in r if c.isalpha() and ord(c) < 128)
        all_letters   = sum(1 for c in r if c.isalpha())
        return all_letters == 0 or ascii_letters / all_letters > 0.9
    return None  # unsupported language → skip


VERIFIERS = {
    "punctuation:no_comma":                                v_no_comma,
    "change_case:english_lowercase":                       v_lowercase,
    "change_case:english_capital":                         v_uppercase,
    "change_case:capital_word_frequency":                  v_capital_word_frequency,
    "length_constraints:number_words":                     v_num_words,
    "length_constraints:number_sentences":                 v_num_sentences,
    "length_constraints:number_paragraphs":                v_num_paragraphs,
    "length_constraints:nth_paragraph_first_word":         v_nth_paragraph_first_word,
    "keywords:existence":                                  v_keyword_existence,
    "keywords:frequency":                                  v_keyword_frequency,
    "keywords:forbidden_words":                            v_forbidden_words,
    "keywords:letter_frequency":                           v_letter_frequency,
    "detectable_format:json_format":                       v_json_format,
    "detectable_format:number_bullet_lists":               v_number_bullets,
    "detectable_format:number_highlighted_sections":       v_number_highlights,
    "detectable_format:multiple_sections":                 v_multiple_sections,
    "detectable_format:title":                             v_title,
    "detectable_format:constrained_response":              v_constrained_response,
    "combination:two_responses":                           v_two_responses,
    "combination:repeat_prompt":                           v_repeat_prompt,
    "detectable_content:postscript":                       v_postscript,
    "detectable_content:number_placeholders":              v_number_placeholders,
    "startend:end_checker":                                v_end_checker,
    "startend:quotation":                                  v_quotation,
    "language:response_language":                          v_response_language,
}


def verify_response(response: str, instruction_ids: list[str], kwargs_list: list[dict]) -> list[bool | None]:
    """Returns a list of pass/fail/None (unsupported) per instruction."""
    out = []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        kw = kw or {}
        fn = VERIFIERS.get(inst_id)
        if fn is None:
            out.append(None)
        else:
            try:
                out.append(bool(fn(response, kw)))
            except Exception:
                out.append(False)
    return out


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str | None,
    eval_file: str,
    model_name: str = "/workspace/models/Qwen3.5-2B",
    n_samples: int = 200,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    enable_thinking: bool = True,
    save_results: str | None = None,
    output_file: str | None = None,
):
    is_zeroshot = (not checkpoint_path) or checkpoint_path == "baseline"
    mode_str = "ZERO-SHOT (no LoRA)" if is_zeroshot else f"CHECKPOINT: {checkpoint_path}"
    logger.info(f"{'='*50}")
    logger.info(f"  Mode: {mode_str}")
    logger.info(f"  Thinking: {'ON' if enable_thinking else 'OFF'}")
    logger.info(f"  Eval file: {eval_file}")
    logger.info(f"  Max tokens: {max_new_tokens}  |  Temp: {temperature}")
    logger.info(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    if checkpoint_path and checkpoint_path != "baseline":
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info(f"Loaded LoRA checkpoint from {checkpoint_path}")
    else:
        model = base_model
        logger.info("Running ZERO-SHOT base model")
    model.eval()

    problems = []
    with open(eval_file) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    if not problems:
        raise RuntimeError(f"No problems loaded from {eval_file}")

    random.seed(42)
    if len(problems) > n_samples:
        problems = random.sample(problems, n_samples)
    logger.info(f"Loaded {len(problems)} prompts")

    do_sample = temperature > 0.0
    results   = []
    for i, p in enumerate(problems):
        prompt          = p["prompt"]
        instruction_ids = p["instruction_id_list"]
        kwargs_list     = p["kwargs"]

        msgs = [
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(raw_ids, skip_special_tokens=False)
        for tok in [tokenizer.eos_token, tokenizer.pad_token]:
            if tok:
                completion = completion.replace(tok, "")
        completion = completion.strip()

        # Strip <think>...</think> if present, eval against the visible response only
        cleaned = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()

        verdicts = verify_response(cleaned, instruction_ids, kwargs_list)
        n_total      = sum(1 for v in verdicts if v is not None)
        n_pass       = sum(1 for v in verdicts if v is True)
        prompt_pass  = (n_total > 0 and n_pass == n_total)

        results.append({
            "key":              p.get("key"),
            "prompt_pass":      prompt_pass,
            "n_instructions":   n_total,
            "n_passed":         n_pass,
            "instruction_ids":  instruction_ids,
            "verdicts":         verdicts,
            "completion":       completion,
            "completion_length":len(completion.split()),
        })

        marker = "✓" if prompt_pass else "✗"
        logger.info(f"  [{i+1}/{len(problems)}] {marker}  {n_pass}/{n_total} instructions  ({','.join(instruction_ids)[:60]})")
        if (i + 1) % 25 == 0:
            so_far = sum(r["prompt_pass"] for r in results)
            logger.info(f"  --> running prompt-level acc: {so_far}/{i+1} = {so_far/(i+1):.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────
    n               = len(results)
    n_prompt_pass   = sum(r["prompt_pass"] for r in results)
    prompt_acc      = n_prompt_pass / n

    n_inst_total    = sum(r["n_instructions"] for r in results)
    n_inst_pass     = sum(r["n_passed"] for r in results)
    inst_acc        = n_inst_pass / n_inst_total if n_inst_total else 0

    # Per-instruction-type breakdown
    by_type = defaultdict(lambda: {"n": 0, "pass": 0, "skip": 0})
    for r in results:
        for inst_id, v in zip(r["instruction_ids"], r["verdicts"]):
            if v is None:
                by_type[inst_id]["skip"] += 1
            else:
                by_type[inst_id]["n"]    += 1
                if v: by_type[inst_id]["pass"] += 1

    print("\n" + "="*60)
    print(f"  IFEval Strict Evaluation Results")
    print("="*60)
    print(f"  Mode:                  {mode_str}")
    print(f"  N prompts:             {n}")
    print(f"  Prompt-level acc:      {n_prompt_pass}/{n} = {prompt_acc:.4f} ({prompt_acc*100:.1f}%)")
    print(f"  Instruction-level acc: {n_inst_pass}/{n_inst_total} = {inst_acc:.4f} ({inst_acc*100:.1f}%)")
    print("="*60)
    print(f"  Per-instruction-type:")
    for t in sorted(by_type.keys()):
        s = by_type[t]
        denom = s["n"]
        pct   = (s["pass"] / denom * 100) if denom else 0.0
        skip_str = f" (+{s['skip']} skipped)" if s["skip"] else ""
        print(f"    {t:50s}  {s['pass']:>3d}/{denom:<3d} = {pct:5.1f}%{skip_str}")
    print("="*60)

    if save_results:
        with open(save_results, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {save_results}")

    if output_file:
        out = {
            "mode":              "zero-shot" if is_zeroshot else "checkpoint",
            "checkpoint":        checkpoint_path,
            "eval_file":         eval_file,
            "n_samples":         n,
            "prompt_accuracy":   prompt_acc,
            "instruction_accuracy": inst_acc,
            "by_instruction_type":  {t: dict(v) for t, v in by_type.items()},
            "results":           results,
        }
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Saved full output to {output_file}")

    return prompt_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=None)
    p.add_argument("--baseline",    action="store_true")
    p.add_argument("--eval-file",   required=True)
    p.add_argument("--model",       default="/workspace/models/Qwen3.5-2B")
    p.add_argument("--n-samples",   type=int, default=200)
    p.add_argument("--max-tokens",  type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-think",    action="store_true")
    p.add_argument("--save-results",default=None)
    p.add_argument("--output-file", default=None)
    args = p.parse_args()

    checkpoint = args.checkpoint
    if args.baseline or checkpoint is None:
        checkpoint = None

    evaluate(
        checkpoint_path = checkpoint,
        eval_file       = args.eval_file,
        model_name      = args.model,
        n_samples       = args.n_samples,
        max_new_tokens  = args.max_tokens,
        temperature     = args.temperature,
        enable_thinking = not args.no_think,
        save_results    = args.save_results,
        output_file     = args.output_file,
    )


if __name__ == "__main__":
    main()
