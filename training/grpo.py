"""
training/grpo.py

GRPO training loop for H100 GPUs using HuggingFace TRL.

On AWS H100s we use:
  - trl.GRPOTrainer (HuggingFace) for the RL loop
  - LoRA via peft

The reward function is our TRS verifier — the model never sees the proof trace.
"""

from __future__ import annotations
import os
import json
import random
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("WARNING: trl not installed. Run: pip install trl>=0.8.0")

from generator.instance import TRSInstance, generate_instance
from generator.curriculum import CurriculumTracker
from agent.prompt import make_chat_messages, SYSTEM
from training.reward import compute_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset factory — loads from pre-generated files if available,
# otherwise falls back to on-the-fly generation
# ---------------------------------------------------------------------------

def load_phase_file(phase: int, train_data_dir: str) -> list[dict]:
    """Load pre-generated instances for a given phase."""
    path = Path(train_data_dir) / f"phase{phase}.jsonl"
    if not path.exists():
        return []
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def make_grpo_dataset(
    curriculum: CurriculumTracker,
    n_instances: int,
    seed_offset: int = 0,
    tokenizer=None,
    train_data_dir: Optional[str] = None,
) -> Dataset:
    """
    Build a HuggingFace Dataset with 'prompt' and '_instance_json' columns.
    Loads from pre-generated phase files if train_data_dir is provided,
    otherwise generates on the fly.
    """
    phase = 2 #curriculum.state.current_phase
    prompts   = []
    inst_jsons = []

    # ── Try loading from pre-generated files ──────────────────────────────
    if train_data_dir:
        pool = load_phase_file(phase, train_data_dir)
        if pool:
            logger.info(f"Loaded {len(pool)} pre-generated instances from phase{phase}.jsonl")
            rng = random.Random(seed_offset + curriculum.state.training_step)
            # Sample n_instances from the pool (with replacement if needed)
            selected = rng.choices(pool, k=n_instances)

            for d in selected:
                inst = TRSInstance.from_dict(d)
                msgs = make_chat_messages(inst)
                if tokenizer is not None:
                    prompt_str = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt_str = f"[SYSTEM]{SYSTEM}[USER]{msgs[1]['content']}"
                prompts.append(prompt_str)
                inst_jsons.append(json.dumps(d))

            return Dataset.from_dict({
                "prompt":         prompts,
                "_instance_json": inst_jsons,
            })
        else:
            logger.warning(f"No pre-generated file found for phase {phase} in {train_data_dir}, falling back to on-the-fly generation")

    # ── Fallback: generate on the fly ─────────────────────────────────────
    logger.info(f"Generating {n_instances} instances on the fly for phase {phase}...")
    rng = random.Random(seed_offset + curriculum.state.training_step)

    for i in range(n_instances):
        kwargs = curriculum.get_instance_kwargs(rng)
        kwargs["seed"] = seed_offset + curriculum.state.training_step * 1000 + i
        try:
            inst = generate_instance(**kwargs)
        except RuntimeError:
            inst = generate_instance(
                n_rules=3, max_depth=2, n_steps=1, domain="boolean",
                seed=kwargs["seed"] + 99999,
            )

        msgs = make_chat_messages(inst)
        if tokenizer is not None:
            prompt_str = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = f"[SYSTEM]{SYSTEM}[USER]{msgs[1]['content']}"

        prompts.append(prompt_str)
        inst_jsons.append(json.dumps(inst.to_dict()))

    return Dataset.from_dict({
        "prompt":         prompts,
        "_instance_json": inst_jsons,
    })


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_reward_fn(instances_by_prompt: dict):
    """
    New TRL API: reward_fn(completions, prompts=None, **kwargs) -> list[float]
    """
    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            prompt = prompts[i] if prompts is not None else None
            inst_json = instances_by_prompt.get(prompt) if prompt else None
            if inst_json is None:
                rewards.append(0.0)
                continue
            inst = TRSInstance.from_dict(json.loads(inst_json))
            r = compute_reward(completion, inst)
            rewards.append(r)
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    model_name:       str   = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir:       str   = "runs/trs_rl",
    train_data_dir:   str   = "data/train",       # pre-generated phase files
    max_steps:        int   = 8000,
    learning_rate:    float = 8e-6,
    batch_size:       int   = 4,
    grad_accumulation:int   = 2,
    group_size:       int   = 8,
    max_new_tokens:   int   = 768,
    temperature:      float = 0.9,
    kl_coeff:         float = 0.04,
    lora_rank:        int   = 16,
    lora_alpha:       int   = 32,
    lora_dropout:     float = 0.05,
    save_every:       int   = 500,
    eval_every:       int   = 250,
    advance_threshold:float = 0.75,
    seed:             int   = 42,
    resume_from:      Optional[str] = None,
) -> None:

    if not TRL_AVAILABLE:
        raise ImportError("Install trl: pip install trl>=0.8.0")

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/train.log"),
            logging.StreamHandler(),
        ]
    )

    logger.info(f"Starting TRS-RL training on {torch.cuda.device_count()} GPU(s)")
    logger.info(f"Model: {model_name}")
    logger.info(f"Train data dir: {train_data_dir}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ───────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = lora_rank,
        lora_alpha     = lora_alpha,
        lora_dropout   = lora_dropout,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Curriculum ──────────────────────────────────────────────────────────
    curriculum = CurriculumTracker(
        advance_threshold = advance_threshold,
        eval_interval     = eval_every,
    )
    if resume_from:
        curriculum = CurriculumTracker.load(f"{resume_from}/curriculum.json")
        logger.info(f"Resumed curriculum: {curriculum.summary()}")

    # ── GRPO Config ──────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir                  = output_dir,
        max_steps                   = max_steps,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accumulation,
        learning_rate               = learning_rate,
        num_generations             = group_size,
        max_completion_length       = max_new_tokens,
        temperature                 = temperature,
        beta                        = kl_coeff,
        save_steps                  = save_every,
        logging_steps               = 10,
        seed                        = seed,
        bf16                        = True,
        gradient_checkpointing      = True,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    # ── Dataset — loaded from pre-generated files ─────────────────────────
    dataset = make_grpo_dataset(
        curriculum,
        n_instances    = max_steps * batch_size,
        seed_offset    = seed,
        tokenizer      = tokenizer,
        train_data_dir = train_data_dir,
    )

    instances_map = {
        row["prompt"]: row["_instance_json"]
        for row in dataset
    }
    reward_fn = make_reward_fn(instances_map)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model            = model,
        reward_funcs     = reward_fn,
        args             = grpo_config,
        train_dataset    = dataset.remove_columns(["_instance_json"]),
        processing_class = tokenizer,
    )

    logger.info("Starting training loop...")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(f"{output_dir}/final")
    curriculum.save(f"{output_dir}/curriculum.json")
    logger.info(f"Training complete. Model saved to {output_dir}/final")