"""
training/grpo.py

GRPO training loop for H100 GPUs using HuggingFace TRL.

On AWS H100s we use:
  - trl.GRPOTrainer (HuggingFace) for the RL loop
  - LoRA via peft

The reward function is our TRS verifier — the model never sees the proof trace.

Changes in this version:
  - Added --start-phase argument to control which phase data to load
  - Added --resume-from-checkpoint to continue from a saved checkpoint
  - Logs saved to disk via tee (run with 2>&1 | tee ~/phaseN_logs.txt)
  - Fixed all TRL API changes (processing_class, max_completion_length, beta)
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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("WARNING: trl not installed. Run: pip install trl>=0.8.0")

from generator.instance import TRSInstance, generate_instance
from generator.curriculum import CurriculumTracker, PHASE_CONFIGS
from agent.prompt import make_chat_messages, SYSTEM
from training.reward import compute_reward

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset factory — loads from pre-generated phase files
# ---------------------------------------------------------------------------

def load_phase_file(phase: int, train_data_dir: str) -> list[dict]:
    """Load pre-generated instances for a given phase."""
    path = Path(train_data_dir) / f"phase{phase}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Phase file not found: {path}")
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    logger.info(f"Loaded {len(instances)} instances from {path}")
    return instances


def make_grpo_dataset(
    phase: int,
    n_instances: int,
    seed_offset: int = 0,
    tokenizer=None,
    train_data_dir: str = "data/train",
) -> Dataset:
    """
    Build a HuggingFace Dataset with 'prompt' and '_instance_json' columns.
    Loads from pre-generated phase files.
    """
    pool = load_phase_file(phase, train_data_dir)
    rng = random.Random(seed_offset)
    selected = rng.choices(pool, k=n_instances)

    prompts = []
    inst_jsons = []

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


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_reward_fn(instances_by_prompt: dict):
    """
    TRL API: reward_fn(completions, prompts=None, **kwargs) -> list[float]
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
    model_name:        str   = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir:        str   = "runs/trs_rl_phase1",
    train_data_dir:    str   = "data/train",
    start_phase:       int   = 1,              # which phase data to load
    resume_checkpoint: Optional[str] = None,   # path to checkpoint to resume from
    max_steps:         int   = 2000,
    learning_rate:     float = 8e-6,
    batch_size:        int   = 4,
    grad_accumulation: int   = 2,
    group_size:        int   = 8,
    max_new_tokens:    int   = 768,
    temperature:       float = 0.9,
    kl_coeff:          float = 0.04,
    lora_rank:         int   = 16,
    lora_alpha:        int   = 32,
    lora_dropout:      float = 0.05,
    save_every:        int   = 500,
    eval_every:        int   = 250,
    seed:              int   = 42,
) -> None:

    if not TRL_AVAILABLE:
        raise ImportError("Install trl: pip install trl>=0.8.0")

    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/train.log"),
            logging.StreamHandler(),
        ]
    )

    logger.info(f"{'='*60}")
    logger.info(f"TRS-RL Training — Phase {start_phase}")
    logger.info(f"Model:      {model_name}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max steps:  {max_steps}")
    logger.info(f"GPUs:       {torch.cuda.device_count()}")
    if resume_checkpoint:
        logger.info(f"Resuming from: {resume_checkpoint}")
    logger.info(f"{'='*60}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────
    if resume_checkpoint:
        # Load base model then apply saved LoRA weights
        logger.info(f"Loading base model + LoRA from checkpoint...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, resume_checkpoint, is_trainable=True)
        logger.info("Checkpoint loaded successfully.")
    else:
        # Fresh model + new LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
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

    # ── GRPO Config ───────────────────────────────────────────────────────
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

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading Phase {start_phase} dataset from {train_data_dir}...")
    dataset = make_grpo_dataset(
        phase          = start_phase,
        n_instances    = max_steps * batch_size,
        seed_offset    = seed,
        tokenizer      = tokenizer,
        train_data_dir = train_data_dir,
    )
    logger.info(f"Dataset ready: {len(dataset)} instances")

    instances_map = {
        row["prompt"]: row["_instance_json"]
        for row in dataset
    }
    reward_fn = make_reward_fn(instances_map)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model            = model,
        reward_funcs     = reward_fn,
        args             = grpo_config,
        train_dataset    = dataset.remove_columns(["_instance_json"]),
        processing_class = tokenizer,
    )

    logger.info("Starting training loop...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(f"{output_dir}/final")
    logger.info(f"Training complete. Model saved to {output_dir}/final")
    logger.info(f"Phase {start_phase} done.")