#!/usr/bin/env python3
"""
RedBlackBench SFT Training Script v2

Fine-tunes Qwen3-14B with LoRA for strategic reasoning in prisoner's dilemma scenarios.
"""

import os
import json
import argparse
import torch
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


@dataclass
class RedBlackTrainingConfig:
    """
    Training configuration for RedBlackBench SFT.
    
    Design rationale:
    1. Strategic reasoning requires strong expression capability -> high LoRA rank
    2. Long context (avg ~5k chars input) -> adequate max_length
    3. Complex task -> careful learning rate and warmup
    """
    # Model - Use Qwen3-8B as default (available locally), can switch to 14B/32B
    model_name: str = "Qwen/Qwen3-8B"
    
    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Data
    train_data_path: str = "data/redblack_train.jsonl"
    val_data_path: str = "data/redblack_val.jsonl"
    max_length: int = 4096
    
    # Training
    output_dir: str = "outputs_redblack"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging & Checkpoints
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001


class RedBlackDataset(Dataset):
    """Dataset for RedBlackBench SFT training."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 4096,
        verbose: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        if verbose:
            print(f"Loaded {len(self.data)} samples from {data_path}")
            self._analyze_lengths()
    
    def _analyze_lengths(self):
        """Analyze token lengths distribution."""
        lengths = []
        truncated = 0
        
        for item in self.data[:min(100, len(self.data))]:  # Sample 100
            text = self.tokenizer.apply_chat_template(
                item['messages'], tokenize=False, add_generation_prompt=False
            )
            tokens = self.tokenizer.encode(text)
            lengths.append(len(tokens))
            if len(tokens) > self.max_length:
                truncated += 1
        
        if lengths:
            print(f"  Token lengths (sample): min={min(lengths)}, max={max(lengths)}, "
                  f"avg={sum(lengths)/len(lengths):.0f}")
            if truncated > 0:
                print(f"  Warning: {truncated}/{len(lengths)} samples exceed max_length")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            item['messages'], tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': encoded['input_ids'].copy()
        }


def collate_fn(batch, pad_token_id):
    """Dynamic padding collate function."""
    max_len = max(len(x['input_ids']) for x in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    
    for x in batch:
        pad_len = max_len - len(x['input_ids'])
        input_ids.append(x['input_ids'] + [pad_token_id] * pad_len)
        attention_mask.append(x['attention_mask'] + [0] * pad_len)
        labels.append(x['labels'] + [-100] * pad_len)
    
    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor(attention_mask),
        'labels': torch.tensor(labels)
    }


def setup_model(config: RedBlackTrainingConfig, tokenizer):
    """Load and configure model with LoRA."""
    print("\nLoading model...")
    print(f"  Model: {config.model_name}")
    
    # Memory allocation for multi-GPU
    max_memory = {0: "60GiB", 1: "60GiB", "cpu": "100GiB"}
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
        )
    except Exception as e:
        print(f"  Warning: Failed to load model from {config.model_name}: {e}")
        print("  This might be a network issue. Please ensure the model is cached locally.")
        raise
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    # Configure LoRA
    print(f"Configuring LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train RedBlackBench model v2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                       help="Model name or path (e.g., Qwen/Qwen3-8B, Qwen/Qwen3-14B, Qwen/Qwen3-32B)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="outputs_redblack")
    parser.add_argument("--train_data", type=str, default="data/redblack_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/redblack_val.jsonl")
    parser.add_argument("--wandb_project", type=str, default="redblack-sft")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    
    # Build config
    config = RedBlackTrainingConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        output_dir=args.output_dir,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
    )
    
    # Print config
    print("=" * 70)
    print("RedBlackBench SFT Training v2")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Max length: {config.max_length}")
    print(f"Output: {config.output_dir}")
    print("=" * 70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load tokenizer from {config.model_name}: {e}")
        print("  Falling back to local tokenizer...")
        # Fallback to local tokenizer from previous training
        fallback_path = "/root/yuxi/sft/outputs_k8s_sim_final/final_model"
        tokenizer = AutoTokenizer.from_pretrained(fallback_path, trust_remote_code=True)
        print(f"  Loaded from: {fallback_path}")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = RedBlackDataset(config.train_data_path, tokenizer, config.max_length)
    val_dataset = RedBlackDataset(config.val_data_path, tokenizer, config.max_length)
    
    # Setup model
    model = setup_model(config, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.bf16,
        gradient_checkpointing=False,  # Already enabled on model
        dataloader_num_workers=0,
        report_to="none" if args.no_wandb else "wandb",
        run_name=f"redblack-r{config.lora_r}-lr{config.learning_rate}",
        remove_unused_columns=False,
        logging_first_step=True,
    )
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=config.early_stopping_threshold,
        )
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting training...")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    
    # Save final model
    final_path = os.path.join(config.output_dir, "final_model")
    print(f"\nSaving final model to: {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save training config
    config_path = os.path.join(final_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "epochs": config.num_train_epochs,
            "max_length": config.max_length,
            "batch_size": config.per_device_train_batch_size,
            "grad_accum": config.gradient_accumulation_steps,
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final model saved to: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

