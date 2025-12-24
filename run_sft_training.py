#!/usr/bin/env python3
"""
SFT训练主脚本 - 支持GPU环境
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from data_utils import SFTDataset
from config import TrainingConfig

def main():
    """主训练函数"""
    print("Starting SFT training...")

    # 创建配置
    config = TrainingConfig()

    # 加载tokenizer
    print(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载训练数据
    print(f"Loading training data from: {config.train_data_path}")
    train_dataset = SFTDataset(config.train_data_path, tokenizer, config.max_length)
    print(f"Loaded {len(train_dataset)} training samples")

    # 加载验证数据（如果有）
    eval_dataset = None
    if config.eval_data_path and os.path.exists(config.eval_data_path):
        print(f"Loading evaluation data from: {config.eval_data_path}")
        eval_dataset = SFTDataset(config.eval_data_path, tokenizer, config.max_length)
        print(f"Loaded {len(eval_dataset)} evaluation samples")

    # 加载模型
    print(f"Loading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto",  # 自动选择设备（GPU优先）
    )

    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    # LoRA配置
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else None,
        run_name=config.wandb_run_name,
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存模型
    print("Saving model...")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final_model"))

    print("Training completed successfully!")
    print(f"Model saved to: {os.path.join(config.output_dir, 'final_model')}")

if __name__ == "__main__":
    main()
