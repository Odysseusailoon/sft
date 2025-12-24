"""
配置文件，用于Qwen模型的LoRA微调
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置"""

    # 模型配置 - 使用14B模型进行GPU训练
    model_name: str = "Qwen/Qwen3-14B"
    tokenizer_name: str = "Qwen/Qwen3-14B"

    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None

    # 初始化LoRA目标模块
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Qwen模型的LoRA目标模块
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    # 训练配置
    train_data_path: str = "data/"  # 指向数据目录，包含多个JSON文件
    eval_data_path: Optional[str] = None
    output_dir: str = "outputs"
    logging_dir: str = "logs"

    # 训练超参数 - 针对GPU环境和14B模型优化
    max_length: int = 4096  # 增加最大长度以适应复杂诊断信息
    batch_size: int = 2     # GPU环境可以使用稍大的批大小
    gradient_accumulation_steps: int = 8  # 梯度累积以支持大模型
    learning_rate: float = 1e-5  # 适合大模型的学习率
    num_train_epochs: int = 3    # 增加训练轮数
    warmup_steps: int = 200     # 增加warmup步骤
    save_steps: int = 1000      # 更合理的保存频率
    eval_steps: int = 1000      # 更合理的评估频率
    logging_steps: int = 50     # 合理的日志频率

    # 优化器配置
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # 精度和设备配置
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # DeepSpeed配置
    use_deepspeed: bool = False
    deepspeed_config: str = "ds_config.json"

    # WandB配置
    use_wandb: bool = True
    wandb_project: str = "qwen-lora-sft"
    wandb_run_name: Optional[str] = None

    # 其他配置
    seed: int = 42
    dataloader_num_workers: int = 4
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "loss"


# Qwen模型的默认LoRA配置
QWEN_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "modules_to_save": []
}

