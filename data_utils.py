"""
数据处理工具，用于Qwen模型的LoRA微调
"""

import json
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any


class SFTDataset(Dataset):
    """用于指令微调的数据集类"""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据，支持单个文件或目录中的多个JSON文件"""
        import os
        data = []

        if os.path.isdir(data_path):
            # 如果是目录，加载所有JSON文件
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(data_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                            data.append(file_data)
                    except Exception as e:
                        print(f"Warning: Failed to load {file_path}: {e}")
        else:
            # 如果是单个文件
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.jsonl'):
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line.strip()))
                elif data_path.endswith('.json'):
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data = file_data
                    else:
                        data = [file_data]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 支持真实诊断数据格式
        if 'problem_id' in item and 'system_state_summary' in item and 'commands' in item:
            # 真实诊断数据格式 - Kubernetes系统故障诊断任务
            problem_id = item['problem_id']
            system_state = item['system_state_summary']
            commands = item['commands']

            # 构建指令微调的prompt和response
            instruction = f"""你是一个Kubernetes系统故障诊断专家。基于以下系统状态信息，请分析问题并提供相应的诊断命令。

问题ID: {problem_id}

系统状态总结:
{system_state}

请提供系统性的诊断步骤和命令来解决这个问题。"""

            # 从commands中提取实际执行的命令序列
            command_sequence = []
            for cmd in commands:
                if cmd.get('success', False):
                    # 使用原始命令或实际执行的命令
                    command_text = cmd.get('original_command', cmd.get('command', ''))
                    if command_text:
                        # 提取命令内容（去掉exec_shell包装）
                        import re
                        match = re.search(r'exec_shell\("([^"]+)"\)', command_text)
                        if match:
                            command_sequence.append(match.group(1))

            # 如果没有成功的命令，使用第一个命令
            if not command_sequence and commands:
                first_cmd = commands[0].get('original_command', commands[0].get('command', ''))
                if first_cmd:
                    match = re.search(r'exec_shell\("([^"]+)"\)', first_cmd)
                    if match:
                        command_sequence = [match.group(1)]

            # 构建完整的对话格式
            if command_sequence:
                response = "让我分析这个问题...\n\n" + "\n".join(f"```bash\n{cmd}\n```" for cmd in command_sequence[:5])  # 限制前5个命令
            else:
                response = "让我分析这个问题...\n\n需要进一步检查系统状态。"

            # 使用对话格式
            full_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"

            # 对完整文本进行编码
            inputs = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 对于因果语言建模，labels就是input_ids
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze()
            }
        else:
            raise ValueError(f"不支持的数据格式。需要包含 'problem_id', 'system_state_summary', 'commands' 字段")


def collate_fn(batch):
    """数据批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def prepare_data(data_path: str, tokenizer_path: str, max_length: int = 2048):
    """准备数据集"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 添加pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SFTDataset(data_path, tokenizer, max_length)

    return dataset, tokenizer

