# SFT Training Project

Supervised Fine-Tuning (SFT) for Qwen3-14B on real Kubernetes diagnostic data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run SFT training
python run_sft_training.py
```

## Project Structure

- `run_sft_training.py` - Main training script
- `config.py` - Training configuration
- `data_utils.py` - Data processing utilities
- `data/` - 49 real diagnostic JSON files
- `requirements.txt` - Python dependencies
- `SFT_README.md` - Detailed documentation

## Data

Contains real Kubernetes system diagnostic data with:
- System state summaries
- Diagnostic commands and results
- Problem identification and solutions

## Training

- Model: Qwen/Qwen3-14B
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Task: Instruction tuning for Kubernetes diagnostics
