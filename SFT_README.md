# SFT Training with Real Data

This repository has been updated to support Supervised Fine-Tuning (SFT) on real Kubernetes diagnostic data.

## Changes Made

### 1. Data Format Support
- **Before**: Expected `command_list` field
- **After**: Now supports `commands` field from real data files
- **Data Loading**: Now loads all JSON files from the `data/` directory automatically

### 2. Model Configuration
- **Model**: Qwen3-14B for full GPU training
- **Training Settings**: Optimized for large model and real data

### 3. Data Processing
- **Input**: System state summary from diagnostic data
- **Output**: Diagnostic commands in conversational format
- **Format**: Uses Qwen chat format with `<|im_start|>user` and `<|im_start|>assistant` tags

## File Structure

```
data/                    # Real diagnostic data (JSON files)
├── k8s_target_port-misconfig-detection-1.json
├── astronomy_shop_ad_service_manual_gc-detection-1.json
└── ... (49 total files)

config.py               # Training configuration
data_utils.py           # Data loading and processing
run_sft_training.py     # Main training script
requirements.txt        # Python dependencies
SFT_README.md          # This documentation
redblack/              # Additional data (preserved)
└── sft_pandemic.json   # Pandemic simulation data
```

## Usage

### GPU Training
```bash
# Install dependencies
pip install -r requirements.txt

# Run full training
python run_sft_training.py
```

### Custom Configuration
Edit `config.py` to modify:
- Model size (`model_name`)
- Training hyperparameters
- Data paths
- LoRA settings

## Data Format

Each JSON file contains:
```json
{
  "problem_id": "unique_problem_identifier",
  "system_state_summary": "Detailed system state description",
  "commands": [
    {
      "original_command": "kubectl command",
      "command": "executed command",
      "result": "command output",
      "success": true
    }
  ]
}
```

## Training Details

- **Task**: Instruction tuning for Kubernetes diagnostics
- **Input**: System state summaries
- **Output**: Diagnostic commands and analysis
- **Format**: Conversational chat format
- **Fine-tuning**: LoRA on Qwen3 models

## Project Cleanup

The project has been cleaned to focus exclusively on SFT training:

### ✅ **Kept Files**
- `run_sft_training.py` - Main SFT training script
- `config.py` - Training configuration
- `data_utils.py` - Data processing utilities
- `data/` - 49 real diagnostic JSON files
- `requirements.txt` - Python dependencies
- `SFT_README.md` - Documentation
- `redblack/` - Preserved additional data

### 🗑️ **Removed Files**
- Old training scripts (`train.py`, `train_diagnosis.py`)
- Old inference scripts (`inference.py`, `diagnosis_inference.py`)
- Shell scripts (`run_training.sh`, `quick_start_diagnosis.sh`)
- Configuration files (`ds_config.json`)
- Logs and documentation (`diagnosis_training.log`, `README.md`, `CLEANUP_COMPLETED.md`)
- Temporary directories (`__pycache__/`, `test_output/`, `venv/`)

## Next Steps

1. Test on GPU with full dataset
2. Adjust hyperparameters for better performance
3. Evaluate model on diagnostic tasks
4. Deploy fine-tuned model for inference
