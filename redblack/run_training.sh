#!/bin/bash
#
# RedBlackBench SFT Training Script
#
# Usage:
#   ./run_training.sh                    # Default: Qwen3-8B
#   ./run_training.sh --model Qwen/Qwen3-14B   # Use 14B model
#   ./run_training.sh --model Qwen/Qwen3-32B   # Use 32B model
#

set -e

cd "$(dirname "$0")"

# Default parameters
MODEL="${MODEL:-Qwen/Qwen3-8B}"
EPOCHS="${EPOCHS:-5}"
LORA_R="${LORA_R:-64}"
LR="${LR:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_redblack}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-wandb)
            NO_WANDB="--no_wandb"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "RedBlackBench SFT Training"
echo "========================================"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "LoRA rank: $LORA_R"
echo "Learning rate: $LR"
echo "Max length: $MAX_LENGTH"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Check if data exists
if [ ! -f "data/redblack_train.jsonl" ]; then
    echo "Data not found. Running data preparation..."
    python prepare_redblack_data_v2.py --data_dir hf_dataset --output_dir data --save_by_scenario
fi

# Run training
python train_redblack_v2.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --lora_r "$LORA_R" \
    --lr "$LR" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --output_dir "$OUTPUT_DIR" \
    $NO_WANDB

echo ""
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"

