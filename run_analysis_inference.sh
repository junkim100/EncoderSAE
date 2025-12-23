#!/usr/bin/env bash
set -euo pipefail

# Paths (can be overridden via environment variables)
SAE_PATH="${SAE_PATH:-checkpoints/intfloat_multilingual-e5-large_4lang_train/exp32_k1024_lr3e-04_aux1e+00_tgt2e-02/final_model.pt}"
MODEL_NAME="${MODEL_NAME:-intfloat/multilingual-e5-large}"
VALIDATION_DATA="${VALIDATION_DATA:-data/4lang_validation.jsonl}"
TEXT_FILE="${TEXT_FILE:-data/4lang_validation.jsonl}"

# Analysis configuration
MASK_THRESHOLD="${MASK_THRESHOLD:-0.95}"  # Threshold (0.0-1.0) for language-specific feature detection
EXCLUDE_OVERLAPPING_FEATURES="${EXCLUDE_OVERLAPPING_FEATURES:-True}"  # Exclude features that are language-specific to multiple languages (default: True)

# Inference configuration
BATCH_SIZE="${BATCH_SIZE:-32}"           # Batch size for text processing
MAX_LENGTH="${MAX_LENGTH:-512}"          # Maximum sequence length

# GPU configuration
NUM_GPUS="${NUM_GPUS:-8}"                # Number of GPUs for vLLM
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"  # GPU memory utilization (0.0-1.0)
USE_VLLM="${USE_VLLM:-True}"             # Use vLLM for faster inference

####################################################################################################

if [ ! -f "${SAE_PATH}" ]; then
    echo "Error: SAE checkpoint not found at ${SAE_PATH}" >&2
    exit 1
fi

if [ ! -f "${VALIDATION_DATA}" ]; then
    echo "Error: Validation data not found at ${VALIDATION_DATA}" >&2
    exit 1
fi

if [ ! -f "${TEXT_FILE}" ]; then
    echo "Error: Text file not found at ${TEXT_FILE}" >&2
    exit 1
fi

SAE_DIR=$(dirname "${SAE_PATH}")
PARENT_DIR=$(basename "${SAE_DIR}")
CHECKPOINT_NAME=$(basename "${SAE_PATH}" .pt)

if [ "${CHECKPOINT_NAME}" = "final_model" ]; then
    ANALYSIS_DIR_NAME="${PARENT_DIR}_final"
elif [[ "${CHECKPOINT_NAME}" == checkpoint_step_* ]]; then
    STEP_NUM="${CHECKPOINT_NAME#checkpoint_step_}"
    ANALYSIS_DIR_NAME="${PARENT_DIR}_step_${STEP_NUM}"
else
    ANALYSIS_DIR_NAME="${PARENT_DIR}_${CHECKPOINT_NAME}"
fi

MASK_PATH="analysis/${ANALYSIS_DIR_NAME}/language_features_combined_mask.pt"

echo "Running language feature analysis..."
uv run -m EncoderSAE.analyze_main \
    --sae_path="${SAE_PATH}" \
    --validation_data="${VALIDATION_DATA}" \
    --model="${MODEL_NAME}" \
    --mask_threshold="${MASK_THRESHOLD}" \
    --exclude_overlapping_features="${EXCLUDE_OVERLAPPING_FEATURES}" \
    --use_vllm \
    --num_gpus="${NUM_GPUS}" \
    --gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"

if [ ! -f "${MASK_PATH}" ]; then
    echo "Error: Mask file not found at ${MASK_PATH}" >&2
    echo "Analysis may have failed or mask path is incorrect." >&2
    exit 1
fi

echo "Running inference with language-agnostic features..."
uv run -m EncoderSAE.inference_main from_text \
    --model_name="${MODEL_NAME}" \
    --sae_path="${SAE_PATH}" \
    --mask_path="${MASK_PATH}" \
    --text_file="${TEXT_FILE}" \
    --batch_size="${BATCH_SIZE}" \
    --max_length="${MAX_LENGTH}" \
    --use_vllm="${USE_VLLM}" \
    --num_gpus="${NUM_GPUS}" \
    --gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"

echo "Analysis and inference complete."
