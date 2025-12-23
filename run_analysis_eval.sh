#!/usr/bin/env bash
set -euo pipefail

# Paths (can be overridden via environment variables)
SAE_PATH="${SAE_PATH:-checkpoints/intfloat_multilingual-e5-large_4lang_train/exp32_k1024_lr3e-04_aux1e+00_tgt2e-02/final_model.pt}"
MODEL_NAME="${MODEL_NAME:-intfloat/multilingual-e5-large}"
VALIDATION_DATA="${VALIDATION_DATA:-data/4lang_validation.jsonl}"

# Analysis configuration
MASK_THRESHOLD="${MASK_THRESHOLD:-0.95}"  # Threshold (0.0-1.0) for language-specific feature detection

# Evaluation configuration
# Default: all Belebele test subsets under data/Belebele
EVAL_DATA_DIRS="${EVAL_DATA_DIRS:-data/Belebele/Belebele_test_en,data/Belebele/Belebele_test_de,data/Belebele/Belebele_test_es,data/Belebele/Belebele_test_hi,data/Belebele/Belebele_test_vi,data/Belebele/Belebele_test_zh}"  # Comma-separated list of dataset directories
RESULTS_ROOT="${RESULTS_ROOT:-./results_sae_eval}"  # Root directory for evaluation results
BATCH_SIZE="${BATCH_SIZE:-128}"  # Batch size for evaluation
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"  # Maximum sequence length for the encoder model
USE_RECONSTRUCTION="${USE_RECONSTRUCTION:-True}"  # Use reconstructed embeddings in the original embedding dimension

# GPU configuration
NUM_GPUS="${NUM_GPUS:-8}"  # Number of GPUs for vLLM
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"  # GPU memory utilization (0.0-1.0)

####################################################################################################

if [ ! -f "${SAE_PATH}" ]; then
    echo "Error: SAE checkpoint not found at ${SAE_PATH}" >&2
    exit 1
fi

if [ ! -f "${VALIDATION_DATA}" ]; then
    echo "Error: Validation data not found at ${VALIDATION_DATA}" >&2
    exit 1
fi

if [ -z "${EVAL_DATA_DIRS}" ]; then
    echo "Error: EVAL_DATA_DIRS must be set (comma-separated list of dataset directories)" >&2
    echo "Example: EVAL_DATA_DIRS='/path/to/dataset1,/path/to/dataset2'" >&2
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

echo "================================================================================"
echo "Step 1: Running language feature analysis..."
echo "================================================================================"
uv run -m EncoderSAE.analyze_main \
    --sae_path="${SAE_PATH}" \
    --validation_data="${VALIDATION_DATA}" \
    --model="${MODEL_NAME}" \
    --mask_threshold="${MASK_THRESHOLD}" \
    --use_vllm \
    --num_gpus="${NUM_GPUS}" \
    --gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"

if [ ! -f "${MASK_PATH}" ]; then
    echo "Error: Mask file not found at ${MASK_PATH}" >&2
    echo "Analysis may have failed or mask path is incorrect." >&2
    exit 1
fi

echo ""
echo "================================================================================"
echo "Step 2: Running evaluation with language-agnostic embeddings..."
echo "================================================================================"

# Convert comma-separated string to Python list format
IFS=',' read -ra DIR_ARRAY <<< "${EVAL_DATA_DIRS}"
EVAL_DIRS_PYTHON="["
for i in "${!DIR_ARRAY[@]}"; do
    if [ $i -gt 0 ]; then
        EVAL_DIRS_PYTHON+=", "
    fi
    EVAL_DIRS_PYTHON+="\"${DIR_ARRAY[i]}\""
done
EVAL_DIRS_PYTHON+="]"

uv run python evaluation/sae_eval.py \
    --model="${MODEL_NAME}" \
    --sae_path="${SAE_PATH}" \
    --mask_path="${MASK_PATH}" \
    --data_dirs="${EVAL_DIRS_PYTHON}" \
    --results_root="${RESULTS_ROOT}" \
    --batch_size="${BATCH_SIZE}" \
    --max_seq_length="${MAX_SEQ_LENGTH}" \
    --use_reconstruction="${USE_RECONSTRUCTION}" \
    --mask_threshold="${MASK_THRESHOLD}"

echo ""
echo "================================================================================"
echo "Analysis and evaluation complete."
echo "Results saved to: ${RESULTS_ROOT}"
echo "================================================================================"

