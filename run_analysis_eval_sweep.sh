#!/usr/bin/env bash
set -euo pipefail

# Wrapper script to run analysis and evaluation with multiple mask thresholds across multiple checkpoints
#
# Configure checkpoints by editing the CHECKPOINT_PATHS array below.
# You can also pass checkpoint paths as command-line arguments (they will be added to the list).
#
# Usage: ./run_analysis_eval_sweep.sh [checkpoint_path1] [checkpoint_path2] ...

####################################################################################################
# CONFIGURATION: Edit this section to specify your checkpoints
####################################################################################################

# Checkpoint paths to evaluate (must point to final_model.pt files)
# Add your checkpoint paths here, one per line
CHECKPOINT_PATHS=(
    "checkpoints/intfloat_multilingual-e5-large_4lang_train/exp32_k1024_lr3e-04_aux1e+00_tgt2e-02/final_model.pt"
    "checkpoints/intfloat_multilingual-e5-large_4lang_train/exp64_k1024_lr3e-04_aux1e+00_tgt2e-02/final_model.pt"
    "checkpoints/intfloat_multilingual-e5-large_4lang_train/exp128_k2048_lr3e-04_aux1e+00_tgt2e-02/final_model.pt"
)

# Mask thresholds to test for each checkpoint
MASK_THRESHOLDS=(0.95 0.98 0.995)

# Whether to exclude overlapping features (features that are language-specific to multiple languages)
# True (default): Only mask features unique to a single language
# False: Mask all features that are language-specific in any language (union behavior)
EXCLUDE_OVERLAPPING_FEATURES="${EXCLUDE_OVERLAPPING_FEATURES:-True}"

# GPU configuration
# Set NUM_GPUS to the number of GPUs to use (leave empty/unset for auto-detection)
# You can also set this via environment variable: export NUM_GPUS=2
NUM_GPUS="${NUM_GPUS:-}"

# Auto-detect GPUs if not set
if [ -z "${NUM_GPUS}" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ "${NUM_GPUS}" -eq 0 ]; then
            NUM_GPUS=1
            echo "Warning: No GPUs detected via nvidia-smi, defaulting to NUM_GPUS=1"
        else
            echo "Auto-detected ${NUM_GPUS} GPU(s) available"
        fi
    else
        NUM_GPUS=1
        echo "Warning: nvidia-smi not found, defaulting to NUM_GPUS=1"
    fi
fi

# Validate and adjust NUM_GPUS for vLLM tensor parallelism
# vLLM requires: num_attention_heads % tensor_parallel_size == 0
# For multilingual-e5-large: 16 attention heads, so valid sizes are: 1, 2, 4, 8, 16
VALID_TP_SIZES=(1 2 4 8 16)  # Divisors of 16 (number of attention heads)
ORIGINAL_NUM_GPUS="${NUM_GPUS}"

# Find largest valid size <= NUM_GPUS
ADJUSTED_NUM_GPUS=1
for size in "${VALID_TP_SIZES[@]}"; do
    if [ "${size}" -le "${NUM_GPUS}" ]; then
        ADJUSTED_NUM_GPUS="${size}"
    fi
done

if [ "${ADJUSTED_NUM_GPUS}" -ne "${ORIGINAL_NUM_GPUS}" ]; then
    echo "Warning: NUM_GPUS=${ORIGINAL_NUM_GPUS} is not compatible with model architecture (16 attention heads)"
    echo "  Valid tensor parallel sizes: ${VALID_TP_SIZES[*]}"
    echo "  Adjusting NUM_GPUS from ${ORIGINAL_NUM_GPUS} to ${ADJUSTED_NUM_GPUS}"
    NUM_GPUS="${ADJUSTED_NUM_GPUS}"
fi

export NUM_GPUS

####################################################################################################

# If checkpoint paths are provided as command-line arguments, add them to the list
if [ $# -gt 0 ]; then
    # Add command-line arguments to the array
    for arg in "$@"; do
        CHECKPOINT_PATHS+=("$arg")
    done
fi

# Check if we have any checkpoint paths
if [ ${#CHECKPOINT_PATHS[@]} -eq 0 ]; then
    echo "Error: No checkpoint paths specified!" >&2
    echo "" >&2
    echo "Please either:" >&2
    echo "  1. Edit this script and add checkpoint paths to the CHECKPOINT_PATHS array, or" >&2
    echo "  2. Pass checkpoint paths as command-line arguments" >&2
    echo "" >&2
    echo "Example: $0 checkpoints/.../exp32_k1024_lr3e-04_aux1e+00_tgt2e-02/final_model.pt" >&2
    exit 1
fi

# Validate all checkpoint paths
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    # Ensure the checkpoint path ends with final_model.pt
    if [[ ! "${CHECKPOINT_PATH}" == */final_model.pt ]]; then
        echo "Error: Checkpoint path must point to final_model.pt" >&2
        echo "Provided path: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi

    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}" >&2
        exit 1
    fi
done

# Get the directory of this script to find run_analysis_eval.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/run_analysis_eval.sh"

if [ ! -f "${ANALYSIS_SCRIPT}" ]; then
    echo "Error: run_analysis_eval.sh not found at ${ANALYSIS_SCRIPT}" >&2
    exit 1
fi

echo "================================================================================"
echo "Running analysis and evaluation sweep"
echo ""
echo "Checkpoints (${#CHECKPOINT_PATHS[@]}):"
for i in "${!CHECKPOINT_PATHS[@]}"; do
    echo "  $((i+1)). ${CHECKPOINT_PATHS[i]}"
done
echo ""
echo "Mask thresholds: ${MASK_THRESHOLDS[*]}"
echo "GPUs: ${NUM_GPUS}"
echo ""
echo "Total combinations: $((${#CHECKPOINT_PATHS[@]} * ${#MASK_THRESHOLDS[@]}))"
echo "================================================================================"
echo ""

# Track progress
TOTAL_COMBINATIONS=$((${#CHECKPOINT_PATHS[@]} * ${#MASK_THRESHOLDS[@]}))
CURRENT_COMBINATION=0

# Run for each checkpoint
for CHECKPOINT_PATH in "${CHECKPOINT_PATHS[@]}"; do
    CHECKPOINT_NAME=$(basename "$(dirname "${CHECKPOINT_PATH}")")

    echo ""
    echo "================================================================================"
    echo "Processing checkpoint: ${CHECKPOINT_NAME}"
    echo "  Path: ${CHECKPOINT_PATH}"
    echo "================================================================================"
    echo ""

    # Run for each mask threshold
    for MASK_THRESHOLD in "${MASK_THRESHOLDS[@]}"; do
        CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))

        echo ""
        echo "------------------------------------------------------------------------"
        echo "[${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] Checkpoint: ${CHECKPOINT_NAME}, Mask Threshold: ${MASK_THRESHOLD}"
        echo "------------------------------------------------------------------------"
        echo ""

        # Set environment variables and run the analysis script
        export SAE_PATH="${CHECKPOINT_PATH}"
        export MASK_THRESHOLD="${MASK_THRESHOLD}"
        export EXCLUDE_OVERLAPPING_FEATURES="${EXCLUDE_OVERLAPPING_FEATURES}"
        export NUM_GPUS="${NUM_GPUS}"  # Ensure NUM_GPUS is passed to the analysis script

        # Run the analysis script (it will use other defaults or environment variables)
        if ! bash "${ANALYSIS_SCRIPT}"; then
            echo "Error: Analysis failed for checkpoint ${CHECKPOINT_NAME} with MASK_THRESHOLD=${MASK_THRESHOLD}" >&2
            exit 1
        fi

        echo ""
        echo "✓ Completed: ${CHECKPOINT_NAME} @ mask_threshold=${MASK_THRESHOLD}"
        echo ""
    done

    echo ""
    echo "✓ Finished all mask thresholds for checkpoint: ${CHECKPOINT_NAME}"
    echo ""
done

echo ""
echo "================================================================================"
echo "Sweep complete! Results for all checkpoints and mask thresholds saved."
echo "  Processed ${#CHECKPOINT_PATHS[@]} checkpoint(s)"
echo "  Tested ${#MASK_THRESHOLDS[@]} mask threshold(s) per checkpoint"
echo "  Total combinations: ${TOTAL_COMBINATIONS}"
echo "================================================================================"

