#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for EncoderSAE with auxiliary loss
# Focus: Address overfitting and dead features using aux_loss_coeff and aux_loss_target
#
# Strategy based on previous results:
# - lr=1e-4 had mixed results (good for exp32, bad for exp64 with high dead features/overfitting)
# - lr=5e-4 and lr=1e-3 performed better overall
# - Overfitting is a key concern, especially for larger expansion factors
# - Auxiliary loss should help reduce dead features and potentially improve generalization

# Fixed settings (edit these as needed)
MODEL="intfloat/multilingual-e5-large"
DATASET="data/4lang_train.jsonl"
VAL_DATASET="data/4lang_validation.jsonl"
VAL_STEP=100
BATCH_SIZE=32768
GRAD_ACC_STEPS=1
LOG_STEPS=10
CHECKPOINT_STEPS=100
NUM_GPUS=8
GPU_MEM_UTIL=0.95

# Encoder output dimension for intfloat/multilingual-e5-large
INPUT_DIM=1024

# Sweep ranges - Focused sweep to test auxiliary loss effectiveness
# Total runs: 2 exp_factors * 1 sparsity * 2 lrs * 3 aux_coeffs * 2 aux_targets = 24 runs

# Expansion factors to try
# Focus on 32 and 64 (skip 128 to reduce sweep size)
EXPANSION_FACTORS=(32 64)

# Target sparsity ratios (k / dict_size)
# Use balanced default ratio
SPARSITY_RATIOS=(0.01)

# Learning rates
# Focus on LRs that performed well: 5e-4 and 1e-3
LRS=(5e-4 1e-3)

# Auxiliary loss coefficient
# 0.0 = no aux loss (baseline to compare against)
# 1e-3 = moderate (default)
# 1e-2 = stronger (more aggressive dead feature reduction)
AUX_LOSS_COEFFS=(0.0 1e-3 1e-2)

# Auxiliary loss target (fraction of samples where each feature should activate)
# 0.01 = 1% (default, balanced)
# 0.02 = 2% (less aggressive, allows some features to be more specialized)
AUX_LOSS_TARGETS=(0.01 0.02)

for ef in "${EXPANSION_FACTORS[@]}"; do
  dict_size=$((INPUT_DIM * ef))

  # Build a list of sparsities (multiples of 1024) for this expansion factor
  declare -a SPARSITIES=()
  declare -A SEEN=()

  for ratio in "${SPARSITY_RATIOS[@]}"; do
    # Compute raw k = ratio * dict_size using python for floating point, then
    # round *up* to the nearest multiple of 1024 to avoid being sparser than requested.
    k_raw=$(python - <<EOF
dict_size = ${dict_size}
ratio = ${ratio}
print(int(dict_size * ratio))
EOF
)
    # Round up to nearest multiple of 1024
    if [ "${k_raw}" -lt 1024 ]; then
      k_rounded=1024
    else
      k_rounded=$(( ( (k_raw + 1023) / 1024 ) * 1024 ))
    fi

    # Cap at dict_size
    if [ "${k_rounded}" -gt "${dict_size}" ]; then
      k_rounded=${dict_size}
    fi

    # Deduplicate
    if [ -z "${SEEN[${k_rounded}]+x}" ]; then
      SEEN[${k_rounded}]=1
      SPARSITIES+=("${k_rounded}")
    fi
  done

  echo "Expansion factor: ${ef}, dict_size=${dict_size}, sparsities=${SPARSITIES[*]}"

  for k in "${SPARSITIES[@]}"; do
    for lr in "${LRS[@]}"; do
      for aux_coeff in "${AUX_LOSS_COEFFS[@]}"; do
        # When aux_loss_coeff is 0.0, aux_loss_target has no effect
        # So only test one target value (use 0.01) to avoid duplicate runs
        if (( $(echo "${aux_coeff} == 0.0" | bc -l) )); then
          TARGETS_TO_TEST=(0.01)
        else
          TARGETS_TO_TEST=("${AUX_LOSS_TARGETS[@]}")
        fi

        for aux_target in "${TARGETS_TO_TEST[@]}"; do
          # Derive a concise run name for WandB / logs
          lr_tag=$(printf "%g" "$lr")
          model_short="${MODEL//\//_}"

          # Build run name: model_exp{ef}_k{k}_lr{lr}_aux{coeff}_tgt{target}
          # Format aux_coeff and aux_target using Python for clean formatting
          aux_tags=$(python - <<EOF
coeff = ${aux_coeff}
target = ${aux_target}

if coeff == 0.0:
    print("noaux")
else:
    # Format coeff: 0.001 -> "1e3", 0.005 -> "5e3", 0.01 -> "1e2"
    if coeff >= 0.001:
        coeff_tag = f"{int(coeff * 1000)}e3"
    elif coeff >= 0.0001:
        coeff_tag = f"{int(coeff * 10000)}e4"
    else:
        coeff_tag = f"{coeff:g}"

    # Format target: 0.005 -> "5e3", 0.01 -> "1e2", 0.02 -> "2e2"
    if target >= 0.01:
        target_tag = f"{int(target * 100)}pct"
    else:
        target_tag = f"{int(target * 1000)}e3"

    print(f"aux{coeff_tag}_tgt{target_tag}")
EOF
)

          run_name="${model_short}_exp${ef}_k${k}_lr${lr_tag}_${aux_tags}"

          echo "==============================================="
          echo "Running EncoderSAE sweep:"
          echo "  exp_factor=${ef}, sparsity=${k}, lr=${lr}"
          echo "  aux_loss_coeff=${aux_coeff}, aux_loss_target=${aux_target}"
          echo "  Run name: ${run_name}"
          echo "==============================================="

          uv run -m EncoderSAE.main \
            --model="${MODEL}" \
            --dataset="${DATASET}" \
            --val_dataset="${VAL_DATASET}" \
            --val_step="${VAL_STEP}" \
            --expansion_factor="${ef}" \
            --sparsity="${k}" \
            --batch_size="${BATCH_SIZE}" \
            --grad_acc_steps="${GRAD_ACC_STEPS}" \
            --lr="${lr}" \
            --log_steps="${LOG_STEPS}" \
            --checkpoint_steps="${CHECKPOINT_STEPS}" \
            --num_gpus="${NUM_GPUS}" \
            --use_vllm=True \
            --gpu_memory_utilization="${GPU_MEM_UTIL}" \
            --aux_loss_coeff="${aux_coeff}" \
            --aux_loss_target="${aux_target}" \
            --wandb_run_name="${run_name}"

          echo "Finished run: ${run_name}"
          echo
        done
      done
    done
  done
done
