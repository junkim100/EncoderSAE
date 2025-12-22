#!/usr/bin/env bash
set -euo pipefail

# Model and dataset configuration
MODEL="intfloat/multilingual-e5-large"
DATASET="data/4lang_train.jsonl"
VAL_DATASET="data/4lang_validation.jsonl"

# Training configuration
VAL_STEP=50                    # Run validation every N steps
BASE_BATCH_SIZE=294912        # Base batch size for expansion factor 32 (scales down for larger factors)
GRAD_ACC_STEPS=1              # Gradient accumulation steps
LOG_STEPS=5                    # Log metrics every N steps
CHECKPOINT_STEPS=50           # Save checkpoint every N steps
NUM_GPUS=2                     # Number of GPUs for distributed training
GPU_MEM_UTIL=0.95             # GPU memory utilization for vLLM (0.0-1.0)

# Model architecture
INPUT_DIM=1024                 # Input dimension (must match encoder model output)

# Hyperparameter sweep ranges
EXPANSION_FACTORS=(32 64 128)  # Dictionary size = INPUT_DIM * expansion_factor
SPARSITY_RATIOS=(0.005 0.01 0.02 0.03)  # Sparsity k = dict_size * ratio (rounded to nearest 1024)
LRS=(3e-4 5e-4)                # Learning rates to sweep
AUX_LOSS_COEFFS=(0.0 0.1 0.5 1.0)  # Auxiliary loss coefficients (0.0 = no aux loss)
AUX_LOSS_TARGETS=(0.02 0.05)   # Target activation fraction for auxiliary loss

####################################################################################################

for ef in "${EXPANSION_FACTORS[@]}"; do
  dict_size=$((INPUT_DIM * ef))

  if [ "${ef}" -eq 32 ]; then
    BATCH_SIZE=${BASE_BATCH_SIZE}
  elif [ "${ef}" -eq 64 ]; then
    BATCH_SIZE=$((BASE_BATCH_SIZE / 2))
  elif [ "${ef}" -eq 128 ]; then
    BATCH_SIZE=$((BASE_BATCH_SIZE / 4))
  elif [ "${ef}" -eq 256 ]; then
    BATCH_SIZE=$((BASE_BATCH_SIZE / 8))
  else
    BATCH_SIZE=$((BASE_BATCH_SIZE * 32 / ef))
  fi

  echo "Expansion factor: ${ef}, dict_size=${dict_size}, batch_size=${BATCH_SIZE}"

  declare -a SPARSITIES=()
  declare -A SEEN=()

  for ratio in "${SPARSITY_RATIOS[@]}"; do
    k_raw=$(python - <<EOF
dict_size = ${dict_size}
ratio = ${ratio}
print(int(dict_size * ratio))
EOF
)
    if [ "${k_raw}" -lt 1024 ]; then
      k_rounded=1024
    else
      k_rounded=$(( ( (k_raw + 1023) / 1024 ) * 1024 ))
    fi

    if [ "${k_rounded}" -gt "${dict_size}" ]; then
      k_rounded=${dict_size}
    fi

    if [ -z "${SEEN[${k_rounded}]+x}" ]; then
      SEEN[${k_rounded}]=1
      SPARSITIES+=("${k_rounded}")
    fi
  done

  echo "  sparsities=${SPARSITIES[*]}"

  for k in "${SPARSITIES[@]}"; do
    for lr in "${LRS[@]}"; do
      for aux_coeff in "${AUX_LOSS_COEFFS[@]}"; do
        if (( $(echo "${aux_coeff} == 0.0" | bc -l) )); then
          TARGETS_TO_TEST=(0.01)
        else
          TARGETS_TO_TEST=("${AUX_LOSS_TARGETS[@]}")
        fi

        for aux_target in "${TARGETS_TO_TEST[@]}"; do
          lr_tag=$(printf "%g" "$lr")
          model_short="${MODEL//\//_}"

          aux_tags=$(python - <<EOF
coeff = ${aux_coeff}
target = ${aux_target}

if coeff == 0.0:
    print("noaux")
else:
    if coeff >= 0.001:
        coeff_tag = f"{int(coeff * 1000)}e3"
    elif coeff >= 0.0001:
        coeff_tag = f"{int(coeff * 10000)}e4"
    else:
        coeff_tag = f"{coeff:g}"

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

          if [ "${NUM_GPUS}" -gt 1 ]; then
            uv run torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m EncoderSAE.main \
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
          else
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
          fi

          echo "Finished run: ${run_name}"
          echo
        done
      done
    done
  done
done
