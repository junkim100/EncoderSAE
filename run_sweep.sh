#!/usr/bin/env bash
set -euo pipefail

# Model and dataset configuration
MODEL="intfloat/multilingual-e5-large"
DATASET="data/train.jsonl"
VAL_DATASET="data/val.jsonl"
# Optional: provide known dataset sizes to avoid scanning huge JSONL files.
# (These are used to pre-allocate the streaming .npy activation cache.)
DATASET_NUM_SAMPLES=124467499
VAL_DATASET_NUM_SAMPLES=31116878

# Training configuration
VAL_STEP=50                   # Run validation every N steps
BASE_BATCH_SIZE=585728        # Base batch size for expansion factor 32 (scales down for larger factors). Multiple of 13 for balanced batch for all 13 languages.
                              # Increased from 292864 to better utilize GPU memory (was at ~38.7% usage)
GRAD_ACC_STEPS=1              # Gradient accumulation steps
LOG_STEPS=5                   # Log metrics every N steps
CHECKPOINT_STEPS=50           # Save checkpoint every N steps
NUM_GPUS=8                    # Number of GPUs for distributed training
GPU_MEM_UTIL=0.95             # GPU memory utilization for vLLM (0.0-1.0)
# Activation extraction configuration (vLLM encode batch size).
# Tune this to drive GPU utilization during activation creation without affecting SAE training batch size.
ACTIVATION_BATCH_SIZE=262144

# Model architecture
INPUT_DIM=1024                 # Input dimension (must match encoder model output)

# Hyperparameter sweep ranges
EXPANSION_FACTORS=(128)            # Dictionary size = INPUT_DIM * expansion_factor (only for expansion factor 128 and 256)
SPARSITY_RATIOS=(0.015 0.02 0.025 0.03)   # Sparsity k = dict_size * ratio (rounded to nearest 1024) (only for expansion factor 128 and 256)
LRS=(3e-4 5e-4)                           # Learning rates to sweep (only for expansion factor 128 and 256)
AUX_LOSS_COEFFS=(0.1 0.3 0.5)             # Auxiliary loss coefficients (0.0 = no aux loss) (only for expansion factor 128 and 256)
AUX_LOSS_TARGETS=(0.02)                   # Target activation fraction for auxiliary loss (only for expansion factor 128 and 256)

####################################################################################################

for ef in "${EXPANSION_FACTORS[@]}"; do
  dict_size=$((INPUT_DIM * ef))

  if [ "${ef}" -le 32 ]; then
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
          # Formatting for run/checkpoint naming as requested
          # Expansion factor: exp32
          # k as is, e.g., k1024
          # lr in scientific (i.e., 0.001 -> 1e-03, 0.5 -> 5e-01)
          # aux/tgt if applicable, otherwise noaux

          lr_clean=$(python -c "print(f'{float(${lr}):.0e}' if abs(float(${lr})) < 1e-2 or abs(float(${lr})) > 1 else f'{float(${lr}):.2g}')" | sed 's/+0/+/; s/-0/-/')
          # But ensure e-03 form for things like 0.001, not 1e-03, not 1e-3. Use three digits in the exponent
          lr_tag=$(python -c "print('{0:.0e}'.format(float(${lr})).replace('e+0', 'e+').replace('e-0', 'e-'))")

          if (( $(echo "${aux_coeff} == 0.0" | bc -l) )); then
            aux_tag="noaux"
            run_name="exp${ef}_k${k}_lr${lr_tag}_${aux_tag}"
          else
            # Format aux_coeff and aux_target: 0.02 -> 1e-02, 0.5 -> 5e-01, 1.0 -> 1e+00
            aux_coeff_tag=$(python -c "print('{:.0e}'.format(float(${aux_coeff})).replace('e+0', 'e+').replace('e-0', 'e-'))")
            aux_target_tag=$(python -c "print('{:.0e}'.format(float(${aux_target})).replace('e+0', 'e+').replace('e-0', 'e-'))")
            run_name="exp${ef}_k${k}_lr${lr_tag}_aux${aux_coeff_tag}_tgt${aux_target_tag}"
          fi

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
              --dataset_num_samples="${DATASET_NUM_SAMPLES}" \
              --val_dataset="${VAL_DATASET}" \
              --val_dataset_num_samples="${VAL_DATASET_NUM_SAMPLES}" \
              --val_step="${VAL_STEP}" \
              --expansion_factor="${ef}" \
              --sparsity="${k}" \
              --batch_size="${BATCH_SIZE}" \
              --activation_batch_size="${ACTIVATION_BATCH_SIZE}" \
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
              --dataset_num_samples="${DATASET_NUM_SAMPLES}" \
              --val_dataset="${VAL_DATASET}" \
              --val_dataset_num_samples="${VAL_DATASET_NUM_SAMPLES}" \
              --val_step="${VAL_STEP}" \
              --expansion_factor="${ef}" \
              --sparsity="${k}" \
              --batch_size="${BATCH_SIZE}" \
              --activation_batch_size="${ACTIVATION_BATCH_SIZE}" \
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
