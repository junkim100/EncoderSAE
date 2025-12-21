"""Main entry point for training EncoderSAE using fire."""

# CRITICAL: Set multiprocessing start method BEFORE any other imports
# This must happen before torch, vllm, or any CUDA-related imports
import os

os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"

import multiprocessing

# Set the start method programmatically - must be before any multiprocessing operations
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set - verify it's spawn
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        import sys

        print(
            f"WARNING: Multiprocessing start method is '{current_method}', not 'spawn'.\n"
            "This may cause CUDA errors with vLLM. Please use 'python run_main.py' instead.",
            file=sys.stderr,
        )

from pathlib import Path
from typing import Optional

import fire
import torch
from torch.utils.data import DataLoader, random_split

from .model import EncoderSAE
from .data import ActivationDataset, load_data, extract_activations
from .train import train_sae
from .utils import set_seed, setup_wandb, get_device
import wandb


def main(
    model: str = "intfloat/multilingual-e5-large",
    dataset: str = "enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5",
    text_column: str = "text",
    expansion_factor: int = 32,
    sparsity: int = 64,
    batch_size: int = 32,
    activation_batch_size: Optional[int] = None,
    grad_acc_steps: int = 1,
    epochs: int = 1,
    lr: float = 3e-4,
    log_steps: int = 10,
    checkpoint_steps: int = 1000,
    val_step: Optional[int] = 1000,
    aux_loss_coeff: float = 1e-3,
    aux_loss_target: float = 0.01,
    val_set: Optional[str] = None,
    val_dataset: Optional[str] = None,
    val_split: float = 0.05,
    save_activations: bool = True,
    activations_dir: Optional[str] = None,
    seed: int = 42,
    wandb_project: str = "EncoderSAE",
    wandb_run_name: Optional[str] = None,
    save_dir: Optional[str] = None,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    num_gpus: Optional[int] = None,
    use_vllm: bool = True,
    gpu_memory_utilization: float = 0.9,
):
    """
    Train a Sparse Autoencoder on encoder-only model activations.

    Args:
        model: HuggingFace model ID or local path (default: "intfloat/multilingual-e5-large")
        dataset: HuggingFace dataset ID or local JSON/JSONL file path
        text_column: Name of text column in dataset
        expansion_factor: SAE expansion factor (default: 32)
        sparsity: Top-k sparsity (default: 64)
        batch_size: SAE training batch size (default: 32)
        activation_batch_size: Batch size for activation / embedding extraction
            (default: None = fall back to `batch_size`)
        grad_acc_steps: Gradient accumulation steps (default: 1)
        epochs: Number of training epochs (default: 1)
        lr: Learning rate (default: 3e-4)
        log_steps: Log metrics every N steps (default: 10)
        checkpoint_steps: Save checkpoints every N training steps (default: 1000)
        val_step: Run validation every N training steps (default: 1000). Set to None or <=0 for end-of-epoch only.
        aux_loss_coeff: Coefficient for auxiliary loss that encourages feature usage (default: 1e-3).
            Set to 0.0 to disable. Higher values more aggressively reduce dead features.
        aux_loss_target: Target fraction of samples where each feature should activate (default: 0.01).
            This is a fraction between 0 and 1 (e.g., 0.01 = 1% of samples).
            Features that activate in fewer samples than this target are penalized.
        val_set: Path to validation activations directory (if precomputed). Ignored if
            val_dataset is provided. If both are None, auto-split from train.
        val_dataset: HuggingFace dataset ID or local JSON/JSONL file for validation.
            If provided, a separate validation activation set is extracted and
            val_split is ignored.
        val_split: Fraction of train set to use for validation when val_dataset and
            val_set are None (default: 0.05)
        save_activations: Whether to cache activations to disk (default: True)
        activations_dir: Directory to save/load activations (auto-generated if None)
        seed: Random seed (default: 42)
        wandb_project: WandB project name (default: "encodersae")
        wandb_run_name: WandB run name (auto-generated if None)
        save_dir: Directory to save model checkpoints
        max_length: Maximum sequence length for tokenization (default: 512)
        max_samples: Maximum number of samples to use (None = all)
        num_gpus: Number of GPUs to use for activation extraction (None = auto-detect all available GPUs). Use 8 to utilize all 8 GPUs on your node.
        use_vllm: Use vLLM for faster activation extraction with task="embed" (default: True)
        gpu_memory_utilization: GPU memory utilization for vLLM, 0.0-1.0 (default: 0.9)
    """
    # Set seed
    set_seed(seed)

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Determine model/dataset short names
    model_short = Path(model).stem if os.path.exists(model) else model.replace("/", "_")
    dataset_short = (
        Path(dataset).stem if os.path.exists(dataset) else dataset.replace("/", "_")
    )

    # Determine activations file path (always set, but overridable)
    if activations_dir is None:
        activations_dir = f"./activations/{model_short}_{dataset_short}"

    # Create directory if it doesn't exist
    activations_dir_path = Path(activations_dir)
    activations_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate activation file path: {model_short}.pt
    activations_file = activations_dir_path / f"{model_short}.pt"

    # Determine batch size for activation extraction (can differ from training batch size)
    activation_bs = activation_batch_size or batch_size

    # Check if activations already exist
    if save_activations and activations_file.exists():
        print(f"Loading cached activations from {activations_file}")
        train_dataset = ActivationDataset(str(activations_file))
    else:
        # Load data
        print(f"Loading dataset: {dataset}")
        texts = load_data(dataset, text_column=text_column, max_samples=max_samples)
        print(f"Loaded {len(texts)} samples")

        # Extract activations
        if save_activations:
            print("Extracting activations...")
            # For vLLM multi-GPU, pass dataset path for on-demand loading
            # For other paths, pass texts (already loaded)
            dataset_path_for_extraction = (
                dataset
                if (use_vllm and num_gpus and num_gpus > 1 and os.path.exists(dataset))
                else None
            )
            activations_file = Path(
                extract_activations(
                    model_name=model,
                    texts=texts,
                    output_dir=str(activations_dir_path),
                    batch_size=activation_bs,
                    max_length=max_length,
                    device=device,
                    num_gpus=num_gpus,
                    use_vllm=use_vllm,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dataset_path=dataset_path_for_extraction,
                    text_column=text_column,
                )
            )
            train_dataset = ActivationDataset(str(activations_file))
        else:
            # Extract on-the-fly (not recommended for large datasets)
            raise NotImplementedError(
                "On-the-fly activation extraction not implemented. Use save_activations=True."
            )

    # Prepare validation dataset (optional separate dataset)
    val_dataset_obj = None
    if val_dataset is not None:
        # Determine validation activations directory
        val_model_short = model_short
        val_dataset_short = (
            Path(val_dataset).stem
            if os.path.exists(val_dataset)
            else val_dataset.replace("/", "_")
        )
        # Use dataset-based suffix only; do not append extra "_val"
        val_activations_dir = f"./activations/{val_model_short}_{val_dataset_short}"
        val_activations_dir_path = Path(val_activations_dir)
        val_activations_dir_path.mkdir(parents=True, exist_ok=True)

        # Generate validation activation file path: {model_short}.pt
        val_activations_file = val_activations_dir_path / f"{val_model_short}.pt"

        if save_activations and val_activations_file.exists():
            print(f"Loading cached validation activations from {val_activations_file}")
            val_dataset_obj = ActivationDataset(str(val_activations_file))
        else:
            print(f"Loading validation dataset: {val_dataset}")
            val_texts = load_data(
                val_dataset, text_column=text_column, max_samples=max_samples
            )
            print(f"Loaded {len(val_texts)} validation samples")

            if save_activations:
                print("Extracting validation activations...")
                # For vLLM multi-GPU, pass dataset path for on-demand loading
                val_dataset_path_for_extraction = (
                    val_dataset
                    if (
                        use_vllm
                        and num_gpus
                        and num_gpus > 1
                        and os.path.exists(val_dataset)
                    )
                    else None
                )
                val_activations_file = Path(
                    extract_activations(
                        model_name=model,
                        texts=val_texts,
                        output_dir=str(val_activations_dir_path),
                        batch_size=activation_bs,
                        max_length=max_length,
                        device=device,
                        num_gpus=num_gpus,
                        use_vllm=use_vllm,
                        gpu_memory_utilization=gpu_memory_utilization,
                        dataset_path=val_dataset_path_for_extraction,
                        text_column=text_column,
                    )
                )
                val_dataset_obj = ActivationDataset(str(val_activations_file))
            else:
                raise NotImplementedError(
                    "On-the-fly validation activation extraction not implemented. "
                    "Use save_activations=True."
                )

    # Split train/val
    if val_dataset_obj is not None:
        val_dataset = val_dataset_obj
    elif val_set is None:
        # Auto-split
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"Split dataset: {train_size} train, {val_size} val")
    else:
        # Load separate validation set from precomputed activations.
        # val_set can be either:
        # - a direct path to a combined .pt file, or
        # - a directory containing a single combined .pt file.
        val_path = Path(val_set)
        if val_path.is_dir():
            pt_files = sorted(val_path.glob("*.pt"))
            if len(pt_files) == 0:
                raise ValueError(
                    f"No .pt files found in validation directory: {val_path}"
                )
            if len(pt_files) > 1:
                raise ValueError(
                    f"Multiple .pt files found in validation directory {val_path}; "
                    f"please specify one explicitly with --val_set."
                )
            val_path = pt_files[0]

        val_dataset = ActivationDataset(str(val_path))
        print(
            f"Using separate validation set from {val_path}: {len(val_dataset)} samples"
        )

    # Determine save directory (always set, but overridable)
    if save_dir is None:
        # Create parent directory: checkpoints/{model_short}_{dataset_short}
        parent_dir = f"./checkpoints/{model_short}_{dataset_short}"
        # Create hyperparameter subfolder: exp{expansion_factor}_k{sparsity}_lr{lr_tag}_aux{coeff}_tgt{target}
        # Format lr using scientific notation: 0.001 -> "1e-3", 0.0005 -> "5e-4", 0.0001 -> "1e-4"
        lr_str = f"{lr:.0e}"
        if "e" in lr_str:
            mantissa, exp = lr_str.split("e")
            # Normalize mantissa: remove decimal point, strip leading zeros
            mantissa_clean = mantissa.replace(".", "").lstrip("0")
            if not mantissa_clean:
                mantissa_clean = "1"
            lr_tag = f"{mantissa_clean}e{exp}"
        else:
            lr_tag = f"{lr:g}"

        # Format aux_loss_coeff and aux_loss_target for folder name
        if aux_loss_coeff == 0.0:
            aux_tag = "noaux"
        else:
            # Format coeff using scientific notation: 0.001 -> "1e-3", 0.005 -> "5e-3", 0.01 -> "1e-2"
            # Convert to proper scientific notation (mantissa between 1 and 10)
            coeff_str = f"{aux_loss_coeff:.0e}"
            if "e" in coeff_str:
                mantissa, exp = coeff_str.split("e")
                # Normalize mantissa: remove decimal point, strip leading zeros
                mantissa_clean = mantissa.replace(".", "").lstrip("0")
                if not mantissa_clean:
                    mantissa_clean = "1"
                coeff_tag = f"{mantissa_clean}e{exp}"
            else:
                coeff_tag = f"{aux_loss_coeff:g}"

            # Format target using scientific notation: 0.005 -> "5e-3", 0.01 -> "1e-2", 0.02 -> "2e-2"
            target_str = f"{aux_loss_target:.0e}"
            if "e" in target_str:
                mantissa, exp = target_str.split("e")
                # Normalize mantissa: remove decimal point, strip leading zeros
                mantissa_clean = mantissa.replace(".", "").lstrip("0")
                if not mantissa_clean:
                    mantissa_clean = "1"
                target_tag = f"{mantissa_clean}e{exp}"
            else:
                target_tag = f"{aux_loss_target:g}"

            aux_tag = f"aux{coeff_tag}_tgt{target_tag}"

        hyperparam_subdir = f"exp{expansion_factor}_k{sparsity}_lr{lr_tag}_{aux_tag}"
        save_dir = f"{parent_dir}/{hyperparam_subdir}"

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid issues with .pt files
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Get input dimension from first sample
    sample_activation = train_dataset[0]
    input_dim = sample_activation.shape[0]
    print(f"Input dimension: {input_dim}")

    # Initialize model
    sae = EncoderSAE(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        sparsity=sparsity,
    )
    print(f"SAE initialized: dict_size={sae.dict_size}, sparsity={sparsity}")

    # Setup WandB
    run = setup_wandb(
        project=wandb_project,
        run_name=wandb_run_name,
        model_name=model,
        dataset_name=dataset,
        expansion_factor=expansion_factor,
        sparsity=sparsity,
        batch_size=batch_size,
        config={
            "input_dim": input_dim,
            "dict_size": sae.dict_size,
            "epochs": epochs,
            "lr": lr,
            "grad_acc_steps": grad_acc_steps,
            "seed": seed,
            "aux_loss_coeff": aux_loss_coeff,
            "aux_loss_target": aux_loss_target,
        },
    )

    # Train
    print("Starting training...")
    trained_model = train_sae(
        model=sae,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_gpus=num_gpus,
        epochs=epochs,
        lr=lr,
        log_steps=log_steps,
        grad_acc_steps=grad_acc_steps,
        save_dir=save_dir,
        checkpoint_steps=checkpoint_steps,
        val_step=val_step,
        aux_loss_coeff=aux_loss_coeff,
        aux_loss_target=aux_loss_target,
    )

    # Save final model
    if save_dir:
        final_path = Path(save_dir) / "final_model.pt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    fire.Fire(main)
