"""Training loop for SAE."""

from pathlib import Path
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .model import EncoderSAE
from .data import ActivationDataset, load_data, extract_activations
from .utils import get_device, setup_wandb


def train_sae(
    model: EncoderSAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    num_gpus: Optional[int] = None,
    epochs: int = 10,
    lr: float = 3e-4,
    log_steps: int = 10,
    grad_acc_steps: int = 1,
    val_step: Optional[int] = 1000,
    save_dir: Optional[str] = None,
    checkpoint_steps: int = 1000,
    aux_loss_coeff: float = 0.0,
    aux_loss_target: float = 0.01,
) -> EncoderSAE:
    """
    Train the SAE model.

    Args:
        model: EncoderSAE model instance
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        device: Device to train on
        num_gpus: Number of GPUs to use for training (None = all visible CUDA devices)
        epochs: Number of training epochs
        lr: Learning rate
        log_steps: Log metrics every N steps
        grad_acc_steps: Gradient accumulation steps
        save_dir: Directory to save checkpoints
        checkpoint_steps: Save checkpoint every N training steps (default: 1000)

    Returns:
        Trained model
    """
    # Move model to target device
    model.to(device)

    # Configure how many GPUs to use for training.
    # - If num_gpus is None: use all visible CUDA devices (if >1) via DataParallel.
    # - If num_gpus is 1: train on a single GPU even if more are visible.
    # - If num_gpus > 1: use that many GPUs (capped by torch.cuda.device_count()) via DataParallel.
    if device.type == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            if num_gpus is None:
                n_gpus = available_gpus
            else:
                n_gpus = max(1, min(num_gpus, available_gpus))

            if n_gpus > 1:
                device_ids = list(range(n_gpus))
                model = torch.nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Helper to unwrap the underlying EncoderSAE (for saving/checkpointing)
    def unwrap(m: torch.nn.Module) -> torch.nn.Module:
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    global_step = 0

    def run_validation(current_epoch: int) -> None:
        """Run validation over the full val_loader and log metrics."""
        if val_loader is None:
            return

        model.eval()
        val_losses = []
        val_metrics = {
            "fvu": [],
            "dead_features": [],
            "l0_norm": [],
        }

            with torch.no_grad():
                for activations in val_loader:
                    activations = activations.to(device)
                    reconstructed, features, l0_norm, topk_mask = model(activations)
                    loss, metrics = unwrap(model).compute_loss(
                        activations, reconstructed, features, topk_mask,
                        aux_loss_coeff=aux_loss_coeff,
                        aux_loss_target=aux_loss_target,
                    )

                val_losses.append(metrics["loss"])
                val_metrics["fvu"].append(metrics["fvu"])
                val_metrics["dead_features"].append(metrics["dead_features"])
                # Handle DataParallel: l0_norm may be a tensor with one value per GPU
                if isinstance(l0_norm, torch.Tensor) and l0_norm.numel() > 1:
                    val_metrics["l0_norm"].append(l0_norm.mean().item())
                else:
                    val_metrics["l0_norm"].append(
                        l0_norm.item() if isinstance(l0_norm, torch.Tensor) else l0_norm
                    )

        if len(val_losses) == 0:
            return

        val_loss = sum(val_losses) / len(val_losses)
        val_fvu = sum(val_metrics["fvu"]) / len(val_metrics["fvu"])
        val_dead = sum(val_metrics["dead_features"]) / len(val_metrics["dead_features"])
        val_l0 = sum(val_metrics["l0_norm"]) / len(val_metrics["l0_norm"])

        wandb.log(
            {
                "val/loss": val_loss,
                "val/fvu": val_fvu,
                "val/dead_features": val_dead,
                "val/l0_norm": val_l0,
                "epoch": current_epoch + 1,
                "step": global_step,
            }
        )

        print(
            f"[val @ step {global_step}] Epoch {current_epoch+1} - "
            f"Val Loss: {val_loss:.4f}, Val FVU: {val_fvu:.4f}, Val DeadFrac: {val_dead:.4f}"
        )

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_metrics = {
            "fvu": [],
            "dead_features": [],
            "l0_norm": [],
        }

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, activations in enumerate(progress_bar):
            activations = activations.to(device)

            # Forward pass
            reconstructed, features, l0_norm, topk_mask = model(activations)

            # Compute loss (unwrap model if wrapped in DataParallel)
            loss, metrics = unwrap(model).compute_loss(
                activations, reconstructed, features, topk_mask,
                aux_loss_coeff=aux_loss_coeff,
                aux_loss_target=aux_loss_target,
            )

            # Scale loss for gradient accumulation
            loss = loss / grad_acc_steps

            # Backward pass
            loss.backward()

            # Update weights (every grad_acc_steps)
            if (batch_idx + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Accumulate metrics
            train_losses.append(metrics["loss"])
            train_metrics["fvu"].append(metrics["fvu"])
            train_metrics["dead_features"].append(metrics["dead_features"])
            # Handle DataParallel: l0_norm may be a tensor with one value per GPU
            if isinstance(l0_norm, torch.Tensor) and l0_norm.numel() > 1:
                train_metrics["l0_norm"].append(l0_norm.mean().item())
            else:
                train_metrics["l0_norm"].append(
                    l0_norm.item() if isinstance(l0_norm, torch.Tensor) else l0_norm
                )

            global_step += 1

            # Log metrics
            if global_step % log_steps == 0:
                avg_loss = sum(train_losses[-log_steps:]) / min(
                    log_steps, len(train_losses)
                )
                avg_fvu = sum(train_metrics["fvu"][-log_steps:]) / min(
                    log_steps, len(train_metrics["fvu"])
                )
                avg_dead = sum(train_metrics["dead_features"][-log_steps:]) / min(
                    log_steps, len(train_metrics["dead_features"])
                )
                avg_l0 = sum(train_metrics["l0_norm"][-log_steps:]) / min(
                    log_steps, len(train_metrics["l0_norm"])
                )

                log_dict = {
                    "train/loss": avg_loss,
                    "train/fvu": avg_fvu,
                    "train/dead_features": avg_dead,
                    "train/l0_norm": avg_l0,
                    "epoch": epoch + 1,
                    "step": global_step,
                }

                wandb.log(log_dict)
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "fvu": f"{avg_fvu:.4f}",
                        "dead_frac": f"{avg_dead:.4f}",
                    }
                )

            # Step-level checkpointing
            if (
                save_dir
                and checkpoint_steps > 0
                and global_step % checkpoint_steps == 0
            ):
                step_ckpt_path = Path(save_dir) / f"checkpoint_step_{global_step}.pt"
                step_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_state_dict": unwrap(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    step_ckpt_path,
                )

            # Step-level validation (if requested)
            if val_step is not None and val_step > 0 and global_step % val_step == 0:
                run_validation(epoch)

        # Epoch-level logging
        epoch_loss = sum(train_losses) / len(train_losses)
        epoch_fvu = sum(train_metrics["fvu"]) / len(train_metrics["fvu"])
        epoch_dead = sum(train_metrics["dead_features"]) / len(
            train_metrics["dead_features"]
        )
        epoch_l0 = sum(train_metrics["l0_norm"]) / len(train_metrics["l0_norm"])

        wandb.log(
            {
                "train/epoch_loss": epoch_loss,
                "train/epoch_fvu": epoch_fvu,
                "train/epoch_dead_features": epoch_dead,
                "train/epoch_l0_norm": epoch_l0,
                "epoch": epoch + 1,
            }
        )

        # End-of-epoch validation (if no val_step or you also want per-epoch eval)
        if val_loader is not None and (val_step is None or val_step <= 0):
            run_validation(epoch)

        # Save checkpoint
        if save_dir:
            save_path = Path(save_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": unwrap(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )

    # Always return the plain EncoderSAE instance (not wrapped in DataParallel)
    return unwrap(model)
