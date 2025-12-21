"""Training loop for SAE."""

from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
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
        aux_loss_coeff: Coefficient for the auxiliary loss
        aux_loss_target: Target for the auxiliary loss

    Returns:
        Trained model
    """
    # Detect whether we're running under torch.distributed (torchrun)
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    is_main_process = rank == 0

    # Move model to target device
    model.to(device)

    # Configure how many GPUs to use for training.
    # - If num_gpus is None: use all visible CUDA devices (if >1) via DataParallel.
    # - If num_gpus is 1: train on a single GPU even if more are visible.
    # - If num_gpus > 1: use that many GPUs (capped by torch.cuda.device_count()) via DataParallel.
    use_data_parallel = False
    if use_ddp:
        # DDP: one process per GPU. Model is replicated by the launcher (torchrun).
        if device.type == "cuda":
            if device.index is None:
                raise ValueError(
                    "DDP requires a CUDA device with an explicit index (e.g., cuda:0). "
                    "Make sure EncoderSAE.main sets torch.cuda.set_device(LOCAL_RANK)."
                )
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
            )
        else:
            model = DDP(model)
    elif device.type == "cuda":
        available_gpus = torch.cuda.device_count()
        if available_gpus > 1:
            if num_gpus is None:
                n_gpus = available_gpus
            else:
                n_gpus = max(1, min(num_gpus, available_gpus))

            if n_gpus > 1:
                device_ids = list(range(n_gpus))
                model = torch.nn.DataParallel(model, device_ids=device_ids)
                use_data_parallel = True

    # Compile model for faster execution (PyTorch 2.0+)
    # Note: Compile after DataParallel wrapping, or skip if using DataParallel
    # (DataParallel and torch.compile can have compatibility issues)
    if (
        hasattr(torch, "compile")
        and device.type == "cuda"
        and (not use_data_parallel)
        and (not use_ddp)
    ):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            if is_main_process:
                print("Model compiled with torch.compile() for faster training")
        except Exception as e:
            if is_main_process:
                print(
                    f"Warning: torch.compile() failed: {e}. Continuing without compilation."
                )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mixed precision training scaler
    scaler = GradScaler() if device.type == "cuda" else None

    # Helper to unwrap the underlying EncoderSAE (for saving/checkpointing)
    def unwrap(m: torch.nn.Module) -> torch.nn.Module:
        return m.module if isinstance(m, (torch.nn.DataParallel, DDP)) else m

    global_step = 0

    def run_validation(current_epoch: int) -> None:
        """Run validation over the full val_loader and log metrics."""
        if val_loader is None:
            return

        model.eval()
        # Accumulate weighted sums for distributed reduction.
        sum_loss = 0.0
        sum_mse = 0.0
        sum_fvu = 0.0
        sum_dead = 0.0
        sum_l0 = 0.0
        sum_aux = 0.0
        total_samples = 0

        with torch.no_grad():
            for activations in val_loader:
                activations = activations.to(device, non_blocking=True)
                reconstructed, features, l0_norm, topk_threshold, raw_features = model(
                    activations
                )
                loss, metrics = unwrap(model).compute_loss(
                    activations,
                    reconstructed,
                    features,
                    topk_threshold,
                    raw_features,
                    aux_loss_coeff=aux_loss_coeff,
                    aux_loss_target=aux_loss_target,
                )

                bsz = int(activations.shape[0])
                total_samples += bsz
                sum_loss += float(metrics["loss"]) * bsz
                sum_mse += float(metrics.get("mse_loss", metrics["loss"])) * bsz
                sum_fvu += float(metrics["fvu"]) * bsz
                sum_dead += float(metrics["dead_features"]) * bsz
                sum_aux += float(metrics["aux_loss"]) * bsz

                # l0_norm can be tensor; reduce to scalar
                if isinstance(l0_norm, torch.Tensor):
                    l0_val = (
                        l0_norm.mean().item() if l0_norm.numel() > 1 else l0_norm.item()
                    )
                else:
                    l0_val = float(l0_norm)
                sum_l0 += float(l0_val) * bsz

        if total_samples == 0:
            return

        if use_ddp:
            t = torch.tensor(
                [
                    sum_loss,
                    sum_mse,
                    sum_fvu,
                    sum_dead,
                    sum_l0,
                    sum_aux,
                    float(total_samples),
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            sum_loss, sum_mse, sum_fvu, sum_dead, sum_l0, sum_aux, total_samples = (
                t.tolist()
            )
            total_samples = int(total_samples)

        val_loss = sum_loss / total_samples
        val_mse = sum_mse / total_samples
        val_fvu = sum_fvu / total_samples
        val_dead = sum_dead / total_samples
        val_l0 = sum_l0 / total_samples
        val_aux = sum_aux / total_samples

        if is_main_process and wandb.run is not None:
            wandb.log(
                {
                    "val/loss": val_loss,
                    "val/mse_loss": val_mse,
                    "val/fvu": val_fvu,
                    "val/dead_features": val_dead,
                    "val/l0_norm": val_l0,
                    "val/aux_loss": val_aux,
                    "epoch": current_epoch + 1,
                    "step": global_step,
                }
            )

        if is_main_process:
            print(
                f"[val @ step {global_step}] Epoch {current_epoch+1} - "
                f"Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val FVU: {val_fvu:.4f}, Val DeadFrac: {val_dead:.4f}, Val AuxLoss: {val_aux:.6f}"
            )

    for epoch in range(epochs):
        # For distributed samplers, ensure a different (but deterministic) shuffle per epoch.
        if (
            use_ddp
            and hasattr(train_loader, "sampler")
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            try:
                train_loader.sampler.set_epoch(epoch)
            except Exception:
                pass

        model.train()
        train_losses = []
        train_metrics = {
            "fvu": [],
            "dead_features": [],
            "l0_norm": [],
            "aux_loss": [],
            "mse_loss": [],
        }

        progress_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            if is_main_process
            else train_loader
        )

        for batch_idx, activations in enumerate(progress_iter):
            activations = activations.to(device, non_blocking=True)

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    reconstructed, features, l0_norm, topk_threshold, raw_features = (
                        model(activations)
                    )
                    # Compute loss (unwrap model if wrapped in DataParallel)
                    loss, metrics = unwrap(model).compute_loss(
                        activations,
                        reconstructed,
                        features,
                        topk_threshold,
                        raw_features,
                        aux_loss_coeff=aux_loss_coeff,
                        aux_loss_target=aux_loss_target,
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / grad_acc_steps

                # Backward pass with scaler
                scaler.scale(loss).backward()

                # Update weights (every grad_acc_steps)
                if (batch_idx + 1) % grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # CPU training without mixed precision
                reconstructed, features, l0_norm, topk_threshold, raw_features = model(
                    activations
                )
                # Compute loss (unwrap model if wrapped in DataParallel)
                loss, metrics = unwrap(model).compute_loss(
                    activations,
                    reconstructed,
                    features,
                    topk_threshold,
                    raw_features,
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
            train_metrics["aux_loss"].append(metrics["aux_loss"])
            train_metrics["mse_loss"].append(metrics.get("mse_loss", metrics["loss"]))
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
                avg_aux = sum(train_metrics["aux_loss"][-log_steps:]) / min(
                    log_steps, len(train_metrics["aux_loss"])
                )
                avg_mse = sum(train_metrics["mse_loss"][-log_steps:]) / min(
                    log_steps, len(train_metrics["mse_loss"])
                )

                # Reduce metrics across ranks so logged values reflect the global average.
                if use_ddp:
                    t = torch.tensor(
                        [avg_loss, avg_mse, avg_fvu, avg_dead, avg_l0, avg_aux],
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    t /= world_size
                    avg_loss, avg_mse, avg_fvu, avg_dead, avg_l0, avg_aux = t.tolist()

                if is_main_process and wandb.run is not None:
                    log_dict = {
                        "train/loss": avg_loss,
                        "train/mse_loss": avg_mse,
                        "train/fvu": avg_fvu,
                        "train/dead_features": avg_dead,
                        "train/l0_norm": avg_l0,
                        "train/aux_loss": avg_aux,
                        "epoch": epoch + 1,
                        "step": global_step,
                    }
                    wandb.log(log_dict)

                if is_main_process:
                    progress_iter.set_postfix(
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
                if is_main_process:
                    step_ckpt_path = (
                        Path(save_dir) / f"checkpoint_step_{global_step}.pt"
                    )
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
        epoch_aux = sum(train_metrics["aux_loss"]) / len(train_metrics["aux_loss"])
        epoch_mse = sum(train_metrics["mse_loss"]) / len(train_metrics["mse_loss"])

        # End-of-epoch validation
        if val_loader is not None:
            # Run if no step-level validation requested, OR if the last step wasn't a validation step
            if (val_step is None or val_step <= 0) or (global_step % val_step != 0):
                run_validation(epoch)

        # Save checkpoint
        if save_dir:
            if is_main_process:
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
