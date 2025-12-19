"""Utility functions for WandB setup, seeding, and helpers."""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_wandb(
    project: str = "encodersae",
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    expansion_factor: int = 32,
    sparsity: int = 64,
    batch_size: int = 32,
    config: Optional[dict] = None,
) -> wandb.Run:
    """
    Initialize WandB run with auto-generated run name if not provided.

    Args:
        project: WandB project name
        run_name: Custom run name (auto-generated if None)
        model_name: Model name for auto-generation
        dataset_name: Dataset name for auto-generation
        expansion_factor: Expansion factor for auto-generation
        sparsity: Sparsity (k) for auto-generation
        batch_size: Batch size for auto-generation
        config: Additional config dictionary

    Returns:
        Initialized WandB run
    """
    if run_name is None:
        # Auto-generate run name
        model_short = Path(model_name or "model").stem if model_name else "model"
        dataset_short = Path(dataset_name or "dataset").stem if dataset_name else "dataset"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_short}_{dataset_short}_exp{expansion_factor}_k{sparsity}_bs{batch_size}_{timestamp}"

    # Merge configs
    wandb_config = {
        "model": model_name,
        "dataset": dataset_name,
        "expansion_factor": expansion_factor,
        "sparsity": sparsity,
        "batch_size": batch_size,
    }
    if config:
        wandb_config.update(config)

    run = wandb.init(
        project=project,
        name=run_name,
        config=wandb_config,
    )

    return run


def get_device() -> torch.device:
    """Auto-detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

