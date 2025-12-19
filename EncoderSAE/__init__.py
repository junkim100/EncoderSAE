"""EncoderSAE: Sparse Autoencoder for Encoder-only Models."""

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before any imports that might initialize CUDA or multiprocessing
import os

os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"

import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Start method already set - that's fine
    pass

__version__ = "0.1.0"

from .model import EncoderSAE
from .data import ActivationDataset, load_data, extract_activations
from .train import train_sae
from .utils import setup_wandb, set_seed
from .analyze import analyze_language_features

__all__ = [
    "EncoderSAE",
    "ActivationDataset",
    "load_data",
    "extract_activations",
    "train_sae",
    "setup_wandb",
    "set_seed",
    "analyze_language_features",
]
