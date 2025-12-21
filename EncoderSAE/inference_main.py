"""Main entry point for language-agnostic inference using fire."""

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

from .inference import (
    LanguageAgnosticEncoder,
    infer_language_agnostic,
    remove_language_features,
)
from .model import EncoderSAE
from .data import ActivationDataset, load_data


def from_activations(
    activations_path: str,
    sae_path: str,
    mask_path: str,
    output_path: str,
    batch_size: int = 32,
    device: Optional[str] = None,
):
    """
    Convert existing activations to language-agnostic embeddings.

    This mode loads pre-computed activations (base model embeddings), passes them through
    the SAE, applies the language mask, and saves the resulting language-agnostic embeddings.

    Args:
        activations_path: Path to .pt file containing activations (shape: [num_samples, input_dim])
        sae_path: Path to trained SAE checkpoint (.pt file)
        mask_path: Path to language mask file (.pt file)
            - Can be individual language mask: `language_features_{lang}_mask.pt`
            - Or combined union mask: `language_features_combined_mask.pt` (recommended)
        output_path: Path to save language-agnostic embeddings (.pt file)
        batch_size: Batch size for processing (default: 32)
        device: Device to run on, e.g., "cuda" or "cpu" (auto-detected if None)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load activations
    print(f"Loading activations from {activations_path}")
    activations_path_obj = Path(activations_path)
    if not activations_path_obj.exists():
        raise FileNotFoundError(f"Activations file not found: {activations_path}")

    activations = torch.load(activations_path, map_location=device)

    # Handle different activation formats
    if isinstance(activations, torch.Tensor):
        activations_tensor = activations
    elif isinstance(activations, list):
        activations_tensor = torch.stack(
            [
                a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                for a in activations
            ]
        )
    elif isinstance(activations, dict) and "activations" in activations:
        inner = activations["activations"]
        if isinstance(inner, torch.Tensor):
            activations_tensor = inner
        elif isinstance(inner, list):
            activations_tensor = torch.stack(
                [
                    a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                    for a in inner
                ]
            )
        else:
            raise ValueError(
                f"Unexpected 'activations' field format in {activations_path}"
            )
    else:
        raise ValueError(f"Unexpected activation format in {activations_path}")

    activations_tensor = activations_tensor.to(device)
    print(
        f"Loaded {len(activations_tensor)} activations of dimension {activations_tensor.shape[1]}"
    )

    # Load SAE
    print(f"Loading SAE from {sae_path}")
    checkpoint = torch.load(sae_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "encoder.weight" in checkpoint:
        state_dict = checkpoint
    else:
        raise ValueError(f"Could not parse checkpoint format from {sae_path}")

    encoder_weight = state_dict["encoder.weight"]
    input_dim = encoder_weight.shape[1]
    dict_size = encoder_weight.shape[0]
    expansion_factor = dict_size // input_dim
    sparsity = checkpoint.get("sparsity", 64) if isinstance(checkpoint, dict) else 64

    if input_dim != activations_tensor.shape[1]:
        raise ValueError(
            f"Input dimension mismatch: SAE expects {input_dim}, but activations have {activations_tensor.shape[1]}"
        )

    sae = EncoderSAE(
        input_dim=input_dim,
        expansion_factor=expansion_factor,
        sparsity=sparsity,
    )
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    print(f"SAE loaded: dict_size={sae.dict_size}, sparsity={sparsity}")

    # Load mask
    print(f"Loading mask from {mask_path}")
    mask = torch.load(mask_path, map_location=device)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.shape[0] != sae.dict_size:
        raise ValueError(
            f"Mask size {mask.shape[0]} does not match SAE dict_size {sae.dict_size}"
        )
    print(
        f"Mask loaded: {mask.sum().item()} language-specific features will be removed"
    )

    # Process activations in batches
    print("Processing activations through SAE and applying mask...")
    all_embeddings = []
    num_batches = (len(activations_tensor) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(activations_tensor))
            batch_activations = activations_tensor[start_idx:end_idx]

            # Pass through SAE
            _, features, _, _, _ = sae(batch_activations)

            # Remove language-specific features
            features_agnostic = remove_language_features(features, mask)

            all_embeddings.append(features_agnostic.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {end_idx}/{len(activations_tensor)} samples")

    # Stack all embeddings
    embeddings = torch.cat(all_embeddings, dim=0)
    print(
        f"Generated {len(embeddings)} language-agnostic embeddings of dimension {embeddings.shape[1]}"
    )

    # Save embeddings
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path_obj)
    print(f"Saved language-agnostic embeddings to {output_path_obj}")


def from_text(
    model_name: str,
    sae_path: str,
    mask_path: str,
    output_path: str,
    texts: Optional[list[str]] = None,
    text_file: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    use_vllm: bool = True,
    num_gpus: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
):
    """
    Generate language-agnostic embeddings from text input.

    This mode takes text input, encodes it with the base model, passes through SAE,
    applies the language mask, and saves the resulting language-agnostic embeddings.

    Args:
        model_name: HuggingFace model ID or local path (same as used for SAE training)
        sae_path: Path to trained SAE checkpoint (.pt file)
        mask_path: Path to language mask file (.pt file)
            - Can be individual language mask: `language_features_{lang}_mask.pt`
            - Or combined union mask: `language_features_combined_mask.pt` (recommended)
        output_path: Path to save language-agnostic embeddings (.pt file)
        texts: List of text strings to encode (if provided, text_file is ignored)
        text_file: Path to text file (one text per line) or JSON/JSONL file with text column
            - If .txt: one text per line
            - If .json/.jsonl: must have a "text" column
        batch_size: Batch size for processing (default: 32)
        max_length: Maximum sequence length (default: 512)
        device: Device to run on, e.g., "cuda" or "cpu" (auto-detected if None)
        use_vllm: Use vLLM for faster base model inference (default: True)
        num_gpus: Number of GPUs to use for vLLM (default: None = auto-detect)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)
    """
    # Set device
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"Using device: {device_obj}")

    # Load texts
    if texts is not None:
        input_texts = texts
        print(f"Using {len(input_texts)} texts from command line argument")
    elif text_file is not None:
        text_file_path = Path(text_file)
        if not text_file_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        # Handle different file formats
        if text_file_path.suffix == ".txt":
            # Plain text file: one text per line
            with open(text_file_path, "r", encoding="utf-8") as f:
                input_texts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(input_texts)} texts from {text_file_path}")
        elif text_file_path.suffix in [".json", ".jsonl"]:
            # JSON/JSONL file: use load_data utility
            input_texts = load_data(str(text_file_path), text_column="text")
            print(f"Loaded {len(input_texts)} texts from {text_file_path}")
        else:
            raise ValueError(
                f"Unsupported file format: {text_file_path.suffix}. "
                "Supported formats: .txt, .json, .jsonl"
            )
    else:
        raise ValueError("Either 'texts' or 'text_file' must be provided")

    if len(input_texts) == 0:
        raise ValueError("No texts to process")

    print(f"Processing {len(input_texts)} texts...")

    # Use the inference function to get language-agnostic embeddings
    embeddings = infer_language_agnostic(
        model_name=model_name,
        sae=sae_path,
        mask_path=mask_path,
        texts=input_texts,
        batch_size=batch_size,
        max_length=max_length,
        device=device_obj,
        use_vllm=use_vllm,
        num_gpus=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print(
        f"Generated {len(embeddings)} language-agnostic embeddings of dimension {embeddings.shape[1]}"
    )

    # Save embeddings
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path_obj)
    print(f"Saved language-agnostic embeddings to {output_path_obj}")


if __name__ == "__main__":
    fire.Fire(
        {
            "from_activations": from_activations,
            "from_text": from_text,
        }
    )
