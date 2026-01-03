"""Standalone activation-cache creation entrypoint (Fire CLI).

This is intentionally a real module (not stdin / -c) so Python multiprocessing
with start_method='spawn' works reliably.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import fire

from .data import extract_activations, load_data


def main(
    model: str,
    dataset: str,
    text_column: str = "text",
    activations_dir: Optional[str] = None,
    activation_batch_size: int = 32,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    dataset_num_samples: Optional[int] = None,
    num_gpus: Optional[int] = None,
    use_vllm: bool = True,
    gpu_memory_utilization: float = 0.9,
    cache_dtype: str = "float16",
) -> str:
    """
    Create (or reuse) an activation cache for a dataset/model pair.

    Returns:
        Path to the activation cache file (.npy for vLLM, or .pt for legacy).
    """
    model_short = Path(model).stem if os.path.exists(model) else model.replace("/", "_")
    dataset_short = Path(dataset).stem if os.path.exists(dataset) else dataset.replace("/", "_")

    if activations_dir is None:
        activations_dir = f"./activations/{model_short}_{dataset_short}"

    out_dir = Path(activations_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer vLLM .npy cache, but fall back to .pt if present.
    npy_file = out_dir / f"{model_short}.npy"
    pt_file = out_dir / f"{model_short}.pt"
    if npy_file.exists():
        print(f"Using cached activations: {npy_file}", flush=True)
        return str(npy_file)
    if pt_file.exists():
        print(f"Using cached activations: {pt_file}", flush=True)
        return str(pt_file)

    # Avoid loading huge local JSONL into RAM: stream from disk when possible.
    if use_vllm and os.path.exists(dataset):
        texts: list[str] = []
        dataset_path = dataset
    else:
        texts = load_data(dataset, text_column=text_column, max_samples=max_samples)
        dataset_path = None

    print(
        f"Progress files: {out_dir / f'{model_short}.progress.json'} "
        f"(and {out_dir / f'{model_short}.error.txt'} on failure)",
        flush=True,
    )

    out_file = extract_activations(
        model_name=model,
        texts=texts,
        output_dir=str(out_dir),
        batch_size=int(activation_batch_size),
        max_length=int(max_length),
        device=None,
        num_gpus=num_gpus,
        use_vllm=use_vllm,
        gpu_memory_utilization=float(gpu_memory_utilization),
        dataset_path=dataset_path,
        text_column=text_column,
        max_samples=max_samples,
        total_samples=dataset_num_samples,
        cache_dtype=cache_dtype,
    )
    print(f"Wrote activations: {out_file}", flush=True)
    return str(out_file)


if __name__ == "__main__":
    fire.Fire(main)


