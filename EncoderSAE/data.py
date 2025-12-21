"""Data loading, tokenization, and activation extraction with mean pooling."""

import os
import sys
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before ANY imports that might initialize CUDA or multiprocessing
# Set environment variable first (before multiprocessing module is fully initialized)
os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"

# Set the start method programmatically
# This must happen before any multiprocessing operations
try:
    # Get current method to log it
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        multiprocessing.set_start_method("spawn", force=True)
except RuntimeError as e:
    # Start method already set - check if it's spawn
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != "spawn":
        # If we can't set it, we'll fail later with a clearer error
        pass

import json
import sys
from pathlib import Path
from typing import Optional, Union
from multiprocessing import Process, Queue
import math

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


class ActivationDataset(Dataset):
    """Dataset for cached activations."""

    def __init__(self, activations_path: str):
        """
        Initialize dataset from cached activations.

        Args:
            activations_path: Path to a single .pt file containing all activations
        """
        activations_path = Path(activations_path)

        # Require a single combined .pt file
        if not (activations_path.is_file() and activations_path.suffix == ".pt"):
            raise ValueError(
                f"activations_path must be a .pt file containing all activations, got {activations_path}"
            )

        # Load combined file format: tensor of shape (num_samples, activation_dim)
        print(f"Loading activations from {activations_path} ...")
        activations = torch.load(activations_path)

        # Common case: already a single tensor
        if isinstance(activations, torch.Tensor):
            self.activations = activations
        else:
            # If it's a list or dict, convert to tensor with a progress bar
            if isinstance(activations, list):
                from tqdm import tqdm

                self.activations = torch.stack(
                    [
                        a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                        for a in tqdm(
                            activations,
                            desc="Stacking activations",
                            unit="samples",
                        )
                    ]
                )
            elif isinstance(activations, dict) and "activations" in activations:
                inner = activations["activations"]
                if isinstance(inner, torch.Tensor):
                    self.activations = inner
                elif isinstance(inner, list):
                    from tqdm import tqdm

                    self.activations = torch.stack(
                        [
                            a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                            for a in tqdm(
                                inner,
                                desc="Stacking activations",
                                unit="samples",
                            )
                        ]
                    )
                else:
                    raise ValueError(
                        f"Unexpected 'activations' field format in {activations_path}"
                    )
            else:
                raise ValueError(f"Unexpected format in {activations_path}")

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]


def load_data(
    dataset: str,
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> list[str]:
    """
    Load dataset from local file or HuggingFace datasets.

    Args:
        dataset: Path to local JSON/JSONL file or HuggingFace dataset ID
        text_column: Name of the text column in the dataset
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of text strings
    """
    # Check if it's a local file first
    dataset_path = dataset
    if os.path.exists(dataset_path):
        texts = []
        if dataset.endswith(".jsonl"):
            # JSONL: one JSON object per line
            print(f"Loading local JSONL dataset from {dataset} ...")

            # Count total lines for progress bar (unless max_samples limits it)
            total_lines = None
            if max_samples is None:
                # Count lines efficiently
                with open(dataset, "r", encoding="utf-8") as f:
                    total_lines = sum(1 for _ in f)
                print(f"Found {total_lines:,} lines")
            else:
                total_lines = max_samples

            # Load data with progress bar
            with open(dataset, "r", encoding="utf-8") as f:
                for line in tqdm(
                    f, desc="Loading data (JSONL)", total=total_lines, unit="lines"
                ):
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if isinstance(data, dict):
                        texts.append(data.get(text_column, ""))
                    elif isinstance(data, list):
                        texts.extend(
                            [
                                (
                                    item.get(text_column, "")
                                    if isinstance(item, dict)
                                    else str(item)
                                )
                                for item in data
                            ]
                        )
                    if max_samples and len(texts) >= max_samples:
                        break
        else:
            # JSON: single file
            with open(dataset, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [
                        (
                            item.get(text_column, "")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        for item in data
                    ]
                elif isinstance(data, dict):
                    texts = [data.get(text_column, "")]
                if max_samples:
                    texts = texts[:max_samples]
        return texts

    # Load from HuggingFace datasets (only if local file doesn't exist)
    try:
        hf_dataset = load_dataset(dataset, split="train")
        if max_samples:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        texts = [item[text_column] for item in hf_dataset]
        return texts
    except Exception as e:
        # Provide a clearer error message
        abs_path = os.path.abspath(dataset)
        raise ValueError(
            f"Failed to load dataset '{dataset}': "
            f"File not found locally (checked: {abs_path}) "
            f"and HuggingFace dataset loading failed: {e}"
        )


def mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply mean pooling across tokens, accounting for attention mask.

    Args:
        hidden_states: Token embeddings of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask of shape (batch_size, seq_len)

    Returns:
        Pooled embeddings of shape (batch_size, hidden_dim)
    """
    # Expand attention mask to match hidden_states dimensions
    attention_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    )

    # Sum hidden states, accounting for attention mask
    sum_hidden = (hidden_states * attention_mask_expanded).sum(dim=1)

    # Sum of attention mask (number of valid tokens per sample)
    sum_mask = attention_mask_expanded.sum(dim=1)

    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    # Mean pooling
    pooled = sum_hidden / sum_mask

    return pooled


def _extract_worker(
    model_name: str,
    texts_chunk: list[str],
    output_dir: str,
    start_idx: int,
    batch_size: int,
    max_length: int,
    gpu_id: int,
    progress_queue: Queue,
):
    """Worker function to extract activations on a specific GPU."""
    # Set the default CUDA device for this process
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_batches = (len(texts_chunk) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(texts_chunk))
            batch_texts = texts_chunk[batch_start:batch_end]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            # Mean pooling
            pooled = mean_pool(hidden_states, attention_mask)

            # Save activations
            for i, activation in enumerate(pooled):
                sample_idx = start_idx + batch_start + i
                save_path = output_path / f"activation_{sample_idx:06d}.pt"
                torch.save(activation.cpu(), save_path)

            # Report progress
            progress_queue.put(1)


def _extract_vllm_worker(
    model_name: str,
    dataset_path: str,
    text_column: str,
    shard_range: tuple[int, int],
    output_dir: str,
    batch_size: int,
    max_length: int,
    gpu_id: int,
    gpu_memory_utilization: float,
    progress_queue: Queue,
):
    """Worker function to extract activations using vLLM on a specific GPU."""
    import os
    import multiprocessing
    from pathlib import Path
    import torch

    # Set CUDA device for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set multiprocessing start method
    os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Patch multiprocessing.get_context to force spawn
    original_get_context = multiprocessing.get_context

    def force_spawn_context(method=None):
        return original_get_context("spawn")

    multiprocessing.get_context = force_spawn_context

    # Load dataset in this worker (on-demand loading)
    try:
        from datasets import load_dataset

        cache_dir = os.path.join("./cache", f"worker_gpu{gpu_id}")
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset(
            "json",
            data_files=dataset_path,
            split="train",
            cache_dir=cache_dir,
            verification_mode="no_checks",
        )
    except Exception as e:
        print(f"[GPU {gpu_id}] Dataset loading failed: {e}", flush=True)
        raise

    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("vLLM is not installed. Install it with: pip install vllm")

    # Create vLLM engine with tensor_parallel_size=1 (full model on this GPU)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,  # Full model on single GPU
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_length,
        task="embed",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect activations for this chunk
    chunk_activations = []
    start_idx, end_idx = shard_range
    total_samples = end_idx - start_idx
    current_idx = start_idx

    # Process in batches using on-demand dataset indexing
    while current_idx < end_idx:
        next_idx = min(current_idx + batch_size, end_idx)

        # Load batch from dataset on-demand
        batch_texts = ds[current_idx:next_idx][text_column]
        batch_len = len(batch_texts)

        if batch_len == 0:
            break

        try:
            # Use llm.encode() instead of llm.embed()
            outputs = llm.encode(batch_texts, pooling_task="embed", use_tqdm=False)

            # Extract embedding tensors with fallback logic
            for output in outputs:
                embedding_tensor = None

                # Try multiple extraction paths (fallback logic)
                if hasattr(output, "outputs"):
                    if hasattr(output.outputs, "embedding"):
                        embedding_data = output.outputs.embedding
                        embedding_tensor = (
                            torch.tensor(embedding_data, dtype=torch.float32)
                            if not isinstance(embedding_data, torch.Tensor)
                            else embedding_data.to(dtype=torch.float32)
                        )
                    elif hasattr(output.outputs, "data"):
                        embedding_data = output.outputs.data
                        embedding_tensor = (
                            torch.tensor(embedding_data, dtype=torch.float32)
                            if not isinstance(embedding_data, torch.Tensor)
                            else embedding_data.to(dtype=torch.float32)
                        )
                    elif hasattr(output.outputs, "__len__") and not isinstance(
                        output.outputs, str
                    ):
                        if len(output.outputs) > 0:
                            if hasattr(output.outputs[0], "embedding"):
                                embedding_data = output.outputs[0].embedding
                            elif hasattr(output.outputs[0], "data"):
                                embedding_data = output.outputs[0].data
                            else:
                                embedding_data = output.outputs[0]
                            embedding_tensor = (
                                torch.tensor(embedding_data, dtype=torch.float32)
                                if not isinstance(embedding_data, torch.Tensor)
                                else embedding_data.to(dtype=torch.float32)
                            )
                        else:
                            raise ValueError("EmbeddingRequestOutput.outputs is empty")
                elif hasattr(output, "embedding"):
                    embedding_data = output.embedding
                    embedding_tensor = (
                        torch.tensor(embedding_data, dtype=torch.float32)
                        if not isinstance(embedding_data, torch.Tensor)
                        else embedding_data.to(dtype=torch.float32)
                    )
                elif isinstance(output, torch.Tensor):
                    embedding_tensor = output.to(dtype=torch.float32)
                elif hasattr(output, "__array__"):
                    import numpy as np

                    embedding_tensor = torch.from_numpy(np.array(output)).to(
                        dtype=torch.float32
                    )
                else:
                    try:
                        embedding_tensor = torch.tensor(output, dtype=torch.float32)
                    except:
                        raise ValueError(
                            f"Cannot extract embedding from {type(output)}. "
                            f"Available attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}"
                        )

                if embedding_tensor is None:
                    raise ValueError(f"Failed to extract embedding from {type(output)}")

                # Ensure it's float32 and on CPU
                if not isinstance(embedding_tensor, torch.Tensor):
                    embedding_tensor = torch.tensor(
                        embedding_tensor, dtype=torch.float32
                    )

                if embedding_tensor.dtype != torch.float32:
                    embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                chunk_activations.append(embedding_tensor.cpu())

        except Exception as e:
            print(f"[GPU {gpu_id}] Error in batch {current_idx}: {e}", flush=True)
            import traceback

            traceback.print_exc()
            raise

        current_idx = next_idx

        # Report progress
        progress_queue.put(1)

    # Save this chunk's activations to a temporary file
    chunk_file = output_path / f"chunk_{gpu_id}.pt"
    if chunk_activations:
        chunk_tensor = torch.stack(chunk_activations)
        torch.save(chunk_tensor, chunk_file)

    return str(chunk_file)


def extract_activations(
    model_name: str,
    texts: list[str],
    output_dir: str,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    num_gpus: Optional[int] = None,
    use_vllm: bool = False,
    gpu_memory_utilization: float = 0.9,
    dataset_path: Optional[str] = None,
    text_column: str = "text",
) -> str:
    """
    Extract sentence-level embeddings from encoder model using mean pooling.
    Supports multi-GPU parallel processing and vLLM for faster inference.

    Process:
    1. Pass input text through the encoder
    2. Extract last layer hidden states
    3. Apply mean pooling across tokens (with attention mask) OR use vLLM's embed task
    4. Save all pooled vectors to a single combined .pt file

    Args:
        model_name: HuggingFace model ID or local path
        texts: List of input text strings
        output_dir: Directory to save the combined activation file
        batch_size: Batch size for processing
        max_length: Maximum sequence length for tokenization
        device: Device to run on (auto-detected if None, ignored if num_gpus > 1 or use_vllm)
        num_gpus: Number of GPUs to use (None = auto-detect all available). For vLLM, this sets tensor_parallel_size.
        use_vllm: If True, use vLLM with task="embed" for faster inference (default: False)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)

    Returns:
        Path to the combined .pt file containing all activations (shape: [num_samples, activation_dim])
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This MUST be done before any CUDA operations or vLLM initialization
    import multiprocessing
    import os

    # Set environment variable as a fallback
    os.environ.setdefault("PYTHON_MULTIPROCESSING_START_METHOD", "spawn")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set, which is fine
        pass

    # Determine number of GPUs to use
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Use vLLM path if requested
    if use_vllm:
        # Force set start method one more time before importing vLLM
        # This is critical because vLLM's multiprocessing executor needs spawn
        os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"
        try:
            # Try to set it again - sometimes it works even if set before
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set - verify it's spawn
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != "spawn":
                error_msg = (
                    f"ERROR: Multiprocessing start method is '{current_method}', not 'spawn'.\n"
                    f"Environment variable PYTHON_MULTIPROCESSING_START_METHOD={os.environ.get('PYTHON_MULTIPROCESSING_START_METHOD', 'not set')}\n"
                    "This will cause CUDA errors with vLLM.\n"
                    "SOLUTION: Use the wrapper script: python run_main.py (instead of uv run -m EncoderSAE.main)\n"
                    "Or ensure PYTHON_MULTIPROCESSING_START_METHOD=spawn is set before Python starts."
                )
                print(error_msg, file=sys.stderr)
                raise RuntimeError(
                    "Multiprocessing start method must be 'spawn' for CUDA compatibility. "
                    "Please use 'python run_main.py' instead of 'uv run -m EncoderSAE.main'"
                )

        # Patch multiprocessing.get_context to force spawn
        # vLLM's executor might create new contexts, so we need to ensure they all use spawn
        original_get_context = multiprocessing.get_context

        def force_spawn_context(method=None):
            # Always use spawn, regardless of what vLLM requests
            return original_get_context("spawn")

        multiprocessing.get_context = force_spawn_context

        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use data parallelism: multiple vLLM engines (one per GPU) instead of tensor parallelism
        if num_gpus > 1:
            print(
                f"Using vLLM with {num_gpus} GPUs in data-parallel mode (one engine per GPU)"
            )
            print(f"Loading model: {model_name}")

            # Determine dataset path and total length
            # Prefer dataset_path if provided (for on-demand loading), otherwise use texts length
            temp_dataset_file = None
            if dataset_path and os.path.exists(dataset_path):
                # Use on-demand dataset loading strategy
                from datasets import load_dataset

                print(f"Loading dataset metadata from {dataset_path}...")
                ds = load_dataset(
                    "json",
                    data_files=dataset_path,
                    split="train",
                    cache_dir="./cache",
                    verification_mode="no_checks",
                )
                total_len = len(ds)
                print(f"Total samples: {total_len}")
            else:
                # Fallback to pre-loaded texts (less efficient but works)
                total_len = len(texts)
                # Save texts to temp file for workers to load
                temp_dataset_file = output_path / "temp_texts.jsonl"
                print(
                    f"Creating temporary dataset file for workers: {temp_dataset_file}"
                )
                with open(temp_dataset_file, "w", encoding="utf-8") as f:
                    for text in texts:
                        f.write(json.dumps({text_column: text}) + "\n")
                dataset_path = str(temp_dataset_file)

            # Split dataset into shard ranges
            chunk_size = math.ceil(total_len / num_gpus)
            shard_ranges = []

            for i in range(num_gpus):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_len)
                if start_idx < end_idx:
                    shard_ranges.append((start_idx, end_idx))

            # Create progress queue and processes
            progress_queue = Queue()
            processes = []

            for gpu_id, shard_range in enumerate(shard_ranges):
                p = Process(
                    target=_extract_vllm_worker,
                    args=(
                        model_name,
                        dataset_path,
                        text_column,
                        shard_range,
                        output_dir,
                        batch_size,
                        max_length,
                        gpu_id,
                        gpu_memory_utilization,
                        progress_queue,
                    ),
                )
                p.start()
                processes.append(p)

            # Monitor progress
            total_batches = sum(
                (end - start + batch_size - 1) // batch_size
                for start, end in shard_ranges
            )
            completed = 0

            with tqdm(
                total=total_batches, desc="Activations (vLLM data-parallel)"
            ) as pbar:
                while completed < total_batches:
                    progress_queue.get()
                    completed += 1
                    pbar.update(1)

            # Wait for all processes to finish
            for p in processes:
                p.join()

            # Check for failures
            if any(p.exitcode != 0 for p in processes):
                raise RuntimeError("One or more worker processes failed.")

            # Load and combine all chunk files
            chunk_files = sorted(output_path.glob("chunk_*.pt"))
            if len(chunk_files) == 0:
                raise ValueError(f"No chunk files found in {output_path}")

            print("Merging chunks...", flush=True)
            all_chunk_activations = []
            for chunk_file in chunk_files:
                chunk_tensor = torch.load(chunk_file)
                all_chunk_activations.append(chunk_tensor)
                # Clean up chunk file
                chunk_file.unlink()

            # Concatenate all chunks in order
            activations_tensor = torch.cat(all_chunk_activations, dim=0)

            # Generate filename: {model_short}.pt
            model_short = (
                Path(model_name).stem
                if os.path.exists(model_name)
                else model_name.replace("/", "_")
            )
            output_file = output_path / f"{model_short}.pt"

            torch.save(activations_tensor, output_file)
            print(f"Saved {len(activations_tensor)} activations to {output_file}")

            # Clean up temp dataset file if we created one
            if temp_dataset_file is not None and temp_dataset_file.exists():
                temp_dataset_file.unlink()
                print(f"Cleaned up temporary dataset file: {temp_dataset_file}")

            return str(output_file)
        else:
            # Single GPU: use tensor_parallel_size=1 (same as data-parallel but simpler)
            print(f"Using vLLM with 1 GPU for activation extraction")
            print(f"Loading model: {model_name}")

        try:
            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_length,
                task="embed",
            )
        except RuntimeError as e:
            if (
                "multiprocessing" in str(e).lower()
                or "cuda" in str(e).lower()
                or "fork" in str(e).lower()
            ):
                raise RuntimeError(
                    f"vLLM initialization failed: {e}\n"
                    "This is likely a multiprocessing/CUDA issue. "
                    "SOLUTION: Use the wrapper script: python run_main.py (instead of uv run -m EncoderSAE.main)\n"
                    "Or ensure PYTHON_MULTIPROCESSING_START_METHOD=spawn is set before Python starts."
                ) from e
            raise

        print(f"Extracting activations for {len(texts)} samples using vLLM...")

        # Collect all activations in a list
        all_activations = []

        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(num_batches), desc="Activations (vLLM)", total=num_batches
        ):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Get embeddings from vLLM using encode() API
            # Use llm.encode() instead of llm.embed() for better performance
            outputs = llm.encode(batch_texts, pooling_task="embed", use_tqdm=False)

            # Extract embedding tensors with fallback logic
            for output in outputs:
                embedding_tensor = None

                # Try multiple extraction paths (fallback logic)
                if hasattr(output, "outputs"):
                    if hasattr(output.outputs, "embedding"):
                        embedding_data = output.outputs.embedding
                        embedding_tensor = (
                            torch.tensor(embedding_data, dtype=torch.float32)
                            if not isinstance(embedding_data, torch.Tensor)
                            else embedding_data.to(dtype=torch.float32)
                        )
                    elif hasattr(output.outputs, "data"):
                        embedding_data = output.outputs.data
                        embedding_tensor = (
                            torch.tensor(embedding_data, dtype=torch.float32)
                            if not isinstance(embedding_data, torch.Tensor)
                            else embedding_data.to(dtype=torch.float32)
                        )
                    elif hasattr(output.outputs, "__len__") and not isinstance(
                        output.outputs, str
                    ):
                        if len(output.outputs) > 0:
                            if hasattr(output.outputs[0], "embedding"):
                                embedding_data = output.outputs[0].embedding
                            elif hasattr(output.outputs[0], "data"):
                                embedding_data = output.outputs[0].data
                            else:
                                embedding_data = output.outputs[0]
                            embedding_tensor = (
                                torch.tensor(embedding_data, dtype=torch.float32)
                                if not isinstance(embedding_data, torch.Tensor)
                                else embedding_data.to(dtype=torch.float32)
                            )
                        else:
                            raise ValueError("EmbeddingRequestOutput.outputs is empty")
                elif hasattr(output, "embedding"):
                    embedding_data = output.embedding
                    embedding_tensor = (
                        torch.tensor(embedding_data, dtype=torch.float32)
                        if not isinstance(embedding_data, torch.Tensor)
                        else embedding_data.to(dtype=torch.float32)
                    )
                elif isinstance(output, torch.Tensor):
                    embedding_tensor = output.to(dtype=torch.float32)
                elif hasattr(output, "__array__"):
                    import numpy as np

                    embedding_tensor = torch.from_numpy(np.array(output)).to(
                        dtype=torch.float32
                    )
                else:
                    try:
                        embedding_tensor = torch.tensor(output, dtype=torch.float32)
                    except:
                        raise ValueError(
                            f"Cannot extract embedding from {type(output)}. "
                            f"Available attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}"
                        )

                    if embedding_tensor is None:
                        raise ValueError(
                            f"Failed to extract embedding from {type(output)}"
                        )

                    # Ensure it's float32 and on CPU
                    if not isinstance(embedding_tensor, torch.Tensor):
                        embedding_tensor = torch.tensor(
                            embedding_tensor, dtype=torch.float32
                        )

                    if embedding_tensor.dtype != torch.float32:
                        embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                    all_activations.append(embedding_tensor.cpu())

            # Stack all activations into a single tensor and save
            activations_tensor = torch.stack(all_activations)

            # Generate filename: {model_short}.pt
            model_short = (
                Path(model_name).stem
                if os.path.exists(model_name)
                else model_name.replace("/", "_")
            )
            output_file = output_path / f"{model_short}.pt"

            torch.save(activations_tensor, output_file)
            print(f"Saved {len(all_activations)} activations to {output_file}")
            return str(output_file)

    if num_gpus <= 1:
        # Single GPU path (original implementation)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        # Compile for faster inference
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect all activations in a list
        all_activations = []

        num_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"Extracting activations for {len(texts)} samples on {device}...")

        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches), desc="Activations", total=num_batches
            ):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                pooled = mean_pool(hidden_states, attention_mask)

                for activation in pooled:
                    all_activations.append(activation.cpu())

        # Stack all activations into a single tensor and save
        activations_tensor = torch.stack(all_activations)

        # Generate filename: {model_short}.pt
        model_short = (
            Path(model_name).stem
            if os.path.exists(model_name)
            else model_name.replace("/", "_")
        )
        output_file = output_path / f"{model_short}.pt"

        torch.save(activations_tensor, output_file)
        print(f"Saved {len(all_activations)} activations to {output_file}")
        return str(output_file)
    else:
        # Multi-GPU path
        print(f"Using {num_gpus} GPUs for activation extraction")
        print(f"Loading model: {model_name}")

        # Split texts across GPUs
        chunk_size = math.ceil(len(texts) / num_gpus)
        chunks = []
        start_indices = []

        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(texts))
            chunks.append(texts[start_idx:end_idx])
            start_indices.append(start_idx)

        # Create progress queue and processes
        progress_queue = Queue()
        processes = []

        for gpu_id in range(num_gpus):
            if len(chunks[gpu_id]) > 0:
                p = Process(
                    target=_extract_worker,
                    args=(
                        model_name,
                        chunks[gpu_id],
                        output_dir,
                        start_indices[gpu_id],
                        batch_size,
                        max_length,
                        gpu_id,
                        progress_queue,
                    ),
                )
                p.start()
                processes.append(p)

        # Monitor progress
        total_batches = sum(
            (len(chunk) + batch_size - 1) // batch_size for chunk in chunks
        )
        completed = 0

        with tqdm(total=total_batches, desc="Activations (multi-GPU)") as pbar:
            while completed < total_batches:
                progress_queue.get()
                completed += 1
                pbar.update(1)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Collect all individual activation files and combine into single file
        output_path = Path(output_dir)
        all_activation_files = sorted(output_path.glob("activation_*.pt"))

        if len(all_activation_files) == 0:
            raise ValueError(f"No activation files found in {output_dir}")

        # Load and stack all activations
        all_activations = []
        for activation_file in all_activation_files:
            activation = torch.load(activation_file)
            all_activations.append(activation)

        activations_tensor = torch.stack(all_activations)

        # Generate filename: {model_short}.pt
        model_short = (
            Path(model_name).stem
            if os.path.exists(model_name)
            else model_name.replace("/", "_")
        )
        output_file = output_path / f"{model_short}.pt"

        torch.save(activations_tensor, output_file)

        # Clean up individual files
        for activation_file in all_activation_files:
            activation_file.unlink()

        print(f"Saved {len(all_activations)} activations to {output_file}")
        return str(output_file)
