"""Data loading, tokenization, and activation extraction with mean pooling."""

import os
import sys
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before ANY imports that might initialize CUDA or multiprocessing
# Set environment variable first (before multiprocessing module is fully initialized)
os.environ["PYTHON_MULTIPROCESSING_START_METHOD"] = "spawn"

# CRITICAL: Force vLLM to use localhost for its internal distributed rendezvous.
#
# vLLM (even for single-GPU engines) initializes a tiny torch.distributed group
# and computes a `distributed_init_method` as:
#   tcp://{get_ip()}:{get_open_port()}
#
# In some cluster / container setups, vLLM's auto-detected IP (from routing to
# 8.8.8.8) is *not bindable* inside the job, which causes TCPStore connect
# timeouts like:
#   torch.distributed.DistNetworkError: client socket timed out ...
#
# Setting `VLLM_HOST_IP=127.0.0.1` forces vLLM to use loopback and avoids those
# timeouts. Users can override this externally if they truly need a different IP.
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("VLLM_LOOPBACK_IP", "127.0.0.1")

# CRITICAL: vLLM V1 defaults to a separate EngineCore subprocess. In torchrun/DDP
# jobs this extra subprocess can occasionally hang during its internal
# torch.distributed TCPStore rendezvous (even for world_size=1), preventing
# activation files from being written. For offline embedding extraction, running
# vLLM in-process is far more reliable.
#
# Users can override externally (set to "1") if they know their environment is
# stable with EngineCore multiprocessing.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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
import shutil
from pathlib import Path
from typing import Optional, Union
from multiprocessing import Process, Queue, Manager
import math
import threading

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


def _cuda_visible_device_for_local_rank(
    local_rank: int, cuda_visible_devices: Optional[str]
) -> str:
    """
    Map a *local* CUDA index (0..N-1 within the current process) to the token
    that should be used for a single-GPU `CUDA_VISIBLE_DEVICES` value.

    Why: on clusters, schedulers often set `CUDA_VISIBLE_DEVICES` to a subset
    like "1,4". In that case, torch device 0 actually refers to physical GPU 1.
    If we naively set `CUDA_VISIBLE_DEVICES=str(local_rank)`, we'd accidentally
    select the wrong physical GPU (e.g. "0" -> physical GPU 0).
    """
    if local_rank < 0:
        raise ValueError(f"local_rank must be >= 0, got {local_rank}")

    if cuda_visible_devices is None or str(cuda_visible_devices).strip() == "":
        # No restriction in the parent -> local rank equals the physical index.
        return str(local_rank)

    parts = [p.strip() for p in str(cuda_visible_devices).split(",") if p.strip()]
    if local_rank >= len(parts):
        raise ValueError(
            f"local_rank={local_rank} is out of range for CUDA_VISIBLE_DEVICES={cuda_visible_devices}"
        )
    return parts[local_rank]


class ActivationDataset(Dataset):
    """Dataset for cached activations."""

    def __init__(self, activations_path: str):
        """
        Initialize dataset from cached activations.

        Args:
            activations_path: Path to a single activation cache file containing all
                activations (.pt or .npy).
        """
        activations_path = Path(activations_path)
        self.activations_path = activations_path
        self.activations: Optional[torch.Tensor] = None
        self._npy_activations = None  # numpy memmap for .npy caches
        self._length: Optional[int] = None
        self._cache_format: str = activations_path.suffix.lstrip(".")

        # Supported cache formats:
        # - .pt: torch.save(tensor) (optionally mmap-loadable)
        # - .npy: numpy open_memmap output (read via np.load(mmap_mode="r"))
        if not (
            activations_path.is_file() and activations_path.suffix in {".pt", ".npy"}
        ):
            raise ValueError(
                "activations_path must be a single activation cache file (.pt or .npy), "
                f"got {activations_path}"
            )

        # Preserve previous behavior: load immediately on construction.
        self._load()

    def _load(self) -> None:
        """Load (or mmap) activations into this process if not already loaded."""
        if self.activations is not None or self._npy_activations is not None:
            return

        # Load combined file format:
        # - .pt: tensor of shape (num_samples, activation_dim)
        # - .npy: numpy array of shape (num_samples, activation_dim)
        #
        # IMPORTANT: activation caches can be extremely large (100GB+). A plain
        # torch.load() will fully deserialize into RAM and can appear "stuck"
        # for a long time. Newer PyTorch supports mmap=True to memory-map the
        # underlying storage for near-instant load and much lower memory usage.
        import time

        try:
            size_gb = self.activations_path.stat().st_size / (1024**3)
            size_str = f"{size_gb:.1f}GB"
        except Exception:
            size_str = "unknown size"

        print(f"Loading activations from {self.activations_path} ({size_str}) ...")
        t0 = time.time()
        if self._cache_format == "pt":
            try:
                activations = torch.load(
                    self.activations_path, map_location="cpu", mmap=True
                )
            except TypeError:
                # Older PyTorch versions may not support mmap.
                activations = torch.load(self.activations_path, map_location="cpu")
        elif self._cache_format == "npy":
            import numpy as np

            activations = np.load(self.activations_path, mmap_mode="r")
        else:
            raise ValueError(
                f"Unsupported activation cache format: {self._cache_format}"
            )
        dt = time.time() - t0
        print(f"Loaded activations in {dt:.2f}s", flush=True)

        if self._cache_format == "pt":
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
                            f"Unexpected 'activations' field format in {self.activations_path}"
                        )
                else:
                    raise ValueError(f"Unexpected format in {self.activations_path}")

            if not isinstance(self.activations, torch.Tensor):
                raise ValueError(
                    f"Activations did not resolve to a torch.Tensor for {self.activations_path}"
                )
            self._length = int(self.activations.shape[0])
        elif self._cache_format == "npy":
            self._npy_activations = activations
            self._length = int(self._npy_activations.shape[0])
        else:
            raise ValueError(
                f"Unsupported activation cache format: {self._cache_format}"
            )

    def __getstate__(self):
        """
        Avoid pickling a huge Tensor when DataLoader uses multiprocessing (spawn).
        Each worker process can cheaply mmap the same file on first access.
        """
        state = dict(self.__dict__)
        state["activations"] = None
        state["_npy_activations"] = None
        state["_length"] = None
        return state

    def __len__(self):
        if self._length is None:
            self._load()
        return self._length

    def __getitem__(self, idx):
        if self.activations is None and self._npy_activations is None:
            self._load()
        if self.activations is not None:
            return self.activations[idx]
        # numpy memmap -> torch tensor (float32 for training stability)
        import numpy as np

        row = self._npy_activations[idx]
        if isinstance(row, np.ndarray):
            return torch.from_numpy(row).to(dtype=torch.float32)
        return torch.tensor(row, dtype=torch.float32)

    # Optional fast-path: DataLoader can call this with a list of indices.
    def __getitems__(self, indices):
        if self.activations is None and self._npy_activations is None:
            self._load()
        if self.activations is not None:
            return self.activations[indices]
        import numpy as np

        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        arr = self._npy_activations[idx]
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).to(dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32)


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

    # Pin this worker to a single GPU.
    #
    # IMPORTANT: `gpu_id` here is a *local* index within the parent's visible
    # GPU set (torchrun / scheduler may have set CUDA_VISIBLE_DEVICES).
    parent_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_device_for_local_rank(
        gpu_id, parent_cuda_visible
    )

    # CRITICAL: If this worker was spawned from a torchrun/DDP parent, it will
    # inherit distributed env vars. Even though vLLM passes its own init_method,
    # these vars can still lead to confusing torch.distributed / NCCL behavior.
    # We want each vLLM engine to be fully isolated.
    dist_vars_to_clear = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "INIT_METHOD",
        "TORCH_DISTRIBUTED_TIMEOUT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
    ]
    for var in dist_vars_to_clear:
        os.environ.pop(var, None)

    # Also clear any already-initialized process group (rare, but safe).
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # IMPORTANT: vLLM (V1) uses a local TCPStore even for world_size=1 and
    # chooses a port via `get_open_port()`. When launching many vLLM instances
    # concurrently (one per GPU), the default random-port selection can race
    # and occasionally deadlock/timeout during init_process_group.
    #
    # To make this deterministic and avoid cross-worker collisions, assign each
    # worker a disjoint port range based on GPU id (unless the user explicitly
    # set VLLM_PORT).
    if "VLLM_PORT" not in os.environ:
        base = int(os.environ.get("ENCODERSAE_VLLM_BASE_PORT", "50000"))
        stride = int(os.environ.get("ENCODERSAE_VLLM_PORT_STRIDE", "100"))
        os.environ["VLLM_PORT"] = str(base + int(gpu_id) * stride)

    # Avoid vLLM V1 EngineCore subprocess in offline activation extraction.
    # (See module-level comment above.)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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


def _process_chunk_worker(
    chunk_queue: Queue,
    result_queue: Queue,
    model_name: str,
    output_dir: str,
    batch_size: int,
    max_length: int,
    use_vllm: bool,
    gpu_memory_utilization: float,
    text_column: str,
    gpu_id: int,
):
    """Worker function to process chunks from a queue, one at a time."""
    import os
    import time
    import json
    import torch
    from pathlib import Path

    # CRITICAL: Clear ALL distributed environment variables FIRST, before any imports or operations
    # This prevents vLLM from detecting distributed setup from parent process
    dist_vars_to_clear = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "INIT_METHOD",
        "TORCH_DISTRIBUTED_TIMEOUT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
    ]
    for var in dist_vars_to_clear:
        os.environ.pop(var, None)  # Remove if exists, ignore if not

    # IMPORTANT: Make vLLM's TCPStore port deterministic per GPU worker to avoid
    # rare port-selection races when launching many vLLM instances concurrently.
    # (See vLLM's `get_open_port()` implementation.)
    if "VLLM_PORT" not in os.environ:
        base = int(os.environ.get("ENCODERSAE_VLLM_BASE_PORT", "50000"))
        stride = int(os.environ.get("ENCODERSAE_VLLM_PORT_STRIDE", "100"))
        os.environ["VLLM_PORT"] = str(base + int(gpu_id) * stride)

    # Avoid vLLM V1 EngineCore subprocess in offline activation extraction.
    # (See module-level comment above.)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Also clear torch.distributed state if it was initialized
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass  # Ignore if not initialized or can't destroy

    # Set CUDA device for this worker process BEFORE any CUDA operations
    # This ensures all CUDA operations in this process use the correct GPU
    if torch.cuda.is_available():
        # Set the default device for this process
        torch.cuda.set_device(gpu_id)
        print(
            f"[GPU {gpu_id}] Worker initialized, CUDA device set to {gpu_id}",
            flush=True,
        )

    while True:
        # Get next chunk from queue
        chunk_data = chunk_queue.get()
        if chunk_data is None:  # Sentinel to stop
            break

        chunk_idx, chunk_texts = chunk_data

        try:
            print(
                f"[GPU {gpu_id}] Starting chunk {chunk_idx + 1} ({len(chunk_texts)} samples, using vLLM)...",
                flush=True,
            )

            chunk_output_dir = Path(output_dir) / f"chunk_{chunk_idx}"
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            # CRITICAL: Set localhost distributed env BEFORE calling extract_activations
            # This ensures vLLM uses localhost instead of detecting remote IP
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                local_port = s.getsockname()[1]

            # Override any distributed environment with localhost
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(local_port)
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "0"

            # Force vLLM usage for chunk processing (more efficient than transformers)
            # This ensures GPU utilization is high
            if not use_vllm:
                print(
                    f"[GPU {gpu_id}] Forcing vLLM usage for chunk processing (was use_vllm=False)",
                    flush=True,
                )

            # Process chunk with single GPU (this worker's GPU) using vLLM
            chunk_output_file = extract_activations(
                model_name=model_name,
                texts=chunk_texts,
                output_dir=str(chunk_output_dir),
                batch_size=batch_size,
                max_length=max_length,
                device=torch.device(f"cuda:{gpu_id}"),
                num_gpus=1,  # Single GPU per chunk
                use_vllm=True,  # Always use vLLM for chunk processing (better GPU utilization)
                gpu_memory_utilization=gpu_memory_utilization,
                dataset_path=None,
                text_column=text_column,
                _chunking_disabled=True,
            )

            print(f"[GPU {gpu_id}] Completed chunk {chunk_idx + 1}", flush=True)

            # Put result in result queue
            result_queue.put((chunk_idx, chunk_output_file, chunk_output_dir))

        except Exception as e:
            # Put error in result queue
            result_queue.put((chunk_idx, None, str(e)))
            print(f"[GPU {gpu_id}] Error processing chunk {chunk_idx}: {e}", flush=True)


def _vllm_output_to_embedding_cpu_float32(output) -> torch.Tensor:
    """
    Normalize vLLM embed outputs to a CPU float32 tensor of shape (hidden_dim,).
    vLLM has changed return types across versions; keep this robust.
    """
    embedding_tensor = None

    if hasattr(output, "outputs"):
        outs = output.outputs
        if hasattr(outs, "embedding"):
            embedding_tensor = outs.embedding
        elif hasattr(outs, "data"):
            embedding_tensor = outs.data
        elif hasattr(outs, "__len__") and not isinstance(outs, str):
            if len(outs) == 0:
                raise ValueError("EmbeddingRequestOutput.outputs is empty")
            first = outs[0]
            if hasattr(first, "embedding"):
                embedding_tensor = first.embedding
            elif hasattr(first, "data"):
                embedding_tensor = first.data
            else:
                embedding_tensor = first
    elif hasattr(output, "embedding"):
        embedding_tensor = output.embedding
    elif isinstance(output, torch.Tensor):
        embedding_tensor = output
    elif hasattr(output, "__array__"):
        import numpy as np

        embedding_tensor = np.array(output)
    else:
        embedding_tensor = output

    if isinstance(embedding_tensor, torch.Tensor):
        t = embedding_tensor.to(dtype=torch.float32, device="cpu")
    else:
        t = torch.as_tensor(embedding_tensor, dtype=torch.float32, device="cpu")

    if t.ndim != 1:
        t = t.view(-1)
    return t


def _count_lines_fast(path: str) -> int:
    """Fast line count for local text files (best-effort)."""
    import subprocess

    try:
        out = subprocess.check_output(["wc", "-l", path], text=True).strip()
        # Format: "<lines> <path>"
        return int(out.split()[0])
    except Exception:
        # Fallback: Python count (slower)
        n = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                n += chunk.count(b"\n")
        return int(n)


def _vllm_write_npy_worker(
    task_queue: Queue,
    progress_queue: Queue,
    mmap_path: str,
    total_samples: int,
    hidden_dim: int,
    np_dtype_str: str,
    model_name: str,
    max_length: int,
    gpu_id: int,
    gpu_memory_utilization: float,
):
    """
    Worker that owns a single GPU, runs vLLM embed, and writes results into a
    shared numpy memmap file at the provided global indices.
    """
    import os
    import multiprocessing
    import torch
    import numpy as np
    from numpy.lib.format import open_memmap

    # Ensure this worker is fully isolated from any parent torchrun env.
    dist_vars_to_clear = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "INIT_METHOD",
        "TORCH_DISTRIBUTED_TIMEOUT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
    ]
    for var in dist_vars_to_clear:
        os.environ.pop(var, None)

    # Pin this worker to one GPU (gpu_id is local to the parent's visible set).
    parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_device_for_local_rank(
        int(gpu_id), parent_visible
    )

    # vLLM: keep in-process and deterministic per worker.
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_LOOPBACK_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if "VLLM_PORT" not in os.environ:
        base = int(os.environ.get("ENCODERSAE_VLLM_BASE_PORT", "50000"))
        stride = int(os.environ.get("ENCODERSAE_VLLM_PORT_STRIDE", "100"))
        os.environ["VLLM_PORT"] = str(base + int(gpu_id) * stride)

    # Multiprocessing must be spawn for CUDA + vLLM.
    os.environ.setdefault("PYTHON_MULTIPROCESSING_START_METHOD", "spawn")
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Open shared mmap for writing.
    mmap = open_memmap(
        mmap_path,
        mode="r+",
        dtype=np.dtype(np_dtype_str),
        shape=(int(total_samples), int(hidden_dim)),
    )

    from vllm import LLM

    # Performance knobs (optional, via env vars).
    # NOTE: GPU memory usage for pooling/embedding models often stays low (mostly weights),
    # so focus on throughput (samples/s) and utilization rather than memory %.
    enforce_eager_env = os.environ.get("ENCODERSAE_VLLM_ENFORCE_EAGER", "1").strip().lower()
    enforce_eager = enforce_eager_env not in {"0", "false", "no", "off"}

    extra_llm_kwargs: dict[str, object] = {}
    max_num_batched_tokens_env = os.environ.get("ENCODERSAE_VLLM_MAX_NUM_BATCHED_TOKENS")
    if max_num_batched_tokens_env:
        try:
            extra_llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens_env)
        except Exception:
            pass
    max_num_seqs_env = os.environ.get("ENCODERSAE_VLLM_MAX_NUM_SEQS")
    if max_num_seqs_env:
        try:
            extra_llm_kwargs["max_num_seqs"] = int(max_num_seqs_env)
        except Exception:
            pass

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_length,
        task="embed",
        disable_log_stats=True,
        **extra_llm_kwargs,
    )

    while True:
        item = task_queue.get()
        if item is None:
            break
        start_idx, batch_texts = item
        outputs = llm.encode(batch_texts, pooling_task="embed", use_tqdm=False)
        batch_emb = torch.stack(
            [_vllm_output_to_embedding_cpu_float32(o) for o in outputs], dim=0
        )  # (B, D) float32 CPU
        np_batch = batch_emb.numpy().astype(mmap.dtype, copy=False)
        end_idx = int(start_idx) + int(np_batch.shape[0])
        mmap[int(start_idx) : end_idx, :] = np_batch
        progress_queue.put(int(np_batch.shape[0]))

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        mmap.flush()
    except Exception:
        pass


def _extract_activations_vllm_to_npy(
    *,
    model_name: str,
    texts: list[str],
    dataset_path: Optional[str],
    text_column: str,
    output_dir: str,
    batch_size: int,
    max_length: int,
    num_gpus: int,
    gpu_memory_utilization: float,
    max_samples: Optional[int],
    total_samples: Optional[int],
    cache_dtype: str,
) -> str:
    """
    Streaming vLLM activation extraction that writes directly to a numpy memmap
    (.npy) file to avoid holding all activations in RAM.
    """
    import multiprocessing
    import os
    import time

    import numpy as np
    from numpy.lib.format import open_memmap
    from transformers import AutoConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_short = (
        Path(model_name).stem
        if os.path.exists(model_name)
        else model_name.replace("/", "_")
    )

    output_file = output_path / f"{model_short}.npy"
    tmp_file = output_path / f"{model_short}.npy.tmp"

    # External progress / error reporting (for torchrun ranks waiting on the cache).
    progress_file = output_path / f"{model_short}.progress.json"
    progress_tmp_file = output_path / f"{model_short}.progress.json.tmp"
    error_file = output_path / f"{model_short}.error.txt"

    try:
        error_file.unlink(missing_ok=True)
    except Exception:
        pass

    def _write_progress(
        *,
        stage: str,
        done: int,
        total: int,
        samples_per_sec: Optional[float] = None,
        eta_seconds: Optional[float] = None,
    ) -> None:
        payload = {
            "stage": str(stage),
            "done": int(done),
            "total": int(total),
            "percent": (float(done) * 100.0 / float(total)) if total > 0 else None,
            "samples_per_sec": float(samples_per_sec)
            if samples_per_sec is not None
            else None,
            "eta_seconds": float(eta_seconds) if eta_seconds is not None else None,
            "updated_at": time.time(),
        }
        try:
            with open(progress_tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(progress_tmp_file, progress_file)
        except Exception:
            # Best-effort only.
            pass

    # If cached, just reuse.
    if output_file.exists():
        return str(output_file)

    done = 0
    sent = 0
    total_samples_final = 0
    workers: list[multiprocessing.Process] = []
    try:
        # Determine total samples.
        #
        # Priority:
        # 1) If `texts` is provided, use its length (capped by max_samples).
        # 2) Else, if `total_samples` is provided, trust it (capped by max_samples).
        # 3) Else, if max_samples is set, use it (avoid scanning huge files).
        # 4) Else, count lines (wc -l).
        if texts and len(texts) > 0:
            total_samples_final = len(texts)
            if max_samples is not None:
                total_samples_final = min(total_samples_final, int(max_samples))
        else:
            if total_samples is not None:
                total_samples_final = int(total_samples)
                if max_samples is not None:
                    total_samples_final = min(total_samples_final, int(max_samples))
            else:
                if dataset_path is None or not os.path.exists(dataset_path):
                    raise ValueError(
                        "For vLLM .npy caching, provide either `texts`, `total_samples`, or a valid `dataset_path`."
                    )
                if max_samples is not None:
                    total_samples_final = int(max_samples)
                else:
                    total_samples_final = _count_lines_fast(dataset_path)

        if total_samples_final <= 0:
            raise ValueError("No samples to process.")

        # Determine embedding dimension.
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        hidden_dim = int(
            getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        )
        if hidden_dim <= 0:
            raise ValueError(f"Could not determine hidden size for model {model_name}")

        _write_progress(stage="initializing", done=0, total=int(total_samples_final))

        # Allocate memmap (tmp file, renamed on success).
        np_dtype = np.dtype(cache_dtype)
        mm = open_memmap(
            tmp_file,
            mode="w+",
            dtype=np_dtype,
            shape=(int(total_samples_final), int(hidden_dim)),
        )
        # Close in parent; workers will reopen.
        del mm

        # Spawn GPU workers.
        os.environ.setdefault("PYTHON_MULTIPROCESSING_START_METHOD", "spawn")
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        ctx = multiprocessing.get_context("spawn")
        task_queue: Queue = ctx.Queue(maxsize=max(8, int(num_gpus) * 8))
        progress_queue: Queue = ctx.Queue()

        for gpu_id in range(int(num_gpus)):
            p = ctx.Process(
                target=_vllm_write_npy_worker,
                args=(
                    task_queue,
                    progress_queue,
                    str(tmp_file),
                    int(total_samples_final),
                    int(hidden_dim),
                    str(np_dtype),
                    model_name,
                    int(max_length),
                    int(gpu_id),
                    float(gpu_memory_utilization),
                ),
            )
            p.start()
            workers.append(p)

        # Producer runs in a background thread so we can report progress while
        # the dataset is still being read/enqueued.
        producer_done = threading.Event()
        producer_err: list[BaseException] = []

        def _producer() -> None:
            nonlocal sent
            try:
                if texts and len(texts) > 0:
                    while sent < total_samples_final:
                        batch_texts = texts[
                            sent : min(sent + batch_size, total_samples_final)
                        ]
                        task_queue.put((sent, batch_texts))
                        sent += len(batch_texts)
                else:
                    assert dataset_path is not None
                    with open(dataset_path, "r", encoding="utf-8") as f:
                        batch: list[str] = []
                        start_idx = 0
                        for line_idx, line in enumerate(f):
                            if line_idx >= total_samples_final:
                                break
                            obj = json.loads(line)
                            batch.append(obj[text_column])
                            if len(batch) >= batch_size:
                                task_queue.put((start_idx, batch))
                                start_idx += len(batch)
                                batch = []
                        if batch:
                            task_queue.put((start_idx, batch))
                            start_idx += len(batch)
                        sent = start_idx
            except BaseException as e:
                producer_err.append(e)
            finally:
                # Always send stop signals so workers can exit.
                try:
                    for _ in workers:
                        task_queue.put(None)
                except Exception:
                    pass
                producer_done.set()

        threading.Thread(target=_producer, daemon=True).start()

        # Progress monitor.
        import queue as py_queue

        start_time = time.time()
        last_write = 0.0
        last_log = 0.0
        write_interval_s = float(os.environ.get("ENCODERSAE_PROGRESS_WRITE_INTERVAL", "5"))
        log_interval_s = float(os.environ.get("ENCODERSAE_PROGRESS_LOG_INTERVAL", "60"))
        is_tty = sys.stderr.isatty()

        # Start as "feeding" until we observe actual completed samples.
        stage = "feeding"
        _write_progress(stage=stage, done=0, total=int(total_samples_final))

        with tqdm(
            total=total_samples_final,
            desc="Activations (vLLM mmap)",
            unit="samples",
        ) as pbar:
            while done < total_samples_final:
                if producer_err:
                    raise RuntimeError("Activation-cache producer failed.") from producer_err[0]

                try:
                    inc = progress_queue.get(timeout=5.0)
                    done += int(inc)
                    pbar.update(int(inc))
                except py_queue.Empty:
                    # If any worker died, fail fast (avoid hanging forever).
                    failed = [p for p in workers if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(
                            "One or more vLLM worker processes failed during activation extraction."
                        )

                now = time.time()
                if stage == "feeding" and done > 0:
                    stage = "running"
                if (now - last_write) >= write_interval_s or done >= total_samples_final:
                    elapsed = max(1e-6, now - start_time)
                    sps = float(done) / elapsed
                    eta = (
                        (float(total_samples_final - done) / sps) if sps > 0 else None
                    )
                    _write_progress(
                        stage=stage,
                        done=int(done),
                        total=int(total_samples_final),
                        samples_per_sec=sps,
                        eta_seconds=eta,
                    )
                    last_write = now

                # When logs are redirected, tqdm's carriage-return updates are often invisible.
                # Emit a newline heartbeat periodically so users know it's alive.
                if (not is_tty) and ((now - last_log) >= log_interval_s):
                    pct = float(done) * 100.0 / float(total_samples_final)
                    elapsed = max(1e-6, now - start_time)
                    sps = float(done) / elapsed
                    eta = (
                        (float(total_samples_final - done) / sps) if sps > 0 else None
                    )
                    eta_str = (
                        f"{int(eta // 3600)}h{int((eta % 3600) // 60)}m{int(eta % 60)}s"
                        if eta is not None
                        else "?"
                    )
                    print(
                        f"[activation_cache] {stage}: {done}/{total_samples_final} ({pct:.2f}%) "
                        f"@ {sps:,.0f} samples/s, ETA {eta_str}",
                        flush=True,
                    )
                    last_log = now

                # If producer finished and workers stopped producing progress, check for failures.
                if producer_done.is_set() and done < total_samples_final:
                    failed = [p for p in workers if p.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(
                            "One or more vLLM worker processes failed during activation extraction."
                        )

        for p in workers:
            p.join()

        if any(p.exitcode != 0 for p in workers):
            raise RuntimeError(
                "One or more vLLM worker processes failed during activation extraction."
            )

        _write_progress(stage="finalizing", done=int(done), total=int(total_samples_final))

        # Rename tmp -> final (atomic on same filesystem).
        os.replace(tmp_file, output_file)
        _write_progress(
            stage="done",
            done=int(total_samples_final),
            total=int(total_samples_final),
            samples_per_sec=None,
            eta_seconds=0.0,
        )
        return str(output_file)
    except Exception:
        import traceback

        try:
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        try:
            _write_progress(stage="error", done=int(done), total=int(total_samples_final))
        except Exception:
            pass
        # Try to tear down workers to avoid long hangs.
        for p in workers:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        raise


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
    max_samples: Optional[int] = None,
    total_samples: Optional[int] = None,
    cache_format: Optional[str] = None,
    cache_dtype: str = "float16",
    _chunking_disabled: bool = False,
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

    # Decide cache format.
    if cache_format is None:
        cache_format = "npy" if use_vllm else "pt"
    cache_format = str(cache_format).lower().strip()

    # vLLM streaming path (memory-safe): write .npy memmap and return early.
    if use_vllm and cache_format == "npy":
        return _extract_activations_vllm_to_npy(
            model_name=model_name,
            texts=texts,
            dataset_path=dataset_path,
            text_column=text_column,
            output_dir=output_dir,
            batch_size=batch_size,
            max_length=max_length,
            num_gpus=int(num_gpus) if num_gpus is not None else 1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_samples=max_samples,
            total_samples=total_samples,
            cache_dtype=cache_dtype,
        )

    # Process in chunks of 10,000,000 lines to avoid memory/timeout issues
    # Only do chunking at the top level (not for recursive calls)
    CHUNK_SIZE = 10_000_000
    all_activations = []

    # Determine total length and split into chunks
    if not _chunking_disabled and texts and len(texts) > 0:
        total_len = len(texts)
        num_chunks = math.ceil(total_len / CHUNK_SIZE)

        if num_chunks > 1:
            print(
                f"Processing {total_len} samples in {num_chunks} chunks of up to {CHUNK_SIZE} lines each"
            )

            # Determine how many GPUs to use for parallel chunk processing
            # Use min(num_chunks, num_gpus) to avoid creating more workers than chunks
            available_gpus = (
                num_gpus
                if num_gpus
                else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
            )
            num_workers = min(num_chunks, available_gpus)

            print(
                f"Using {num_workers} GPUs to process {num_chunks} chunks in parallel"
            )

            # Create queues for chunk distribution and results
            chunk_queue = Queue()
            result_queue = Queue()

            # Prepare all chunks and add to queue
            chunk_data_list = []
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_len)
                chunk_texts = texts[start_idx:end_idx]
                chunk_data_list.append((chunk_idx, chunk_texts))
                chunk_queue.put((chunk_idx, chunk_texts))

            # Add sentinel values to stop workers
            for _ in range(num_workers):
                chunk_queue.put(None)

            # Start worker processes (one per GPU)
            workers = []
            for worker_id in range(num_workers):
                gpu_id = worker_id % available_gpus  # Cycle through available GPUs
                p = Process(
                    target=_process_chunk_worker,
                    args=(
                        chunk_queue,
                        result_queue,
                        model_name,
                        output_dir,
                        batch_size,
                        max_length,
                        use_vllm,
                        gpu_memory_utilization,
                        text_column,
                        gpu_id,
                    ),
                )
                p.start()
                workers.append(p)

            # Collect results as they complete (maintain order)
            chunk_results = {}
            completed_chunks = 0

            with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                while completed_chunks < num_chunks:
                    chunk_idx, chunk_output_file, chunk_output_dir = result_queue.get()

                    if chunk_output_file is None:
                        # Error occurred
                        raise RuntimeError(
                            f"Chunk {chunk_idx} failed: {chunk_output_dir}"
                        )

                    chunk_results[chunk_idx] = (chunk_output_file, chunk_output_dir)
                    completed_chunks += 1
                    pbar.update(1)
                    print(f"Completed chunk {chunk_idx + 1}/{num_chunks}")

            # Wait for all workers to finish
            for p in workers:
                p.join()

            # Check for worker failures
            if any(p.exitcode != 0 for p in workers):
                raise RuntimeError("One or more worker processes failed.")

            # Load chunks in order and concatenate
            print("\nLoading and concatenating chunks...")
            for chunk_idx in sorted(chunk_results.keys()):
                chunk_output_file, chunk_output_dir = chunk_results[chunk_idx]
                chunk_activations = torch.load(chunk_output_file)
                all_activations.append(chunk_activations)

                # Clean up chunk directory
                shutil.rmtree(chunk_output_dir)

            # Concatenate all chunks in order
            print("Concatenating all chunks...")
            activations_tensor = torch.cat(all_activations, dim=0)

            # Generate final output filename
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            model_short = (
                Path(model_name).stem
                if os.path.exists(model_name)
                else model_name.replace("/", "_")
            )
            output_file = output_path / f"{model_short}.pt"

            torch.save(activations_tensor, output_file)
            print(f"Saved {len(activations_tensor)} activations to {output_file}")

            return str(output_file)
        # If only one chunk needed, fall through to normal processing

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

        # Avoid vLLM V1 EngineCore subprocess in offline activation extraction.
        # (See module-level comment above.)
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

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
            # If texts are already loaded, use that length to avoid reloading the dataset
            temp_dataset_file = None
            if texts and len(texts) > 0:
                # Use pre-loaded texts length (avoids disk space issues from reloading)
                total_len = len(texts)
                print(f"Using pre-loaded texts length: {total_len} samples")
                # Save texts to temp file for workers to load on-demand
                temp_dataset_file = output_path / "temp_texts.jsonl"
                print(
                    f"Creating temporary dataset file for workers: {temp_dataset_file}"
                )
                with open(temp_dataset_file, "w", encoding="utf-8") as f:
                    for text in texts:
                        f.write(json.dumps({text_column: text}) + "\n")
                dataset_path = str(temp_dataset_file)
            elif dataset_path and os.path.exists(dataset_path):
                # Only load dataset if texts not provided (shouldn't happen in normal flow)
                from datasets import load_dataset

                print(f"Loading dataset metadata from {dataset_path}...")
                # Use cache directory on /data_x to avoid /tmp space issues
                cache_dir = os.path.join(
                    ".cache",
                    f"vllm_cache_{os.getpid()}",
                )
                os.makedirs(cache_dir, exist_ok=True)
                ds = load_dataset(
                    "json",
                    data_files=dataset_path,
                    split="train",
                    cache_dir=cache_dir,
                    verification_mode="no_checks",
                )
                total_len = len(ds)
                print(f"Total samples: {total_len}")
            else:
                raise ValueError("Either texts or dataset_path must be provided")

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

            # In distributed mode, ensure vLLM only sees the rank's GPU to avoid NCCL conflicts
            original_cuda_visible = None
            original_dist_env = {}
            gpu_id = None
            if device is not None and device.type == "cuda":
                gpu_id = device.index if device.index is not None else 0
                # Save original CUDA_VISIBLE_DEVICES
                original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                # Set to only this GPU - vLLM will see it as GPU 0.
                # Note: gpu_id is a *local* index within original_cuda_visible.
                target_visible = _cuda_visible_device_for_local_rank(
                    gpu_id, original_cuda_visible
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = target_visible
                print(
                    f"Setting CUDA_VISIBLE_DEVICES={target_visible} for vLLM "
                    f"(local_gpu={gpu_id} -> vLLM sees GPU 0)"
                )
                # Also set the default CUDA device for this process
                torch.cuda.set_device(0)  # vLLM sees it as 0 after CUDA_VISIBLE_DEVICES

                # Completely remove all distributed environment variables AGAIN (in case they were set)
                # This prevents vLLM from trying to use distributed initialization at all
                # vLLM will initialize its own isolated distributed environment internally
                dist_vars = [
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "INIT_METHOD",
                    "TORCH_DISTRIBUTED_TIMEOUT",
                    "NCCL_SOCKET_IFNAME",
                    "NCCL_IB_DISABLE",
                    "NCCL_DEBUG",
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                ]
                for var in dist_vars:
                    if var in os.environ:
                        original_dist_env[var] = os.environ[var]
                        del os.environ[var]

                # Force vLLM to use localhost for distributed init (even though it's single GPU)
                # This prevents it from trying to connect to remote servers
                import socket

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    local_port = s.getsockname()[1]

                # Set minimal localhost distributed env - vLLM requires this for initialization
                # But use localhost so it doesn't try to connect remotely
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(local_port)
                os.environ["RANK"] = "0"
                os.environ["WORLD_SIZE"] = "1"
                os.environ["LOCAL_RANK"] = "0"
                # Disable timeout completely
                os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "0"

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
                # Restore CUDA_VISIBLE_DEVICES on error
                if original_cuda_visible is not None:
                    if original_cuda_visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                # Restore distributed environment variables
                for var in [
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "INIT_METHOD",
                    "TORCH_DISTRIBUTED_TIMEOUT",
                ]:
                    if var in original_dist_env:
                        os.environ[var] = original_dist_env[var]
                    elif var in os.environ:
                        os.environ.pop(var, None)
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

            try:
                print(f"Extracting activations for {len(texts)} samples using vLLM...")

                # Collect all activations in a list
                all_activations = []

                # Process in batches
                num_batches = (len(texts) + batch_size - 1) // batch_size

                # Create progress bar with more descriptive info
                progress_desc = f"Chunk activations (vLLM"
                if gpu_id is not None:
                    progress_desc += f", GPU {gpu_id}"
                progress_desc += f", batch={batch_size}"
                progress_desc += ")"
                if len(texts) > 1000000:
                    progress_desc += f" [{len(texts)//1000000}M samples]"

                for batch_idx in tqdm(
                    range(num_batches),
                    desc=progress_desc,
                    total=num_batches,
                    unit="batch",
                ):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    # Get embeddings from vLLM using encode() API
                    # Use llm.encode() instead of llm.embed() for better performance
                    # vLLM will process this batch on the GPU
                    outputs = llm.encode(
                        batch_texts, pooling_task="embed", use_tqdm=False
                    )

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
                                        torch.tensor(
                                            embedding_data, dtype=torch.float32
                                        )
                                        if not isinstance(embedding_data, torch.Tensor)
                                        else embedding_data.to(dtype=torch.float32)
                                    )
                                else:
                                    raise ValueError(
                                        "EmbeddingRequestOutput.outputs is empty"
                                    )
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
                                embedding_tensor = torch.tensor(
                                    output, dtype=torch.float32
                                )
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
                                embedding_tensor = embedding_tensor.to(
                                    dtype=torch.float32
                                )

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
            finally:
                # vLLM initializes a torch.distributed process group even for
                # world_size=1. Tear it down so callers (e.g., torchrun/DDP) can
                # safely re-initialize their own process group afterwards.
                try:
                    import torch.distributed as dist

                    if dist.is_initialized():
                        dist.destroy_process_group()
                except Exception:
                    pass

                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    if original_cuda_visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                # Restore distributed environment variables
                for var in [
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "RANK",
                    "WORLD_SIZE",
                    "LOCAL_RANK",
                    "INIT_METHOD",
                    "TORCH_DISTRIBUTED_TIMEOUT",
                ]:
                    if var in original_dist_env:
                        os.environ[var] = original_dist_env[var]
                    elif var in os.environ:
                        os.environ.pop(var, None)

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
