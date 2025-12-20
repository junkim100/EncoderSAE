"""Language-agnostic inference utilities for removing language-specific features from embeddings."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModel, AutoTokenizer

from .model import EncoderSAE
from .data import mean_pool


def remove_language_features(
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Remove language-specific features from SAE feature activations.

    Args:
        features: SAE feature activations of shape (batch_size, dict_size)
        mask: Boolean mask of shape (dict_size,) where True indicates language-specific features

    Returns:
        Language-agnostic features with language-specific features set to 0
    """
    # Apply mask: set language-specific features to 0
    # mask is True for language-specific features, so ~mask is True for contextual features
    return features * (~mask)


def infer_language_agnostic(
    model_name: str,
    sae: Union[EncoderSAE, str],
    mask_path: str,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    use_vllm: bool = True,
    num_gpus: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
) -> torch.Tensor:
    """
    Full inference pipeline: text -> base model -> SAE -> remove language features -> output.

    This creates language-agnostic embeddings by:
    1. Encoding text with the base model
    2. Passing through SAE to get sparse features
    3. Masking out language-specific features (setting them to 0)
    4. Returning the language-agnostic feature activations

    Args:
        model_name: HuggingFace model ID or local path (same as used for SAE training)
        sae: EncoderSAE model instance or path to checkpoint (.pt file)
        mask_path: Path to language mask file (.pt file)
            - Can be individual language mask: `language_features_{lang}_mask.pt`
            - Or combined union mask: `language_features_combined_mask.pt`
        texts: List of input text strings
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on (auto-detected if None)
        use_vllm: Use vLLM for faster base model inference (default: True)
        num_gpus: Number of GPUs to use for vLLM (default: None = auto-detect)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)

    Returns:
        Language-agnostic feature activations of shape (len(texts), dict_size)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SAE if path provided
    if isinstance(sae, str):
        sae_path = Path(sae)
        checkpoint = torch.load(sae_path, map_location=device)

        # Determine input_dim from checkpoint
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
        sparsity = (
            checkpoint.get("sparsity", 64) if isinstance(checkpoint, dict) else 64
        )

        sae = EncoderSAE(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            sparsity=sparsity,
        )
        sae.load_state_dict(state_dict)
        sae.to(device)
        sae.eval()

    # Load mask
    mask = torch.load(mask_path, map_location=device)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.shape[0] != sae.dict_size:
        raise ValueError(
            f"Mask size {mask.shape[0]} does not match SAE dict_size {sae.dict_size}"
        )

    # Extract base embeddings from texts
    if use_vllm:
        try:
            # Suppress vLLM logs
            import logging

            vllm_logger = logging.getLogger("vllm")
            vllm_logger.setLevel(logging.WARNING)

            from vllm import LLM

            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=num_gpus or 1,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_length,
                task="embed",
                disable_log_stats=True,
            )

            all_activations = []
            num_batches = (len(texts) + batch_size - 1) // batch_size

            @contextmanager
            def suppress_output():
                """Temporarily suppress stdout/stderr to hide vLLM progress bars."""
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                with open(os.devnull, "w") as devnull:
                    sys.stdout = devnull
                    sys.stderr = devnull
                    try:
                        yield
                    finally:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Get embeddings from vLLM (suppress progress bars)
                with suppress_output():
                    outputs = llm.encode(
                        batch_texts, pooling_task="embed", use_tqdm=False
                    )

                # Extract embeddings (same logic as in data.py)
                batch_activations = []
                for output in outputs:
                    embedding_tensor = None

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
                    elif hasattr(output, "embedding"):
                        embedding_data = output.embedding
                        embedding_tensor = (
                            torch.tensor(embedding_data, dtype=torch.float32)
                            if not isinstance(embedding_data, torch.Tensor)
                            else embedding_data.to(dtype=torch.float32)
                        )
                    elif isinstance(output, torch.Tensor):
                        embedding_tensor = output.to(dtype=torch.float32)
                    else:
                        embedding_tensor = torch.tensor(output, dtype=torch.float32)

                    if embedding_tensor.dtype != torch.float32:
                        embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                    batch_activations.append(embedding_tensor)

                batch_activations = torch.stack(batch_activations).to(device)
                all_activations.append(batch_activations)

            base_embeddings = torch.cat(all_activations, dim=0)

        except ImportError:
            print("vLLM not available, falling back to HuggingFace")
            use_vllm = False

    if not use_vllm:
        # Use HuggingFace transformers
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        all_activations = []

        with torch.no_grad():
            num_batches = (len(texts) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Tokenize and encode
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # Get embeddings
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                batch_activations = mean_pool(hidden_states, attention_mask)
                all_activations.append(batch_activations)

        base_embeddings = torch.cat(all_activations, dim=0)

    # Pass through SAE to get features
    with torch.no_grad():
        _, features, _, _ = sae(base_embeddings)

    # Remove language-specific features
    features_agnostic = remove_language_features(features, mask)

    return features_agnostic


class LanguageAgnosticEncoder:
    """
    Pipeline for creating language-agnostic embeddings from text.

    Usage:
        encoder = LanguageAgnosticEncoder(
            model_name="intfloat/multilingual-e5-large",
            sae_path="checkpoints/.../final_model.pt",
            mask_path="analysis/.../language_features_combined_mask.pt",
        )
        embeddings = encoder.encode(["Hello world", "Bonjour le monde"])
    """

    def __init__(
        self,
        model_name: str,
        sae_path: str,
        mask_path: str,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[torch.device] = None,
        use_vllm: bool = True,
        num_gpus: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Initialize the language-agnostic encoder pipeline.

        Args:
            model_name: HuggingFace model ID or local path
            sae_path: Path to trained SAE checkpoint (.pt file)
            mask_path: Path to language mask file (.pt file)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            device: Device to run on (auto-detected if None)
            use_vllm: Use vLLM for faster inference (default: True)
            num_gpus: Number of GPUs to use for vLLM (default: None = auto-detect)
            gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)
        """
        self.model_name = model_name
        self.sae_path = sae_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_vllm = use_vllm
        self.num_gpus = num_gpus
        self.gpu_memory_utilization = gpu_memory_utilization

        # Load SAE
        checkpoint = torch.load(sae_path, map_location=self.device)
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
        sparsity = (
            checkpoint.get("sparsity", 64) if isinstance(checkpoint, dict) else 64
        )

        self.sae = EncoderSAE(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            sparsity=sparsity,
        )
        self.sae.load_state_dict(state_dict)
        self.sae.to(self.device)
        self.sae.eval()

        # Load mask
        self.mask = torch.load(mask_path, map_location=self.device)
        if self.mask.dtype != torch.bool:
            self.mask = self.mask.bool()
        if self.mask.shape[0] != self.sae.dict_size:
            raise ValueError(
                f"Mask size {self.mask.shape[0]} does not match SAE dict_size {self.sae.dict_size}"
            )

    def encode(
        self,
        texts: Union[str, list[str]],
    ) -> torch.Tensor:
        """
        Encode texts into language-agnostic embeddings.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Language-agnostic feature activations of shape (len(texts), dict_size)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Extract base embeddings from texts
        if self.use_vllm:
            try:
                # Suppress vLLM logs
                import logging

                vllm_logger = logging.getLogger("vllm")
                vllm_logger.setLevel(logging.WARNING)

                from vllm import LLM

                llm = LLM(
                    model=self.model_name,
                    trust_remote_code=True,
                    enforce_eager=True,
                    tensor_parallel_size=self.num_gpus or 1,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_length,
                    task="embed",
                    disable_log_stats=True,
                )

                all_activations = []
                num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

                @contextmanager
                def suppress_output():
                    """Temporarily suppress stdout/stderr to hide vLLM progress bars."""
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    with open(os.devnull, "w") as devnull:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        try:
                            yield
                        finally:
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    # Get embeddings from vLLM (suppress progress bars)
                    with suppress_output():
                        outputs = llm.encode(
                            batch_texts, pooling_task="embed", use_tqdm=False
                        )

                    # Extract embeddings (same logic as in data.py)
                    batch_activations = []
                    for output in outputs:
                        embedding_tensor = None

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
                        elif hasattr(output, "embedding"):
                            embedding_data = output.embedding
                            embedding_tensor = (
                                torch.tensor(embedding_data, dtype=torch.float32)
                                if not isinstance(embedding_data, torch.Tensor)
                                else embedding_data.to(dtype=torch.float32)
                            )
                        elif isinstance(output, torch.Tensor):
                            embedding_tensor = output.to(dtype=torch.float32)
                        else:
                            embedding_tensor = torch.tensor(output, dtype=torch.float32)

                        if embedding_tensor.dtype != torch.float32:
                            embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                        batch_activations.append(embedding_tensor)

                    batch_activations = torch.stack(batch_activations).to(self.device)
                    all_activations.append(batch_activations)

                base_embeddings = torch.cat(all_activations, dim=0)

            except ImportError:
                print("vLLM not available, falling back to HuggingFace")
                self.use_vllm = False

        if not self.use_vllm:
            # Use HuggingFace transformers
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()

            all_activations = []

            with torch.no_grad():
                num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    # Tokenize and encode
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                    input_ids = encoded["input_ids"].to(self.device)
                    attention_mask = encoded["attention_mask"].to(self.device)

                    # Get embeddings
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    hidden_states = outputs.last_hidden_state
                    batch_activations = mean_pool(hidden_states, attention_mask)
                    all_activations.append(batch_activations)

            base_embeddings = torch.cat(all_activations, dim=0)

        # Pass through SAE to get features
        with torch.no_grad():
            _, features, _, _ = self.sae(base_embeddings)

        # Remove language-specific features using the already-loaded mask
        features_agnostic = remove_language_features(features, self.mask)

        return features_agnostic
