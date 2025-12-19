"""Analysis tools for identifying language-specific SAE features."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .model import EncoderSAE
from .data import load_data, mean_pool, extract_activations
from .utils import get_device


def analyze_language_features(
    sae_path: str,
    model_name: str,
    validation_data: str,
    text_column: str = "text",
    language_column: str = "language",
    batch_size: int = 32,
    max_length: int = 512,
    top_k_features: int = 20,
    output_dir: Optional[str] = None,
    use_vllm: bool = True,
    num_gpus: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
) -> dict:
    """
    Analyze which SAE features correspond to which languages.

    Args:
        sae_path: Path to trained SAE checkpoint (.pt file)
        model_name: HuggingFace model ID or local path (same as used for training)
        validation_data: Path to validation JSONL file with language labels
        text_column: Name of text column in validation data
        language_column: Name of language column in validation data
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        top_k_features: Number of top features to report per language
        output_dir: Directory to save analysis results (None = auto-generate)
        use_vllm: Use vLLM for faster inference
        num_gpus: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization for vLLM

    Returns:
        Dictionary mapping language -> list of top feature indices
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load validation data with language labels
    print(f"Loading validation data from {validation_data}")
    texts_by_language = defaultdict(list)

    with open(validation_data, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get(text_column, "")
            language = data.get(language_column, "unknown")
            if text:
                texts_by_language[language].append(text)

    print(f"Found {len(texts_by_language)} languages:")
    for lang, texts in texts_by_language.items():
        print(f"  {lang}: {len(texts)} samples")

    # Load trained SAE
    print(f"\nLoading trained SAE from {sae_path}")
    checkpoint = torch.load(sae_path, map_location=device)

    # Determine input_dim and other parameters from checkpoint
    if "model_state_dict" in checkpoint:
        # Full checkpoint with state dict
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "encoder.weight" in checkpoint:
        # Just state dict
        state_dict = checkpoint
    else:
        raise ValueError(f"Could not parse checkpoint format from {sae_path}")

    encoder_weight = state_dict["encoder.weight"]
    input_dim = encoder_weight.shape[1]
    dict_size = encoder_weight.shape[0]
    expansion_factor = dict_size // input_dim

    # Try to get sparsity from checkpoint config, otherwise infer or use default
    if "model_state_dict" in checkpoint:
        sparsity = checkpoint.get("sparsity", 64)
    else:
        # Try to infer from decoder if available, or use default
        sparsity = 64  # Default

    print(
        f"SAE parameters: input_dim={input_dim}, dict_size={dict_size}, expansion_factor={expansion_factor}"
    )

    sae = EncoderSAE(
        input_dim=input_dim, expansion_factor=expansion_factor, sparsity=sparsity
    )
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()

    # Extract activations and features for each language
    print("\nExtracting features for each language...")
    language_features = defaultdict(
        lambda: defaultdict(int)
    )  # lang -> feature_idx -> count

    if use_vllm:
        # Use vLLM for faster extraction
        try:
            from vllm import LLM

            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=num_gpus or torch.cuda.device_count(),
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_length,
                task="embed",
            )

            for language, texts in texts_by_language.items():
                print(f"\nProcessing {language} ({len(texts)} samples)...")

                # Process in batches
                num_batches = (len(texts) + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(num_batches), desc=f"{language}"):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    # Get embeddings from vLLM
                    # vLLM's embed() returns EmbeddingRequestOutput objects
                    embedding_outputs = llm.embed(batch_texts)

                    # Extract embeddings from EmbeddingRequestOutput objects
                    # Structure: EmbeddingRequestOutput.outputs.embedding (list of floats)
                    # Note: outputs can be a single EmbeddingOutput or a list
                    embedding_tensors = []
                    for embedding_output in embedding_outputs:
                        if hasattr(embedding_output, "outputs"):
                            # outputs might be a single EmbeddingOutput object or a list
                            if hasattr(
                                embedding_output.outputs, "__len__"
                            ) and not isinstance(embedding_output.outputs, str):
                                # It's a list/sequence - get first element
                                if len(embedding_output.outputs) > 0:
                                    embedding_data = embedding_output.outputs[
                                        0
                                    ].embedding
                                else:
                                    raise ValueError(
                                        "EmbeddingRequestOutput.outputs is empty"
                                    )
                            else:
                                # It's a single EmbeddingOutput object
                                embedding_data = embedding_output.outputs.embedding
                            embedding_tensor = torch.tensor(
                                embedding_data, dtype=torch.float32
                            )
                        elif hasattr(embedding_output, "embedding"):
                            embedding_data = embedding_output.embedding
                            embedding_tensor = (
                                torch.tensor(embedding_data, dtype=torch.float32)
                                if not isinstance(embedding_data, torch.Tensor)
                                else embedding_data
                            )
                        elif isinstance(embedding_output, torch.Tensor):
                            embedding_tensor = embedding_output
                        else:
                            embedding_tensor = torch.tensor(
                                embedding_output, dtype=torch.float32
                            )
                        embedding_tensors.append(embedding_tensor)

                    # Stack into batch tensor
                    batch_activations = torch.stack(embedding_tensors)

                    batch_activations = batch_activations.to(device)

                    # Pass through SAE to get features
                    with torch.no_grad():
                        _, features, _ = sae(batch_activations)

                        # Count which features fired for this language
                        for sample_features in features:
                            # Get top-k active features
                            active_features = (sample_features > 0).nonzero(
                                as_tuple=True
                            )[0]
                            for feat_idx in active_features:
                                language_features[language][feat_idx.item()] += 1
        except ImportError:
            print("vLLM not available, falling back to HuggingFace")
            use_vllm = False

    if not use_vllm:
        # Use HuggingFace transformers
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()

        for language, texts in texts_by_language.items():
            print(f"\nProcessing {language} ({len(texts)} samples)...")

            num_batches = (len(texts) + batch_size - 1) // batch_size

            with torch.no_grad():
                for batch_idx in tqdm(range(num_batches), desc=f"{language}"):
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

                    # Pass through SAE
                    _, features, _ = sae(batch_activations)

                    # Count which features fired
                    for sample_features in features:
                        active_features = (sample_features > 0).nonzero(as_tuple=True)[
                            0
                        ]
                        for feat_idx in active_features:
                            language_features[language][feat_idx.item()] += 1

    # Analyze and rank features per language
    print("\n" + "=" * 60)
    print("Language-Specific Feature Analysis")
    print("=" * 60)

    results = {}

    for language in sorted(language_features.keys()):
        feature_counts = language_features[language]
        total_samples = len(texts_by_language[language])

        # Sort features by frequency
        sorted_features = sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Get top-k features
        top_features = [
            feat_idx for feat_idx, count in sorted_features[:top_k_features]
        ]
        top_features_with_counts = [
            (feat_idx, count, count / total_samples * 100)
            for feat_idx, count in sorted_features[:top_k_features]
        ]

        results[language] = {
            "top_features": top_features,
            "top_features_detailed": top_features_with_counts,
            "total_samples": total_samples,
            "unique_features": len(feature_counts),
        }

        print(
            f"\n{language.upper()} ({total_samples} samples, {len(feature_counts)} unique features):"
        )
        print(f"  Top {top_k_features} features:")
        for feat_idx, count, pct in top_features_with_counts[:10]:
            print(
                f"    Feature {feat_idx:5d}: {count:4d} times ({pct:5.2f}% of samples)"
            )

    # Save results
    if output_dir is None:
        output_dir = f"./analysis/{Path(sae_path).stem}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = output_path / "language_features.json"
    json_results = {
        lang: {
            "top_features": res["top_features"],
            "top_features_detailed": [
                {"feature": int(f), "count": int(c), "percentage": float(p)}
                for f, c, p in res["top_features_detailed"]
            ],
            "total_samples": res["total_samples"],
            "unique_features": res["unique_features"],
        }
        for lang, res in results.items()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_path}")

    return results
