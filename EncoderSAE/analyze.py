"""Analysis tools for identifying language-specific SAE features."""

import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from .model import EncoderSAE
from .data import load_data, mean_pool, extract_activations
from .utils import get_device


@contextmanager
def suppress_output():
    """Temporarily suppress stdout/stderr to hide vLLM progress bars."""
    import os
    import sys

    # Save original file descriptors
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect to devnull
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def analyze_language_features(
    sae_path: str,
    model_name: str,
    validation_data: str,
    text_column: str = "text",
    language_column: str = "language",
    batch_size: int = 32,
    max_length: int = 512,
    top_k_features: Optional[int] = None,
    mask_threshold: float = 0.5,
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
        top_k_features: Number of top features to report per language in JSON (None = show all features, for reporting only, does not affect masks)
        mask_threshold: Percentage threshold (0.0-1.0) for mask generation. Features firing above this threshold
            for a language are considered language-specific and included in the mask. Default: 0.5 (50%)
        output_dir: Directory to save analysis results (None = auto-generate)
        use_vllm: Use vLLM for faster inference
        num_gpus: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization for vLLM

    Returns:
        Dictionary mapping language -> analysis results
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
            # Suppress vLLM INFO logs and progress bars
            import logging

            vllm_logger = logging.getLogger("vllm")
            vllm_logger.setLevel(logging.WARNING)

            from vllm import LLM

            llm = LLM(
                model=model_name,
                trust_remote_code=True,
                enforce_eager=True,
                tensor_parallel_size=num_gpus or torch.cuda.device_count(),
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_length,
                task="embed",
                disable_log_stats=True,  # Disable vLLM's internal progress stats
            )

            for language, texts in texts_by_language.items():
                print(f"\nProcessing {language} ({len(texts)} samples)...")

                # Process in batches
                num_batches = (len(texts) + batch_size - 1) // batch_size

                for batch_idx in tqdm(
                    range(num_batches), desc=f"{language}", leave=True
                ):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]

                    # Get embeddings from vLLM using encode() API
                    # Suppress vLLM's internal progress bars ("Adding requests", "Processed prompts")
                    # by redirecting stdout/stderr during the encode call
                    try:
                        with suppress_output():
                            outputs = llm.encode(
                                batch_texts, pooling_task="embed", use_tqdm=False
                            )
                    except Exception as e:
                        # Re-raise with original stdout/stderr restored so error is visible
                        print(f"Error during vLLM encode: {e}", file=sys.stderr)
                        raise

                    # Extract embedding tensors with fallback logic (same as data.py)
                    embedding_tensors = []
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

                        # Ensure it's float32
                        if not isinstance(embedding_tensor, torch.Tensor):
                            embedding_tensor = torch.tensor(
                                embedding_tensor, dtype=torch.float32
                            )

                        if embedding_tensor.dtype != torch.float32:
                            embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                        embedding_tensors.append(embedding_tensor)

                    # Stack into batch tensor
                    batch_activations = torch.stack(embedding_tensors)

                    batch_activations = batch_activations.to(device)

                    # Pass through SAE to get features
                    with torch.no_grad():
                        _, features, _, _ = sae(batch_activations)

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
                    _, features, _, _ = sae(batch_activations)

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

        # Get top features (all if top_k_features is None, otherwise top k)
        if top_k_features is None:
            # Show all features
            top_features = [feat_idx for feat_idx, count in sorted_features]
            top_features_with_counts = [
                (feat_idx, count, count / total_samples * 100)
                for feat_idx, count in sorted_features
            ]
        else:
            # Limit to top k
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
        if top_k_features is None:
            print(f"  All {len(top_features)} features:")
            display_count = min(10, len(top_features_with_counts))
        else:
            print(f"  Top {top_k_features} features:")
            display_count = min(10, len(top_features_with_counts))

        for feat_idx, count, pct in top_features_with_counts[:display_count]:
            print(
                f"    Feature {feat_idx:5d}: {count:4d} times ({pct:5.2f}% of samples)"
            )

    # Save results
    if output_dir is None:
        # Generate output directory name from checkpoint path
        # e.g., "checkpoints/model_dataset/exp64_k2048_lr0.001/final_model.pt"
        # -> "analysis/model_dataset_exp64_k2048_lr0.001_final"
        sae_path_obj = Path(sae_path)

        # Get parent directory name (e.g., "exp64_k2048_lr0.001")
        parent_dir = sae_path_obj.parent.name

        # Get checkpoint name from filename (e.g., "final_model" or "checkpoint_step_6000")
        checkpoint_name = sae_path_obj.stem

        # Combine: model_dataset_exp64_k2048_lr0.001_final or model_dataset_exp64_k2048_lr0.001_step_6000
        if checkpoint_name == "final_model":
            analysis_dir_name = f"{parent_dir}_final"
        elif checkpoint_name.startswith("checkpoint_step_"):
            step_num = checkpoint_name.replace("checkpoint_step_", "")
            analysis_dir_name = f"{parent_dir}_step_{step_num}"
        else:
            # Fallback: use checkpoint name as-is
            analysis_dir_name = f"{parent_dir}_{checkpoint_name}"

        output_dir = f"./analysis/{analysis_dir_name}"

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

    # Generate feature masks for each language
    # Mask: 1 for features that fire above threshold percentage, 0 otherwise
    # Note: This checks ALL features, not just top_k_features (top_k is only for JSON reporting)
    print("\n" + "=" * 60)
    print("Generating Language Feature Masks")
    print("=" * 60)
    print(f"Mask threshold: >{mask_threshold * 100:.1f}% of samples")
    print(f"Dictionary size: {dict_size}")
    if top_k_features is not None:
        print(
            f"Note: All features are checked for masks, not just top {top_k_features} (top_k is for JSON reporting only)"
        )
    else:
        print("Note: All features are included in JSON output and checked for masks")

    mask_threshold_pct = mask_threshold * 100.0
    all_masks = {}  # Dictionary to store all language masks

    for language in sorted(language_features.keys()):
        # Create mask tensor: 1 for features with >50% usage, 0 otherwise
        mask = torch.zeros(dict_size, dtype=torch.bool)

        # Get all feature counts for this language (not just top_k)
        feature_counts = language_features[language]
        total_samples = len(texts_by_language[language])

        # Check ALL features (not just top_k_features) and filter by percentage threshold
        # This ensures we capture all language-specific features, regardless of how many exist
        for feat_idx, count in feature_counts.items():
            percentage = (count / total_samples) * 100.0
            if percentage > mask_threshold_pct:
                mask[feat_idx] = 1

        # Count how many features are enabled for this language
        num_enabled = mask.sum().item()

        print(
            f"\n{language.upper()}: {num_enabled} features enabled "
            f"(>{mask_threshold_pct}% of {total_samples} samples)"
        )

        # Save individual mask as .pt file
        mask_path = output_path / f"language_features_{language}_mask.pt"
        torch.save(mask, mask_path)
        print(f"  Saved mask to {mask_path}")

        # Store in combined dictionary
        all_masks[language] = mask

    # Save combined mask file with all languages
    combined_mask_path = output_path / "language_features_combined_masks.pt"
    torch.save(all_masks, combined_mask_path)
    print(f"\nCombined masks saved to {combined_mask_path}")
    print(
        f"  Contains masks for {len(all_masks)} languages: {', '.join(sorted(all_masks.keys()))}"
    )

    # Create combined index of all language-specific features (union across all languages)
    # This is the set of all features that fire >threshold% for at least one language
    all_language_specific_features = set()
    for language, mask in all_masks.items():
        # Get indices where mask is True (feature fires >threshold% for this language)
        language_feature_indices = mask.nonzero(as_tuple=True)[0].tolist()
        all_language_specific_features.update(language_feature_indices)

    # Convert to sorted list for easy use
    all_language_specific_features_list = sorted(list(all_language_specific_features))

    # Save combined feature index
    combined_index_path = output_path / "language_features_combined_index.pt"
    torch.save(
        {
            "feature_indices": torch.tensor(
                all_language_specific_features_list, dtype=torch.long
            ),
            "num_features": len(all_language_specific_features_list),
            "languages": sorted(all_masks.keys()),
            "mask_threshold": mask_threshold,
        },
        combined_index_path,
    )
    print(f"\nCombined feature index saved to {combined_index_path}")
    print(
        f"  Total language-specific features (union across all languages): {len(all_language_specific_features_list)}"
    )
    print(
        f"  These features fire >{mask_threshold * 100:.1f}% for at least one language"
    )

    # Also create a combined union mask
    # This mask is 1 for ANY feature that is language-specific in ANY language
    # Use this to remove ALL language-specific features at once
    combined_mask = torch.zeros(dict_size, dtype=torch.bool)
    for mask in all_masks.values():
        combined_mask |= (
            mask  # Union: 1 if feature is language-specific in any language
        )

    combined_mask_path_union = output_path / "language_features_combined_mask.pt"
    torch.save(combined_mask, combined_mask_path_union)
    print(f"\nCombined union mask saved to {combined_mask_path_union}")
    print(
        f"  Contains {combined_mask.sum().item()} features that are language-specific in at least one language"
    )
    print(
        f"  Usage: features_agnostic = features * (~combined_mask)  # Remove all language-specific features"
    )

    print(f"\nAll language masks and indices saved to {output_path}")

    return results
