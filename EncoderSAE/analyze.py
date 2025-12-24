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
    batch_size: int = 256,  # Increased default for better GPU utilization
    max_length: int = 512,
    top_k_features: Optional[int] = None,
    mask_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    use_vllm: bool = True,
    num_gpus: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    exclude_overlapping_features: bool = True,
) -> dict:
    """
    Analyze which SAE features correspond to which languages.

    Args:
        sae_path: Path to trained SAE checkpoint (.pt file)
        model_name: HuggingFace model ID or local path (same as used for training)
        validation_data: Path to validation JSONL file with language labels
        text_column: Name of text column in validation data
        language_column: Name of language column in validation data
        batch_size: Batch size for SAE processing (vLLM processes all texts at once internally).
            This parameter mainly affects GPU memory usage during SAE feature extraction, not vLLM encoding speed.
        max_length: Maximum sequence length
        top_k_features: Number of top features to report per language in JSON (None = show all features, for reporting only, does not affect masks)
        mask_threshold: Percentage threshold (0.0-1.0) for mask generation. Features firing above this threshold
            for a language are considered language-specific and included in the mask. Default: 0.5 (50%)
        output_dir: Directory to save analysis results (None = auto-generate)
        use_vllm: Use vLLM for faster inference
        num_gpus: Number of GPUs to use
        gpu_memory_utilization: GPU memory utilization for vLLM
        exclude_overlapping_features: If True (default), exclude features that are language-specific to multiple
            languages from the combined mask. Only features unique to a single language will be masked.
            If False, include all features that are language-specific in any language (union behavior).

    Returns:
        Dictionary mapping language -> analysis results
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load validation data with language labels
    print(f"Loading validation data from {validation_data}")
    texts_by_language = defaultdict(list)
    all_texts = []  # Store all texts for batch processing
    text_to_language = []  # Map text index to language

    with open(validation_data, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get(text_column, "")
            language = data.get(language_column, "unknown")
            if text:
                texts_by_language[language].append(text)
                all_texts.append(text)
                text_to_language.append(language)

    print(f"Found {len(texts_by_language)} languages:")
    for lang, texts in texts_by_language.items():
        print(f"  {lang}: {len(texts)} samples")

    total_samples = len(all_texts)
    print(f"Total samples: {total_samples}")

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

    # Extract activations and features for all texts (much faster than per-language)
    print("\nExtracting features for all texts (batch processing)...")
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

            # Process all texts together - vLLM handles batching internally efficiently
            # Note: batch_size parameter doesn't significantly affect vLLM's internal processing
            # vLLM's encode() method processes requests efficiently regardless of input batch size
            print(f"Processing {total_samples} samples with vLLM (internal batching)...")

            # Pass all texts at once - vLLM will handle batching internally for optimal GPU utilization
            # This is more efficient than manual batching since vLLM can optimize its internal scheduling
            try:
                with suppress_output():
                    outputs = llm.encode(
                        all_texts, pooling_task="embed", use_tqdm=False
                    )
            except Exception as e:
                # If memory error, fall back to batched processing
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"Memory error with full batch, falling back to batched processing (batch_size={batch_size})...")
                    num_batches = (total_samples + batch_size - 1) // batch_size
                    all_outputs = []
                    for batch_idx in tqdm(
                        range(num_batches), desc="Extracting features", leave=True
                    ):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, total_samples)
                        batch_texts = all_texts[start_idx:end_idx]
                        with suppress_output():
                            batch_outputs = llm.encode(
                                batch_texts, pooling_task="embed", use_tqdm=False
                            )
                        all_outputs.extend(batch_outputs)
                    outputs = all_outputs
                else:
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
                                    torch.tensor(embedding_data, dtype=torch.float32)
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

                    # Ensure it's float32
                    if not isinstance(embedding_tensor, torch.Tensor):
                        embedding_tensor = torch.tensor(
                            embedding_tensor, dtype=torch.float32
                        )

                    if embedding_tensor.dtype != torch.float32:
                        embedding_tensor = embedding_tensor.to(dtype=torch.float32)

                    embedding_tensors.append(embedding_tensor)

                # Stack into batch tensor
                all_activations = torch.stack(embedding_tensors)
                all_activations = all_activations.to(device)

                # Process through SAE in batches to avoid memory issues
                # Use batch_size for SAE processing (this actually matters for GPU memory)
                sae_batch_size = batch_size if batch_size > 0 else 1024
                num_sae_batches = (len(all_activations) + sae_batch_size - 1) // sae_batch_size

                for sae_batch_idx in range(num_sae_batches):
                    sae_start_idx = sae_batch_idx * sae_batch_size
                    sae_end_idx = min(sae_start_idx + sae_batch_size, len(all_activations))
                    batch_activations = all_activations[sae_start_idx:sae_end_idx]
                    batch_languages_chunk = text_to_language[sae_start_idx:sae_end_idx]

                    # Pass through SAE to get features
                    with torch.no_grad():
                        _, features, _, _, _ = sae(batch_activations)

                        # Count which features fired for each language in this batch
                        for i, sample_features in enumerate(features):
                            language = batch_languages_chunk[i]
                            # Get top-k active features
                            active_features = (sample_features > 0).nonzero(as_tuple=True)[
                                0
                            ]
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

        # Process all texts together in batches (much faster than per-language)
        print(f"Processing {total_samples} samples in batches of {batch_size}...")
        num_batches = (total_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_texts = all_texts[start_idx:end_idx]
                batch_languages = text_to_language[start_idx:end_idx]

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
                _, features, _, _, _ = sae(batch_activations)

                # Count which features fired for each language in this batch
                for i, sample_features in enumerate(features):
                    language = batch_languages[i]
                    active_features = (sample_features > 0).nonzero(as_tuple=True)[0]
                    for feat_idx in active_features:
                        language_features[language][feat_idx.item()] += 1

    # Analyze and rank features per language
    print("\n" + "=" * 60)
    print("Language-Specific Feature Analysis")
    print("=" * 60)

    results = {}
    mask_threshold_pct = mask_threshold * 100.0

    for language in sorted(language_features.keys()):
        feature_counts = language_features[language]
        total_samples = len(texts_by_language[language])

        # Sort features by frequency
        sorted_features = sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Filter features by threshold: only include features with percentage >= mask_threshold
        threshold_features_with_counts = [
            (feat_idx, count, count / total_samples * 100)
            for feat_idx, count in sorted_features
            if (count / total_samples * 100) >= mask_threshold_pct
        ]

        # Get the feature indices
        threshold_features = [
            feat_idx for feat_idx, _, _ in threshold_features_with_counts
        ]

        # Count how many features meet the threshold
        num_features_above_threshold = len(threshold_features)

        results[language] = {
            "top_features": threshold_features,
            "top_features_detailed": threshold_features_with_counts,
            "total_samples": total_samples,
            "unique_features": len(feature_counts),
            "features_above_threshold": num_features_above_threshold,
            "threshold_percentage": mask_threshold_pct,
        }

        print(
            f"\n{language.upper()} ({total_samples} samples, {len(feature_counts)} unique features):"
        )
        print(
            f"  Features above threshold ({mask_threshold_pct:.1f}%): {num_features_above_threshold}"
        )
        display_count = min(10, len(threshold_features_with_counts))

        for feat_idx, count, pct in threshold_features_with_counts[:display_count]:
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
            base_name = f"{parent_dir}_final"
        elif checkpoint_name.startswith("checkpoint_step_"):
            step_num = checkpoint_name.replace("checkpoint_step_", "")
            base_name = f"{parent_dir}_step_{step_num}"
        else:
            # Fallback: use checkpoint name as-is
            base_name = f"{parent_dir}_{checkpoint_name}"

        # Add mask threshold to directory name
        # Format: mask0_95 for threshold 0.95, mask0_995 for threshold 0.995
        mask_str = f"mask{str(mask_threshold).replace('.', '_')}"
        analysis_dir_name = f"{base_name}_{mask_str}"

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
            "features_above_threshold": res["features_above_threshold"],
            "threshold_percentage": res["threshold_percentage"],
        }
        for lang, res in results.items()
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_path}")

    # Generate feature masks for each language
    # Mask semantics: mask[i] = True means feature i fires in >= threshold% of samples for this language
    # When used in evaluation: features where mask[i] = True are zeroed out (disabled)
    # This removes language-specific features to create language-agnostic embeddings
    print("\n" + "=" * 60)
    print("Generating Language Feature Masks")
    print("=" * 60)
    print(f"Mask threshold: >= {mask_threshold * 100:.1f}% of samples")
    print(f"Dictionary size: {dict_size}")
    print(
        f"Note: Features firing in >= {mask_threshold_pct:.1f}% of samples will be masked (disabled)"
    )

    all_masks = {}  # Dictionary to store all language masks

    for language in sorted(language_features.keys()):
        # Create mask tensor: True for features that fire >= threshold% of samples
        # mask[i] = True means feature i should be disabled (zeroed out) during inference
        # Example: If mask_threshold=0.99 (99%), features firing in >=99% of samples are marked True
        mask = torch.zeros(dict_size, dtype=torch.bool)

        # Get all feature counts for this language (not just top_k)
        feature_counts = language_features[language]
        total_samples = len(texts_by_language[language])

        # Check ALL features (not just top_k_features) and filter by percentage threshold
        # This ensures we capture all language-specific features, regardless of how many exist
        #
        # IMPORTANT: "Activation percentage" here means FREQUENCY (how often a feature fires),
        # NOT the activation value magnitude. A feature is considered language-specific if it
        # fires (activation > 0) in >= threshold% of samples for that language.
        #
        # Example: If mask_threshold=0.99 (99%), features that fire in >=99% of samples are masked.
        # This identifies features that are consistently active for a language, regardless of
        # their activation value magnitude.
        #
        # count = number of samples where this feature had activation > 0 (fired)
        # percentage = (count / total_samples) * 100.0 = frequency percentage
        for feat_idx, count in feature_counts.items():
            percentage = (count / total_samples) * 100.0
            # Use >= to include features at exactly threshold (e.g., exactly 99% for threshold 0.99)
            # mask_threshold_pct is already converted from decimal (0.99) to percentage (99.0)
            if percentage >= mask_threshold_pct:
                mask[feat_idx] = True  # Mark this feature for masking (disabling)

        # Count how many features are marked for masking (disabling) for this language
        num_enabled = mask.sum().item()

        print(
            f"\n{language.upper()}: {num_enabled} features marked for masking "
            f"(>= {mask_threshold_pct:.1f}% of {total_samples} samples)"
        )

        # Store in combined dictionary (individual files not saved - use per_language_masks.pt instead)
        all_masks[language] = mask

    # Save per-language masks dictionary (all individual language masks in one file)
    per_language_masks_path = output_path / "language_features_per_language_masks.pt"
    torch.save(all_masks, per_language_masks_path)
    print(f"\nPer-language masks saved to {per_language_masks_path}")
    print(
        f"  Contains masks for {len(all_masks)} languages: {', '.join(sorted(all_masks.keys()))}"
    )
    print(
        f"  Usage: Load this file to get individual masks for each language, "
        f"then combine them to disable specific languages"
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

    # Create a combined mask
    # Combined mask semantics: mask[i] = True means feature i should be disabled (zeroed out)
    # Option 1: Exclude overlapping features (default) - only features unique to a single language
    # Option 2: Include all features (union) - features language-specific in any language
    # When applied: z_masked[:, mask] = 0 zeros out features where mask is True
    combined_mask_path_union = output_path / "language_features_combined_mask.pt"

    if exclude_overlapping_features:
        # Count how many languages each feature appears in
        feature_language_count = torch.zeros(dict_size, dtype=torch.long)
        for mask in all_masks.values():
            feature_language_count += mask.long()

        # Only include features that appear in exactly ONE language mask
        combined_mask = (feature_language_count == 1).bool()

        # Count overlapping features for reporting
        overlapping_count = (feature_language_count > 1).sum().item()
        unique_count = combined_mask.sum().item()

        print(
            f"\nCombined mask (excluding overlapping features) saved to {combined_mask_path_union}"
        )
        print(f"  Unique features (single language only): {unique_count}")
        print(
            f"  Overlapping features (multiple languages): {overlapping_count} (excluded)"
        )
        print(f"  Total language-specific features: {unique_count + overlapping_count}")
    else:
        # Union: include all features that are language-specific in any language
        combined_mask = torch.zeros(dict_size, dtype=torch.bool)
        for mask in all_masks.values():
            combined_mask |= (
                mask  # Union: 1 if feature is language-specific in any language
            )

        print(
            f"\nCombined union mask (including all language-specific features) saved to {combined_mask_path_union}"
        )
        print(
            f"  Contains {combined_mask.sum().item()} features that are language-specific in at least one language"
        )

    torch.save(combined_mask, combined_mask_path_union)
    print(
        f"  Usage: features_agnostic = features * (~combined_mask)  # Remove language-specific features"
    )

    print(f"\nAll language masks and indices saved to {output_path}")

    return results
