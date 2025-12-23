"""CLI entry point for language feature analysis."""

from typing import Optional

import fire

from .analyze import analyze_language_features


def main(
    sae_path: str,
    model: str = "intfloat/multilingual-e5-large",
    validation_data: str = "data/4lang_validation.jsonl",
    text_column: str = "text",
    language_column: str = "language",
    batch_size: int = 256,  # Increased default for better GPU utilization
    max_length: int = 512,
    top_k_features: Optional[int] = None,
    mask_threshold: float = 0.8,
    output_dir: str = None,
    use_vllm: bool = True,
    num_gpus: int = None,
    gpu_memory_utilization: float = 0.9,
    exclude_overlapping_features: bool = True,
):
    """
    Analyze which SAE features correspond to which languages.

    Args:
        sae_path: Path to trained SAE checkpoint (.pt file)
        model: HuggingFace model ID or local path (default: "intfloat/multilingual-e5-large")
        validation_data: Path to validation JSONL file with language labels
        text_column: Name of text column in validation data (default: "text")
        language_column: Name of language column in validation data (default: "language")
        batch_size: Batch size for processing (default: 32)
        max_length: Maximum sequence length (default: 512)
        top_k_features: Number of top features to report per language in JSON (default: None = show all features, for reporting only, does not affect masks)
        mask_threshold: Percentage threshold (0.0-1.0) for mask generation (default: 0.8 = 80%).
            Features firing above this threshold for a language are considered language-specific.
        output_dir: Directory to save analysis results (None = auto-generate)
        use_vllm: Use vLLM for faster inference (default: True)
        num_gpus: Number of GPUs to use (None = auto-detect all)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)
        exclude_overlapping_features: If True (default), exclude features that are language-specific to multiple
            languages from the combined mask. Only features unique to a single language will be masked.
            If False, include all features that are language-specific in any language (union behavior).
    """
    results = analyze_language_features(
        sae_path=sae_path,
        model_name=model,
        validation_data=validation_data,
        text_column=text_column,
        language_column=language_column,
        batch_size=batch_size,
        max_length=max_length,
        top_k_features=top_k_features,
        mask_threshold=mask_threshold,
        output_dir=output_dir,
        use_vllm=use_vllm,
        num_gpus=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        exclude_overlapping_features=exclude_overlapping_features,
    )

    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    fire.Fire(main)
