"""CLI entry point for language feature analysis."""

import fire

from .analyze import analyze_language_features


def main(
    sae_path: str,
    model: str = "intfloat/multilingual-e5-large",
    validation_data: str = "data/4lang_validation.jsonl",
    text_column: str = "text",
    language_column: str = "language",
    batch_size: int = 32,
    max_length: int = 512,
    top_k_features: int = 20,
    output_dir: str = None,
    use_vllm: bool = True,
    num_gpus: int = None,
    gpu_memory_utilization: float = 0.9,
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
        top_k_features: Number of top features to report per language (default: 20)
        output_dir: Directory to save analysis results (None = auto-generate)
        use_vllm: Use vLLM for faster inference (default: True)
        num_gpus: Number of GPUs to use (None = auto-detect all)
        gpu_memory_utilization: GPU memory utilization for vLLM (default: 0.9)
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
        output_dir=output_dir,
        use_vllm=use_vllm,
        num_gpus=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    fire.Fire(main)

