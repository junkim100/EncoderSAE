# EncoderSAE

A production-ready Python library for training Sparse Autoencoders (SAEs) on **encoder-only models** (e.g., BERT, RoBERTa). Unlike standard SAEs that train on token-level residual streams, EncoderSAE trains on **sentence-level embeddings** using mean pooling.

## Features

- 🎯 **Sentence-level embeddings**: Extracts and trains on pooled sentence representations (not token-level)
- 🔄 **Mean pooling**: Automatically applies mean pooling across tokens with attention mask support
- 💾 **Activation caching**: Optionally cache extracted activations to disk for faster training
- 📊 **WandB integration**: Automatic experiment tracking with auto-generated run names
- 🔧 **Flexible data loading**: Supports both local JSON/JSONL files and HuggingFace datasets
- ⚡ **Top-K sparsity**: Efficient sparse activation via top-k feature selection
- 🎛️ **Fire CLI**: Clean command-line interface with sensible defaults

## Installation

Install the package:

```bash
uv pip install -e .
```

## Quick Start

### Basic Usage

Train with default settings:

```bash
uv run -m EncoderSAE.main
```

This will:
- Use model: `intfloat/multilingual-e5-large`
- Use dataset: `enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5`
- Extract sentence-level embeddings via mean pooling
- Cache activations to `./activations/`
- Auto-split 5% for validation
- Log to WandB project `encodersae`

### Custom Configuration

```bash
uv run -m EncoderSAE.main \
    --model="intfloat/multilingual-e5-large" \
    --dataset="path/to/data.jsonl" \
    --val_dataset="path/to/val.jsonl" \
    --expansion_factor=64 \
    --sparsity=128 \
    --batch_size=64 \
    --epochs=1 \
    --lr=1e-3 \
    --wandb_project="my-sae-project" \
    --checkpoint_steps=1000 \
    --num_gpus=8 \
    --use_vllm \
    --gpu_memory_utilization=0.9
```

### Using the CLI Command

After installation, you can also use:

```bash
encodersae --model="roberta-base" --dataset="my-dataset" --epochs=15
```

## Architecture

EncoderSAE uses a simple but effective architecture:

1. **Encoder**: Linear layer that expands input dimension by `expansion_factor`
2. **Top-K Activation**: Keeps only the top `sparsity` features per sample
3. **Decoder**: Linear layer that reconstructs the original dimension

The model is initialized with tied weights (decoder = encoder^T).

## Key Arguments

### Model & Data
- `model`: HuggingFace model ID or local path (default: `"intfloat/multilingual-e5-large"`)
- `dataset`: HuggingFace dataset ID or local JSON/JSONL file (default: `"enjalot/fineweb-edu-sample-10BT-chunked-500-nomic-text-v1.5"`)
- `text_column`: Name of text column in dataset (default: `"text"`)

### SAE Hyperparameters
- `expansion_factor`: Dictionary size multiplier (default: `32`)
- `sparsity`: Number of top features to keep (default: `64`)
- `batch_size`: SAE training batch size (default: `32`)
- `epochs`: Number of training epochs (default: `1`)
- `lr`: Learning rate (default: `3e-4`)
- `grad_acc_steps`: Gradient accumulation steps (default: `1`)

### Data & Caching
- `save_activations`: Cache activations to disk (default: `True`)
- `activations_dir`: Directory for cached activations (auto-generated as `./activations/{model}_{dataset}` if `None`)
- `val_set`: Path to validation activations directory (if precomputed). Ignored if `val_dataset` is provided; if both are `None`, 5% of train is auto-split for validation.
- `val_dataset`: HuggingFace dataset ID or local JSON/JSONL file for validation; if provided, a separate validation activation set is extracted and `val_split` is ignored.
- `val_split`: Fraction for validation split when `val_dataset` and `val_set` are `None` (default: `0.05`)
- `max_length`: Maximum sequence length (default: `512`)
- `max_samples`: Limit number of samples (default: `None` = all)
- `activation_batch_size`: Batch size for activation / embedding extraction
  (default: `None` = falls back to `batch_size`). You can often set this **larger**
  than the training batch size to maximize vLLM/HF throughput during offline
  embedding creation.
- `num_gpus`: Number of GPUs to use (default: `None` = auto-detect all available GPUs).
  - **Activation / embedding extraction**:
    - HF path (`use_vllm=False`): controls how many GPUs are used in the custom multi-process data-parallel extraction.
    - vLLM path (`use_vllm=True`): passed as `tensor_parallel_size` to `vllm.LLM` (tensor parallelism).
  - **SAE training**:
    - If CUDA is available and more than 1 GPU is visible, the training loop uses `torch.nn.DataParallel` with:
      - `num_gpus=None`: all visible GPUs
      - `num_gpus=1`: single-GPU training
      - `num_gpus>1`: that many GPUs (capped by `torch.cuda.device_count()`).
- `use_vllm`: Use vLLM for faster activation extraction with `task="embed"` (default: `False`).
- `gpu_memory_utilization`: GPU memory utilization for vLLM, 0.0-1.0 (default: `0.9`)

### Troubleshooting: vLLM Multiprocessing Error

If you encounter the error `RuntimeError: Cannot re-initialize CUDA in forked subprocess`, this is because vLLM requires the multiprocessing start method to be set to `'spawn'` for CUDA compatibility.

**Solution**: Set the environment variable before running:
```bash
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
uv run -m EncoderSAE.main --model="..." --dataset="..." --num_gpus=8
```

### Logging
- `wandb_project`: WandB project name (default: `"encodersae"`)
- `wandb_run_name`: Custom run name (auto-generated if `None`)
- `log_steps`: Log metrics every N steps (default: `10`)
- `seed`: Random seed (default: `42`)

### Output
- `save_dir`: Directory to save model checkpoints (auto-generated as `./checkpoints/{model}_{dataset}` if `None`)
- `checkpoint_steps`: Save checkpoints every N training steps (default: `1000`)

## How It Works

1. **Data Loading**: Automatically detects if `dataset` is a local file or HuggingFace dataset ID
2. **Activation Extraction**:
   - Passes text through the encoder model
   - Extracts last layer hidden states
   - Applies **mean pooling** across tokens (with attention mask)
   - Optionally caches pooled vectors to `.pt` files
3. **Training**: Trains SAE on the pooled sentence embeddings
4. **Metrics**: Tracks `loss`, `fvu` (Fraction of Variance Unexplained), `dead_features`, and `l0_norm`

## Metrics

The training loop logs the following metrics to WandB:

- **loss**: Mean squared error reconstruction loss
- **fvu**: Fraction of Variance Unexplained (lower is better)
- **dead_features**: Fraction of features that never fired in the batch
- **l0_norm**: Average number of active features per sample

## Example: Training on Custom Data

```bash
# Train on local JSONL file
uv run -m EncoderSAE.main \
    --dataset="./data/my_texts.jsonl" \
    --model="bert-base-uncased" \
    --expansion_factor=32 \
    --sparsity=64 \
    --epochs=1

# Train on HuggingFace dataset
uv run -m EncoderSAE.main \
    --dataset="wikitext" \
    --text_column="text" \
    --model="roberta-base"
```

## Language Feature Analysis

After training an SAE, you can analyze which features correspond to which languages:

```bash
uv run -m EncoderSAE.analyze_main \
    --sae_path="./checkpoints/model_dataset/final_model.pt" \
    --validation_data="data/4lang_validation.jsonl" \
    --model="intfloat/multilingual-e5-large" \
    --top_k_features=20 \
    --use_vllm \
    --num_gpus=8
```

This will:
- Load your trained SAE checkpoint
- Process the validation set grouped by language
- Identify which SAE features fire most frequently for each language
- Save results to `./analysis/{checkpoint_name}/language_features.json`

The output includes:
- Top feature indices per language
- Feature activation frequencies
- Percentage of samples where each feature fires

## Project Structure

```
EncoderSAE/
├── EncoderSAE/
│   ├── __init__.py      # Package exports
│   ├── model.py         # SAE architecture
│   ├── data.py          # Data loading & mean pooling
│   ├── train.py         # Training loop
│   ├── utils.py         # WandB setup & utilities
│   ├── main.py          # CLI entry point (training)
│   ├── analyze.py       # Language feature analysis
│   └── analyze_main.py  # CLI entry point (analysis)
├── pyproject.toml       # Package configuration (uv/pip)
└── README.md
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA/MPS support (optional, for GPU acceleration)

## License

EncoderSAE is released under the MIT License. See the `LICENSE` file for details.
