# EncoderSAE

A production-ready Python library for training Sparse Autoencoders (SAEs) on **encoder-only models** (e.g., BERT, RoBERTa, multilingual E5). Unlike standard SAEs that train on token-level residual streams, EncoderSAE trains on **sentence-level embeddings** using mean pooling.

## Features

- 🎯 **Sentence-level embeddings**: Extracts and trains on pooled sentence representations (not token-level)
- 🔄 **Mean pooling**: Automatically applies mean pooling across tokens with attention mask support
- 💾 **Activation caching**: Optionally cache extracted activations to combined `.pt` files for faster training
- 📊 **WandB integration**: Automatic experiment tracking with auto-generated run names
- 🔧 **Flexible data loading**: Supports both local JSON/JSONL files and HuggingFace datasets
- ⚡ **Top-K sparsity**: Efficient sparse activation via top-k feature selection
- 🎛️ **Fire CLI**: Clean command-line interface with sensible defaults
- 🚀 **Multi-GPU support**: Efficient data-parallel training and activation extraction
- ⚙️ **Auxiliary loss**: Configurable regularization to reduce dead features and improve feature utilization
- 🔍 **Language analysis**: Built-in tools to analyze which SAE features correspond to different languages
- 🔄 **Language-agnostic inference**: Full pipeline API for creating embeddings without language-specific features

## Installation

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA/MPS support (optional, for GPU acceleration)
- vLLM (optional, for faster activation extraction with `--use_vllm`)

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
  - Dictionary size = `input_dim × expansion_factor`
  - Larger values create more features but may increase dead features
- `sparsity`: Number of top features to keep per sample (default: `64`)
  - Controls sparsity ratio: `sparsity / dict_size`
  - Recommended: 0.5% - 2% of dictionary size for good feature utilization
- `batch_size`: SAE training batch size (default: `32`)
- `epochs`: Number of training epochs (default: `1`)
- `lr`: Learning rate (default: `3e-4`)
  - Recommended: `1e-3` to `5e-4` for most cases
- `grad_acc_steps`: Gradient accumulation steps (default: `1`)
- `aux_loss_coeff`: Coefficient for auxiliary loss that encourages feature usage (default: `1e-3`)
  - Set to `0.0` to disable auxiliary loss
  - Higher values more aggressively reduce dead features
  - Recommended: `1e-3` to `1e-2` for models with high dead feature rates
- `aux_loss_target`: Target fraction of samples where each feature should appear in top-k (default: `0.01`)
  - Features used less than this fraction are penalized
  - Recommended: `0.005` (0.5%) to `0.02` (2%) depending on desired feature specialization

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

**Solution**: The library automatically sets this, but if you encounter issues, set the environment variable before running:

```bash
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
uv run -m EncoderSAE.main --model="..." --dataset="..." --num_gpus=8
```

### Training & Validation

- `val_step`: Run validation every N training steps (default: `1000`)
  - Set to `None` or `<=0` for end-of-epoch validation only
  - Useful for monitoring overfitting during training
- `val_set`: Path to validation activations directory (if precomputed)
  - Auto-detects `.pt` file if directory is provided
  - Ignored if `val_dataset` is provided
- `val_dataset`: HuggingFace dataset ID or local JSON/JSONL file for validation
  - If provided, a separate validation activation set is extracted
  - `val_split` is ignored when this is set

### Logging

- `wandb_project`: WandB project name (default: `"encodersae"`)
- `wandb_run_name`: Custom run name (auto-generated if `None`)
- `log_steps`: Log metrics every N steps (default: `10`)
- `seed`: Random seed (default: `42`)

### Output

- `save_dir`: Directory to save model checkpoints
  - Auto-generated as `./checkpoints/{model}_{dataset}/exp{expansion_factor}_k{sparsity}_lr{lr}` if `None`
  - Creates hyperparameter-specific subdirectories for easy organization
- `checkpoint_steps`: Save checkpoints every N training steps (default: `1000`)
  - Checkpoints saved as `checkpoint_step_{step}.pt`
  - Final model saved as `final_model.pt`

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
  - Measures reconstruction quality: `1 - (explained_variance / total_variance)`
  - Lower values indicate better reconstruction
- **dead_features**: Fraction of features that never fired in the batch (0-1)
  - Lower values indicate better feature utilization
  - Target: < 0.3 (30% dead features) for good models
- **l0_norm**: Average number of active features per sample
  - Should approximately equal `sparsity` when top-k is working correctly
- **aux_loss**: Auxiliary loss value (only when `aux_loss_coeff > 0`)
  - Penalizes features that are used less than `aux_loss_target`
  - Helps reduce dead features during training

All metrics are logged for both training (`train/*`) and validation (`val/*`) sets.

## Example: Training on Custom Data

```bash
# Train on local JSONL file with auxiliary loss
uv run -m EncoderSAE.main \
    --dataset="./data/my_texts.jsonl" \
    --model="bert-base-uncased" \
    --expansion_factor=32 \
    --sparsity=64 \
    --epochs=1 \
    --aux_loss_coeff=1e-3 \
    --aux_loss_target=0.01

# Train on HuggingFace dataset with multi-GPU
uv run -m EncoderSAE.main \
    --dataset="wikitext" \
    --text_column="text" \
    --model="roberta-base" \
    --num_gpus=8 \
    --use_vllm \
    --val_step=500
```

## Hyperparameter Sweeping

A sweep script is included for systematic hyperparameter exploration:

```bash
bash run_sweep.sh
```

The script sweeps over:

- **Expansion factors**: 32, 64, 128
- **Sparsity ratios**: 0.5%, 1%, 2% of dictionary size
- **Learning rates**: 1e-3, 5e-4, 2e-4, 1e-4
- **Auxiliary loss coefficients**: 0.0, 1e-3, 5e-3, 1e-2
- **Auxiliary loss targets**: 0.005, 0.01, 0.02

Edit `run_sweep.sh` to customize the sweep ranges. Each run is automatically logged to WandB with a descriptive run name.

## Language Feature Analysis

After training an SAE, you can analyze which features correspond to which languages and generate masks to remove language-specific information:

```bash
uv run -m EncoderSAE.analyze_main \
    --sae_path="./checkpoints/model_dataset/exp64_k2048_lr0.001/final_model.pt" \
    --validation_data="data/4lang_validation.jsonl" \
    --model="intfloat/multilingual-e5-large" \
    --mask_threshold=0.8 \
    --use_vllm \
    --num_gpus=8
```

This will:

- Load your trained SAE checkpoint
- Process the validation set grouped by language (automatically detects all languages)
- Identify which SAE features fire most frequently for each language
- Generate feature masks for language-specific feature removal
- Save results to `./analysis/{hyperparams}_{checkpoint}/`

### Output Files

1. **`language_features.json`**: Analysis results with feature frequencies per language
   - `top_features`: List of feature indices (all features if `top_k_features=None`)
   - `top_features_detailed`: Feature indices with counts and percentages
   - `total_samples`: Number of samples per language
   - `unique_features`: Number of unique features that fired

2. **`language_features_{language}_mask.pt`**: Individual mask per language
   - Boolean tensor (dict_size,) where 1 = feature fires >threshold% for that language
   - Use to remove language-specific features for a specific language

3. **`language_features_combined_masks.pt`**: Dictionary of all language masks
   - `{language: mask_tensor}` for all languages
   - Convenient way to load all masks at once

4. **`language_features_combined_index.pt`**: Combined feature index
   - Contains tensor of all language-specific feature indices (union across languages)
   - Metadata: number of features, languages, mask_threshold used

5. **`language_features_combined_mask.pt`**: Union mask for all languages
   - Boolean tensor where 1 = language-specific in ANY language
   - Use this to remove ALL language-specific features at once

### Analysis Arguments

- `mask_threshold`: Percentage threshold (0.0-1.0) for mask generation (default: 0.8 = 80%)
  - Features firing above this threshold for a language are considered language-specific
  - Higher values = more strict (fewer features marked as language-specific)
- `top_k_features`: Number of features to show in JSON (default: None = show all)
  - Only affects JSON reporting, does NOT affect mask generation
  - Masks always check ALL features, filtered by `mask_threshold`

## Language-Agnostic Inference

After training an SAE and analyzing language-specific features, you can create language-agnostic embeddings by removing language-specific features. The library provides three ways to do this:

### Option 1: Class-Based Pipeline (Recommended for Repeated Use)

The `LanguageAgnosticEncoder` class is ideal when you need to encode multiple batches, as it loads the model and SAE once and reuses them:

```python
from EncoderSAE import LanguageAgnosticEncoder

# Initialize the encoder (loads SAE and mask once)
encoder = LanguageAgnosticEncoder(
    model_name="intfloat/multilingual-e5-large",
    sae_path="checkpoints/model_dataset/exp32_k1024_lr0.001/final_model.pt",
    mask_path="analysis/.../language_features_combined_mask.pt",
    batch_size=32,
    max_length=512,
    use_vllm=True,              # Use vLLM for faster inference
    num_gpus=8,                 # Number of GPUs for vLLM
    gpu_memory_utilization=0.9,
)

# Encode single text or list of texts
texts = ["Hello world", "Bonjour le monde", "Hola mundo"]
embeddings = encoder.encode(texts)
# Shape: (3, dict_size) - language-specific features already removed

# Can encode multiple batches efficiently
more_texts = ["Another sentence", "Une autre phrase"]
more_embeddings = encoder.encode(more_texts)
```

**Parameters:**
- `model_name`: HuggingFace model ID (must match the model used for SAE training)
- `sae_path`: Path to trained SAE checkpoint (`.pt` file)
- `mask_path`: Path to language mask file (`.pt` file)
  - Can use individual language mask: `language_features_{lang}_mask.pt`
  - Or combined union mask: `language_features_combined_mask.pt` (recommended)
- `batch_size`: Batch size for processing (default: 32)
- `max_length`: Maximum sequence length (default: 512)
- `use_vllm`: Use vLLM for faster base model inference (default: True)
- `num_gpus`: Number of GPUs for vLLM (default: None = auto-detect)
- `gpu_memory_utilization`: GPU memory utilization for vLLM (default: 0.9)

### Option 2: Function-Based API (Convenient for One-Time Use)

The `infer_language_agnostic` function provides a simple one-call interface:

```python
from EncoderSAE import infer_language_agnostic
import torch

# Full pipeline: text -> base model -> SAE -> mask -> output
embeddings = infer_language_agnostic(
    model_name="intfloat/multilingual-e5-large",
    sae="checkpoints/.../final_model.pt",  # Can be path or EncoderSAE instance
    mask_path="analysis/.../language_features_combined_mask.pt",
    texts=["Hello world", "Bonjour le monde"],
    batch_size=32,
    max_length=512,
    use_vllm=True,
    num_gpus=8,
    gpu_memory_utilization=0.9,
)
# Returns: torch.Tensor of shape (len(texts), dict_size)
```

**Note**: This function loads the model and SAE each time it's called, so use `LanguageAgnosticEncoder` if you need to encode multiple batches.

### Option 3: Manual Feature Masking (For Custom Pipelines)

If you already have SAE features and just need to apply the mask:

```python
from EncoderSAE import EncoderSAE, remove_language_features
import torch

# Load SAE and process activations
sae = EncoderSAE(input_dim=1024, expansion_factor=32, sparsity=64)
sae.load_state_dict(torch.load("checkpoints/.../final_model.pt"))
sae.eval()

# Get SAE features from activations
activations = ...  # Your base model activations (batch_size, input_dim)
with torch.no_grad():
    _, features, _, _, _ = sae(activations)  # Get sparse features

# Load mask and apply
mask = torch.load("analysis/.../language_features_combined_mask.pt")
language_agnostic_features = remove_language_features(features, mask)
# Shape: (batch_size, dict_size) - language-specific features set to 0
```

### Understanding Masks

Masks are boolean tensors of shape `(dict_size,)` where:
- `True` (1) = language-specific feature (will be set to 0)
- `False` (0) = contextual feature (preserved)

**Mask Types:**
- **Individual language mask** (`language_features_{lang}_mask.pt`): Features specific to one language
- **Combined union mask** (`language_features_combined_mask.pt`): Features specific to ANY language (recommended for general use)

### Use Cases

1. **Cross-lingual semantic search**: Create embeddings that focus on meaning rather than language
2. **Language-agnostic classification**: Train models that work across languages
3. **Multilingual retrieval**: Find semantically similar content regardless of language
4. **Bias reduction**: Remove language-specific artifacts from embeddings

### Example: Complete Workflow

```python
from EncoderSAE import LanguageAgnosticEncoder

# Step 1: Initialize encoder with trained SAE and language mask
encoder = LanguageAgnosticEncoder(
    model_name="intfloat/multilingual-e5-large",
    sae_path="checkpoints/.../final_model.pt",
    mask_path="analysis/.../language_features_combined_mask.pt",
    use_vllm=True,
    num_gpus=8,
)

# Step 2: Encode texts in any language
queries = [
    "What is machine learning?",
    "Qu'est-ce que l'apprentissage automatique?",
    "¿Qué es el aprendizaje automático?",
]
embeddings = encoder.encode(queries)

# Step 3: Use embeddings for downstream tasks
# These embeddings focus on semantic content, not language identity
# Similar meanings across languages will have similar embeddings
```

## Project Structure

```text
EncoderSAE/
├── EncoderSAE/
│   ├── __init__.py      # Package exports & multiprocessing setup
│   ├── model.py         # SAE architecture with auxiliary loss
│   ├── data.py          # Data loading, activation extraction (vLLM/HF)
│   ├── train.py         # Training loop with multi-GPU support
│   ├── utils.py         # WandB setup & utilities
│   ├── main.py          # CLI entry point (training)
│   ├── analyze.py       # Language feature analysis
│   ├── analyze_main.py  # CLI entry point (analysis)
│   └── inference.py     # Language-agnostic inference pipeline
├── run_main.py          # Wrapper script for multiprocessing compatibility
├── run_sweep.sh         # Hyperparameter sweep script
├── pyproject.toml       # Package configuration (uv/pip)
├── LICENSE              # MIT License
└── README.md
```

## Performance Tips

### Activation Extraction

- **Use vLLM** (`--use_vllm=True`) for significantly faster embedding extraction, especially for large datasets
- **Set `activation_batch_size`** larger than `batch_size` (e.g., 32768 vs 8192) to maximize throughput
- **Multi-GPU extraction**: With `num_gpus > 1`, vLLM uses data-parallel processing (one engine per GPU)

### Training

- **Monitor `dead_features`**: Should decrease over training, target < 0.3
- **Use `aux_loss_coeff`**: Helps reduce dead features, especially for large `expansion_factor` values
- **Validation frequency**: Set `val_step=500` to catch overfitting early
- **Multi-GPU training**: Automatically uses `DataParallel` when `num_gpus > 1`

### Hyperparameter Guidelines

- **Sparsity ratio**: Aim for 0.5% - 2% of dictionary size (`sparsity / dict_size`)
- **Learning rate**: Start with `1e-3` or `5e-4`, adjust based on convergence
- **Expansion factor**: Larger values (64, 128) may need stronger `aux_loss_coeff` to avoid dead features

## Troubleshooting

### vLLM Multiprocessing Error

If you encounter `RuntimeError: Cannot re-initialize CUDA in forked subprocess`:

**Solution**: Use the wrapper script or set environment variable:

```bash
# Option 1: Use wrapper script
python run_main.py --model="..." --dataset="..."

# Option 2: Set environment variable
export PYTHON_MULTIPROCESSING_START_METHOD=spawn
uv run -m EncoderSAE.main --model="..." --dataset="..."
```

### High Dead Features

If `dead_features > 0.5`:

- Increase `sparsity` (higher sparsity ratio)
- Increase `aux_loss_coeff` (e.g., `1e-2`)
- Decrease `expansion_factor` (smaller dictionary)
- Train for more epochs

### Overfitting

If validation loss increases while training loss decreases:

- Reduce learning rate
- Increase `val_step` frequency to monitor earlier
- Consider reducing `expansion_factor` or `sparsity`
- Train for fewer epochs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

EncoderSAE is released under the MIT License. See the `LICENSE` file for details.
