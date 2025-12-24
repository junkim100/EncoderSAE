import os
import gc
import math
import json
import tempfile
import shutil
from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    Features,
    Value,
    load_from_disk,
)
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# =========================================================
# 1. Configuration
# =========================================================
CACHE_DIR = "/data_x/aa007878/encodersae/data/cache"
TEMP_DIR = "/data_x/aa007878/encodersae/data/temp_processed"  # Temp directory for per-language files
NUM_PROC = 60
TOKENIZER_NAME = "intfloat/multilingual-e5-large"
MODEL_MAX_LENGTH = 512  # Model's maximum sequence length
TARGET_MIN_LEN = 250
# Reserve space for special tokens (E5 typically adds 2 tokens: start + end)
# Using 510 to be safe (512 - 2)
TARGET_MAX_LEN = MODEL_MAX_LENGTH - 2
NUM_LANGUAGES = 13
# Sample size for quick count estimation (process first N items to estimate chunking ratio)
ESTIMATION_SAMPLE_SIZE = 10000

# 13 languages from Belebele corpus
LANG_CODES = [
    "ar",
    "dt",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "pt",
    "ru",
    "vi",
    "zh",
]

# Mapping from language codes to mmarco full names
MMARCO_LANG_MAP = {
    "ar": "arabic",
    "dt": "dutch",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "pt": "portuguese",
    "ru": "russian",
    "vi": "vietnamese",
    "zh": "chinese",
}

# MIRACL language codes (only languages available in MIRACL)
# MIRACL available languages: ar, en, es, fr, hi, id, ja, ru, zh
# Missing from MIRACL: dt (Dutch), it (Italian), pt (Portuguese), vi (Vietnamese)
# These will use mMARCO data only
MIRACL_LANG_MAP = {
    "ar": "ar",  # Arabic - available
    "en": "en",  # English - available
    "es": "es",  # Spanish - available
    "fr": "fr",  # French - available
    "hi": "hi",  # Hindi - available
    "id": "id",  # Indonesian - available
    "ja": "ja",  # Japanese - available
    "ru": "ru",  # Russian - available
    "zh": "zh",  # Chinese - available
    # dt, it, pt, vi are NOT in MIRACL - will use mMARCO only
}

# File counts for MIRACL languages (approximate, may vary)
MIRACL_LANG_FILES = {
    "en": 66,  # English has the most files
    "fr": 30,
    "es": 21,
    "ar": 21,
    "hi": 21,
    "id": 21,
    "ja": 21,
    "ru": 21,
    "zh": 10,
}

# Tokenizer (global for fork-safe multiprocessing)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# =========================================================
# 2. Data Loading Functions
# =========================================================


def load_miracl_lang(lang_code):
    """
    Load MIRACL corpus for a language if available.

    Returns None if language is not available in MIRACL (expected for dt, it, pt, vi).
    """
    # Map Belebele language code to MIRACL language code
    miracl_lang_code = MIRACL_LANG_MAP.get(lang_code)
    if miracl_lang_code is None:
        # Language not in MIRACL - this is expected for dt, it, pt, vi
        return None

    n_files = MIRACL_LANG_FILES.get(miracl_lang_code, 0)
    if n_files == 0:
        return None

    base_url = f"https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/miracl-corpus-v1.0-{miracl_lang_code}/docs-{{i}}.jsonl.gz"
    file_urls = [base_url.format(i=i) for i in range(n_files)]
    miracl_features = Features(
        {"docid": Value("string"), "title": Value("string"), "text": Value("string")}
    )

    try:
        return load_dataset(
            "json",
            data_files=file_urls,
            features=miracl_features,
            split="train",
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(
            f"Warning: Could not load MIRACL for {lang_code} (MIRACL code: {miracl_lang_code}): {e}"
        )
        return None


def load_lookup_dict(file_path):
    """Load TSV file as ID -> Text dictionary."""
    lookup = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 2:
                lookup[parts[0]] = parts[1]
    return lookup


def mmarco_generator(triples_path, queries_path, collection_path):
    """Generator for mMARCO dataset."""
    queries_dict = load_lookup_dict(queries_path)
    collection_dict = load_lookup_dict(collection_path)

    with open(triples_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                qid, pid, nid = line.rstrip().split("\t")
                yield {
                    "query": queries_dict[qid],
                    "positive": collection_dict[pid],
                    "negative": collection_dict[nid],
                }
            except (KeyError, ValueError):
                continue


def get_mmarco_lang(full_lang_name):
    """Load mMARCO dataset for a language."""
    try:
        coll_file = hf_hub_download(
            repo_id="unicamp-dl/mmarco",
            filename=f"data/google/collections/{full_lang_name}_collection.tsv",
            repo_type="dataset",
            cache_dir=CACHE_DIR,
        )
        query_file = hf_hub_download(
            repo_id="unicamp-dl/mmarco",
            filename=f"data/google/queries/train/{full_lang_name}_queries.train.tsv",
            repo_type="dataset",
            cache_dir=CACHE_DIR,
        )
        triples_file = hf_hub_download(
            repo_id="unicamp-dl/mmarco",
            filename="data/triples.train.ids.small.tsv",
            repo_type="dataset",
            cache_dir=CACHE_DIR,
        )

        mmarco_features = Features(
            {
                "query": Value("string"),
                "positive": Value("string"),
                "negative": Value("string"),
            }
        )

        return Dataset.from_generator(
            generator=mmarco_generator,
            gen_kwargs={
                "triples_path": triples_file,
                "queries_path": query_file,
                "collection_path": coll_file,
            },
            features=mmarco_features,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"Warning: Could not load mMARCO for {full_lang_name}: {e}")
        return None


# =========================================================
# 3. Data Processing Functions
# =========================================================


def standardize_miracl(batch, lang):
    """Standardize MIRACL format: title + text."""
    new_texts = [f"{t}\n{txt}" for t, txt in zip(batch["title"], batch["text"])]
    return {
        "text": new_texts,
        "source": ["miracl"] * len(batch["text"]),
        "language": [lang] * len(batch["text"]),
    }


def standardize_mmarco(batch, lang):
    """Standardize mMARCO format: query + positive."""
    new_texts = [f"{q}\n{p}" for q, p in zip(batch["query"], batch["positive"])]
    return {
        "text": new_texts,
        "source": ["mmarco"] * len(batch["query"]),
        "language": [lang] * len(batch["query"]),
    }


def smart_chunking_v2(batch):
    """
    Smart chunking: split long docs, merge short docs.
    Ensures all chunks are between TARGET_MIN_LEN and TARGET_MAX_LEN tokens
    (without special tokens), and verifies they don't exceed MODEL_MAX_LENGTH
    when special tokens are added.
    """
    encodings = tokenizer(batch["text"], add_special_tokens=False, truncation=False)
    input_ids_list = encodings["input_ids"]
    sources = batch["source"]
    langs = batch["language"]

    out_text, out_source, out_language = [], [], []
    frag_ids, frag_sources = [], set()
    frag_lang = langs[0] if langs else ""

    def verify_and_add_chunk(chunk_ids, source, lang):
        """Verify chunk doesn't exceed model max length with special tokens."""
        # Decode the chunk
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Re-tokenize with special tokens to verify actual length
        verified_encoding = tokenizer(
            chunk_text, add_special_tokens=True, truncation=False
        )
        actual_length = len(verified_encoding["input_ids"])

        if actual_length > MODEL_MAX_LENGTH:
            # This shouldn't happen if TARGET_MAX_LEN is conservative enough
            # But if it does, truncate the text to fit (safety measure)
            print(
                f"Warning: Chunk exceeds max length ({actual_length} > {MODEL_MAX_LENGTH}), truncating"
            )
            # Truncate by re-encoding with truncation
            truncated_encoding = tokenizer(
                chunk_text,
                add_special_tokens=True,
                truncation=True,
                max_length=MODEL_MAX_LENGTH,
            )
            chunk_text = tokenizer.decode(
                truncated_encoding["input_ids"][
                    1:-1
                ],  # Remove special tokens for storage
                skip_special_tokens=True,
            )

        out_text.append(chunk_text)
        out_source.append(source)
        out_language.append(lang)
        return True

    for i, ids in enumerate(input_ids_list):
        doc_len = len(ids)
        chunks = []

        # Split long documents
        if doc_len > TARGET_MAX_LEN:
            num_splits = math.ceil(doc_len / TARGET_MAX_LEN)
            split_size = math.ceil(doc_len / num_splits)
            for k in range(0, doc_len, split_size):
                chunks.append(ids[k : k + split_size])
        else:
            chunks.append(ids)

        # Process each chunk
        for chunk in chunks:
            chunk_len = len(chunk)

            # Case A: Sufficiently long (TARGET_MIN_LEN to TARGET_MAX_LEN)
            if chunk_len >= TARGET_MIN_LEN:
                verify_and_add_chunk(chunk, sources[i], langs[i])

            # Case B: Too short (<TARGET_MIN_LEN) - add to fragment buffer
            else:
                # If buffer would exceed max, flush it
                if len(frag_ids) + chunk_len > TARGET_MAX_LEN:
                    if frag_ids:
                        verify_and_add_chunk(
                            frag_ids, ",".join(sorted(frag_sources)), frag_lang
                        )
                    frag_ids, frag_sources = [], set()

                # Add to buffer
                frag_ids.extend(chunk)
                frag_sources.add(sources[i])
                frag_lang = langs[i]

                # If buffer is now sufficient, flush it
                if len(frag_ids) >= TARGET_MIN_LEN:
                    verify_and_add_chunk(
                        frag_ids, ",".join(sorted(frag_sources)), frag_lang
                    )
                    frag_ids, frag_sources = [], set()

    # Flush remaining fragments
    if frag_ids:
        verify_and_add_chunk(frag_ids, ",".join(sorted(frag_sources)), frag_lang)

    return {"text": out_text, "source": out_source, "language": out_language}


# =========================================================
# 4. Quick Count Estimation
# =========================================================


def estimate_chunk_count(lang_code, sample_size=ESTIMATION_SAMPLE_SIZE):
    """
    Quickly estimate the number of chunks that will be generated.
    Processes a sample and extrapolates.
    """
    print(f"  Estimating chunk count for [{lang_code.upper()}]...")

    full_lang_name = MMARCO_LANG_MAP[lang_code]

    # Load datasets
    ds_mmarco = get_mmarco_lang(full_lang_name)
    if ds_mmarco is None:
        return 0

    # MIRACL may not be available for all languages (dt, it, pt, vi are not in MIRACL)
    ds_miracl = load_miracl_lang(lang_code)

    # Standardize formats (quick, no chunking yet)
    mmarco_std = ds_mmarco.map(
        standardize_mmarco,
        fn_kwargs={"lang": lang_code},
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds_mmarco.column_names,
    )

    datasets_to_combine = [mmarco_std]
    if ds_miracl is not None:
        miracl_std = ds_miracl.map(
            standardize_miracl,
            fn_kwargs={"lang": lang_code},
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=ds_miracl.column_names,
        )
        datasets_to_combine.append(miracl_std)

    combined = concatenate_datasets(datasets_to_combine)
    total_items = len(combined)

    # Sample and estimate chunking ratio
    if total_items > sample_size:
        sample = combined.select(range(sample_size))
        sample_chunked = sample.map(
            smart_chunking_v2,
            batched=True,
            batch_size=2000,
            num_proc=NUM_PROC,
            remove_columns=sample.column_names,
        )
        sample_chunk_count = len(sample_chunked)
        chunk_ratio = sample_chunk_count / sample_size
        estimated_count = int(total_items * chunk_ratio)

        # Clean sample memory
        del sample, sample_chunked
    else:
        # If dataset is small, just process it all
        rechunked = combined.map(
            smart_chunking_v2,
            batched=True,
            batch_size=2000,
            num_proc=NUM_PROC,
            remove_columns=combined.column_names,
        )
        estimated_count = len(rechunked)
        del rechunked

    # Clean memory
    del ds_mmarco, mmarco_std
    if ds_miracl is not None:
        del ds_miracl, miracl_std
    del combined
    gc.collect()

    print(
        f"  -> [{lang_code}] Estimated {estimated_count:,} chunks from {total_items:,} items"
    )
    return estimated_count


# =========================================================
# 5. Main Processing Pipeline
# =========================================================


def process_language(lang_code, max_items=None, save_path=None):
    """
    Process a single language: load, standardize, chunk.
    Balances MIRACL and mMARCO sources to 50/50 if possible.

    Args:
        lang_code: Language code
        max_items: Maximum number of items to process (None = all)
        save_path: Path to save processed data (None = return in memory)

    Returns:
        Tuple of (dataset, stats_dict) if save_path is None, else (None, stats_dict)
        stats_dict contains: available_mmarco, available_miracl, used_mmarco, used_miracl
    """
    print(f"\n>>> Processing [{lang_code.upper()}]...")
    if max_items:
        print(f"  Limiting to {max_items:,} items")

    full_lang_name = MMARCO_LANG_MAP[lang_code]

    # Load datasets
    ds_mmarco = get_mmarco_lang(full_lang_name)
    if ds_mmarco is None:
        raise ValueError(f"mMARCO data not available for {lang_code}")

    # MIRACL may not be available for all languages (dt, it, pt, vi are not in MIRACL)
    # This is expected and fine - we'll use mMARCO data only for those languages
    ds_miracl = load_miracl_lang(lang_code)
    if ds_miracl is None:
        print(f"  [{lang_code}]: MIRACL not available, using mMARCO only")

    # Standardize formats separately to track sources
    mmarco_std = ds_mmarco.map(
        standardize_mmarco,
        fn_kwargs={"lang": lang_code},
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds_mmarco.column_names,
    )

    miracl_std = None
    if ds_miracl is not None:
        miracl_std = ds_miracl.map(
            standardize_miracl,
            fn_kwargs={"lang": lang_code},
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=ds_miracl.column_names,
        )

    # Apply smart chunking separately to track sources
    mmarco_chunked = mmarco_std.map(
        smart_chunking_v2,
        batched=True,
        batch_size=2000,
        num_proc=NUM_PROC,
        remove_columns=mmarco_std.column_names,
    )
    mmarco_chunked = mmarco_chunked.shuffle(seed=42)
    available_mmarco = len(mmarco_chunked)

    miracl_chunked = None
    available_miracl = 0
    if miracl_std is not None:
        miracl_chunked = miracl_std.map(
            smart_chunking_v2,
            batched=True,
            batch_size=2000,
            num_proc=NUM_PROC,
            remove_columns=miracl_std.column_names,
        )
        miracl_chunked = miracl_chunked.shuffle(seed=42)
        available_miracl = len(miracl_chunked)

    # Balance sources to 50/50 if possible
    if max_items:
        if miracl_chunked is not None and available_miracl > 0:
            # Try to balance 50/50, but use all available from smaller source if needed
            target_per_source = max_items // 2

            # Start with 50/50 target
            used_mmarco = min(target_per_source, available_mmarco)
            used_miracl = min(target_per_source, available_miracl)

            # If one source has less than target, use all of it and fill rest from other
            if available_miracl < target_per_source:
                # Use all MIRACL available, fill rest with mMARCO
                used_miracl = available_miracl
                used_mmarco = min(max_items - used_miracl, available_mmarco)
            elif available_mmarco < target_per_source:
                # Use all mMARCO available, fill rest with MIRACL
                used_mmarco = available_mmarco
                used_miracl = min(max_items - used_mmarco, available_miracl)
            else:
                # Both have enough - try to balance, fill remainder if needed
                total_used = used_mmarco + used_miracl
                if total_used < max_items:
                    remaining = max_items - total_used
                    # Prefer balancing - add half to each if possible
                    add_mmarco = min(remaining // 2, available_mmarco - used_mmarco)
                    add_miracl = min(
                        remaining - add_mmarco, available_miracl - used_miracl
                    )
                    used_mmarco += add_mmarco
                    used_miracl += add_miracl
                    # If still room, fill with whichever has more available
                    if used_mmarco + used_miracl < max_items:
                        remaining = max_items - used_mmarco - used_miracl
                        if (
                            available_mmarco - used_mmarco
                            >= available_miracl - used_miracl
                        ):
                            used_mmarco = min(used_mmarco + remaining, available_mmarco)
                        else:
                            used_miracl = min(used_miracl + remaining, available_miracl)

            # Select balanced amounts
            mmarco_selected = mmarco_chunked.select(range(used_mmarco))
            miracl_selected = miracl_chunked.select(range(used_miracl))
            rechunked = concatenate_datasets([mmarco_selected, miracl_selected])
            rechunked = rechunked.shuffle(seed=42)
        else:
            # MIRACL not available, use mMARCO only
            used_mmarco = min(max_items, available_mmarco)
            used_miracl = 0
            rechunked = mmarco_chunked.select(range(used_mmarco))
    else:
        # No limit - use all available
        if miracl_chunked is not None:
            rechunked = concatenate_datasets([mmarco_chunked, miracl_chunked])
            rechunked = rechunked.shuffle(seed=42)
            used_mmarco = available_mmarco
            used_miracl = available_miracl
        else:
            rechunked = mmarco_chunked
            used_mmarco = available_mmarco
            used_miracl = 0

    # Statistics
    stats = {
        "available_mmarco": available_mmarco,
        "available_miracl": available_miracl,
        "used_mmarco": used_mmarco,
        "used_miracl": used_miracl,
        "total_used": len(rechunked),
        "unused_mmarco": available_mmarco - used_mmarco,
        "unused_miracl": available_miracl - used_miracl,
    }

    # Clean memory before saving/returning
    del ds_mmarco, mmarco_std, mmarco_chunked
    if ds_miracl is not None:
        del ds_miracl, miracl_std, miracl_chunked
    gc.collect()

    # Save to disk if path provided
    if save_path:
        print(f"  -> Saving [{lang_code}] to {save_path}")
        rechunked.save_to_disk(save_path)
        print(f"  -> [{lang_code}] Saved {len(rechunked):,} chunks")
        print(f"      mMARCO: {used_mmarco:,}/{available_mmarco:,} used")
        if available_miracl > 0:
            print(f"      MIRACL: {used_miracl:,}/{available_miracl:,} used")
        return None, stats
    else:
        print(f"  -> [{lang_code}] Generated {len(rechunked):,} chunks")
        return rechunked, stats


def arrange_in_sets(data_by_lang):
    """
    Arrange data into sets of 13 (one per language).
    This ensures each batch (multiple of 13) contains equal language representation.
    """
    # Find minimum count
    counts = {lang: len(data) for lang, data in data_by_lang.items()}
    min_count = min(counts.values())

    print(f"\n>>> Balancing to minimum count: {min_count:,}")

    # Trim all languages to minimum count
    balanced_data = {}
    for lang_code in LANG_CODES:
        ds = data_by_lang[lang_code]
        balanced_data[lang_code] = ds.select(range(min_count))
        print(f"  [{lang_code}]: {len(balanced_data[lang_code]):,} items")

    # Arrange in sets of 13
    num_sets = min_count
    arranged_data = []

    for i in range(num_sets):
        # One item from each language (in fixed order)
        for lang_code in LANG_CODES:
            item = balanced_data[lang_code][i]
            # Handle both list and scalar values from dataset
            text = item["text"] if isinstance(item["text"], str) else item["text"][0]
            lang = (
                item["language"]
                if isinstance(item["language"], str)
                else item["language"][0]
            )
            source = item.get("source", "unknown")
            if isinstance(source, list):
                source = source[0] if source else "unknown"

            arranged_data.append({"text": text, "language": lang, "source": source})

    return arranged_data, num_sets


def split_train_val(data, train_ratio=0.8):
    """
    Split data into train/val with 8:2 ratio.
    Ensures both splits are multiples of 13.
    """
    total_items = len(data)

    # Calculate split sizes that are multiples of 13
    total_sets = total_items // NUM_LANGUAGES
    train_sets = int(total_sets * train_ratio)
    val_sets = total_sets - train_sets

    # Ensure we have at least some validation data
    if val_sets == 0:
        val_sets = 1
        train_sets = total_sets - 1

    train_size = train_sets * NUM_LANGUAGES
    val_size = val_sets * NUM_LANGUAGES

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]

    print(
        f"\n>>> Split: Train={len(train_data):,} ({train_sets} sets), Val={len(val_data):,} ({val_sets} sets)"
    )

    return train_data, val_data


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    print(f"Writing {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(data):,} items to {filepath}")


def count_sources_in_data(data):
    """
    Count how many items come from each source in the final dataset.
    Note: Merged fragments (with comma-separated sources) are counted for all their sources.
    """
    source_counts = {}
    merged_count = 0
    for item in data:
        source = item.get("source", "unknown")
        # Handle comma-separated sources (merged fragments)
        sources = [s.strip() for s in source.split(",")]
        if len(sources) > 1:
            merged_count += 1
        for s in sources:
            source_counts[s] = source_counts.get(s, 0) + 1
    source_counts["_merged_fragments"] = merged_count
    return source_counts


def save_statistics(stats_by_lang, train_data, val_data, output_dir):
    """Save statistics JSON file with detailed information."""
    # Count sources in final datasets
    train_source_counts = count_sources_in_data(train_data)
    val_source_counts = count_sources_in_data(val_data)

    # Count per language in final datasets
    train_lang_counts = {}
    val_lang_counts = {}
    for item in train_data:
        lang = item.get("language", "unknown")
        train_lang_counts[lang] = train_lang_counts.get(lang, 0) + 1
    for item in val_data:
        lang = item.get("language", "unknown")
        val_lang_counts[lang] = val_lang_counts.get(lang, 0) + 1

    # Build statistics summary
    statistics = {
        "summary": {
            "total_languages": NUM_LANGUAGES,
            "languages": LANG_CODES,
            "total_train_items": len(train_data),
            "total_val_items": len(val_data),
            "total_items": len(train_data) + len(val_data),
        },
        "per_language": {},
        "dataset_distribution": {
            "train": train_source_counts,
            "val": val_source_counts,
            "total": {
                k: train_source_counts.get(k, 0) + val_source_counts.get(k, 0)
                for k in set(
                    list(train_source_counts.keys()) + list(val_source_counts.keys())
                )
            },
        },
    }

    # Add per-language statistics
    for lang_code in LANG_CODES:
        lang_stats = stats_by_lang.get(lang_code, {})
        train_count = train_lang_counts.get(lang_code, 0)
        val_count = val_lang_counts.get(lang_code, 0)

        statistics["per_language"][lang_code] = {
            "available": {
                "mmarco": lang_stats.get("available_mmarco", 0),
                "miracl": lang_stats.get("available_miracl", 0),
                "total": lang_stats.get("available_mmarco", 0)
                + lang_stats.get("available_miracl", 0),
            },
            "used": {
                "mmarco": lang_stats.get("used_mmarco", 0),
                "miracl": lang_stats.get("used_miracl", 0),
                "total": lang_stats.get("total_used", 0),
            },
            "unused": {
                "mmarco": lang_stats.get("unused_mmarco", 0),
                "miracl": lang_stats.get("unused_miracl", 0),
                "total": lang_stats.get("unused_mmarco", 0)
                + lang_stats.get("unused_miracl", 0),
            },
            "final_dataset": {
                "train": train_count,
                "val": val_count,
                "total": train_count + val_count,
            },
            "source_distribution": {
                "mmarco_percent": round(
                    100
                    * lang_stats.get("used_mmarco", 0)
                    / max(lang_stats.get("total_used", 1), 1),
                    2,
                ),
                "miracl_percent": round(
                    100
                    * lang_stats.get("used_miracl", 0)
                    / max(lang_stats.get("total_used", 1), 1),
                    2,
                ),
            },
        }

    # Save statistics
    stats_path = os.path.join(output_dir, "statistics.json")
    print(f"\nWriting statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved statistics to {stats_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Statistics Summary:")
    print("=" * 80)
    print(
        f"Total items: {statistics['summary']['total_items']:,} (Train: {statistics['summary']['total_train_items']:,}, Val: {statistics['summary']['total_val_items']:,})"
    )
    print(f"\nPer-language breakdown:")
    for lang_code in LANG_CODES:
        lang_stat = statistics["per_language"][lang_code]
        print(f"  [{lang_code}]:")
        print(
            f"    Available: mMARCO={lang_stat['available']['mmarco']:,}, MIRACL={lang_stat['available']['miracl']:,}"
        )
        print(
            f"    Used: mMARCO={lang_stat['used']['mmarco']:,}, MIRACL={lang_stat['used']['miracl']:,} ({lang_stat['source_distribution']['mmarco_percent']:.1f}%/{lang_stat['source_distribution']['miracl_percent']:.1f}%)"
        )
        print(
            f"    Unused: mMARCO={lang_stat['unused']['mmarco']:,}, MIRACL={lang_stat['unused']['miracl']:,}"
        )
        print(
            f"    Final: Train={lang_stat['final_dataset']['train']:,}, Val={lang_stat['final_dataset']['val']:,}"
        )

    return statistics


def main():
    """Main execution pipeline with optimizations."""
    print("=" * 80)
    print("Downloading and Processing Data for 13 Languages")
    print("=" * 80)

    # Create temp directory
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    try:
        # Phase 1: Quick estimation pass - find minimum count
        print("\n" + "=" * 80)
        print("Phase 1: Estimating chunk counts for all languages...")
        print("=" * 80)

        # Check if temp files already exist (resumability)
        existing_counts = {}
        for lang_code in LANG_CODES:
            temp_path = os.path.join(TEMP_DIR, lang_code)
            if os.path.exists(temp_path):
                try:
                    ds = load_from_disk(temp_path)
                    existing_counts[lang_code] = len(ds)
                    print(
                        f"  [{lang_code}]: Found existing temp file with {len(ds):,} items"
                    )
                except:
                    pass

        estimated_counts = {}
        for lang_code in LANG_CODES:
            if lang_code in existing_counts:
                # Use existing count if temp file exists
                estimated_counts[lang_code] = existing_counts[lang_code]
            else:
                try:
                    count = estimate_chunk_count(lang_code)
                    estimated_counts[lang_code] = count
                except Exception as e:
                    print(f"Error estimating {lang_code}: {e}")
                    raise

        min_count = min(estimated_counts.values())
        print(f"\n>>> Minimum estimated count: {min_count:,}")
        print(">>> All languages will be limited to this count for efficiency")

        # Phase 2: Process each language and save to temp files (with early stopping)
        print("\n" + "=" * 80)
        print("Phase 2: Processing languages and saving to temp files...")
        print("=" * 80)

        temp_paths = {}
        stats_by_lang = {}  # Collect statistics for each language

        for lang_code in LANG_CODES:
            temp_path = os.path.join(TEMP_DIR, lang_code)
            temp_paths[lang_code] = temp_path

            # Skip if temp file already exists and has correct count
            if lang_code in existing_counts:
                existing_count = existing_counts[lang_code]
                if existing_count >= min_count:
                    print(
                        f"  [{lang_code}]: Skipping (already processed with {existing_count:,} items)"
                    )
                    # Trim to min_count if needed
                    if existing_count > min_count:
                        ds = load_from_disk(temp_path)
                        ds_trimmed = ds.select(range(min_count))
                        ds_trimmed.save_to_disk(temp_path)
                        print(f"  [{lang_code}]: Trimmed to {min_count:,} items")

                    # Load stats from existing file (count sources accurately)
                    ds = load_from_disk(temp_path)
                    source_counts = {}
                    for i in range(len(ds)):
                        item = ds[i]
                        source = item.get("source", "unknown")
                        # Handle comma-separated sources (merged fragments)
                        sources = [s.strip() for s in source.split(",")]
                        for s in sources:
                            source_counts[s] = source_counts.get(s, 0) + 1

                    # Count sources in the existing file
                    total = len(ds)
                    used_miracl = source_counts.get("miracl", 0)
                    used_mmarco = source_counts.get("mmarco", 0)
                    # Note: merged fragments may have both sources, count separately
                    # For merged fragments, we count them as contributing to both
                    # But for "used" counts, we'll use the actual counts

                    stats_by_lang[lang_code] = {
                        "available_mmarco": used_mmarco,  # From existing file, approximate
                        "available_miracl": used_miracl,  # From existing file, approximate
                        "used_mmarco": used_mmarco,
                        "used_miracl": used_miracl,
                        "total_used": total,
                        "unused_mmarco": 0,  # Unknown for existing files
                        "unused_miracl": 0,  # Unknown for existing files
                    }
                    continue

            try:
                _, stats = process_language(
                    lang_code, max_items=min_count, save_path=temp_path
                )
                stats_by_lang[lang_code] = stats
            except Exception as e:
                print(f"Error processing {lang_code}: {e}")
                raise

        # Phase 3: Load from temp files and arrange in sets
        print("\n" + "=" * 80)
        print("Phase 3: Loading from temp files and arranging in sets...")
        print("=" * 80)

        data_by_lang = {}
        for lang_code in LANG_CODES:
            temp_path = temp_paths[lang_code]
            ds = load_from_disk(temp_path)
            data_by_lang[lang_code] = ds
            print(f"  [{lang_code}]: Loaded {len(ds):,} items")

        arranged_data, num_sets = arrange_in_sets(data_by_lang)

        # Phase 4: Split train/val
        print("\n" + "=" * 80)
        print("Phase 4: Splitting train/val...")
        print("=" * 80)

        train_data, val_data = split_train_val(arranged_data, train_ratio=0.8)

        # Phase 5: Save to JSONL files and statistics
        print("\n" + "=" * 80)
        print("Phase 5: Saving final JSONL files and statistics...")
        print("=" * 80)

        output_dir = os.path.dirname(os.path.abspath(__file__))
        train_path = os.path.join(output_dir, "train.jsonl")
        val_path = os.path.join(output_dir, "val.jsonl")

        save_jsonl(train_data, train_path)
        save_jsonl(val_data, val_path)

        # Save statistics
        save_statistics(stats_by_lang, train_data, val_data, output_dir)

        # Phase 6: Cleanup temp files
        print("\n" + "=" * 80)
        print("Phase 6: Cleaning up temp files...")
        print("=" * 80)

        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(f"  -> Removed {TEMP_DIR}")

        print("\n" + "=" * 80)
        print("Done! Output files:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"  - {os.path.join(output_dir, 'statistics.json')}")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during processing: {e}")
        print(f"Temp files preserved in {TEMP_DIR} for debugging")
        raise


if __name__ == "__main__":
    main()
