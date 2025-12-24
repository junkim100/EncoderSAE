import os
import gc
import math
import json
from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# =========================================================
# 1. Configuration
# =========================================================
CACHE_DIR = "/data_x/aa007878/encodersae/data/cache"
NUM_PROC = 60
TOKENIZER_NAME = "intfloat/multilingual-e5-large"
TARGET_MIN_LEN = 250
TARGET_MAX_LEN = 500
NUM_LANGUAGES = 13

# 13 languages from Belebele corpus
LANG_CODES = ['ar', 'dt', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'pt', 'ru', 'vi', 'zh']

# Mapping from language codes to mmarco full names
MMARCO_LANG_MAP = {
    'ar': 'arabic',
    'dt': 'dutch',
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    'hi': 'hindi',
    'id': 'indonesian',
    'it': 'italian',
    'ja': 'japanese',
    'pt': 'portuguese',
    'ru': 'russian',
    'vi': 'vietnamese',
    'zh': 'chinese'
}

# MIRACL language codes and file counts (may not have all languages)
# Note: MIRACL uses ISO 639-1 codes, 'dt' (Dutch) maps to 'nl' in MIRACL
MIRACL_LANG_MAP = {
    'ar': 'ar',
    'dt': 'nl',  # Dutch in Belebele is 'dt', but MIRACL uses 'nl'
    'en': 'en',
    'es': 'es',
    'fr': 'fr',
    'hi': 'hi',
    'id': 'id',
    'it': 'it',
    'ja': 'ja',
    'pt': 'pt',
    'ru': 'ru',
    'vi': 'vi',
    'zh': 'zh',
}

MIRACL_LANG_FILES = {
    'en': 66,
    'zh': 10,
    'fr': 30,
    'es': 21,
    'ar': 21,
    'hi': 21,
    'it': 21,
    'ja': 21,
    'pt': 21,
    'ru': 21,
    'nl': 21,  # Dutch
    'vi': 21,
    'id': 21,
}

# Tokenizer (global for fork-safe multiprocessing)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

# =========================================================
# 2. Data Loading Functions
# =========================================================

def load_miracl_lang(lang_code):
    """Load MIRACL corpus for a language if available."""
    # Map Belebele language code to MIRACL language code
    miracl_lang_code = MIRACL_LANG_MAP.get(lang_code)
    if miracl_lang_code is None:
        return None

    n_files = MIRACL_LANG_FILES.get(miracl_lang_code, 0)
    if n_files == 0:
        return None

    base_url = f'https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/miracl-corpus-v1.0-{miracl_lang_code}/docs-{{i}}.jsonl.gz'
    file_urls = [base_url.format(i=i) for i in range(n_files)]
    miracl_features = Features({
        'docid': Value('string'),
        'title': Value('string'),
        'text': Value('string')
    })

    try:
        return load_dataset("json", data_files=file_urls, features=miracl_features, split="train", cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"Warning: Could not load MIRACL for {lang_code} (MIRACL code: {miracl_lang_code}): {e}")
        return None

def load_lookup_dict(file_path):
    """Load TSV file as ID -> Text dictionary."""
    lookup = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split('\t')
            if len(parts) >= 2:
                lookup[parts[0]] = parts[1]
    return lookup

def mmarco_generator(triples_path, queries_path, collection_path):
    """Generator for mMARCO dataset."""
    queries_dict = load_lookup_dict(queries_path)
    collection_dict = load_lookup_dict(collection_path)

    with open(triples_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qid, pid, nid = line.rstrip().split('\t')
                yield {
                    "query": queries_dict[qid],
                    "positive": collection_dict[pid],
                    "negative": collection_dict[nid]
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
            cache_dir=CACHE_DIR
        )
        query_file = hf_hub_download(
            repo_id="unicamp-dl/mmarco",
            filename=f"data/google/queries/train/{full_lang_name}_queries.train.tsv",
            repo_type="dataset",
            cache_dir=CACHE_DIR
        )
        triples_file = hf_hub_download(
            repo_id="unicamp-dl/mmarco",
            filename="data/triples.train.ids.small.tsv",
            repo_type="dataset",
            cache_dir=CACHE_DIR
        )

        mmarco_features = Features({
            "query": Value("string"),
            "positive": Value("string"),
            "negative": Value("string")
        })

        return Dataset.from_generator(
            generator=mmarco_generator,
            gen_kwargs={
                "triples_path": triples_file,
                "queries_path": query_file,
                "collection_path": coll_file
            },
            features=mmarco_features,
            cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"Warning: Could not load mMARCO for {full_lang_name}: {e}")
        return None

# =========================================================
# 3. Data Processing Functions
# =========================================================

def standardize_miracl(batch, lang):
    """Standardize MIRACL format: title + text."""
    new_texts = [f"{t}\n{txt}" for t, txt in zip(batch['title'], batch['text'])]
    return {
        "text": new_texts,
        "source": ["miracl"] * len(batch['text']),
        "language": [lang] * len(batch['text'])
    }

def standardize_mmarco(batch, lang):
    """Standardize mMARCO format: query + positive."""
    new_texts = [f"{q}\n{p}" for q, p in zip(batch['query'], batch['positive'])]
    return {
        "text": new_texts,
        "source": ["mmarco"] * len(batch['query']),
        "language": [lang] * len(batch['query'])
    }

def smart_chunking_v2(batch):
    """
    Smart chunking: split long docs (>500), merge short docs (<250).
    Ensures all chunks are between 250-500 tokens.
    """
    encodings = tokenizer(batch['text'], add_special_tokens=False, truncation=False)
    input_ids_list = encodings['input_ids']
    sources = batch['source']
    langs = batch['language']

    out_text, out_source, out_language = [], [], []
    frag_ids, frag_sources = [], set()
    frag_lang = langs[0] if langs else ""

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

            # Case A: Sufficiently long (250-500)
            if chunk_len >= TARGET_MIN_LEN:
                out_text.append(tokenizer.decode(chunk, skip_special_tokens=True))
                out_source.append(sources[i])
                out_language.append(langs[i])

            # Case B: Too short (<250) - add to fragment buffer
            else:
                # If buffer would exceed max, flush it
                if len(frag_ids) + chunk_len > TARGET_MAX_LEN:
                    out_text.append(tokenizer.decode(frag_ids, skip_special_tokens=True))
                    out_source.append(",".join(sorted(frag_sources)))
                    out_language.append(frag_lang)
                    frag_ids, frag_sources = [], set()

                # Add to buffer
                frag_ids.extend(chunk)
                frag_sources.add(sources[i])
                frag_lang = langs[i]

                # If buffer is now sufficient, flush it
                if len(frag_ids) >= TARGET_MIN_LEN:
                    out_text.append(tokenizer.decode(frag_ids, skip_special_tokens=True))
                    out_source.append(",".join(sorted(frag_sources)))
                    out_language.append(frag_lang)
                    frag_ids, frag_sources = [], set()

    # Flush remaining fragments
    if frag_ids:
        out_text.append(tokenizer.decode(frag_ids, skip_special_tokens=True))
        out_source.append(",".join(sorted(frag_sources)))
        out_language.append(frag_lang)

    return {
        "text": out_text,
        "source": out_source,
        "language": out_language
    }

# =========================================================
# 4. Main Processing Pipeline
# =========================================================

def process_language(lang_code):
    """Process a single language: load, standardize, chunk."""
    print(f"\n>>> Processing [{lang_code.upper()}]...")

    full_lang_name = MMARCO_LANG_MAP[lang_code]

    # Load datasets
    ds_mmarco = get_mmarco_lang(full_lang_name)
    if ds_mmarco is None:
        raise ValueError(f"mMARCO data not available for {lang_code}")

    ds_miracl = load_miracl_lang(lang_code)

    # Standardize formats
    mmarco_std = ds_mmarco.map(
        standardize_mmarco,
        fn_kwargs={"lang": lang_code},
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=ds_mmarco.column_names
    )

    datasets_to_combine = [mmarco_std]

    if ds_miracl is not None:
        miracl_std = ds_miracl.map(
            standardize_miracl,
            fn_kwargs={"lang": lang_code},
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=ds_miracl.column_names
        )
        datasets_to_combine.append(miracl_std)

    # Combine datasets
    combined = concatenate_datasets(datasets_to_combine)

    # Apply smart chunking
    rechunked = combined.map(
        smart_chunking_v2,
        batched=True,
        batch_size=2000,
        num_proc=NUM_PROC,
        remove_columns=combined.column_names
    )

    # Shuffle to randomize between datasets
    rechunked = rechunked.shuffle(seed=42)

    # Clean memory
    del ds_mmarco, mmarco_std
    if ds_miracl is not None:
        del ds_miracl, miracl_std
    del combined
    gc.collect()

    print(f"  -> [{lang_code}] Generated {len(rechunked):,} chunks")
    return rechunked

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
            lang = item["language"] if isinstance(item["language"], str) else item["language"][0]
            source = item.get("source", "unknown")
            if isinstance(source, list):
                source = source[0] if source else "unknown"

            arranged_data.append({
                "text": text,
                "language": lang,
                "source": source
            })

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
    val_data = data[train_size:train_size + val_size]

    print(f"\n>>> Split: Train={len(train_data):,} ({train_sets} sets), Val={len(val_data):,} ({val_sets} sets)")

    return train_data, val_data

def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    print(f"Writing {filepath}...")
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  -> Saved {len(data):,} items to {filepath}")

def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("Downloading and Processing Data for 13 Languages")
    print("=" * 80)

    # Phase 1: Process each language
    data_by_lang = {}
    for lang_code in LANG_CODES:
        try:
            processed = process_language(lang_code)
            data_by_lang[lang_code] = processed
        except Exception as e:
            print(f"Error processing {lang_code}: {e}")
            raise

    # Phase 2: Arrange in sets of 13
    arranged_data, num_sets = arrange_in_sets(data_by_lang)

    # Phase 3: Split train/val
    train_data, val_data = split_train_val(arranged_data, train_ratio=0.8)

    # Phase 4: Save to JSONL files
    output_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)

    print("\n" + "=" * 80)
    print("Done! Output files:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

