"""Utility functions for SAE evaluation."""
import torch
import json
from pathlib import Path
from typing import List


def load_text_data(jsonl_path: str):
    """Load texts and languages from JSONL file."""
    texts = []
    langs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data.get("text", ""))
            langs.append(data.get("language", "unknown"))
    return texts, langs


def run_base_encoding(
    texts: List[str],
    model_name: str = "intfloat/multilingual-e5-large",
    batch_size: int = 256,
    max_seq_length: int = 512,
) -> torch.Tensor:
    """
    Encode texts using vLLM if available, otherwise HuggingFace transformers.
    Returns tensor on CPU.
    Matches the extraction method used during SAE training.
    """
    # Attempt to use vLLM first (matching training)
    try:
        from vllm import LLM
        import logging

        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.WARNING)

        print(f"Encoding {len(texts)} texts with vLLM ({model_name})")
        llm = LLM(
            model=model_name,
            task="embed",
            enforce_eager=True,
            gpu_memory_utilization=0.85,
            max_model_len=max_seq_length,
        )

        all_embs = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_texts = texts[i * batch_size : (i + 1) * batch_size]
            outputs = llm.encode(batch_texts, pooling_task="embed", use_tqdm=False)

            for output in outputs:
                embedding_tensor = None
                # Extract embedding using same logic as data.py
                if hasattr(output, "outputs"):
                    if hasattr(output.outputs, "embedding"):
                        embedding_data = output.outputs.embedding
                    elif hasattr(output.outputs, "data"):
                        embedding_data = output.outputs.data
                    elif hasattr(output.outputs, "__len__") and len(output.outputs) > 0:
                        embedding_data = output.outputs[0]
                    else:
                        raise ValueError("Could not extract embedding from vLLM output")
                elif hasattr(output, "embedding"):
                    embedding_data = output.embedding
                elif isinstance(output, torch.Tensor):
                    embedding_data = output
                else:
                    embedding_data = output

                if isinstance(embedding_data, torch.Tensor):
                    embedding_tensor = embedding_data.to(dtype=torch.float32)
                else:
                    embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)

                all_embs.append(embedding_tensor)

        # Clean up vLLM
        import gc

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        return torch.stack(all_embs).cpu()

    except ImportError:
        print("vLLM not found, falling back to HuggingFace transformers")
    except Exception as e:
        print(f"vLLM failed ({e}), falling back to HuggingFace transformers")

    # Fallback to HuggingFace transformers with manual mean pooling (matching training)
    from transformers import AutoTokenizer, AutoModel
    from EncoderSAE.data import mean_pool

    print(f"Encoding {len(texts)} texts with HuggingFace transformers ({model_name})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            batch_embs = mean_pool(hidden_states, attention_mask)
            all_embs.append(batch_embs.cpu())

    return torch.cat(all_embs, dim=0)


def get_e5_query_prefix(model_name: str) -> str:
    """Get query prefix for E5 models."""
    lower = model_name.lower()
    if "e5" in lower or "snowflake-arctic-embed" in lower:
        return "query: "
    if model_name == "nvidia/llama-nemotron-embed-1b-v2":
        return "query: "
    return ""


def get_e5_passage_prefix(model_name: str) -> str:
    """Get passage prefix for E5 models."""
    lower = model_name.lower()
    if "e5" in lower:
        return "passage: "
    if model_name == "nvidia/llama-nemotron-embed-1b-v2":
        return "passage: "
    return ""

