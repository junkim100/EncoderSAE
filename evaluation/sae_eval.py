"""Standalone SAE Evaluation (vLLM Optimized) - Rewritten from scratch."""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add evaluation directory to path for imports
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

# Import modules (works when run as script or module)
import sae_runtime
import utils

SAERuntime = sae_runtime.SAERuntime
run_base_encoding = utils.run_base_encoding
get_e5_query_prefix = utils.get_e5_query_prefix
get_e5_passage_prefix = utils.get_e5_passage_prefix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Standalone SAE Evaluation (vLLM Optimized)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name (e.g. intfloat/multilingual-e5-large)",
    )
    parser.add_argument("--sae_path", type=str, required=True, help="Path to SAE checkpoint directory")
    parser.add_argument(
        "--mask_path", type=str, default=None, help="Path to mask file (.pt)"
    )
    parser.add_argument(
        "--data_dirs",
        type=str,
        required=True,
        help="Comma separated list of data directories (or Python list format)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (for compatibility, not used in vLLM path)",
    )
    parser.add_argument("--results_root", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--use_reconstruction",
        type=str,
        default="True",
        help="If 'True', use SAE decoder output. If 'False', use latent features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for SAE processing",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=None,
        help="Mask threshold used (for output directory naming)",
    )
    return parser.parse_args()


def compute_metrics(qrels: Dict[str, Dict[str, int]], results: Dict[str, Dict[str, float]], k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000]):
    """
    Compute NDCG, Recall, MAP, Precision etc.
    Uses BEIR's evaluation framework.
    """
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval

        class MockRetriever:
            def __init__(self):
                pass

        evaluator = EvaluateRetrieval(retriever=MockRetriever())
        return evaluator.evaluate(qrels, results, k_values)

    except ImportError:
        logger.error("BEIR not installed. Cannot compute metrics.")
        return {}, {}, {}, {}


def process_embeddings(
    runtime: SAERuntime,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    use_reconstruction: bool = True,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Apply SAE -> Mask -> (Optional) Reconstruction to a batch of embeddings.

    Args:
        runtime: SAERuntime instance
        embeddings: [N, D] tensor of base embeddings
        mask: Boolean mask of shape [dict_size] where True means disable (zero out) feature.
            Features where mask[i] = True fired in >= threshold% of samples (e.g., >=99% for threshold=0.99)
            and will be zeroed out to create language-agnostic embeddings.
            Example: If mask_threshold=0.99, features that activated in >=99% of language samples are marked True.
        use_reconstruction: If True, return reconstructed embeddings; if False, return masked features
        batch_size: Batch size for processing

    Returns:
        Processed embeddings: [N, D] tensor (if reconstruction) or [N, S] (if latent)
    """
    runtime.model.to(runtime.device)
    if mask is not None:
        mask = mask.to(runtime.device)

    results = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size].to(runtime.device)

            # SAE Forward with mask
            # Returns: x_hat_orig (unmasked reconstruction), z (unmasked features), x_hat_masked (masked reconstruction)
            x_hat_orig, z, x_hat_masked = runtime.forward(batch, mask=mask)

            if use_reconstruction:
                # Use the masked reconstruction (language-agnostic embeddings)
                res = x_hat_masked
            else:
                # Return masked features (latent space) instead of reconstruction
                if mask is not None:
                    # Apply mask to features: zero out features where mask[i] = True
                    # This disables SAE neurons (features) that fired in >= threshold% of language samples
                    # Use explicit multiplication to ensure correct column selection
                    z_masked = z.clone()
                    z_masked = z_masked * (~mask).float().unsqueeze(0)  # Zero out language-specific features
                    res = z_masked
                else:
                    res = z

            results.append(res.cpu())

    return torch.cat(results, dim=0)


def main():
    """Main evaluation function."""
    args = parse_args()

    # 1. Setup Output Dir
    out_dir = Path(args.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load SAE Runtime
    logger.info(f"Loading SAE from {args.sae_path}")
    sae_path_obj = Path(args.sae_path)
    if sae_path_obj.is_file():
        ckpt_dir = sae_path_obj.parent
    else:
        ckpt_dir = sae_path_obj

    runtime = SAERuntime(str(ckpt_dir))

    # Extract checkpoint folder name for output directory naming
    checkpoint_folder_name = ckpt_dir.name

    # 3. Load Mask
    # Mask semantics: mask[i] = True means feature i fired in >= threshold% of samples for a language
    # These features (SAE neurons) will be zeroed out (disabled) to create language-agnostic embeddings
    # Example: If mask_threshold=0.99, features that activated in >=99% of language samples are marked True
    # During inference: z_masked[:, mask] = 0 zeros out these language-specific features
    mask = None
    if args.mask_path:
        logger.info(f"Loading Mask from {args.mask_path}")
        mask = torch.load(args.mask_path, map_location="cpu")
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # Mask Stats
        total = mask.numel()
        masked_cnt = mask.sum().item()
        logger.info(
            f"Mask Stats: {masked_cnt}/{total} ({(masked_cnt/total)*100:.2f}%) features will be disabled"
        )
        logger.info(
            f"  These features fired in >= threshold% of language samples and will be zeroed out"
        )

    # 4. Process Datasets
    # Handle both comma-separated and Python list format
    data_dirs_str = args.data_dirs.strip()
    if data_dirs_str.startswith("[") and data_dirs_str.endswith("]"):
        # Python list format: ["dir1", "dir2"]
        import ast
        data_dirs = ast.literal_eval(data_dirs_str)
    else:
        # Comma-separated format
        data_dirs = [d.strip() for d in data_dirs_str.split(",")]

    model_name = args.model
    # Parse use_reconstruction (handle string from shell script)
    use_reconstruction = args.use_reconstruction.lower() in ("true", "1", "yes")

    # Query/Passage Prefixes
    q_prefix = get_e5_query_prefix(model_name)
    p_prefix = get_e5_passage_prefix(model_name)

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        dataset_name = data_path.name
        logger.info(f"Evaluating on {dataset_name}...")

        # Load Data
        queries = {}
        with open(data_path / "queries.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "_id" in item and "text" in item:
                    queries[item["_id"]] = q_prefix + item["text"]
                else:
                    # Fallback for {qid: text} or similar
                    for k, v in item.items():
                        if isinstance(v, str):
                            queries[k] = q_prefix + v
                        elif isinstance(v, dict) and "text" in v:
                            queries[k] = q_prefix + v["text"]

        corpus = {}
        with open(data_path / "corpus.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Handle standard BEIR {_id: ..., text: ...}
                if "_id" in item:
                    text = item.get("text", "")
                    title = item.get("title", "")
                    full_text = f"{title} {text}".strip() if title else text.strip()
                    corpus[item["_id"]] = p_prefix + full_text
                else:
                    # Handle {doc_id: {text: ...}} format
                    for did, doc in item.items():
                        if isinstance(doc, dict):
                            text = doc.get("text", "")
                            title = doc.get("title", "")
                            full_text = (
                                f"{title} {text}".strip() if title else text.strip()
                            )
                            corpus[did] = p_prefix + full_text

        qrels = {}
        with open(data_path / "qrels.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Handle BEIR flat format: {query-id: ..., corpus-id: ..., score: ...}
                if "query-id" in item and "corpus-id" in item:
                    qid = item["query-id"]
                    did = item["corpus-id"]
                    score = item["score"]
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][did] = int(score)
                else:
                    # Handle nested format: {qid: {did: score}}
                    for qid, docs in item.items():
                        if qid not in qrels:
                            qrels[qid] = {}
                        for did, score in docs.items():
                            qrels[qid][did] = int(score)

        # Sort IDs to ensure alignment
        q_ids = sorted(list(queries.keys()))
        c_ids = sorted(list(corpus.keys()))

        q_texts = [queries[qid] for qid in q_ids]
        c_texts = [corpus[cid] for cid in c_ids]

        # Encode (Base -> SAE -> Mask -> Recon)
        logger.info("Encoding Queries...")
        q_base = run_base_encoding(q_texts, model_name, max_seq_length=args.max_seq_length)
        q_embs = process_embeddings(
            runtime, q_base, mask, use_reconstruction, args.batch_size
        )

        logger.info("Encoding Corpus...")
        c_base = run_base_encoding(c_texts, model_name, max_seq_length=args.max_seq_length)
        c_embs = process_embeddings(
            runtime, c_base, mask, use_reconstruction, args.batch_size
        )

        # Similarity Search
        logger.info("Computing Similarity...")
        # Normalize for Cosine Similarity
        q_embs = F.normalize(q_embs, p=2, dim=1)
        c_embs = F.normalize(c_embs, p=2, dim=1)

        # Score Matrix: [Q, C]
        scores = torch.mm(q_embs, c_embs.transpose(0, 1))

        # Convert to Dictionary for BEIR
        results_b = {}
        # Optimization: Top-K only
        top_k = 1000  # Evaluate up to @1000

        logger.info("Formatting Results...")
        for i, qid in enumerate(q_ids):
            # Top K
            row_scores = scores[i]
            # sort descending
            top_vals, top_inds = torch.topk(row_scores, k=min(top_k, len(c_ids)))

            results_b[qid] = {}
            for v, idx in zip(top_vals, top_inds):
                doc_id = c_ids[int(idx)]
                results_b[qid][doc_id] = float(v)

        # Evaluate
        logger.info("Calculating Metrics...")
        ndcg, _map, recall, precision = compute_metrics(qrels, results_b)

        # Print & Save
        logger.info(f"[{dataset_name}] NDCG@20: {ndcg.get('NDCG@20', 0.0):.4f}")
        logger.info(f"[{dataset_name}] Recall@20: {recall.get('Recall@20', 0.0):.4f}")

        # Build output directory name with checkpoint folder and mask threshold
        model_safe_name = args.model.replace("/", "_").replace(" ", "_")
        if args.mask_threshold is not None:
            mask_str = f"mask{str(args.mask_threshold).replace('.', '_')}"
            output_subdir = out_dir / f"{model_safe_name}_{checkpoint_folder_name}_{mask_str}"
        else:
            output_subdir = out_dir / f"{model_safe_name}_{checkpoint_folder_name}"

        output_subdir.mkdir(parents=True, exist_ok=True)
        out_file = output_subdir / f"{dataset_name}_results.json"

        final_res = {
            "model": model_name,
            "sae": str(args.sae_path),
            "mask": str(args.mask_path) if args.mask_path else None,
            "mask_threshold": args.mask_threshold,
            "checkpoint_folder": checkpoint_folder_name,
            "dataset": dataset_name,
            "use_reconstruction": use_reconstruction,
            "ndcg": ndcg,
            "recall": recall,
            "map": _map,
            "precision": precision,
        }
        with open(out_file, "w") as f:
            json.dump(final_res, f, indent=2)

        logger.info(f"Results saved to: {out_file}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
