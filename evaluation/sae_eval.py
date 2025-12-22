import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import fire
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.base import BaseSearch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from EncoderSAE.model import EncoderSAE
from EncoderSAE.inference import remove_language_features

logger = logging.getLogger(__name__)


def load_st_model(model_name: str) -> SentenceTransformer:
    """Load SentenceTransformer model with appropriate settings."""
    lower = model_name.lower()

    if "gte" in lower:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            tokenizer_kwargs={"fix_mistral_regex": True},
        )
        if "qwen" in lower:
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={"dtype": torch.bfloat16},
            )
    elif "jina" in lower or "llama-nemotron-embed" in lower:
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
    elif "qwen" in lower:
        model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.bfloat16})
    elif "mxbai" in lower:
        model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.float16})
    else:
        model = SentenceTransformer(
            model_name,
            tokenizer_kwargs={"fix_mistral_regex": True},
        )

    model.max_seq_length = 512

    for module in model.modules():
        if hasattr(module, "config") and hasattr(module.config, "use_cache"):
            module.config.use_cache = False

    return model


def get_safe_model_name(model_name: str) -> str:
    """Generate filesystem-safe model name."""
    name = model_name.rstrip("/")
    if os.path.exists(name):
        name = os.path.basename(name)
    return name.replace("/", "_").replace(" ", "_")


def add_prefix_to_queries(queries: Dict[str, str], model_name: str) -> Dict[str, str]:
    """Add model-specific prefix to queries."""
    lower = model_name.lower()
    if "e5" in lower or "snowflake-arctic-embed" in lower:
        return {qid: f"query: {query}" for qid, query in queries.items()}
    if "qwen" in lower or "gte-qwen" in lower:
        return {
            qid: (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                f"Query: {query}"
            )
            for qid, query in queries.items()
        }
    if "jina" in lower:
        return {
            qid: f"Represent the query for retrieving evidence documents: {query}"
            for qid, query in queries.items()
        }
    if "embeddinggemma" in lower:
        return {
            qid: f"task: search result | query: {query}"
            for qid, query in queries.items()
        }
    if model_name == "nvidia/llama-nemotron-embed-1b-v2":
        return {qid: f"query: {query}" for qid, query in queries.items()}
    if "mxbai" in lower:
        return {
            qid: f"Represent this sentence for searching relevant passages: {query}"
            for qid, query in queries.items()
        }
    return queries


def add_prefix_to_corpus(
    corpus: Dict[str, Dict[str, Any]],
    model_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Add model-specific prefix to corpus texts while preserving other fields."""
    lower = model_name.lower()
    new_corpus: Dict[str, Dict[str, Any]] = {}

    for doc_id, doc in corpus.items():
        new_doc = dict(doc)
        text = doc.get("text", "")

        if "e5" in lower:
            new_doc["text"] = f"passage: {text}"
        elif "jina" in lower:
            new_doc["text"] = f"Represent the document for retrieval: {text}"
        elif "embeddinggemma" in lower:
            new_doc["text"] = f"title: none | text: {text}"
        elif model_name == "nvidia/llama-nemotron-embed-1b-v2":
            new_doc["text"] = f"passage: {text}"
        else:
            new_doc["text"] = text

        new_corpus[doc_id] = new_doc

    return new_corpus


class SAERetriever(BaseSearch):
    """Retriever that uses SAE and language mask for language-agnostic embeddings."""

    def __init__(
        self,
        model: SentenceTransformer,
        sae: EncoderSAE,
        mask: torch.Tensor,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        use_reconstruction: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.sae = sae
        self.mask = mask
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.use_reconstruction = use_reconstruction
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.results: Dict[str, Dict[str, float]] = {}

        if torch.cuda.is_available():
            self.device = "cuda"
            num_gpus = torch.cuda.device_count()
            logger.info(f"Using CUDA with {num_gpus} visible GPU(s) for encoding.")
        else:
            self.device = "cpu"
            logger.info("CUDA not available; using CPU for encoding.")

        self.sae.to(self.device)
        self.sae.eval()
        self.mask = self.mask.to(self.device)

    def search(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Search corpus for queries and return top_k results."""
        logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )

        query_embeddings = self.apply_sae_transformation(query_embeddings)

        logger.info("Encoding corpus...")
        corpus_ids = list(corpus.keys())
        corpus_texts = []
        for cid in corpus_ids:
            doc = corpus[cid]
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                corpus_texts.append(f"{title} {text}".strip())
            else:
                corpus_texts.append(text.strip())

        corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )

        corpus_embeddings = self.apply_sae_transformation(corpus_embeddings)

        logger.info("Computing similarities...")
        cos_scores = cos_sim(query_embeddings, corpus_embeddings)

        self.results = {}
        for idx, qid in enumerate(query_ids):
            scores = cos_scores[idx]
            top_results = torch.topk(scores, k=min(top_k, len(corpus_ids)))

            self.results[qid] = {}
            for score, corpus_idx in zip(
                top_results.values.tolist(), top_results.indices.tolist()
            ):
                corpus_id = corpus_ids[corpus_idx]
                if corpus_id != qid:
                    self.results[qid][corpus_id] = float(score)

        return self.results

    def apply_sae_transformation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process embeddings through SAE and apply language mask."""
        transformed_list = []

        with torch.no_grad():
            for i in range(0, len(embeddings), self.batch_size):
                batch = embeddings[i : i + self.batch_size].to(self.device)

                _, features, _, _, _ = self.sae(batch)
                features_agnostic = remove_language_features(features, self.mask)

                if self.use_reconstruction:
                    output = self.sae.decoder(features_agnostic)
                else:
                    output = features_agnostic

                transformed_list.append(output)

        return torch.cat(transformed_list, dim=0)

    def encode(
        self,
        corpus,
        queries,
        encode_output_path="./embeddings/",
        overwrite=False,
        query_filename="queries.pkl",
        corpus_filename="corpus.*.pkl",
        **kwargs,
    ):
        raise NotImplementedError("encode method not implemented")

    def search_from_files(
        self, query_embeddings_file, corpus_embeddings_files, top_k, **kwargs
    ):
        raise NotImplementedError("search_from_files method not implemented")


def run_sae_eval(
    model: str,
    sae_path: str,
    mask_path: str,
    data_dirs: list[str],
    results_root: str = "./results_sae_eval",
    batch_size: int = 128,
    use_reconstruction: bool = False,
) -> None:
    """
    Evaluate language-agnostic embeddings using SAE and language mask.

    Args:
        model: Base model name (must match model used for SAE training).
        sae_path: Path to trained SAE checkpoint (.pt file).
        mask_path: Path to language mask file (.pt file).
        data_dirs: List of dataset directory paths. Each directory must contain
            queries.jsonl, corpus.jsonl, and qrels.jsonl.
        results_root: Root directory where result JSON files are stored.
        batch_size: Batch size for encoding and SAE processing.
        use_reconstruction: If True, reconstruct to base embedding space;
            otherwise use sparse SAE features directly.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    results_path = Path(results_root)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Evaluating model: {model}")
    print(f"Using SAE: {sae_path}")
    print(f"Using Mask: {mask_path}")
    print(f"Reconstruction: {use_reconstruction}")
    print(f"{'='*80}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st_model = load_st_model(model)

        checkpoint = torch.load(sae_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "encoder.weight" in checkpoint:
            state_dict = checkpoint
        else:
            raise ValueError(f"Could not parse checkpoint format from {sae_path}")

        encoder_weight = state_dict["encoder.weight"]
        input_dim = encoder_weight.shape[1]
        dict_size = encoder_weight.shape[0]
        expansion_factor = dict_size // input_dim
        sparsity = (
            checkpoint.get("sparsity", 64) if isinstance(checkpoint, dict) else 64
        )

        sae = EncoderSAE(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            sparsity=sparsity,
        )
        sae.load_state_dict(state_dict)
        sae.to(device)
        sae.eval()

        mask = torch.load(mask_path, map_location=device)
        if mask.dtype != torch.bool:
            mask = mask.bool()

        retriever_kwargs = {
            "model": st_model,
            "sae": sae,
            "mask": mask,
            "batch_size": batch_size,
            "use_reconstruction": use_reconstruction,
        }
    except Exception as e:
        print(f"[ERROR] Failed to load model/SAE: {e}")
        return

    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir.rstrip("/"))
        print(f"\nEvaluating on dataset: {dataset_name}")

        model_safe_name = get_safe_model_name(model)
        sae_safe_name = get_safe_model_name(sae_path)
        mode_name = "reconstructed" if use_reconstruction else "features"
        output_dir = results_path / f"{model_safe_name}_{sae_safe_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}_{mode_name}_results.json"

        if output_file.exists():
            print(f"[SKIP] Results already exist at: {output_file}")
            try:
                with output_file.open("r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    ndcg = existing_results.get("ndcg", {})
                    recall = existing_results.get("recall", {})
                    print(f"       NDCG@20: {ndcg.get('NDCG@20', 'N/A')}")
                    print(f"       Recall@20: {recall.get('Recall@20', 'N/A')}")
            except Exception as e:
                print(f"       Warning: Error reading existing results: {e}")
            continue

        try:
            queries = {}
            corpus = {}
            qrels = {}

            with open(
                os.path.join(data_dir, "queries.jsonl"), "r", encoding="utf-8"
            ) as f:
                for line in f:
                    data = json.loads(line)
                    if "_id" in data and "text" in data:
                        queries[data["_id"]] = data["text"]
                    else:
                        queries.update(data)

            with open(
                os.path.join(data_dir, "corpus.jsonl"), "r", encoding="utf-8"
            ) as f:
                for line in f:
                    data = json.loads(line)
                    corpus.update(data)

            with open(
                os.path.join(data_dir, "qrels.jsonl"), "r", encoding="utf-8"
            ) as f:
                for line in f:
                    data = json.loads(line)
                    qrels.update(data)

            queries = add_prefix_to_queries(queries, model)
            corpus = add_prefix_to_corpus(corpus, model)

            retriever = SAERetriever(**retriever_kwargs)
            evaluator = EvaluateRetrieval(retriever=retriever)
            results = evaluator.retrieve(corpus, queries)
            ndcg, _map, recall, precision = evaluator.evaluate(
                qrels, results, k_values=[1, 3, 5, 10, 20, 100, 1000]
            )

            results_dict = {
                "model": model,
                "sae": sae_path,
                "mask": mask_path,
                "dataset": dataset_name,
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precision": precision,
                "use_reconstruction": use_reconstruction,
            }

            with output_file.open("w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)

            print(f"Results saved to: {output_file}")
            print(f"NDCG@20: {ndcg.get('NDCG@20', 'N/A')}")
            print(f"Recall@20: {recall.get('Recall@20', 'N/A')}")
        except FileNotFoundError as e:
            print(f"[WARN] Missing file for dataset '{dataset_name}': {e}")
        except Exception as e:
            print(f"[ERROR] Evaluation failed for dataset '{dataset_name}': {e}")


if __name__ == "__main__":
    fire.Fire(run_sae_eval)
