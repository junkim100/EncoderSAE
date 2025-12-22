import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import fire
import torch
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.base import BaseSearch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)


class DenseRetriever(BaseSearch):
    """Dense retriever that wraps a SentenceTransformer model for BEIR evaluation."""

    def __init__(
        self,
        model: SentenceTransformer,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        **kwargs,
    ):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.results: Dict[str, Dict[str, float]] = {}

    def search(
        self,
        corpus: Dict[str, Dict[str, Any]],
        queries: Dict[str, str],
        top_k: int,
        score_function: str = "cos_sim",
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """Search the corpus for each query and return top_k results."""
        logger.info("Encoding queries...")
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
        )

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
        )

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

    def encode(
        self,
        corpus,
        queries,
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ):
        raise NotImplementedError("encode method not implemented")

    def search_from_files(
        self, query_embeddings_file, corpus_embeddings_files, top_k: int, **kwargs
    ):
        raise NotImplementedError("search_from_files method not implemented")


def load_queries_from_jsonl(file_path: str) -> Dict[str, str]:
    """Load queries from JSONL as {qid: query_text}."""
    queries: Dict[str, str] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "_id" in data and "text" in data:
                queries[data["_id"]] = data["text"]
            else:
                queries.update(data)
    return queries


def load_corpus_from_jsonl(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load corpus from JSONL as {doc_id: {"text": context, ...}}."""
    corpus: Dict[str, Dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.update(data)
    return corpus


def load_qrels_from_jsonl(file_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels from JSONL as {qid: {doc_id: relevance}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qrels.update(data)
    return qrels


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
    corpus: Dict[str, Dict[str, Any]], model_name: str
) -> Dict[str, Dict[str, Any]]:
    """Add model-specific prefix to corpus texts."""
    lower = model_name.lower()
    if "e5" in lower:
        return {
            doc_id: {"text": f"passage: {doc['text']}"}
            for doc_id, doc in corpus.items()
        }
    if "jina" in lower:
        return {
            doc_id: {"text": f"Represent the document for retrieval: {doc['text']}"}
            for doc_id, doc in corpus.items()
        }
    if "embeddinggemma" in lower:
        return {
            doc_id: {"text": f"title: none | text: {doc['text']}"}
            for doc_id, doc in corpus.items()
        }
    if model_name == "nvidia/llama-nemotron-embed-1b-v2":
        return {
            doc_id: {"text": f"passage: {doc['text']}"}
            for doc_id, doc in corpus.items()
        }
    return corpus


def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
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

    if hasattr(model[0], "auto_model") and hasattr(model[0].auto_model, "config"):
        model[0].auto_model.config.use_cache = False

    return model


def run(
    model: str,
    data_dirs: list[str],
    results_root: str = "./results_base",
    batch_size: int = 1024,
    overwrite: bool = False,
) -> None:
    """
    Run BEIR-style evaluation for dense embedding models.

    Args:
        model: Model name to evaluate (HuggingFace ID or local path).
        data_dirs: List of dataset directory paths. Each directory must contain
            queries.jsonl, corpus.jsonl, and qrels.jsonl.
        results_root: Root directory where result JSON files are stored.
        batch_size: Batch size for encoding corpus/queries.
        overwrite: If False, skip datasets for which results already exist.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    results_path = Path(results_root)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Evaluating model: {model}")
    print(f"{'=' * 80}")

    try:
        st_model = _load_sentence_transformer(model)
        retriever = DenseRetriever(st_model, batch_size=batch_size)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{model}': {e}")
        return

    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir.rstrip("/"))
        print(f"\nEvaluating on dataset: {dataset_name}")

        model_safe_name = model.split("/")[-1]
        output_dir = results_path / model_safe_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_name}_results.json"

        if output_file.exists() and not overwrite:
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
            queries = load_queries_from_jsonl(os.path.join(data_dir, "queries.jsonl"))
            corpus = load_corpus_from_jsonl(os.path.join(data_dir, "corpus.jsonl"))
            qrels = load_qrels_from_jsonl(os.path.join(data_dir, "qrels.jsonl"))

            queries = add_prefix_to_queries(queries, model)
            corpus = add_prefix_to_corpus(corpus, model)

            evaluator = EvaluateRetrieval(retriever=retriever)
            results = evaluator.retrieve(corpus, queries)
            ndcg, _map, recall, precision = evaluator.evaluate(
                qrels, results, k_values=[1, 3, 5, 10, 20, 100, 1000]
            )

            results_dict = {
                "model": model,
                "dataset": dataset_name,
                "ndcg": ndcg,
                "map": _map,
                "recall": recall,
                "precision": precision,
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
    fire.Fire(run)
