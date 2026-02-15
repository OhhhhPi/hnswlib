#!/usr/bin/env python3
"""
Phase 2: Generate embeddings using BAAI/bge-large-en-v1.5 with sentence-transformers.

Uses multi-process inference with each worker loading its own model copy.
Documents are encoded without instruction prefix, queries with search instruction.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def encode_worker(args: tuple) -> dict[str, Any]:
    """Worker function to encode a shard of texts."""
    (
        shard_idx,
        texts,
        model_name,
        batch_size,
        threads_per_worker,
        is_query,
        shard_output_path,
    ) = args
    
    import torch
    torch.set_num_threads(threads_per_worker)
    
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 512
    
    all_embeddings = []
    total_texts = len(texts)
    
    if is_query:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        embeddings = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        all_embeddings.append(embeddings.astype(np.float32))
    
    all_embeddings = np.vstack(all_embeddings)
    
    np.save(shard_output_path, all_embeddings)
    
    del model
    
    return {
        "shard_idx": shard_idx,
        "num_texts": total_texts,
        "embedding_shape": all_embeddings.shape,
        "output_path": shard_output_path,
    }


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def generate_embeddings(
    corpus_path: str,
    queries_path: str,
    model_name: str,
    output_dir: str,
    num_workers: int,
    threads_per_worker: int,
    batch_size: int,
    max_length: int = 512,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading corpus from {corpus_path}...")
    corpus = load_jsonl(corpus_path)
    corpus_texts = [doc["text"] for doc in corpus]
    print(f"Loaded {len(corpus_texts)} documents")
    
    print(f"Loading queries from {queries_path}...")
    queries = load_jsonl(queries_path)
    query_texts = [q["question"] for q in queries]
    print(f"Loaded {len(query_texts)} queries")
    
    total_start = time.time()
    
    print(f"\n=== Encoding Corpus ({len(corpus_texts)} docs, {num_workers} workers) ===")
    corpus_shard_dir = os.path.join(output_dir, "corpus_shards")
    os.makedirs(corpus_shard_dir, exist_ok=True)
    
    corpus_shard_size = (len(corpus_texts) + num_workers - 1) // num_workers
    corpus_tasks = []
    for i in range(num_workers):
        start_idx = i * corpus_shard_size
        end_idx = min(start_idx + corpus_shard_size, len(corpus_texts))
        if start_idx >= len(corpus_texts):
            break
        shard_texts = corpus_texts[start_idx:end_idx]
        shard_path = os.path.join(corpus_shard_dir, f"shard_{i:04d}.npy")
        corpus_tasks.append((
            i,
            shard_texts,
            model_name,
            batch_size,
            threads_per_worker,
            False,
            shard_path,
        ))
    
    with Pool(num_workers) as pool:
        corpus_results = list(tqdm(
            pool.imap(encode_worker, corpus_tasks),
            total=len(corpus_tasks),
            desc="Corpus shards"
        ))
    
    print(f"\n=== Encoding Queries ({len(query_texts)} queries, {num_workers} workers) ===")
    query_shard_dir = os.path.join(output_dir, "query_shards")
    os.makedirs(query_shard_dir, exist_ok=True)
    
    query_shard_size = (len(query_texts) + num_workers - 1) // num_workers
    query_tasks = []
    for i in range(num_workers):
        start_idx = i * query_shard_size
        end_idx = min(start_idx + query_shard_size, len(query_texts))
        if start_idx >= len(query_texts):
            break
        shard_texts = query_texts[start_idx:end_idx]
        shard_path = os.path.join(query_shard_dir, f"shard_{i:04d}.npy")
        query_tasks.append((
            i,
            shard_texts,
            model_name,
            batch_size,
            threads_per_worker,
            True,
            shard_path,
        ))
    
    with Pool(num_workers) as pool:
        query_results = list(tqdm(
            pool.imap(encode_worker, query_tasks),
            total=len(query_tasks),
            desc="Query shards"
        ))
    
    print("\n=== Merging Shards ===")
    
    corpus_embeddings = []
    for r in sorted(corpus_results, key=lambda x: x["shard_idx"]):
        shard = np.load(r["output_path"])
        corpus_embeddings.append(shard)
    corpus_embeddings = np.vstack(corpus_embeddings)
    
    query_embeddings = []
    for r in sorted(query_results, key=lambda x: x["shard_idx"]):
        shard = np.load(r["output_path"])
        query_embeddings.append(shard)
    query_embeddings = np.vstack(query_embeddings)
    
    corpus_output_path = os.path.join(output_dir, "corpus_embeddings.npy")
    query_output_path = os.path.join(output_dir, "query_embeddings.npy")
    
    print(f"Saving corpus embeddings to {corpus_output_path}...")
    np.save(corpus_output_path, corpus_embeddings)
    
    print(f"Saving query embeddings to {query_output_path}...")
    np.save(query_output_path, query_embeddings)
    
    total_time = time.time() - total_start
    
    dim = corpus_embeddings.shape[1]
    
    meta = {
        "model_name": model_name,
        "backend": "sentence-transformers + pytorch-cpu",
        "dimension": dim,
        "normalized": True,
        "distance_metric": "ip",
        "num_documents": len(corpus_embeddings),
        "num_queries": len(query_embeddings),
        "encoding_time_sec": total_time,
        "num_workers": num_workers,
        "threads_per_worker": threads_per_worker,
        "batch_size": batch_size,
        "max_length": max_length,
        "query_instruction": QUERY_INSTRUCTION,
        "corpus_embeddings_file": "corpus_embeddings.npy",
        "query_embeddings_file": "query_embeddings.npy",
    }
    
    meta_path = os.path.join(output_dir, "embedding_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Corpus embeddings shape: {corpus_embeddings.shape}")
    print(f"Query embeddings shape: {query_embeddings.shape}")
    print(f"Total encoding time: {total_time:.2f} seconds")
    print(f"Corpus throughput: {len(corpus_embeddings) / total_time:.2f} docs/sec")
    print(f"Query throughput: {len(query_embeddings) / total_time:.2f} queries/sec")
    print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl")
    parser.add_argument("--model_name", default="BAAI/bge-large-en-v1.5", help="Model name")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--threads_per_worker", type=int, default=4, help="Threads per worker")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()
    
    generate_embeddings(
        corpus_path=args.corpus,
        queries_path=args.queries,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
