#!/usr/bin/env python3
"""
Phase 2: Generate embeddings using BAAI/bge-large-en-v1.5 with sentence-transformers.
Simple sequential version for debugging.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_jsonl(path: str) -> list[dict]:
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
    batch_size: int = 64,
    max_length: int = 512,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = max_length
    
    print(f"Loading corpus from {corpus_path}...")
    corpus = load_jsonl(corpus_path)
    corpus_texts = [doc["text"] for doc in corpus]
    print(f"Loaded {len(corpus_texts)} documents")
    
    print(f"Loading queries from {queries_path}...")
    queries = load_jsonl(queries_path)
    query_texts = [q["question"] for q in queries]
    print(f"Loaded {len(query_texts)} queries")
    
    total_start = time.time()
    
    print(f"\n=== Encoding Corpus ({len(corpus_texts)} docs) ===")
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    corpus_embeddings = corpus_embeddings.astype(np.float32)
    
    print(f"\n=== Encoding Queries ({len(query_texts)} queries) ===")
    query_texts_with_instruction = [QUERY_INSTRUCTION + t for t in query_texts]
    query_embeddings = model.encode(
        query_texts_with_instruction,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embeddings = query_embeddings.astype(np.float32)
    
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
    parser.add_argument("--model_name", default="BAAI/bge-large-en-v1.5", help="Model name or path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()
    
    generate_embeddings(
        corpus_path=args.corpus,
        queries_path=args.queries,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
