#!/usr/bin/env python3
"""
Phase 2.6: Export embeddings and ground truth to fvecs/ivecs format for C++ benchmark.

fvecs format: [dim (int32)] [v1 (float32)] ... [v_dim (float32)] per vector
ivecs format: [K (int32)] [id1 (int32)] ... [id_K (int32)] per vector
"""

import argparse
import json
import os
from typing import Any

import numpy as np
from tqdm import tqdm


def write_fvecs(filename: str, vectors: np.ndarray) -> None:
    """Write vectors to fvecs format."""
    n, dim = vectors.shape
    print(f"Writing {n} vectors of dim {dim} to {filename}...")
    
    with open(filename, "wb") as f:
        for i in tqdm(range(n), desc="Writing fvecs"):
            f.write(np.array([dim], dtype=np.int32).tobytes())
            f.write(vectors[i].astype(np.float32).tobytes())
    
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size / (1024**3):.2f} GB")


def write_ivecs(filename: str, indices: np.ndarray) -> None:
    """Write indices to ivecs format."""
    n, k = indices.shape
    print(f"Writing {n} vectors of k={k} to {filename}...")
    
    with open(filename, "wb") as f:
        for i in tqdm(range(n), desc="Writing ivecs"):
            f.write(np.array([k], dtype=np.int32).tobytes())
            f.write(indices[i].astype(np.int32).tobytes())
    
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size / (1024**2):.2f} MB")


def export_vectors(
    corpus_emb_path: str,
    query_emb_path: str,
    gt_indices_path: str,
    output_dir: str,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading corpus embeddings from {corpus_emb_path}...")
    corpus_embeddings = np.load(corpus_emb_path)
    num_docs, dim = corpus_embeddings.shape
    print(f"Corpus: {num_docs} vectors, dim={dim}")
    
    print(f"Loading query embeddings from {query_emb_path}...")
    query_embeddings = np.load(query_emb_path)
    num_queries = query_embeddings.shape[0]
    print(f"Queries: {num_queries} vectors")
    
    print(f"Loading ground truth indices from {gt_indices_path}...")
    gt_indices = np.load(gt_indices_path)
    gt_k = gt_indices.shape[1]
    print(f"Ground truth: {gt_indices.shape[0]} queries, k={gt_k}")
    
    corpus_fvecs = os.path.join(output_dir, "corpus_vectors.fvecs")
    query_fvecs = os.path.join(output_dir, "query_vectors.fvecs")
    gt_ivecs = os.path.join(output_dir, "ground_truth.ivecs")
    
    write_fvecs(corpus_fvecs, corpus_embeddings)
    write_fvecs(query_fvecs, query_embeddings)
    write_ivecs(gt_ivecs, gt_indices)
    
    corpus_npy = os.path.join(output_dir, "corpus_embeddings.npy")
    query_npy = os.path.join(output_dir, "query_embeddings.npy")
    
    if corpus_emb_path != corpus_npy:
        print(f"Copying corpus embeddings to {corpus_npy}...")
        np.save(corpus_npy, corpus_embeddings)
    
    if query_emb_path != query_npy:
        print(f"Copying query embeddings to {query_npy}...")
        np.save(query_npy, query_embeddings)
    
    meta = {
        "corpus_fvecs": "corpus_vectors.fvecs",
        "query_fvecs": "query_vectors.fvecs",
        "ground_truth_ivecs": "ground_truth.ivecs",
        "corpus_npy": "corpus_embeddings.npy",
        "query_npy": "query_embeddings.npy",
        "num_documents": num_docs,
        "num_queries": num_queries,
        "dimension": dim,
        "ground_truth_k": gt_k,
    }
    
    meta_path = os.path.join(output_dir, "export_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n=== Export Summary ===")
    print(f"Corpus vectors: {corpus_fvecs} ({num_docs} x {dim})")
    print(f"Query vectors: {query_fvecs} ({num_queries} x {dim})")
    print(f"Ground truth: {gt_ivecs} ({num_queries} x {gt_k})")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Export vectors to fvecs/ivecs format")
    parser.add_argument("--corpus_emb", required=True, help="Path to corpus_embeddings.npy")
    parser.add_argument("--query_emb", required=True, help="Path to query_embeddings.npy")
    parser.add_argument("--gt", required=True, help="Path to knn_gt_indices.npy")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    export_vectors(
        corpus_emb_path=args.corpus_emb,
        query_emb_path=args.query_emb,
        gt_indices_path=args.gt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
