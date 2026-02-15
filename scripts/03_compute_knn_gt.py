#!/usr/bin/env python3
"""
Phase 2.5: Compute exact KNN ground truth for recall evaluation.

Uses batched matrix multiplication to avoid memory issues with large datasets.
Computes top-K nearest neighbors using inner product (cosine similarity on normalized vectors).
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np
from tqdm import tqdm


def compute_knn_ground_truth(
    corpus_emb_path: str,
    query_emb_path: str,
    output_dir: str,
    top_k: int = 100,
    batch_size: int = 1000,
) -> dict[str, Any]:
    print(f"Loading corpus embeddings from {corpus_emb_path}...")
    corpus_embeddings = np.load(corpus_emb_path)
    num_docs, dim = corpus_embeddings.shape
    print(f"Corpus: {num_docs} documents, {dim} dimensions")
    
    print(f"Loading query embeddings from {query_emb_path}...")
    query_embeddings = np.load(query_emb_path)
    num_queries = query_embeddings.shape[0]
    print(f"Queries: {num_queries}")
    
    norms = np.linalg.norm(corpus_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print("Corpus embeddings not normalized, normalizing...")
        corpus_embeddings = corpus_embeddings / np.clip(
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True), a_min=1e-9, a_max=None
        )
    
    norms = np.linalg.norm(query_embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print("Query embeddings not normalized, normalizing...")
        query_embeddings = query_embeddings / np.clip(
            np.linalg.norm(query_embeddings, axis=1, keepdims=True), a_min=1e-9, a_max=None
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_indices = np.zeros((num_queries, top_k), dtype=np.int32)
    all_distances = np.zeros((num_queries, top_k), dtype=np.float32)
    
    print(f"\nComputing top-{top_k} neighbors with batch size {batch_size}...")
    start_time = time.time()
    
    num_batches = (num_queries + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing KNN"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_queries)
        
        query_batch = query_embeddings[start:end]
        
        scores = query_batch @ corpus_embeddings.T
        
        if top_k >= num_docs:
            batch_indices = np.argsort(-scores, axis=1)[:, :top_k]
        else:
            batch_indices = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
            batch_scores = np.take_along_axis(scores, batch_indices, axis=1)
            sorted_order = np.argsort(-batch_scores, axis=1)
            batch_indices = np.take_along_axis(batch_indices, sorted_order, axis=1)
        
        batch_distances = np.take_along_axis(scores, batch_indices, axis=1)
        
        all_indices[start:end] = batch_indices
        all_distances[start:end] = batch_distances
    
    total_time = time.time() - start_time
    
    indices_path = os.path.join(output_dir, "knn_gt_indices.npy")
    distances_path = os.path.join(output_dir, "knn_gt_distances.npy")
    
    print(f"\nSaving indices to {indices_path}...")
    np.save(indices_path, all_indices)
    
    print(f"Saving distances to {distances_path}...")
    np.save(distances_path, all_distances)
    
    meta = {
        "num_queries": num_queries,
        "num_documents": num_docs,
        "top_k": top_k,
        "batch_size": batch_size,
        "distance_metric": "ip",
        "computation_time_sec": total_time,
        "indices_file": "knn_gt_indices.npy",
        "distances_file": "knn_gt_distances.npy",
    }
    
    meta_path = os.path.join(output_dir, "knn_gt_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print("\n=== Summary ===")
    print(f"Total queries: {num_queries}")
    print(f"Total documents: {num_docs}")
    print(f"Top-K: {top_k}")
    print(f"Computation time: {total_time:.2f} seconds")
    print(f"Throughput: {num_queries / total_time:.2f} queries/sec")
    print(f"Sample distances (query 0, top 5): {all_distances[0, :5]}")
    print(f"Sample indices (query 0, top 5): {all_indices[0, :5]}")
    
    return meta


def main():
    parser = argparse.ArgumentParser(description="Compute exact KNN ground truth")
    parser.add_argument("--corpus_emb", required=True, help="Path to corpus_embeddings.npy")
    parser.add_argument("--query_emb", required=True, help="Path to query_embeddings.npy")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--top_k", type=int, default=100, help="Number of neighbors to compute")
    parser.add_argument("--batch_size", type=int, default=1000, help="Query batch size")
    args = parser.parse_args()
    
    compute_knn_ground_truth(
        corpus_emb_path=args.corpus_emb,
        query_emb_path=args.query_emb,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
