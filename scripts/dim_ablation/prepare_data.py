#!/usr/bin/env python3
"""
Prepare data for dimensionality ablation experiment on SIFT1M.

This script:
1. Reads SIFT1M vectors (128-dim) from fvecs format
2. Creates zero-padded versions at target dimensions (256, 512, 1024)
3. Copies ground truth (unchanged: zero-padding preserves L2 distances)
4. Performs PCA intrinsic dimensionality analysis on SIFT1M and (optionally) HotpotQA

Zero-padding rationale:
  L2(pad(x), pad(y)) = sqrt(sum((xi-yi)^2, i=1..128) + sum(0^2, i=129..D))
                      = L2(x, y)
  So nearest-neighbor ordering is identical, ground truth is reusable,
  and only the per-distance FLOP cost changes.
"""

import argparse
import json
import os
import shutil
import struct
import time

import numpy as np


# ---------------------------------------------------------------------------
# fvecs I/O (fast, numpy-based)
# ---------------------------------------------------------------------------

def read_fvecs(filename: str) -> np.ndarray:
    """Read fvecs file into numpy array (fast path)."""
    with open(filename, "rb") as f:
        dim = struct.unpack("<i", f.read(4))[0]
    raw = np.fromfile(filename, dtype=np.float32)
    num = len(raw) // (dim + 1)
    raw = raw.reshape(num, dim + 1)
    return raw[:, 1:].copy()


def write_fvecs(filename: str, data: np.ndarray) -> None:
    """Write numpy array to fvecs format (fast path)."""
    n, dim = data.shape
    data = np.ascontiguousarray(data, dtype=np.float32)
    dim_col = np.full((n, 1), dim, dtype=np.int32).view(np.float32)
    out = np.hstack([dim_col, data])
    out.tofile(filename)
    size_mb = os.path.getsize(filename) / (1024 ** 2)
    print(f"  Written {filename} ({n} x {dim}, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Zero-padding
# ---------------------------------------------------------------------------

def zero_pad(data: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad vectors from orig_dim to target_dim."""
    n, orig_dim = data.shape
    if target_dim <= orig_dim:
        raise ValueError(f"target_dim ({target_dim}) must be > orig_dim ({orig_dim})")
    padded = np.zeros((n, target_dim), dtype=np.float32)
    padded[:, :orig_dim] = data
    return padded


# ---------------------------------------------------------------------------
# PCA intrinsic dimensionality analysis
# ---------------------------------------------------------------------------

def pca_analysis(data: np.ndarray, name: str, max_samples: int = 100_000) -> dict:
    """Compute PCA explained-variance spectrum."""
    n, dim = data.shape
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        data = data[idx]
        n = max_samples

    print(f"  PCA on {name}: {n} samples, {dim} dims ...")
    t0 = time.time()

    mean = data.mean(axis=0)
    centered = data - mean

    # Covariance matrix approach: O(n*d^2 + d^3), efficient when d << n
    cov = (centered.T @ centered) / (n - 1)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending
    eigenvalues = eigenvalues[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability

    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
    cumulative = np.cumsum(explained_ratio)

    # Intrinsic dimensionality at standard thresholds
    thresholds = [0.80, 0.90, 0.95, 0.99]
    intrinsic_dims = {}
    for t in thresholds:
        intrinsic_dims[f"dim_at_{int(t * 100)}pct"] = int(np.searchsorted(cumulative, t) + 1)

    elapsed = time.time() - t0
    print(f"  PCA done in {elapsed:.1f}s")
    for t in thresholds:
        key = f"dim_at_{int(t * 100)}pct"
        print(f"    {int(t * 100)}% variance captured by {intrinsic_dims[key]} / {dim} components")

    return {
        "name": name,
        "ambient_dim": dim,
        "num_samples": n,
        "intrinsic_dims": intrinsic_dims,
        "total_variance": float(total_var),
        "top_eigenvalues": eigenvalues[:min(dim, 200)].tolist(),
        "explained_variance_ratio": explained_ratio[:min(dim, 200)].tolist(),
        "cumulative_variance_ratio": cumulative.tolist(),
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare zero-padded SIFT1M data for dimensionality ablation"
    )
    parser.add_argument(
        "--sift1m_dir", default="data/sift1m",
        help="Path to original SIFT1M data directory"
    )
    parser.add_argument(
        "--hotpotqa_dir", default="data/hotpotqa",
        help="Path to HotpotQA data directory (for PCA comparison)"
    )
    parser.add_argument(
        "--output_dir", default="data/sift1m_dim_ablation",
        help="Output directory for padded data"
    )
    parser.add_argument(
        "--target_dims", default="256,512,1024",
        help="Comma-separated target dimensions for zero-padding"
    )
    parser.add_argument(
        "--pca_samples", type=int, default=100_000,
        help="Max samples for PCA analysis"
    )
    parser.add_argument(
        "--skip_pca", action="store_true",
        help="Skip PCA analysis"
    )
    args = parser.parse_args()

    target_dims = [int(d) for d in args.target_dims.split(",")]

    base_path = os.path.join(args.sift1m_dir, "sift1m_base.fvecs")
    query_path = os.path.join(args.sift1m_dir, "sift1m_query.fvecs")
    gt_path = os.path.join(args.sift1m_dir, "sift1m_groundtruth.ivecs")

    for p in [base_path, query_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # ------------------------------------------------------------------
    # Step 1: Load original SIFT1M data
    # ------------------------------------------------------------------
    print("=" * 60)
    print(" Step 1: Loading SIFT1M data")
    print("=" * 60)
    base_data = read_fvecs(base_path)
    query_data = read_fvecs(query_path)
    orig_dim = base_data.shape[1]
    print(f"  Base:  {base_data.shape[0]} vectors, {orig_dim} dims")
    print(f"  Query: {query_data.shape[0]} vectors, {query_data.shape[1]} dims")

    # ------------------------------------------------------------------
    # Step 2: Create zero-padded variants
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Step 2: Creating zero-padded variants")
    print("=" * 60)

    for target_dim in target_dims:
        dim_dir = os.path.join(args.output_dir, f"dim_{target_dim}")
        base_out = os.path.join(dim_dir, "base.fvecs")
        query_out = os.path.join(dim_dir, "query.fvecs")
        gt_out = os.path.join(dim_dir, "groundtruth.ivecs")

        if os.path.exists(base_out) and os.path.exists(query_out) and os.path.exists(gt_out):
            print(f"\n  [dim={target_dim}] Already exists, skipping.")
            continue

        os.makedirs(dim_dir, exist_ok=True)
        print(f"\n  [dim={target_dim}] Zero-padding {orig_dim} -> {target_dim} ...")

        t0 = time.time()
        padded_base = zero_pad(base_data, target_dim)
        padded_query = zero_pad(query_data, target_dim)
        print(f"    Padding done in {time.time() - t0:.1f}s")

        write_fvecs(base_out, padded_base)
        write_fvecs(query_out, padded_query)

        # Ground truth: copy original (L2 distances preserved by zero-padding)
        shutil.copy2(gt_path, gt_out)
        print(f"    Copied ground truth to {gt_out}")

        del padded_base, padded_query

    # ------------------------------------------------------------------
    # Step 3: PCA intrinsic dimensionality analysis
    # ------------------------------------------------------------------
    if not args.skip_pca:
        print("\n" + "=" * 60)
        print(" Step 3: PCA intrinsic dimensionality analysis")
        print("=" * 60)

        pca_results = {}

        # SIFT1M analysis
        pca_results["sift1m"] = pca_analysis(
            base_data, "SIFT1M (128-dim)", max_samples=args.pca_samples
        )

        # HotpotQA analysis (if embeddings available)
        hotpotqa_emb = os.path.join(args.hotpotqa_dir, "corpus_embeddings.npy")
        if os.path.exists(hotpotqa_emb):
            print(f"\n  Loading HotpotQA embeddings from {hotpotqa_emb} ...")
            hotpotqa_data = np.load(hotpotqa_emb)
            print(f"  HotpotQA: {hotpotqa_data.shape[0]} vectors, {hotpotqa_data.shape[1]} dims")
            pca_results["hotpotqa"] = pca_analysis(
                hotpotqa_data, "HotpotQA (1024-dim)", max_samples=args.pca_samples
            )
            del hotpotqa_data
        else:
            # Try loading from fvecs
            hotpotqa_fvecs = os.path.join(args.hotpotqa_dir, "corpus_vectors.fvecs")
            if os.path.exists(hotpotqa_fvecs):
                print(f"\n  Loading HotpotQA from {hotpotqa_fvecs} ...")
                hotpotqa_data = read_fvecs(hotpotqa_fvecs)
                print(f"  HotpotQA: {hotpotqa_data.shape[0]} vectors, {hotpotqa_data.shape[1]} dims")
                pca_results["hotpotqa"] = pca_analysis(
                    hotpotqa_data, "HotpotQA (1024-dim)", max_samples=args.pca_samples
                )
                del hotpotqa_data
            else:
                print(f"\n  HotpotQA data not found at {hotpotqa_emb} or {hotpotqa_fvecs}, skipping.")

        # Save PCA analysis
        os.makedirs(args.output_dir, exist_ok=True)
        pca_path = os.path.join(args.output_dir, "pca_analysis.json")
        with open(pca_path, "w") as f:
            json.dump(pca_results, f, indent=2)
        print(f"\n  PCA analysis saved to {pca_path}")

    # ------------------------------------------------------------------
    # Step 4: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  Original data: {args.sift1m_dir} (dim={orig_dim})")
    print(f"  Padded data:   {args.output_dir}")
    for target_dim in target_dims:
        dim_dir = os.path.join(args.output_dir, f"dim_{target_dim}")
        print(f"    dim_{target_dim}/  (base.fvecs, query.fvecs, groundtruth.ivecs)")

    meta = {
        "source": "SIFT1M",
        "original_dim": orig_dim,
        "num_base": int(base_data.shape[0]),
        "num_query": int(query_data.shape[0]),
        "target_dims": target_dims,
        "padding_method": "zero",
        "gt_preserved": True,
        "distance_metric": "L2",
    }
    meta_path = os.path.join(args.output_dir, "ablation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
