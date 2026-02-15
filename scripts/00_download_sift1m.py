#!/usr/bin/env python3
"""
Download and extract SIFT1M dataset.

SIFT1M is a standard benchmark dataset for approximate nearest neighbor search.
- 1,000,000 base vectors (128 dimensions)
- 10,000 query vectors
- Ground truth neighbors for each query

Data source: http://corpus-texmex.irisa.fr/
"""

import argparse
import gzip
import os
import struct
import urllib.request
from pathlib import Path


def download_file(url: str, output_path: str) -> None:
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def read_fvecs(filepath: str) -> tuple:
    """Read fvecs file and return (n, dim) array."""
    with open(filepath, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0, 2)
        n = f.tell() // (4 + dim * 4)
        f.seek(0)
        
        data = []
        for _ in range(n):
            d = struct.unpack("i", f.read(4))[0]
            if d != dim:
                raise ValueError(f"Dimension mismatch: expected {dim}, got {d}")
            vec = struct.unpack(f"{dim}f", f.read(dim * 4))
            data.append(vec)
        
        return data, dim


def read_ivecs(filepath: str) -> tuple:
    """Read ivecs file and return (n, dim) array."""
    with open(filepath, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
        f.seek(0, 2)
        n = f.tell() // (4 + dim * 4)
        f.seek(0)
        
        data = []
        for _ in range(n):
            d = struct.unpack("i", f.read(4))[0]
            if d != dim:
                raise ValueError(f"Dimension mismatch: expected {dim}, got {d}")
            vec = struct.unpack(f"{dim}i", f.read(dim * 4))
            data.append(vec)
        
        return data, dim


def write_fvecs(filepath: str, data: list, dim: int) -> None:
    """Write data to fvecs file."""
    with open(filepath, "wb") as f:
        for vec in data:
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}f", *vec))


def write_ivecs(filepath: str, data: list, dim: int) -> None:
    """Write data to ivecs file."""
    with open(filepath, "wb") as f:
        for vec in data:
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}i", *vec))


def download_sift1m(output_dir: str) -> None:
    """Download and extract SIFT1M dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "ftp://ftp.irisa.fr/local/texmex/corpus"
    files = {
        "sift_base.fvecs": f"{base_url}/sift.tar.gz",
    }
    
    tar_path = os.path.join(output_dir, "sift.tar.gz")
    if not os.path.exists(tar_path):
        download_file(files["sift_base.fvecs"], tar_path)
    else:
        print(f"{tar_path} already exists, skipping download")
    
    import tarfile
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)
    
    print("\nRenaming files to standard format...")
    file_mapping = {
        "sift/sift_base.fvecs": "sift1m_base.fvecs",
        "sift/sift_query.fvecs": "sift1m_query.fvecs",
        "sift/sift_groundtruth.ivecs": "sift1m_groundtruth.ivecs",
        "sift/sift_learn.fvecs": "sift1m_learn.fvecs",
    }
    
    for src, dst in file_mapping.items():
        src_path = os.path.join(output_dir, src)
        dst_path = os.path.join(output_dir, dst)
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)
            print(f"  {src} -> {dst}")
    
    sift_dir = os.path.join(output_dir, "sift")
    if os.path.exists(sift_dir):
        os.rmdir(sift_dir)
    
    base_path = os.path.join(output_dir, "sift1m_base.fvecs")
    query_path = os.path.join(output_dir, "sift1m_query.fvecs")
    gt_path = os.path.join(output_dir, "sift1m_groundtruth.ivecs")
    
    print("\nVerifying data...")
    
    base_data, base_dim = read_fvecs(base_path)
    print(f"  Base: {len(base_data)} vectors, dim={base_dim}")
    
    query_data, query_dim = read_fvecs(query_path)
    print(f"  Query: {len(query_data)} vectors, dim={query_dim}")
    
    gt_data, gt_dim = read_ivecs(gt_path)
    print(f"  Ground truth: {len(gt_data)} queries, k={gt_dim}")
    
    print(f"\nSIFT1M dataset ready in {output_dir}")
    print(f"  - sift1m_base.fvecs: 1,000,000 base vectors (128-dim)")
    print(f"  - sift1m_query.fvecs: 10,000 query vectors (128-dim)")
    print(f"  - sift1m_groundtruth.ivecs: 100 nearest neighbors per query")


def main():
    parser = argparse.ArgumentParser(description="Download SIFT1M dataset")
    parser.add_argument(
        "--output_dir",
        default="data/sift1m",
        help="Output directory for SIFT1M data"
    )
    args = parser.parse_args()
    
    download_sift1m(args.output_dir)


if __name__ == "__main__":
    main()
