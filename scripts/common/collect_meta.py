#!/usr/bin/env python3
"""
Phase 4: Collect run environment metadata.

Gathers system information and dataset statistics for reproducibility.
"""

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path


def run_command(cmd: str, default: str = "") -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() or default
    except Exception:
        return default


def get_cpu_info() -> dict:
    """Get CPU information."""
    info = {}
    
    model_name = run_command("lscpu | grep 'Model name' | cut -d: -f2", "Unknown")
    info["model"] = model_name.strip()
    
    cores = run_command("nproc", "0")
    info["cores"] = int(cores) if cores.isdigit() else 0
    
    sockets = run_command("lscpu | grep 'Socket(s)' | cut -d: -f2", "1")
    info["sockets"] = int(sockets.strip()) if sockets.strip().isdigit() else 1
    
    threads_per_core = run_command("lscpu | grep 'Thread(s) per core' | cut -d: -f2", "1")
    info["threads_per_core"] = int(threads_per_core.strip()) if threads_per_core.strip().isdigit() else 1
    
    mhz = run_command("lscpu | grep 'CPU MHz' | cut -d: -f2", "0")
    try:
        info["mhz"] = float(mhz.strip())
    except ValueError:
        info["mhz"] = 0.0
    
    cache = run_command("lscpu | grep 'L3 cache' | cut -d: -f2", "Unknown")
    info["l3_cache"] = cache.strip()
    
    return info


def get_memory_info() -> dict:
    """Get memory information."""
    info = {}
    
    mem_total = run_command("free -g | grep Mem | awk '{print $2}'", "0")
    try:
        info["total_gb"] = int(mem_total)
    except ValueError:
        info["total_gb"] = 0
    
    mem_available = run_command("free -g | grep Mem | awk '{print $7}'", "0")
    try:
        info["available_gb"] = int(mem_available)
    except ValueError:
        info["available_gb"] = 0
    
    return info


def get_os_info() -> dict:
    """Get OS information."""
    info = {}
    
    info["system"] = platform.system()
    info["release"] = platform.release()
    info["version"] = platform.version()
    info["machine"] = platform.machine()
    
    distro = run_command("lsb_release -d | cut -d: -f2", "Unknown")
    info["distribution"] = distro.strip()
    
    return info


def get_compiler_info() -> dict:
    """Get compiler information."""
    info = {}
    
    gpp_version = run_command("g++ --version | head -1", "Unknown")
    info["gpp"] = gpp_version
    
    cmake_version = run_command("cmake --version | head -1", "Unknown")
    info["cmake"] = cmake_version
    
    return info


def get_python_info() -> dict:
    """Get Python and package versions."""
    info = {}
    
    info["version"] = platform.python_version()
    info["implementation"] = platform.python_implementation()
    
    packages = [
        "numpy",
        "onnxruntime",
        "transformers",
        "optimum",
        "matplotlib",
        "pandas",
        "tqdm",
    ]
    
    for pkg in packages:
        version = run_command(f"python -c \"import {pkg}; print({pkg}.__version__)\" 2>/dev/null", "not installed")
        info[pkg] = version
    
    return info


def get_dataset_info(data_dir: str) -> dict:
    """Get dataset information from metadata files."""
    info = {
        "name": "HotpotQA-Train",
    }
    
    embedding_meta = os.path.join(data_dir, "embedding_meta.json")
    if os.path.exists(embedding_meta):
        with open(embedding_meta, "r") as f:
            meta = json.load(f)
            info["num_docs"] = meta.get("num_documents", 0)
            info["num_queries"] = meta.get("num_queries", 0)
            info["dim"] = meta.get("dimension", 0)
            info["embedding_model"] = meta.get("model_name", "Unknown")
            info["distance_metric"] = meta.get("distance_metric", "Unknown")
    
    stats_file = os.path.join(data_dir, "stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            stats = json.load(f)
            info["total_queries"] = stats.get("total_queries", 0)
            info["total_documents"] = stats.get("total_documents", 0)
    
    return info


def get_hnswlib_info(hnswlib_dir: str) -> dict:
    """Get hnswlib git information."""
    info = {}
    
    commit = run_command(f"cd {hnswlib_dir} && git rev-parse --short HEAD 2>/dev/null", "unknown")
    info["commit"] = commit
    
    branch = run_command(f"cd {hnswlib_dir} && git rev-parse --abbrev-ref HEAD 2>/dev/null", "unknown")
    info["branch"] = branch
    
    return info


def collect_metadata(data_dir: str, hnswlib_dir: str, output_path: str) -> dict:
    """Collect all metadata and save to file."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hostname": platform.node(),
    }
    
    metadata["cpu"] = get_cpu_info()
    metadata["memory"] = get_memory_info()
    metadata["os"] = get_os_info()
    metadata["compiler"] = get_compiler_info()
    metadata["python"] = get_python_info()
    metadata["dataset"] = get_dataset_info(data_dir)
    metadata["hnswlib"] = get_hnswlib_info(hnswlib_dir)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {output_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Collect run metadata")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--hnswlib_dir", required=True, help="hnswlib directory")
    args = parser.parse_args()
    
    collect_metadata(args.data_dir, args.hnswlib_dir, args.output)


if __name__ == "__main__":
    main()
