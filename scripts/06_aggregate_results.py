#!/usr/bin/env python3
"""
Phase 4: Aggregate C++ benchmark results from multiple JSON files.

Creates summary.json, summary.csv, and best_configs.json.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def load_json_files(input_dir: str) -> list[dict[str, Any]]:
    """Load all JSON files from input directory."""
    results = []
    input_path = Path(input_dir)
    
    for json_file in sorted(input_path.glob("*.json")):
        print(f"Loading {json_file}...")
        with open(json_file, "r") as f:
            data = json.load(f)
            results.append(data)
    
    return results


def flatten_results(all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested results into a list of row dictionaries."""
    rows = []
    
    for result in all_results:
        base_info = {
            "dataset": result.get("dataset", ""),
            "embedding_model": result.get("embedding_model", ""),
            "dim": result.get("dim", 0),
            "num_base": result.get("num_base", 0),
            "num_query": result.get("num_query", 0),
        }
        
        index_info = result.get("index", {})
        base_info["M"] = index_info.get("M", 0)
        base_info["ef_construction"] = index_info.get("ef_construction", 0)
        base_info["build_time_sec"] = index_info.get("build_time_sec", 0)
        base_info["build_throughput_pts_per_sec"] = index_info.get("build_throughput_pts_per_sec", 0)
        base_info["index_memory_mb"] = index_info.get("index_memory_mb", 0)
        base_info["data_memory_mb"] = index_info.get("data_memory_mb", 0)
        
        for res in result.get("results", []):
            row = base_info.copy()
            row["ef_search"] = int(res.get("ef_search", 0))
            row["K"] = int(res.get("K", 0))
            row["recall"] = float(res.get("recall", 0))
            row["mean_us"] = float(res.get("mean_us", 0))
            row["p50_us"] = float(res.get("p50_us", 0))
            row["p95_us"] = float(res.get("p95_us", 0))
            row["p99_us"] = float(res.get("p99_us", 0))
            row["max_us"] = float(res.get("max_us", 0))
            
            for key, value in res.items():
                if key.startswith("qps_"):
                    row[key] = float(value)
            
            rows.append(row)
    
    return rows


def find_best_configs(rows: list[dict[str, Any]], k_filter: int = 10) -> dict[str, dict[str, Any]]:
    """Find best configurations by various criteria."""
    k10_rows = [r for r in rows if r["K"] == k_filter]
    
    if not k10_rows:
        return {}
    
    best = {}
    
    sorted_by_recall = sorted(k10_rows, key=lambda x: -x["recall"])
    if sorted_by_recall:
        best["highest_recall_at_k10"] = sorted_by_recall[0]
    
    above_95 = [r for r in k10_rows if r["recall"] >= 0.95]
    
    if above_95:
        sorted_by_p99 = sorted(above_95, key=lambda x: x["p99_us"])
        best["lowest_p99_above_95_recall"] = sorted_by_p99[0]
        
        sorted_by_qps = sorted(above_95, key=lambda x: -x.get("qps_1t", 0))
        best["highest_qps_above_95_recall"] = sorted_by_qps[0]
        
        sorted_by_mem = sorted(above_95, key=lambda x: x["index_memory_mb"])
        best["best_memory_efficiency"] = sorted_by_mem[0]
    
    sorted_by_latency = sorted(k10_rows, key=lambda x: x["mean_us"])
    if sorted_by_latency:
        best["lowest_mean_latency"] = sorted_by_latency[0]
    
    return best


def aggregate_results(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = load_json_files(input_dir)
    
    if not all_results:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"\nLoaded {len(all_results)} result files")
    
    rows = flatten_results(all_results)
    print(f"Flattened to {len(rows)} result rows")
    
    summary = {
        "num_configs": len(all_results),
        "num_result_rows": len(rows),
        "configs": [],
        "results": rows,
    }
    
    for result in all_results:
        config = {
            "M": result.get("index", {}).get("M"),
            "ef_construction": result.get("index", {}).get("ef_construction"),
            "build_time_sec": result.get("index", {}).get("build_time_sec"),
            "index_memory_mb": result.get("index", {}).get("index_memory_mb"),
        }
        summary["configs"].append(config)
    
    summary_path = os.path.join(output_dir, "summary.json")
    print(f"Writing {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    if rows:
        csv_path = os.path.join(output_dir, "summary.csv")
        print(f"Writing {csv_path}...")
        
        fieldnames = [
            "M", "ef_construction", "ef_search", "K",
            "recall", "mean_us", "p50_us", "p95_us", "p99_us", "max_us",
            "qps_1t", "qps_2t", "qps_4t", "qps_8t", "qps_16t", "qps_32t", "qps_64t",
            "build_time_sec", "build_throughput_pts_per_sec",
            "index_memory_mb", "data_memory_mb",
            "dim", "num_base", "num_query"
        ]
        
        fieldnames = [f for f in fieldnames if f in rows[0] or any(f in r for r in rows)]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    
    best_configs = find_best_configs(rows)
    best_path = os.path.join(output_dir, "best_configs.json")
    print(f"Writing {best_path}...")
    with open(best_path, "w") as f:
        json.dump(best_configs, f, indent=2)
    
    print("\n=== Best Configurations ===")
    for target, cfg in best_configs.items():
        print(f"\n{target}:")
        print(f"  M={cfg['M']}, efc={cfg['ef_construction']}, efs={cfg['ef_search']}")
        print(f"  recall@{cfg['K']}={cfg['recall']:.4f}")
        print(f"  p99={cfg['p99_us']:.1f}us")
        qps_1t = cfg.get("qps_1t", 0)
        if qps_1t:
            print(f"  QPS(1t)={qps_1t:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument("--input_dir", required=True, help="Directory with raw JSON results")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    aggregate_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
