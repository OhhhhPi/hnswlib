#!/usr/bin/env python3
"""
Phase 4: Generate visualization plots from benchmark results.

Creates 7 plots:
1. recall_vs_qps.png - Pareto curve
2. recall_vs_latency_p99.png
3. ef_search_vs_recall.png
4. ef_search_vs_latency.png
5. thread_scalability.png
6. memory_comparison.png
7. build_time_comparison.png
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style():
    """Setup matplotlib style."""
    plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_recall_vs_qps(df: pd.DataFrame, output_path: str) -> None:
    """Plot recall vs QPS (Pareto curve)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_k10 = df[df["K"] == 10].copy()
    
    if df_k10.empty:
        print("No data for K=10, skipping recall_vs_qps")
        return
    
    unique_configs = sorted(df_k10[["M", "ef_construction"]].drop_duplicates().values.tolist())
    
    color_map = {
        (8, 100): "#8B0000",
        (8, 200): "#DC143C",
        (8, 400): "#FF6347",
        (16, 100): "#00008B",
        (16, 200): "#1E90FF",
        (16, 400): "#87CEEB",
        (32, 100): "#006400",
        (32, 200): "#32CD32",
        (32, 400): "#90EE90",
        (48, 100): "#8B4513",
        (48, 200): "#DAA520",
        (48, 400): "#FFD700"
    }
    
    marker_map = {
        (8, 100): "o",
        (8, 200): "o",
        (8, 400): "o",
        (16, 100): "s",
        (16, 200): "s",
        (16, 400): "s",
        (32, 100): "D",
        (32, 200): "D",
        (32, 400): "D",
        (48, 100): "^",
        (48, 200): "^",
        (48, 400): "^"
    }
    
    linestyle_map = {
        (8, 100): "-",
        (8, 200): "--",
        (8, 400): ":",
        (16, 100): "-",
        (16, 200): "--",
        (16, 400): ":",
        (32, 100): "-",
        (32, 200): "--",
        (32, 400): ":",
        (48, 100): "-",
        (48, 200): "--",
        (48, 400): ":"
    }
    
    for (m, efc) in unique_configs:
        df_subset = df_k10[(df_k10["M"] == m) & (df_k10["ef_construction"] == efc)].sort_values("recall")
        
        if df_subset.empty:
            continue
        
        label = f"M={m}, efc={efc}"
        color = color_map.get((m, efc), "#000000")
        marker = marker_map.get((m, efc), "o")
        linestyle = linestyle_map.get((m, efc), "-")
        
        ax.scatter(
            df_subset["recall"].values,
            df_subset["qps_1t"].values,
            label=label,
            color=color,
            marker=marker,
            s=70,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5
        )
        ax.plot(
            df_subset["recall"].values,
            df_subset["qps_1t"].values,
            color=color,
            linestyle=linestyle,
            alpha=0.7,
            linewidth=2
        )
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            if idx % 2 == 0 or idx == len(df_subset) - 1:
                ax.annotate(
                    str(int(row["ef_search"])),
                    (row["recall"], row["qps_1t"]),
                    fontsize=7,
                    alpha=0.9,
                    xytext=(6, 6),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor="none")
                )
    
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (1 thread)")
    ax.set_yscale("log")
    ax.set_title("Recall vs QPS (Higher and more right is better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_recall_vs_latency_p99(df: pd.DataFrame, output_path: str) -> None:
    """Plot recall vs P99 latency."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_k10 = df[df["K"] == 10].copy()
    
    if df_k10.empty:
        print("No data for K=10, skipping recall_vs_latency_p99")
        return
    
    unique_configs = sorted(df_k10[["M", "ef_construction"]].drop_duplicates().values.tolist())
    
    color_map = {
        (8, 100): "#8B0000",
        (8, 200): "#DC143C",
        (8, 400): "#FF6347",
        (16, 100): "#00008B",
        (16, 200): "#1E90FF",
        (16, 400): "#87CEEB",
        (32, 100): "#006400",
        (32, 200): "#32CD32",
        (32, 400): "#90EE90",
        (48, 100): "#8B4513",
        (48, 200): "#DAA520",
        (48, 400): "#FFD700"
    }
    
    marker_map = {
        (8, 100): "o",
        (8, 200): "o",
        (8, 400): "o",
        (16, 100): "s",
        (16, 200): "s",
        (16, 400): "s",
        (32, 100): "D",
        (32, 200): "D",
        (32, 400): "D",
        (48, 100): "^",
        (48, 200): "^",
        (48, 400): "^"
    }
    
    linestyle_map = {
        (8, 100): "-",
        (8, 200): "--",
        (8, 400): ":",
        (16, 100): "-",
        (16, 200): "--",
        (16, 400): ":",
        (32, 100): "-",
        (32, 200): "--",
        (32, 400): ":",
        (48, 100): "-",
        (48, 200): "--",
        (48, 400): ":"
    }
    
    for (m, efc) in unique_configs:
        df_subset = df_k10[(df_k10["M"] == m) & (df_k10["ef_construction"] == efc)].sort_values("recall")
        
        if df_subset.empty:
            continue
        
        label = f"M={m}, efc={efc}"
        color = color_map.get((m, efc), "#000000")
        marker = marker_map.get((m, efc), "o")
        linestyle = linestyle_map.get((m, efc), "-")
        
        ax.scatter(
            df_subset["recall"].values,
            df_subset["p99_us"].values,
            label=label,
            color=color,
            marker=marker,
            s=70,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5
        )
        ax.plot(
            df_subset["recall"].values,
            df_subset["p99_us"].values,
            color=color,
            linestyle=linestyle,
            alpha=0.7,
            linewidth=2
        )
        
        for idx, (_, row) in enumerate(df_subset.iterrows()):
            if idx % 2 == 0 or idx == len(df_subset) - 1:
                ax.annotate(
                    str(int(row["ef_search"])),
                    (row["recall"], row["p99_us"]),
                    fontsize=7,
                    alpha=0.9,
                    xytext=(6, 6),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor="none")
                )
    
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("P99 Latency (μs)")
    ax.set_yscale("log")
    ax.set_title("Recall vs P99 Latency (Lower and more right is better)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_ef_search_vs_recall(df: pd.DataFrame, output_path: str) -> None:
    """Plot ef_search vs recall for different M values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_k10 = df[df["K"] == 10].copy()
    df_k10 = df_k10[df_k10["ef_construction"] == df_k10["ef_construction"].mode()[0]]
    
    if df_k10.empty:
        print("No data for K=10, skipping ef_search_vs_recall")
        return
    
    m_values = sorted(df_k10["M"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    for i, m in enumerate(m_values):
        df_m = df_k10[df_k10["M"] == m].sort_values("ef_search")
        
        if df_m.empty:
            continue
        
        ax.plot(
            df_m["ef_search"].values,
            df_m["recall"].values,
            label=f"M={m}",
            color=colors[i],
            marker="o",
            markersize=5
        )
    
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Recall@10")
    ax.set_xscale("log")
    ax.set_title("ef_search vs Recall@10")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_ef_search_vs_latency(df: pd.DataFrame, output_path: str) -> None:
    """Plot ef_search vs latency with p50-p99 range."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_k10 = df[df["K"] == 10].copy()
    df_k10 = df_k10[df_k10["ef_construction"] == df_k10["ef_construction"].mode()[0]]
    
    if df_k10.empty:
        print("No data for K=10, skipping ef_search_vs_latency")
        return
    
    m_values = sorted(df_k10["M"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    for i, m in enumerate(m_values):
        df_m = df_k10[df_k10["M"] == m].sort_values("ef_search")
        
        if df_m.empty:
            continue
        
        ax.plot(
            df_m["ef_search"].values,
            df_m["mean_us"].values,
            label=f"M={m}",
            color=colors[i],
            marker="o",
            markersize=5
        )
        
        ax.fill_between(
            df_m["ef_search"].values,
            df_m["p50_us"].values,
            df_m["p99_us"].values,
            color=colors[i],
            alpha=0.2
        )
    
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Latency (μs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("ef_search vs Latency (shaded: p50 to p99)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_thread_scalability(df: pd.DataFrame, output_path: str) -> None:
    """Plot QPS vs thread count."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    df_k10 = df[df["K"] == 10].copy()
    df_k10 = df_k10[df_k10["M"] == 16]
    df_k10 = df_k10[df_k10["ef_construction"] == df_k10["ef_construction"].mode()[0]]
    
    if df_k10.empty:
        print("No data for K=10, M=16, skipping thread_scalability")
        return
    
    ef_values = sorted(df_k10["ef_search"].unique())[:5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ef_values)))
    
    thread_cols = [c for c in df.columns if c.startswith("qps_") and c.endswith("t")]
    thread_counts = [int(c.replace("qps_", "").replace("t", "")) for c in thread_cols]
    
    ax2 = ax1.twinx()
    
    for i, ef in enumerate(ef_values):
        df_ef = df_k10[df_k10["ef_search"] == ef]
        
        if df_ef.empty:
            continue
        
        qps_values = []
        for tc in sorted(thread_counts):
            col = f"qps_{tc}t"
            if col in df_ef.columns:
                qps_values.append(df_ef[col].values[0])
            else:
                qps_values.append(np.nan)
        
        valid_tc = [tc for tc, qps in zip(sorted(thread_counts), qps_values) if not np.isnan(qps)]
        valid_qps = [qps for qps in qps_values if not np.isnan(qps)]
        
        if valid_qps:
            ax1.plot(
                valid_tc,
                valid_qps,
                label=f"ef_search={ef}",
                color=colors[i],
                marker="o",
                markersize=5
            )
            
            if valid_qps[0] > 0:
                speedup = [q / valid_qps[0] for q in valid_qps]
                ax2.plot(
                    valid_tc,
                    speedup,
                    color=colors[i],
                    linestyle="--",
                    alpha=0.5
                )
    
    if valid_qps:
        ideal_tc = [1] + [tc for tc in sorted(thread_counts) if tc <= max(valid_tc)]
        ideal_speedup = ideal_tc
        ax2.plot(
            ideal_tc,
            ideal_speedup,
            "k--",
            alpha=0.3,
            label="Ideal scaling"
        )
    
    ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("QPS")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_title("Thread Scalability (solid: QPS, dashed: speedup)")
    ax1.legend(loc="upper left")
    ax2.set_ylabel("Speedup")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_memory_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Plot memory usage comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    idx_cols = ["M", "ef_construction"]
    memory_df = df[idx_cols + ["index_memory_mb"]].drop_duplicates()
    
    if memory_df.empty:
        print("No memory data, skipping memory_comparison")
        return
    
    pivot = memory_df.pivot_table(
        index="M",
        columns="ef_construction",
        values="index_memory_mb",
        aggfunc="mean"
    )
    
    pivot.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_xlabel("M")
    ax.set_ylabel("Index Memory (MB)")
    ax.set_title("Index Memory by M and ef_construction")
    ax.legend(title="ef_construction")
    ax.grid(True, alpha=0.3, axis="y")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_build_time_comparison(df: pd.DataFrame, output_path: str) -> None:
    """Plot build time comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    idx_cols = ["M", "ef_construction"]
    time_df = df[idx_cols + ["build_time_sec"]].drop_duplicates()
    
    if time_df.empty:
        print("No build time data, skipping build_time_comparison")
        return
    
    pivot = time_df.pivot_table(
        index="M",
        columns="ef_construction",
        values="build_time_sec",
        aggfunc="mean"
    )
    
    pivot.plot(kind="bar", ax=ax, width=0.8)
    
    ax.set_xlabel("M")
    ax.set_ylabel("Build Time (seconds)")
    ax.set_title("Index Build Time by M and ef_construction")
    ax.legend(title="ef_construction")
    ax.grid(True, alpha=0.3, axis="y")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def generate_plots(summary_csv: str, output_dir: str) -> None:
    setup_style()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {summary_csv}...")
    df = pd.read_csv(summary_csv)
    
    print(f"Loaded {len(df)} rows")
    
    plot_recall_vs_qps(df, os.path.join(output_dir, "recall_vs_qps.png"))
    plot_recall_vs_latency_p99(df, os.path.join(output_dir, "recall_vs_latency_p99.png"))
    plot_ef_search_vs_recall(df, os.path.join(output_dir, "ef_search_vs_recall.png"))
    plot_ef_search_vs_latency(df, os.path.join(output_dir, "ef_search_vs_latency.png"))
    plot_thread_scalability(df, os.path.join(output_dir, "thread_scalability.png"))
    plot_memory_comparison(df, os.path.join(output_dir, "memory_comparison.png"))
    plot_build_time_comparison(df, os.path.join(output_dir, "build_time_comparison.png"))
    
    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument("--summary_csv", required=True, help="Path to summary.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    generate_plots(args.summary_csv, args.output_dir)


if __name__ == "__main__":
    main()
