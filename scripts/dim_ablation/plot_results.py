#!/usr/bin/env python3
"""
Visualize dimensionality ablation experiment results.

Generates plots that isolate the effect of vector dimensionality on HNSW
performance by comparing zero-padded SIFT1M variants (128, 256, 512, 1024 dim).

Plots produced:
1. qps_vs_dim.png          - QPS scaling with dimension
2. latency_vs_dim.png      - Latency scaling with dimension
3. recall_vs_dim.png       - Recall stability (should be flat for zero-padding)
4. build_vs_dim.png        - Build time / throughput scaling
5. memory_vs_dim.png       - Memory scaling
6. scaling_analysis.png    - Observed vs theoretical scaling factor
7. pca_explained_var.png   - PCA cumulative explained variance (if data provided)
8. thread_scalability.png  - Thread scalability at different dimensions
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_style():
    """Setup matplotlib style."""
    plt.style.use("seaborn-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


DIM_COLORS = {
    128: "#1f77b4",
    256: "#ff7f0e",
    512: "#2ca02c",
    1024: "#d62728",
}


def get_color(dim: int) -> str:
    return DIM_COLORS.get(dim, "#7f7f7f")


# ---------------------------------------------------------------------------
# Plot 1: QPS vs Dimension
# ---------------------------------------------------------------------------

def plot_qps_vs_dim(df: pd.DataFrame, output_path: str) -> None:
    """QPS (single-thread and multi-thread) vs dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_k10 = df[df["K"] == 10].copy()
    if df_k10.empty:
        print("  No K=10 data, skipping qps_vs_dim")
        return

    dims = sorted(df_k10["dim"].unique())
    m_values = sorted(df_k10["M"].unique())

    # Left: single-thread QPS at different ef_search
    ax = axes[0]
    for m in m_values:
        for ef in [50, 100, 200]:
            sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)]
            if sub.empty:
                continue
            sub = sub.sort_values("dim")
            label = f"M={m}, ef={ef}"
            marker = "o" if m <= 16 else ("s" if m <= 32 else "^")
            ax.plot(sub["dim"].values, sub["qps_1t"].values, marker=marker,
                    label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("QPS (1 thread)")
    ax.set_title("Single-Thread QPS vs Dimension (K=10)")
    ax.set_xticks(dims)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: multi-thread QPS (32t) at ef=100
    ax = axes[1]
    qps_cols = ["qps_1t", "qps_4t", "qps_8t", "qps_16t", "qps_32t", "qps_64t"]
    qps_cols = [c for c in qps_cols if c in df_k10.columns]
    for m in m_values:
        sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == 100)]
        if sub.empty:
            continue
        sub = sub.sort_values("dim")
        for col in ["qps_1t", "qps_16t", "qps_32t"]:
            if col not in sub.columns:
                continue
            threads = col.replace("qps_", "").replace("t", "")
            linestyle = "-" if col == "qps_1t" else ("--" if col == "qps_16t" else ":")
            ax.plot(sub["dim"].values, sub[col].values, marker="o", linewidth=2,
                    label=f"M={m}, {threads}t", linestyle=linestyle)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("QPS")
    ax.set_title("Multi-Thread QPS vs Dimension (ef=100, K=10)")
    ax.set_xticks(dims)
    ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: Latency vs Dimension
# ---------------------------------------------------------------------------

def plot_latency_vs_dim(df: pd.DataFrame, output_path: str) -> None:
    """Mean and P99 latency vs dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_k10 = df[df["K"] == 10].copy()
    if df_k10.empty:
        return

    dims = sorted(df_k10["dim"].unique())
    m_values = sorted(df_k10["M"].unique())

    for ax, ef in zip(axes, [100, 200]):
        for m in m_values:
            sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)]
            if sub.empty:
                continue
            sub = sub.sort_values("dim")
            marker = "o" if m <= 16 else ("s" if m <= 32 else "^")
            ax.plot(sub["dim"].values, sub["mean_us"].values, marker=marker,
                    linewidth=2, label=f"M={m} mean")
            ax.fill_between(sub["dim"].values, sub["p50_us"].values,
                            sub["p99_us"].values, alpha=0.15)

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Latency (μs)")
        ax.set_title(f"Latency vs Dimension (ef={ef}, K=10)\n(shaded: p50–p99)")
        ax.set_xticks(dims)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: Recall vs Dimension
# ---------------------------------------------------------------------------

def plot_recall_vs_dim(df: pd.DataFrame, output_path: str) -> None:
    """Recall vs dimension — should be flat for zero-padding."""
    fig, ax = plt.subplots(figsize=(12, 6))

    df_k10 = df[df["K"] == 10].copy()
    if df_k10.empty:
        return

    dims = sorted(df_k10["dim"].unique())
    m_values = sorted(df_k10["M"].unique())

    for m in m_values:
        for ef in sorted(df_k10["ef_search"].unique()):
            sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)]
            if sub.empty:
                continue
            sub = sub.sort_values("dim")
            marker = "o" if m <= 16 else ("s" if m <= 32 else "^")
            ax.plot(sub["dim"].values, sub["recall"].values, marker=marker,
                    label=f"M={m}, ef={ef}", linewidth=2, markersize=6)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall vs Dimension (zero-padding, should be constant)")
    ax.set_xticks(dims)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))
    ax.legend(fontsize=8, ncol=3, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: Build Time vs Dimension
# ---------------------------------------------------------------------------

def plot_build_vs_dim(df: pd.DataFrame, output_path: str) -> None:
    """Build time and throughput vs dimension."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    build_df = df[["dim", "M", "ef_construction", "build_time_sec",
                    "build_throughput_pts_per_sec"]].drop_duplicates()
    dims = sorted(build_df["dim"].unique())
    m_values = sorted(build_df["M"].unique())

    # Left: build time
    ax = axes[0]
    for m in m_values:
        sub = build_df[build_df["M"] == m].sort_values("dim")
        if sub.empty:
            continue
        ax.plot(sub["dim"].values, sub["build_time_sec"].values, marker="o",
                linewidth=2, label=f"M={m}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Build Time (sec)")
    ax.set_title("Index Build Time vs Dimension")
    ax.set_xticks(dims)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: build throughput
    ax = axes[1]
    for m in m_values:
        sub = build_df[build_df["M"] == m].sort_values("dim")
        if sub.empty:
            continue
        ax.plot(sub["dim"].values, sub["build_throughput_pts_per_sec"].values,
                marker="o", linewidth=2, label=f"M={m}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Build Throughput (pts/sec)")
    ax.set_title("Build Throughput vs Dimension")
    ax.set_xticks(dims)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 5: Memory vs Dimension
# ---------------------------------------------------------------------------

def plot_memory_vs_dim(df: pd.DataFrame, output_path: str) -> None:
    """Index memory vs dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))

    mem_df = df[["dim", "M", "ef_construction", "index_memory_mb",
                  "data_memory_mb"]].drop_duplicates()
    dims = sorted(mem_df["dim"].unique())
    m_values = sorted(mem_df["M"].unique())

    for m in m_values:
        sub = mem_df[mem_df["M"] == m].sort_values("dim")
        if sub.empty:
            continue
        ax.plot(sub["dim"].values, sub["index_memory_mb"].values, marker="o",
                linewidth=2, label=f"M={m} (index)")
        ax.plot(sub["dim"].values, sub["data_memory_mb"].values, marker="x",
                linewidth=1.5, linestyle="--", label=f"M={m} (data only)", alpha=0.6)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage vs Dimension")
    ax.set_xticks(dims)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 6: Scaling Analysis
# ---------------------------------------------------------------------------

def plot_scaling_analysis(df: pd.DataFrame, output_path: str) -> None:
    """Observed vs theoretical scaling factor (relative to dim=128)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_k10 = df[df["K"] == 10].copy()
    if df_k10.empty:
        return

    base_dim = df_k10["dim"].min()
    dims = sorted(df_k10["dim"].unique())
    m_values = sorted(df_k10["M"].unique())

    # Left: QPS scaling factor
    ax = axes[0]
    theoretical_factors = [d / base_dim for d in dims]
    ax.plot(dims, theoretical_factors, "k--", linewidth=2, label="Theoretical (dim/128)",
            alpha=0.5, zorder=10)

    for m in m_values:
        for ef in [50, 100, 200]:
            base_row = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)
                              & (df_k10["dim"] == base_dim)]
            if base_row.empty:
                continue
            base_qps = base_row["qps_1t"].values[0]

            sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)].sort_values("dim")
            if len(sub) < 2:
                continue
            observed_factor = base_qps / sub["qps_1t"].values
            ax.plot(sub["dim"].values, observed_factor, marker="o", linewidth=2,
                    label=f"M={m}, ef={ef}", markersize=6)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("QPS Slowdown Factor (QPS_128 / QPS_dim)")
    ax.set_title("QPS Scaling: Observed vs Theoretical")
    ax.set_xticks(dims)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Latency scaling factor
    ax = axes[1]
    ax.plot(dims, theoretical_factors, "k--", linewidth=2, label="Theoretical (dim/128)",
            alpha=0.5, zorder=10)

    for m in m_values:
        for ef in [50, 100, 200]:
            base_row = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)
                              & (df_k10["dim"] == base_dim)]
            if base_row.empty:
                continue
            base_lat = base_row["mean_us"].values[0]

            sub = df_k10[(df_k10["M"] == m) & (df_k10["ef_search"] == ef)].sort_values("dim")
            if len(sub) < 2:
                continue
            observed_factor = sub["mean_us"].values / base_lat
            ax.plot(sub["dim"].values, observed_factor, marker="o", linewidth=2,
                    label=f"M={m}, ef={ef}", markersize=6)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Latency Increase Factor (lat_dim / lat_128)")
    ax.set_title("Latency Scaling: Observed vs Theoretical")
    ax.set_xticks(dims)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 7: PCA Explained Variance
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(pca_data: dict, output_path: str) -> None:
    """Cumulative explained variance for SIFT1M and HotpotQA."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: cumulative explained variance
    ax = axes[0]
    colors = {"sift1m": "#1f77b4", "hotpotqa": "#d62728"}
    labels = {"sift1m": "SIFT1M (128-dim)", "hotpotqa": "HotpotQA (1024-dim)"}

    for key in ["sift1m", "hotpotqa"]:
        if key not in pca_data:
            continue
        cum = np.array(pca_data[key]["cumulative_variance_ratio"])
        components = np.arange(1, len(cum) + 1)
        ax.plot(components, cum, label=labels[key], color=colors[key], linewidth=2)

        # Mark thresholds
        for threshold in [0.90, 0.95, 0.99]:
            idx = np.searchsorted(cum, threshold)
            if idx < len(cum):
                ax.axhline(y=threshold, color="gray", linestyle=":", alpha=0.3)
                ax.plot(idx + 1, cum[idx], "x", color=colors[key], markersize=8)
                ax.annotate(f"{idx + 1}d", (idx + 1, cum[idx]),
                            fontsize=8, xytext=(5, -10), textcoords="offset points")

    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Intrinsic Dimensionality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: eigenvalue spectrum (log scale)
    ax = axes[1]
    for key in ["sift1m", "hotpotqa"]:
        if key not in pca_data:
            continue
        eigenvals = np.array(pca_data[key]["explained_variance_ratio"])
        components = np.arange(1, len(eigenvals) + 1)
        ax.semilogy(components, eigenvals, label=labels[key], color=colors[key],
                     linewidth=2)

    ax.set_xlabel("Component Index")
    ax.set_ylabel("Explained Variance Ratio (log)")
    ax.set_title("PCA Eigenvalue Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Plot 8: Thread Scalability at Different Dimensions
# ---------------------------------------------------------------------------

def plot_thread_scalability(df: pd.DataFrame, output_path: str) -> None:
    """Thread scalability comparison across dimensions."""
    fig, ax = plt.subplots(figsize=(12, 7))

    df_k10 = df[(df["K"] == 10) & (df["ef_search"] == 100)].copy()
    if df_k10.empty:
        return

    # Pick one M value
    m_val = 16 if 16 in df_k10["M"].values else df_k10["M"].mode()[0]
    df_k10 = df_k10[df_k10["M"] == m_val]

    thread_cols = sorted(
        [c for c in df.columns if c.startswith("qps_") and c.endswith("t")],
        key=lambda c: int(c.replace("qps_", "").replace("t", ""))
    )
    thread_counts = [int(c.replace("qps_", "").replace("t", "")) for c in thread_cols]

    dims = sorted(df_k10["dim"].unique())

    for dim in dims:
        sub = df_k10[df_k10["dim"] == dim]
        if sub.empty:
            continue
        qps_vals = [sub[c].values[0] for c in thread_cols if c in sub.columns]
        valid_tc = thread_counts[:len(qps_vals)]
        color = get_color(dim)
        ax.plot(valid_tc, qps_vals, marker="o", linewidth=2, color=color,
                label=f"dim={dim}", markersize=6)

    # Ideal scaling reference
    if dims:
        sub = df_k10[df_k10["dim"] == dims[0]]
        if not sub.empty and "qps_1t" in sub.columns:
            base_qps = sub["qps_1t"].values[0]
            ideal_qps = [base_qps * tc for tc in thread_counts]
            ax.plot(thread_counts, ideal_qps, "k--", alpha=0.3, label="Ideal (linear)")

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("QPS")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title(f"Thread Scalability at Different Dimensions (M={m_val}, ef=100, K=10)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Write analysis summary
# ---------------------------------------------------------------------------

def write_scaling_summary(df: pd.DataFrame, output_path: str) -> None:
    """Write a JSON summary of observed scaling factors."""
    df_k10 = df[df["K"] == 10].copy()
    if df_k10.empty:
        return

    base_dim = int(df_k10["dim"].min())
    dims = sorted(df_k10["dim"].unique())
    summary = {"base_dim": base_dim, "configs": []}

    for _, row in df_k10.iterrows():
        base_row = df_k10[
            (df_k10["M"] == row["M"]) &
            (df_k10["ef_search"] == row["ef_search"]) &
            (df_k10["dim"] == base_dim)
        ]
        if base_row.empty:
            continue
        base_qps = base_row["qps_1t"].values[0]
        base_lat = base_row["mean_us"].values[0]

        entry = {
            "dim": int(row["dim"]),
            "M": int(row["M"]),
            "ef_construction": int(row["ef_construction"]),
            "ef_search": int(row["ef_search"]),
            "recall": round(float(row["recall"]), 6),
            "qps_1t": round(float(row["qps_1t"]), 1),
            "mean_us": round(float(row["mean_us"]), 1),
            "theoretical_factor": round(float(row["dim"]) / base_dim, 2),
            "observed_qps_factor": round(base_qps / float(row["qps_1t"]), 2) if row["qps_1t"] > 0 else None,
            "observed_latency_factor": round(float(row["mean_us"]) / base_lat, 2) if base_lat > 0 else None,
        }
        summary["configs"].append(entry)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dimensionality ablation experiment plots"
    )
    parser.add_argument("--summary_csv", required=True, help="Path to summary.csv")
    parser.add_argument("--pca_json", default=None,
                        help="Path to pca_analysis.json (optional)")
    parser.add_argument("--output_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.summary_csv} ...")
    df = pd.read_csv(args.summary_csv)
    print(f"  {len(df)} rows, dims: {sorted(df['dim'].unique())}")

    plot_qps_vs_dim(df, os.path.join(args.output_dir, "qps_vs_dim.png"))
    plot_latency_vs_dim(df, os.path.join(args.output_dir, "latency_vs_dim.png"))
    plot_recall_vs_dim(df, os.path.join(args.output_dir, "recall_vs_dim.png"))
    plot_build_vs_dim(df, os.path.join(args.output_dir, "build_vs_dim.png"))
    plot_memory_vs_dim(df, os.path.join(args.output_dir, "memory_vs_dim.png"))
    plot_scaling_analysis(df, os.path.join(args.output_dir, "scaling_analysis.png"))
    plot_thread_scalability(df, os.path.join(args.output_dir, "thread_scalability.png"))

    # PCA plot (if analysis available)
    if args.pca_json and os.path.exists(args.pca_json):
        with open(args.pca_json) as f:
            pca_data = json.load(f)
        plot_pca_explained_variance(pca_data, os.path.join(args.output_dir, "pca_explained_var.png"))

    # Scaling summary JSON
    write_scaling_summary(df, os.path.join(args.output_dir, "..", "scaling_analysis.json"))

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
