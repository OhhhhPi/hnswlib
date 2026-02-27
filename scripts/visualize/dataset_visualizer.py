#!/usr/bin/env python3
"""
数据集整体可视化

将整个数据集的向量投影到 2D/3D 空间，以交互方式展示数据分布。
支持 HotpotQA 和 SIFT1M，可叠加查询点及其 Ground Truth 近邻。

Usage:
    python scripts/visualize/dataset_visualizer.py --dataset hotpotqa --data_dir data
    python scripts/visualize/dataset_visualizer.py --dataset sift1m --data_dir data
    python scripts/visualize/dataset_visualizer.py --dataset hotpotqa --max_base 50000 --max_queries 5000 --dim 3

Output:
    results/dataset_viz_<timestamp>/dataset.html  (自包含、可交互)

Dependencies:
    pip install plotly numpy scikit-learn tqdm
    Optional: pip install umap-learn
"""

import argparse
import struct
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    sys.exit("Missing: plotly. pip install plotly")
try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
try:
    import umap as umap_mod
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x


# ============================================================================
#  Data I/O
# ============================================================================

def read_fvecs(path):
    buf = open(path, "rb").read()
    off, vecs = 0, []
    while off < len(buf):
        d = struct.unpack_from("i", buf, off)[0]
        off += 4
        vecs.append(np.frombuffer(buf, np.float32, d, off).copy())
        off += d * 4
    return np.vstack(vecs)


def read_ivecs(path):
    buf = open(path, "rb").read()
    off, vecs = 0, []
    while off < len(buf):
        d = struct.unpack_from("i", buf, off)[0]
        off += 4
        vecs.append(np.frombuffer(buf, np.int32, d, off).copy())
        off += d * 4
    return np.vstack(vecs)


def load_dataset(dataset, data_dir):
    dd = Path(data_dir)
    if dataset == "hotpotqa":
        hd = dd / "hotpotqa"
        base = np.load(hd / "corpus_embeddings.npy")
        queries = np.load(hd / "query_embeddings.npy")
        gtp = hd / "knn_gt_indices.npy"
        gt = np.load(gtp) if gtp.exists() else None
        if gt is None and (hd / "ground_truth.ivecs").exists():
            buf = open(hd / "ground_truth.ivecs", "rb").read()
            off, rows = 0, []
            while off < len(buf):
                d = struct.unpack_from("i", buf, off)[0]
                off += 4
                rows.append(np.frombuffer(buf, np.int32, d, off).copy())
                off += d * 4
            gt = np.vstack(rows) if rows else None
    elif dataset == "locomo":
        ld = dd / "locomo"
        base = np.load(ld / "corpus_embeddings.npy")
        queries = np.load(ld / "query_embeddings.npy")
        gtp = ld / "knn_gt_indices.npy"
        gt = np.load(gtp) if gtp.exists() else None
        if gt is None and (ld / "ground_truth.ivecs").exists():
            buf = open(ld / "ground_truth.ivecs", "rb").read()
            off, rows = 0, []
            while off < len(buf):
                d = struct.unpack_from("i", buf, off)[0]
                off += 4
                rows.append(np.frombuffer(buf, np.int32, d, off).copy())
                off += d * 4
            gt = np.vstack(rows) if rows else None
    elif dataset == "sift1m":
        base = read_fvecs(dd / "sift1m" / "sift1m_base.fvecs")
        queries = read_fvecs(dd / "sift1m" / "sift1m_query.fvecs")
        gt = read_ivecs(dd / "sift1m" / "sift1m_groundtruth.ivecs")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"[data] base={base.shape}  queries={queries.shape}  gt={gt.shape if gt is not None else 'N/A'}")
    return base, queries, gt


# ============================================================================
#  Projection
# ============================================================================

def project(X, n_dims, method="auto"):
    if method == "auto":
        method = "umap" if HAS_UMAP else ("pca" if PCA else "random")
    print(f"[proj] {len(X)} vectors → {n_dims}D via {method} ...")
    if method == "umap" and HAS_UMAP:
        reducer = umap_mod.UMAP(n_components=n_dims, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
        return reducer.fit_transform(X).astype(np.float32)
    if method == "pca" and PCA:
        pca = PCA(n_dims, random_state=42)
        out = pca.fit_transform(X).astype(np.float32)
        print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        return out
    rng = np.random.RandomState(42)
    P = rng.randn(X.shape[1], n_dims).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    return (X @ P).astype(np.float32)


# ============================================================================
#  Visualization
# ============================================================================

def build_fig_2d(base_2d, query_2d, gt_indices_per_query, base_sample_idx, query_sample_idx,
                 n_base_total, n_query_total, dataset):
    """2D scatter: base (gray) + queries (blue) + optional GT highlight."""
    fig = go.Figure()

    # Base vectors
    fig.add_trace(go.Scatter(
        x=base_2d[:, 0], y=base_2d[:, 1],
        mode="markers", name=f"文档 ({len(base_2d):,} / {n_base_total:,})",
        marker=dict(size=0.8, color="#9e9e9e", opacity=0.5),
        text=[f"doc {i}" for i in base_sample_idx], hoverinfo="text"))

    # Query vectors
    fig.add_trace(go.Scatter(
        x=query_2d[:, 0], y=query_2d[:, 1],
        mode="markers", name=f"查询 ({len(query_2d):,} / {n_query_total:,})",
        marker=dict(size=1.2, color="#2196f3"),
        text=[f"query {i}" for i in query_sample_idx], hoverinfo="text"))

    # GT neighbors for first 5 queries (optional highlight)
    if gt_indices_per_query is not None and len(gt_indices_per_query) > 0:
        base_idx_set = set(base_sample_idx)
        for qi in range(min(5, len(query_2d))):
            q_orig = query_sample_idx[qi]
            gt_ids = gt_indices_per_query.get(q_orig, [])
            hit = [i for i in gt_ids if i in base_idx_set]
            if not hit:
                continue
            pos = np.array([base_sample_idx.index(i) for i in hit])
            pts = base_2d[pos]
            fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1],
                mode="markers", name=f"Q{q_orig} GT ({len(hit)})",
                marker=dict(size=1.2, color="#ff9800", opacity=0.9),
                hoverinfo="skip"))

    fig.update_layout(
        title=f"{dataset} 数据集分布 (2D)",
        xaxis_title="D1", yaxis_title="D2",
        height=700, showlegend=True, legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=20, t=50, b=60))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_fig_3d(base_3d, query_3d, gt_indices_per_query, base_sample_idx, query_sample_idx,
                 n_base_total, n_query_total, dataset):
    """3D scatter: base (gray) + queries (blue) + optional GT highlight."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=base_3d[:, 0], y=base_3d[:, 1], z=base_3d[:, 2],
        mode="markers", name=f"文档 ({len(base_3d):,} / {n_base_total:,})",
        marker=dict(size=0.8, color="#9e9e9e", opacity=0.5),
        text=[f"doc {i}" for i in base_sample_idx], hoverinfo="text"))

    fig.add_trace(go.Scatter3d(
        x=query_3d[:, 0], y=query_3d[:, 1], z=query_3d[:, 2],
        mode="markers", name=f"查询 ({len(query_3d):,} / {n_query_total:,})",
        marker=dict(size=1.2, color="#2196f3"),
        text=[f"query {i}" for i in query_sample_idx], hoverinfo="text"))

    if gt_indices_per_query is not None and len(gt_indices_per_query) > 0:
        base_idx_set = set(base_sample_idx)
        for qi in range(min(5, len(query_3d))):
            q_orig = query_sample_idx[qi]
            gt_ids = gt_indices_per_query.get(q_orig, [])
            hit = [i for i in gt_ids if i in base_idx_set]
            if not hit:
                continue
            pos = np.array([base_sample_idx.index(i) for i in hit])
            pts = base_3d[pos]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers", name=f"Q{q_orig} GT ({len(hit)})",
                marker=dict(size=1.2, color="#ff9800", opacity=0.9),
                hoverinfo="skip"))

    fig.update_layout(
        title=f"{dataset} 数据集分布 (3D)",
        scene=dict(xaxis_title="D1", yaxis_title="D2", zaxis_title="D3", aspectmode="data"),
        height=750, showlegend=True, legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=50, b=0))
    return fig


def build_html(fig, params, path):
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{params.get('dataset','')} 数据集可视化</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font-family:-apple-system,sans-serif;margin:20px;background:#f5f7fa}}
h1{{color:#1a1a2e;border-bottom:2px solid #4361ee;padding-bottom:10px}}
.card{{background:#fff;border-radius:12px;padding:20px;margin:15px 0;box-shadow:0 2px 10px rgba(0,0,0,.08)}}
.params{{display:flex;flex-wrap:wrap;gap:10px;font-size:14px}}
.pm{{background:#e8eaf6;padding:6px 12px;border-radius:6px}}
</style></head><body><div class="card">
<h1>{params.get('dataset','')} 数据集可视化</h1>
<div class="params">
"""
    for k, v in params.items():
        html += f'<span class="pm"><b>{k}</b>: {v}</span>'
    html += f"""
</div></div>
<div class="card">
<p style="color:#666;font-size:14px">拖动旋转 | 滚轮缩放 | 点击图例切换图层</p>
{pio.to_html(fig, full_html=False, include_plotlyjs=False)}
</div></body></html>"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    print(f"✓ Output → {path}")


# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="数据集整体可视化")
    ap.add_argument("--dataset", choices=["hotpotqa", "locomo", "sift1m"], default="hotpotqa")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--dim", type=int, choices=[2, 3], default=2, help="投影维度")
    ap.add_argument("--max_base", type=int, default=50000, help="基向量最大采样数")
    ap.add_argument("--max_queries", type=int, default=5000, help="查询最大采样数")
    ap.add_argument("--projection", choices=["auto", "pca", "umap"], default="auto")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--no_gt", action="store_true", help="不叠加 GT 高亮")
    args = ap.parse_args()

    base, queries, gt = load_dataset(args.dataset, args.data_dir)
    n_base, n_query = len(base), len(queries)

    # Sample
    rng = np.random.RandomState(42)
    n_base_use = min(n_base, args.max_base)
    n_query_use = min(n_query, args.max_queries)
    base_idx = rng.choice(n_base, n_base_use, replace=False) if n_base_use < n_base else np.arange(n_base)
    query_idx = rng.choice(n_query, n_query_use, replace=False) if n_query_use < n_query else np.arange(n_query)
    base_samp = base[base_idx]
    query_samp = queries[query_idx]

    # Project jointly for consistent coordinates
    combined = np.vstack([base_samp, query_samp])
    combined_proj = project(combined, args.dim, args.projection)
    base_proj = combined_proj[:n_base_use]
    query_proj = combined_proj[n_base_use:]

    # GT indices per query (for highlight)
    gt_per_query = None
    if not args.no_gt and gt is not None:
        gt_per_query = {}
        for qid in query_idx:
            if qid < gt.shape[0]:
                row = np.asarray(gt[qid]).flatten()
                gt_per_query[int(qid)] = row[: min(10, len(row))].tolist()

    # Build figure
    if args.dim == 2:
        fig = build_fig_2d(
            base_proj, query_proj, gt_per_query,
            base_idx.tolist(), query_idx.tolist(),
            n_base, n_query, args.dataset)
    else:
        fig = build_fig_3d(
            base_proj, query_proj, gt_per_query,
            base_idx.tolist(), query_idx.tolist(),
            n_base, n_query, args.dataset)

    out_dir = args.output_dir or f"results/dataset_viz_{datetime.now():%Y%m%d_%H%M%S}"
    out_path = str(Path(out_dir) / "dataset.html")
    params = dict(
        dataset=args.dataset,
        dim=f"{args.dim}D",
        base_total=n_base,
        query_total=n_query,
        base_shown=n_base_use,
        query_shown=n_query_use,
        projection=args.projection,
    )
    build_html(fig, params, out_path)
    print(f"\n完成! 在浏览器中打开 {out_path}")


if __name__ == "__main__":
    main()
