#!/usr/bin/env python3
"""
HNSW Search Trace Visualizer

Interactive 3D visualization of HNSW search paths for low-recall queries.
Compares HNSW search with BFS minimum-hop paths to diagnose recall failures.
Generates statistical charts: distance decay, degree-hitrate heatmap, etc.

Usage:
    python scripts/visualize/search_trace_visualizer.py --dataset hotpotqa --data_dir data
    python scripts/visualize/search_trace_visualizer.py --dataset sift1m --data_dir data

Output:
    results/search_trace_<timestamp>/dashboard.html  (self-contained, interactive)

Dependencies:
    pip install plotly numpy scikit-learn tqdm hnswlib
    Optional: pip install umap-learn  (better 3D projection, fallback to PCA)
"""

import argparse
import struct
import heapq
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
_missing = []
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    _missing.append("plotly")
try:
    import hnswlib
except ImportError:
    _missing.append("hnswlib")
if _missing:
    sys.exit(f"Missing: {', '.join(_missing)}. pip install {' '.join(_missing)}")

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
try:
    import umap as umap_mod
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ============================================================================
#  Data I/O
# ============================================================================

def read_fvecs(path):
    buf = open(path, "rb").read()
    off, vecs = 0, []
    while off < len(buf):
        d = struct.unpack_from("i", buf, off)[0]; off += 4
        vecs.append(np.frombuffer(buf, np.float32, d, off).copy()); off += d * 4
    return np.vstack(vecs)


def read_ivecs(path):
    buf = open(path, "rb").read()
    off, vecs = 0, []
    while off < len(buf):
        d = struct.unpack_from("i", buf, off)[0]; off += 4
        vecs.append(np.frombuffer(buf, np.int32, d, off).copy()); off += d * 4
    return np.vstack(vecs)


def load_dataset(dataset, data_dir):
    dd = Path(data_dir)
    if dataset == "hotpotqa":
        hd = dd / "hotpotqa"
        base = np.load(hd / "corpus_embeddings.npy")
        queries = np.load(hd / "query_embeddings.npy")
        gtp = hd / "knn_gt_indices.npy"
        gt = np.load(gtp) if gtp.exists() else read_ivecs(hd / "ground_truth.ivecs")
        space = "ip"
    elif dataset == "locomo":
        ld = dd / "locomo"
        base = np.load(ld / "corpus_embeddings.npy")
        queries = np.load(ld / "query_embeddings.npy")
        gtp = ld / "knn_gt_indices.npy"
        gt = np.load(gtp) if gtp.exists() else read_ivecs(ld / "ground_truth.ivecs")
        space = "ip"
    elif dataset == "sift1m":
        base = read_fvecs(dd / "sift1m" / "sift1m_base.fvecs")
        queries = read_fvecs(dd / "sift1m" / "sift1m_query.fvecs")
        gt = read_ivecs(dd / "sift1m" / "sift1m_groundtruth.ivecs")
        space = "l2"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    print(f"[data] base={base.shape}  queries={queries.shape}  gt={gt.shape}  space={space}")
    return base, queries, gt, space

# ============================================================================
#  Distance helpers
# ============================================================================

def _dist_l2(a, b):
    d = a - b
    return float(d @ d)

def _dist_ip(a, b):
    return 1.0 - float(a @ b)

def get_dist(space):
    return _dist_l2 if space == "l2" else _dist_ip

# ============================================================================
#  HNSW Graph Extraction (from hnswlib __getstate__)
# ============================================================================

class HNSWGraph:
    """Parse the internal HNSW graph topology from a live hnswlib index."""

    def __init__(self, index):
        p = index.__getstate__()[0]

        self.n           = p["cur_element_count"]
        self.M           = p["M"]
        self.max_M0      = p["max_M0"]        # 2*M for level-0
        self.max_M       = p["max_M"]          # M for upper levels
        self.efc         = p["ef_construction"]
        self.enterpoint  = p["enterpoint_node"]
        self.max_level   = p["max_level"]
        self.sz_per_elem = p["size_data_per_element"]
        self.sz_link_up  = p["size_links_per_element"]

        el = np.asarray(p["element_levels"], dtype=np.int32)
        self.element_levels = el[: self.n]

        # label mapping --------------------------------------------------
        ek = np.asarray(p["label_lookup_external"], dtype=np.int64)
        iv = np.asarray(p["label_lookup_internal"], dtype=np.int32)
        self.i2e = np.zeros(self.n, dtype=np.int64)
        self.e2i = {}
        for e, i in zip(ek, iv):
            self.i2e[i] = e
            self.e2i[int(e)] = int(i)

        # level-0 neighbors -----------------------------------------------
        raw0 = np.frombuffer(p["data_level0"], dtype=np.uint8)
        self.adj0 = []
        self.deg0 = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            off = i * self.sz_per_elem
            cnt = int(np.frombuffer(raw0[off : off + 4], np.uint32)[0])
            cnt = min(cnt, self.max_M0)
            nbrs = np.frombuffer(raw0[off + 4 : off + 4 + cnt * 4], np.uint32).tolist()
            self.adj0.append(nbrs)
            self.deg0[i] = cnt

        # upper-level neighbors --------------------------------------------
        self.adj_up = {}  # (internal_id, level) -> [neighbors]
        if len(p["link_lists"]) > 0:
            rl = np.frombuffer(p["link_lists"], dtype=np.uint8)
            ptr = 0
            for i in range(self.n):
                lev = self.element_levels[i]
                if lev > 0:
                    for l in range(1, lev + 1):
                        cnt = int(np.frombuffer(rl[ptr : ptr + 4], np.uint32)[0])
                        cnt = min(cnt, self.max_M)
                        nbrs = np.frombuffer(rl[ptr + 4 : ptr + 4 + cnt * 4], np.uint32).tolist()
                        self.adj_up[(i, l)] = nbrs
                        ptr += self.sz_link_up

        print(f"[graph] n={self.n}  M={self.M}  max_level={self.max_level}  "
              f"entry={self.enterpoint}  avg_deg0={self.deg0.mean():.1f}")

    def nbrs(self, node, level=0):
        if level == 0:
            return self.adj0[node]
        return self.adj_up.get((node, level), [])

    def deg(self, node, level=0):
        return len(self.nbrs(node, level))

# ============================================================================
#  HNSW Search Tracer
# ============================================================================

class HNSWTracer:
    """Faithfully replicate searchKnn + searchBaseLayerST and record every step."""

    def __init__(self, graph, vectors, space):
        self.g = graph
        self.v = vectors
        self.dfn = get_dist(space)
        self.i2e = graph.i2e

    def _d(self, q, iid):
        return self.dfn(q, self.v[int(self.i2e[iid])])

    def trace(self, query, k, ef):
        g = self.g
        upper_path = []

        # --- upper-level greedy walk ---
        cur, cur_d = g.enterpoint, self._d(query, g.enterpoint)
        upper_path.append(dict(node=int(cur), dist=cur_d, level=g.max_level))
        for lev in range(g.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for nb in g.nbrs(cur, lev):
                    d = self._d(query, nb)
                    if d < cur_d:
                        cur_d, cur = d, nb
                        changed = True
                        upper_path.append(dict(node=int(cur), dist=cur_d, level=lev))

        ep_l0 = cur
        ep_d  = self._d(query, ep_l0)

        # --- base-layer beam search ---
        cand = [(ep_d, ep_l0)]           # min-heap (closest first)
        top  = [(-ep_d, ep_l0)]          # max-heap via negation
        lb   = ep_d
        visited = {ep_l0}
        vlog = [dict(order=0, node=int(ep_l0), dist=ep_d)]
        steps, hop, order = [], 0, 0

        while cand:
            cd, cn = heapq.heappop(cand)
            if cd > lb:
                break
            hop += 1
            nbrs = g.nbrs(cn, 0)
            st = dict(hop=hop, expanded=int(cn), exp_dist=cd, checked=[])
            for nb in nbrs:
                if nb in visited:
                    continue
                visited.add(nb)
                d = self._d(query, nb)
                order += 1
                added = False
                if len(top) < ef or d < lb:
                    heapq.heappush(cand, (d, nb))
                    heapq.heappush(top, (-d, nb))
                    added = True
                    if len(top) > ef:
                        heapq.heappop(top)
                    lb = -top[0][0]
                st["checked"].append(dict(node=int(nb), dist=d, added=added))
                vlog.append(dict(order=order, node=int(nb), dist=d))
            steps.append(st)

        res = []
        while top:
            nd, nid = heapq.heappop(top)
            res.append((nid, -nd))
        res.reverse()
        res = res[:k]

        return (
            [int(self.i2e[r[0]]) for r in res],
            dict(upper_path=upper_path, entry_l0=int(ep_l0),
                 steps=steps, vlog=vlog, visited=visited,
                 total_hops=hop, total_visited=len(visited)),
        )

# ============================================================================
#  BFS shortest path (minimum hop count)
# ============================================================================

def bfs_paths(graph, src, targets):
    """Unweighted BFS → hop-count shortest path from src to each target."""
    tgt = set(targets)
    found, vis, prev = {}, {src}, {src: None}
    q = deque([src])
    while q and len(found) < len(tgt):
        n = q.popleft()
        if n in tgt:
            p, c = [], n
            while c is not None:
                p.append(c); c = prev.get(c)
            found[n] = p[::-1]
        for nb in graph.nbrs(n, 0):
            if nb not in vis:
                vis.add(nb); prev[nb] = n; q.append(nb)
    return found


# ============================================================================
#  Low-recall query finder
# ============================================================================

def find_low_recall(index, queries, gt, k, ef, n_select=5):
    index.set_ef(ef)
    labels, _ = index.knn_query(queries, k=k, num_threads=1)
    recalls = np.array([
        len(set(gt[i, :k].tolist()) & set(labels[i].tolist())) / k
        for i in range(len(queries))
    ])
    order = np.argsort(recalls)
    sel = []
    for idx in order[:n_select]:
        sel.append(dict(query_idx=int(idx), recall=float(recalls[idx]),
                        hnsw_results=labels[idx].tolist(),
                        gt_results=gt[idx, :k].tolist()))
    print(f"[recall] avg recall@{k} (ef={ef}): {recalls.mean():.4f}")
    for s in sel:
        print(f"  Q#{s['query_idx']}: recall={s['recall']:.4f}")
    return sel, recalls

# ============================================================================
#  3-D projection
# ============================================================================

def project_3d(vectors, method="auto"):
    if method == "auto":
        method = "umap" if HAS_UMAP else ("pca" if PCA else "random")
    print(f"[proj] {len(vectors)} vectors → 3D via {method}")
    if method == "umap" and HAS_UMAP:
        return umap_mod.UMAP(3, n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=42).fit_transform(vectors).astype(np.float32)
    if method == "pca" and PCA:
        pca = PCA(3, random_state=42)
        c = pca.fit_transform(vectors).astype(np.float32)
        print(f"  explained var: {pca.explained_variance_ratio_.sum():.3f}")
        return c
    rng = np.random.RandomState(42)
    P = rng.randn(vectors.shape[1], 3).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    return (vectors @ P).astype(np.float32)

# ============================================================================
#  3-D search-trace figure  (Plotly)
# ============================================================================

def fig_3d_trace(qi_ext, q3d, base3d, graph, trace, bfs_paths, gt_ext, hnsw_ext,
                 recall, space, base_vecs, query_vec):
    """bfs_paths: {tgt_int: [path_list]} for missed GT targets (BFS minimum-hop path)."""
    fig = go.Figure()
    i2e, e2i, dfn = graph.i2e, graph.e2i, get_dist(space)

    gt_int   = {e2i[g] for g in gt_ext   if g in e2i}
    hnsw_int = {e2i[g] for g in hnsw_ext if g in e2i}
    found_gt  = gt_int & hnsw_int
    missed_gt = gt_int - hnsw_int
    visited   = trace["visited"]

    def c(iid):
        return base3d[int(i2e[iid])]

    # 1 — context: 1-hop neighborhood of visited (capped)
    ctx = set()
    for v in visited:
        ctx.update(graph.nbrs(v, 0))
    ctx -= visited | gt_int
    if len(ctx) > 2000:
        ctx = set(np.random.choice(list(ctx), 2000, replace=False).tolist())
    if ctx:
        cl = list(ctx)
        fig.add_trace(go.Scatter3d(
            x=[c(n)[0] for n in cl], y=[c(n)[1] for n in cl], z=[c(n)[2] for n in cl],
            mode="markers", name="邻域上下文",
            marker=dict(size=1.5, color="lightgray", opacity=0.25), hoverinfo="skip"))

    # 2 — visited (non-GT)
    vis_only = list(visited - gt_int)
    if vis_only:
        vd = [dfn(query_vec, base_vecs[int(i2e[n])]) for n in vis_only]
        fig.add_trace(go.Scatter3d(
            x=[c(n)[0] for n in vis_only], y=[c(n)[1] for n in vis_only],
            z=[c(n)[2] for n in vis_only],
            mode="markers", name=f"已访问 ({len(vis_only)})",
            marker=dict(size=3, color=vd, colorscale="Blues_r", opacity=0.55,
                        colorbar=dict(title="dist", len=0.4, x=1.02)),
            text=[f"ID:{int(i2e[n])} d={d:.4f} deg={graph.deg(n)}"
                  for n, d in zip(vis_only, vd)], hoverinfo="text"))

    # 3 — HNSW search path
    path_n = [s["expanded"] for s in trace["steps"]]
    if path_n:
        fig.add_trace(go.Scatter3d(
            x=[c(n)[0] for n in path_n], y=[c(n)[1] for n in path_n],
            z=[c(n)[2] for n in path_n],
            mode="lines+markers", name="HNSW 搜索路径",
            marker=dict(size=4, color=list(range(len(path_n))),
                        colorscale="Reds", showscale=False),
            line=dict(color="red", width=3),
            text=[f"Step {i+1} ID:{int(i2e[n])} d={s['exp_dist']:.4f}"
                  for i, (n, s) in enumerate(zip(path_n, trace["steps"]))],
            hoverinfo="text"))

    # 4 — BFS minimum-hop paths to missed GT
    pal = ["#2ecc71", "#27ae60", "#16a085", "#1abc9c", "#00b894"]
    ci = 0
    for tgt_int, path in bfs_paths.items():
        if tgt_int in missed_gt:
            hops = len(path) - 1
            fig.add_trace(go.Scatter3d(
                x=[c(n)[0] for n in path], y=[c(n)[1] for n in path], z=[c(n)[2] for n in path],
                mode="lines+markers",
                name=f"BFS→GT:{int(i2e[tgt_int])} ({hops}跳)",
                marker=dict(size=4, color=pal[ci % len(pal)]),
                line=dict(color=pal[ci % len(pal)], width=2, dash="dash"),
                text=[f"Hop {j} ID:{int(i2e[n])}" for j, n in enumerate(path)],
                hoverinfo="text"))
            ci += 1

    # 5 — GT found
    if found_gt:
        fg = list(found_gt)
        fig.add_trace(go.Scatter3d(
            x=[c(n)[0] for n in fg], y=[c(n)[1] for n in fg], z=[c(n)[2] for n in fg],
            mode="markers", name=f"GT 命中 ({len(fg)})",
            marker=dict(size=8, color="gold", symbol="diamond"),
            text=[f"GT命中 ID:{int(i2e[n])}" for n in fg], hoverinfo="text"))

    # 6 — GT missed
    if missed_gt:
        mg = list(missed_gt)
        fig.add_trace(go.Scatter3d(
            x=[c(n)[0] for n in mg], y=[c(n)[1] for n in mg], z=[c(n)[2] for n in mg],
            mode="markers", name=f"GT 未命中 ({len(mg)})",
            marker=dict(size=10, color="red", symbol="x"),
            text=[f"GT未命中 ID:{int(i2e[n])} d={dfn(query_vec, base_vecs[int(i2e[n])]):.4f}"
                  for n in mg], hoverinfo="text"))

    # 7 — entry point (level-0)
    ep = trace["entry_l0"]
    fig.add_trace(go.Scatter3d(
        x=[c(ep)[0]], y=[c(ep)[1]], z=[c(ep)[2]],
        mode="markers", name="入口点 (L0)",
        marker=dict(size=9, color="orange", symbol="square"),
        text=[f"Entry L0 ID:{int(i2e[ep])}"], hoverinfo="text"))

    # 8 — query
    fig.add_trace(go.Scatter3d(
        x=[q3d[0]], y=[q3d[1]], z=[q3d[2]],
        mode="markers", name="查询点",
        marker=dict(size=12, color="blue", symbol="diamond"),
        text=[f"Query #{qi_ext}"], hoverinfo="text"))

    fig.update_layout(
        title=f"Query #{qi_ext} — Recall@K = {recall:.2%}",
        scene=dict(xaxis_title="D1", yaxis_title="D2", zaxis_title="D3",
                   aspectmode="data"),
        height=700, legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ============================================================================
#  Statistical figures
# ============================================================================

def fig_distance_decay(traces_info):
    """Running-min distance over hops for each traced query."""
    fig = go.Figure()
    for tr, qext in traces_info:
        steps = tr["steps"]
        if not steps:
            continue
        hops = [s["hop"] for s in steps]
        running = []
        m = float("inf")
        for s in steps:
            m = min(m, s["exp_dist"]); running.append(m)
        fig.add_trace(go.Scatter(x=hops, y=running, mode="lines+markers",
                                 name=f"Q#{qext}", line=dict(width=2), marker=dict(size=3)))
    fig.update_layout(title="距离衰减曲线 (Distance Decay over Hops)",
                      xaxis_title="跳数", yaxis_title="到查询的最优距离", height=420)
    return fig


def fig_degree_hitrate_heatmap(graph, qdata, base_vecs, queries_vecs, space):
    """2-D heatmap: node-degree × distance-to-query → hit-rate for GT targets."""
    dfn = get_dist(space)
    pts_found, pts_missed = [], []
    for qd in qdata:
        qi = qd["query_idx"]
        hset = set(qd["hnsw_results"])
        for g in qd["gt_results"]:
            if g not in graph.e2i:
                continue
            iid = graph.e2i[g]
            d = dfn(queries_vecs[qi], base_vecs[g])
            deg = graph.deg(iid)
            (pts_found if g in hset else pts_missed).append((deg, d))
    if not pts_found and not pts_missed:
        return None
    all_pts = pts_found + pts_missed
    degs = [p[0] for p in all_pts]
    dists = [p[1] for p in all_pts]
    dbins = np.linspace(min(degs) - 0.5, max(degs) + 0.5, 12)
    tbins = np.linspace(min(dists), max(dists) * 1.01, 12)
    h_f, _, _ = np.histogram2d([p[0] for p in pts_found],  [p[1] for p in pts_found],  [dbins, tbins])
    h_m, _, _ = np.histogram2d([p[0] for p in pts_missed], [p[1] for p in pts_missed], [dbins, tbins])
    total = h_f + h_m
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(total > 0, h_f / total, np.nan)
    fig = go.Figure(go.Heatmap(
        z=rate.T, x=(dbins[:-1] + dbins[1:]) / 2, y=(tbins[:-1] + tbins[1:]) / 2,
        colorscale="RdYlGn", zmin=0, zmax=1,
        colorbar_title="命中率",
        hovertemplate="度数: %{x}<br>距离: %{y:.4f}<br>命中率: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="出度 × 距离 → 命中率热图 (Degree vs Hit Rate)",
                      xaxis_title="节点出度 (Level 0)", yaxis_title="到查询的距离", height=450)
    return fig


def fig_degree_distribution(graph, qdata):
    """Degree histogram: found GT vs missed GT targets."""
    d_f, d_m = [], []
    for qd in qdata:
        hset = set(qd["hnsw_results"])
        for g in qd["gt_results"]:
            if g in graph.e2i:
                deg = graph.deg(graph.e2i[g])
                (d_f if g in hset else d_m).append(deg)
    fig = go.Figure()
    if d_f:
        fig.add_trace(go.Histogram(x=d_f, name="GT 命中", opacity=0.7,
                                   marker_color="#2ecc71", nbinsx=20))
    if d_m:
        fig.add_trace(go.Histogram(x=d_m, name="GT 未命中", opacity=0.7,
                                   marker_color="#e74c3c", nbinsx=20))
    fig.update_layout(title="GT 目标度数分布: 命中 vs 未命中",
                      xaxis_title="节点出度 (Level 0)", yaxis_title="计数",
                      barmode="overlay", height=400)
    return fig


def fig_bfs_hop_dist(qdata, graph):
    """BFS hop distance to GT targets: found vs missed."""
    h_f, h_m = [], []
    for qd in qdata:
        bfs = qd.get("bfs_paths", {})
        hset = set(qd["hnsw_results"])
        for g in qd["gt_results"]:
            if g in graph.e2i:
                iid = graph.e2i[g]
                if iid in bfs:
                    hops = len(bfs[iid]) - 1
                    (h_f if g in hset else h_m).append(hops)
    fig = go.Figure()
    if h_f:
        fig.add_trace(go.Histogram(x=h_f, name="GT 命中", opacity=0.7,
                                   marker_color="#2ecc71", nbinsx=30))
    if h_m:
        fig.add_trace(go.Histogram(x=h_m, name="GT 未命中", opacity=0.7,
                                   marker_color="#e74c3c", nbinsx=30))
    fig.update_layout(title="BFS 最短路径跳数分布 (到 GT 目标)",
                      xaxis_title="最短路径跳数", yaxis_title="计数",
                      barmode="overlay", height=400)
    return fig


def fig_frontier_evolution(trace, q_ext):
    """How the distance of visited nodes evolves over visit order."""
    vl = trace["vlog"]
    orders = [v["order"] for v in vl]
    dists  = [v["dist"]  for v in vl]
    rmin, m = [], float("inf")
    for d in dists:
        m = min(m, d); rmin.append(m)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=orders, y=dists, mode="markers",
                             name="访问节点距离",
                             marker=dict(size=2, color="lightblue", opacity=0.4)))
    fig.add_trace(go.Scatter(x=orders, y=rmin, mode="lines",
                             name="当前最优距离",
                             line=dict(color="red", width=2)))
    fig.update_layout(title=f"搜索前沿演化 — Query #{q_ext}",
                      xaxis_title="访问顺序", yaxis_title="到查询的距离", height=380)
    return fig


def fig_recall_vs_difficulty(qdata, base_vecs, queries_vecs, space):
    """Scatter: average GT distance (difficulty) vs achieved recall."""
    dfn = get_dist(space)
    xs, ys, txt = [], [], []
    for qd in qdata:
        qi = qd["query_idx"]
        ds = [dfn(queries_vecs[qi], base_vecs[g]) for g in qd["gt_results"]]
        xs.append(np.mean(ds)); ys.append(qd["recall"]); txt.append(f"Q#{qi}")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers", text=txt,
                               marker=dict(size=9, color="steelblue"),
                               hoverinfo="text+x+y"))
    fig.update_layout(title="查询难度 vs Recall (低 recall 查询)",
                      xaxis_title="平均 GT 距离 (越大越难)", yaxis_title="Recall",
                      height=400)
    return fig


def fig_local_connectivity(graph, qdata, base_vecs, space):
    """GT target: degree vs fraction of its neighbors that were visited."""
    data_f, data_m = [], []
    for qd in qdata:
        visited = qd.get("trace", {}).get("visited", set())
        hset = set(qd["hnsw_results"])
        for g in qd["gt_results"]:
            if g not in graph.e2i:
                continue
            iid = graph.e2i[g]
            nbrs = graph.nbrs(iid, 0)
            vis_frac = sum(1 for nb in nbrs if nb in visited) / max(len(nbrs), 1)
            deg = graph.deg(iid)
            (data_f if g in hset else data_m).append((deg, vis_frac))
    fig = go.Figure()
    if data_f:
        fig.add_trace(go.Scatter(
            x=[d[0] for d in data_f], y=[d[1] for d in data_f],
            mode="markers", name="命中", marker=dict(size=6, color="#2ecc71", opacity=0.6)))
    if data_m:
        fig.add_trace(go.Scatter(
            x=[d[0] for d in data_m], y=[d[1] for d in data_m],
            mode="markers", name="未命中", marker=dict(size=8, color="#e74c3c", symbol="x")))
    fig.update_layout(title="GT 目标局部连通性: 度数 vs 邻居被访问比例",
                      xaxis_title="节点出度", yaxis_title="邻居被搜索访问的比例",
                      height=400)
    return fig


def fig_edge_graph_around_query(base3d, graph, trace, gt_ext, qi_ext, q3d,
                                base_vecs, query_vec, space):
    """Show actual graph edges near the search area for structural insight."""
    i2e, e2i = graph.i2e, graph.e2i
    visited = trace["visited"]
    gt_int = {e2i[g] for g in gt_ext if g in e2i}
    show_nodes = visited | gt_int
    if len(show_nodes) > 1500:
        keep = set(list(visited)[:800]) | gt_int
        show_nodes = keep

    edges_x, edges_y, edges_z = [], [], []
    for n in show_nodes:
        c1 = base3d[int(i2e[n])]
        for nb in graph.nbrs(n, 0):
            if nb in show_nodes:
                c2 = base3d[int(i2e[nb])]
                edges_x += [c1[0], c2[0], None]
                edges_y += [c1[1], c2[1], None]
                edges_z += [c1[2], c2[2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode="lines", name="图边",
        line=dict(color="rgba(180,180,180,0.15)", width=1), hoverinfo="skip"))

    ns = list(show_nodes)
    dfn = get_dist(space)
    ds = [dfn(query_vec, base_vecs[int(i2e[n])]) for n in ns]
    clr = ["red" if n in gt_int and n not in trace["visited"] else
           "gold" if n in gt_int else "steelblue" for n in ns]
    fig.add_trace(go.Scatter3d(
        x=[base3d[int(i2e[n])][0] for n in ns],
        y=[base3d[int(i2e[n])][1] for n in ns],
        z=[base3d[int(i2e[n])][2] for n in ns],
        mode="markers", name="节点",
        marker=dict(size=3, color=clr, opacity=0.7),
        text=[f"ID:{int(i2e[n])} d={d:.4f} deg={graph.deg(n)}"
              for n, d in zip(ns, ds)], hoverinfo="text"))

    fig.add_trace(go.Scatter3d(
        x=[q3d[0]], y=[q3d[1]], z=[q3d[2]],
        mode="markers", name="查询点",
        marker=dict(size=12, color="blue", symbol="diamond"),
        text=[f"Query #{qi_ext}"], hoverinfo="text"))

    fig.update_layout(title=f"Query #{qi_ext} 局部图结构 (Graph Edges)",
                      scene=dict(xaxis_title="D1", yaxis_title="D2", zaxis_title="D3",
                                 aspectmode="data"),
                      height=650, margin=dict(l=0, r=0, t=40, b=0))
    return fig

# ============================================================================
#  HTML dashboard
# ============================================================================

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f5f7fa;color:#333;line-height:1.6}
.ctn{max-width:1500px;margin:0 auto;padding:20px}
h1{color:#1a1a2e;margin:20px 0;font-size:28px;border-bottom:3px solid #4361ee;padding-bottom:10px}
h2{color:#2d3436;margin:20px 0 10px;font-size:22px}
.card{background:#fff;border-radius:12px;padding:20px;margin:15px 0;
      box-shadow:0 2px 10px rgba(0,0,0,.08)}
.params{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin:10px 0}
.pm{background:#e8eaf6;padding:8px 12px;border-radius:6px;font-size:14px}
.pm .lb{color:#666;font-size:12px}.pm .vl{font-weight:bold;color:#1a1a2e}
table{width:100%;border-collapse:collapse;margin:10px 0}
th,td{padding:10px 15px;text-align:left;border-bottom:1px solid #eee}
th{background:#4361ee;color:#fff;font-weight:500}
tr:hover{background:#f0f0f0}
.tbs{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:10px}
.tb{padding:8px 16px;border:2px solid #4361ee;border-radius:6px;background:#fff;
    color:#4361ee;cursor:pointer;font-size:14px;transition:.2s}
.tb:hover{background:#e8eaf6}.tb.a{background:#4361ee;color:#fff}
.tc{display:none}.tc.a{display:block}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:15px}
.insight{background:#fff3cd;border-left:4px solid #ffc107;padding:12px;margin:10px 0;
         border-radius:0 6px 6px 0;font-size:14px}
"""

_JS = """
function sw(e,id){
  var c=e.target.closest('.twrap');
  c.querySelectorAll('.tc').forEach(t=>t.classList.remove('a'));
  c.querySelectorAll('.tb').forEach(b=>b.classList.remove('a'));
  document.getElementById(id).classList.add('a');
  e.target.classList.add('a');
  var p=document.getElementById(id).querySelector('.plotly-graph-div');
  if(p)Plotly.Plots.resize(p);
}
"""


def _pm(label, val):
    return f'<div class="pm"><div class="lb">{label}</div><div class="vl">{val}</div></div>'


def build_html(figs_3d, figs_edge, stat_figs, qdata, params, path):
    to_div = lambda f: pio.to_html(f, full_html=False, include_plotlyjs=False)

    trows = ""
    for qd in qdata:
        missed = len(set(qd["gt_results"]) - set(qd["hnsw_results"]))
        tr = qd.get("trace", {})
        trows += (f'<tr><td>{qd["query_idx"]}</td><td>{qd["recall"]:.4f}</td>'
                  f'<td>{tr.get("total_visited","?")}</td><td>{tr.get("total_hops","?")}</td>'
                  f'<td>{missed}</td></tr>')

    html = (f'<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>HNSW Search Trace Analysis</title>'
            f'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
            f'<style>{_CSS}</style></head><body><div class="ctn">'
            f'<h1>HNSW 搜索路径分析 (Search Trace Analysis)</h1>')

    # --- params ---
    html += '<div class="card"><h2>参数配置</h2><div class="params">'
    for k, v in params.items():
        html += _pm(k, f"{v:.4f}" if isinstance(v, float) else v)
    html += '</div></div>'

    # --- query table ---
    html += ('<div class="card"><h2>低 Recall 查询</h2><table>'
             '<tr><th>Query ID</th><th>Recall@K</th><th>访问节点</th><th>跳数</th><th>GT 未命中</th></tr>'
             f'{trows}</table></div>')

    # --- insights ---
    html += '<div class="card"><h2>诊断洞察</h2>'
    for qd in qdata:
        qi, rec = qd["query_idx"], qd["recall"]
        missed = set(qd["gt_results"]) - set(qd["hnsw_results"])
        bfs = qd.get("bfs_paths", {})
        missed_hops = [len(bfs[graph_e2i]) - 1 for g in missed
                       if (graph_e2i := qd.get("_e2i", {}).get(g)) is not None
                       and graph_e2i in bfs] if bfs else []
        avg_hop = np.mean(missed_hops) if missed_hops else -1
        tr = qd.get("trace", {})
        msg = f"<b>Q#{qi}</b> (recall={rec:.2%}): "
        if avg_hop > 6:
            msg += "未命中目标在图中距离远 (平均 {:.1f} 跳) → <b>图质量/连通性</b>可能不足。".format(avg_hop)
        elif avg_hop > 0:
            msg += "未命中目标图距离适中 ({:.1f} 跳) 但搜索未到达 → <b>ef 不足</b>或搜索陷入局部最优。".format(avg_hop)
        else:
            msg += f"未命中 {len(missed)} 个目标。"
        html += f'<div class="insight">{msg}</div>'
    html += '</div>'

    # --- 3D search trace ---
    html += '<div class="card"><h2>3D 搜索路径可视化</h2>'
    html += '<p style="color:#666;font-size:14px;margin-bottom:8px">拖动旋转 | 滚轮缩放 | 点击图例切换图层 | 悬停查看详情</p>'
    html += '<div class="twrap"><div class="tbs">'
    for i, qd in enumerate(qdata):
        a = " a" if i == 0 else ""
        html += f'<button class="tb{a}" onclick="sw(event,\'t3d_{i}\')">Q#{qd["query_idx"]} (R={qd["recall"]:.2f})</button>'
    html += '</div>'
    for i, f3d in enumerate(figs_3d):
        a = " a" if i == 0 else ""
        html += f'<div class="tc{a}" id="t3d_{i}">{to_div(f3d)}</div>'
    html += '</div></div>'

    # --- graph edge view ---
    if figs_edge:
        html += '<div class="card"><h2>局部图结构 (Graph Edges)</h2>'
        html += '<div class="twrap"><div class="tbs">'
        for i, qd in enumerate(qdata):
            a = " a" if i == 0 else ""
            html += f'<button class="tb{a}" onclick="sw(event,\'tge_{i}\')">Q#{qd["query_idx"]}</button>'
        html += '</div>'
        for i, fe in enumerate(figs_edge):
            a = " a" if i == 0 else ""
            html += f'<div class="tc{a}" id="tge_{i}">{to_div(fe)}</div>'
        html += '</div></div>'

    # --- statistics ---
    html += '<div class="card"><h2>统计分析</h2><div class="sg">'
    for sf in stat_figs:
        if sf is not None:
            html += f'<div>{to_div(sf)}</div>'
    html += '</div></div>'

    html += f'</div><script>{_JS}</script></body></html>'

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    print(f"\n✓ Dashboard → {path}")

# ============================================================================
#  Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="HNSW Search Trace Visualizer")
    ap.add_argument("--dataset", choices=["hotpotqa", "locomo", "sift1m"], default="hotpotqa")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef_construction", type=int, default=200)
    ap.add_argument("--ef_search", type=int, default=10)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--num_queries", type=int, default=5)
    ap.add_argument("--projection", choices=["auto", "pca", "umap"], default="auto")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--index_path", default=None)
    ap.add_argument("--max_base", type=int, default=None, help="Cap base vectors (for quick test)")
    args = ap.parse_args()

    base, queries, gt, space = load_dataset(args.dataset, args.data_dir)
    if args.max_base and len(base) > args.max_base:
        print(f"[cap] base → {args.max_base}")
        base = base[: args.max_base]
        # Recompute GT for the capped subset via brute-force
        print("[cap] recomputing ground truth for subset ...")
        if space == "ip":
            sims = queries @ base.T
            gt_new = np.argsort(-sims, axis=1)[:, : gt.shape[1]]
        else:
            from sklearn.metrics import pairwise_distances
            dists_all = pairwise_distances(queries, base, metric="sqeuclidean")
            gt_new = np.argsort(dists_all, axis=1)[:, : gt.shape[1]]
        gt = gt_new.astype(np.int32)
        print(f"  gt updated: {gt.shape}")

    n, dim = base.shape

    # --- build / load index ---
    print(f"\n[index] building M={args.M} efc={args.ef_construction} ...")
    idx = hnswlib.Index(space=space, dim=dim)
    if args.index_path and Path(args.index_path).exists():
        idx.init_index(max_elements=n, M=args.M,
                       ef_construction=args.ef_construction, random_seed=42)
        idx.load_index(args.index_path, max_elements=n)
        print(f"  loaded from {args.index_path}")
    else:
        idx.init_index(max_elements=n, M=args.M,
                       ef_construction=args.ef_construction, random_seed=42)
        idx.add_items(base, ids=np.arange(n), num_threads=1)
        if args.index_path:
            idx.save_index(args.index_path)
            print(f"  saved to {args.index_path}")

    # --- low-recall queries ---
    sel, all_recalls = find_low_recall(idx, queries, gt, args.K, args.ef_search, args.num_queries)
    if not sel:
        sys.exit("No queries selected — try lowering ef_search.")

    # --- extract graph ---
    print("\n[graph] extracting ...")
    graph = HNSWGraph(idx)

    # --- trace ---
    tracer = HNSWTracer(graph, base, space)
    print("\n[trace] HNSW search traces ...")
    for qd in sel:
        qi = qd["query_idx"]
        res_ext, tr = tracer.trace(queries[qi], args.K, args.ef_search)
        qd["trace"] = tr
        qd["traced_results"] = res_ext
        qd["_e2i"] = graph.e2i
        print(f"  Q#{qi}: visited={tr['total_visited']}  hops={tr['total_hops']}")

    # --- BFS minimum-hop paths ---
    print("[path] BFS minimum-hop paths ...")
    for qd in sel:
        ep = qd["trace"]["entry_l0"]
        gt_int = [graph.e2i[g] for g in qd["gt_results"] if g in graph.e2i]
        qd["bfs_paths"] = bfs_paths(graph, ep, gt_int)

    # --- 3D projection ---
    qi_list = [qd["query_idx"] for qd in sel]
    all_vecs = np.vstack([base, queries[qi_list]])
    c3d = project_3d(all_vecs, args.projection)
    base3d = c3d[:n]
    q3d_map = {qi: c3d[n + i] for i, qi in enumerate(qi_list)}

    # --- 3D figures ---
    print("[viz] 3D figures ...")
    figs_3d, figs_edge = [], []
    for qd in sel:
        qi = qd["query_idx"]
        missed_gt = {graph.e2i[g] for g in set(qd["gt_results"]) - set(qd["hnsw_results"]) if g in graph.e2i}
        bfs_missed = {k: v for k, v in qd.get("bfs_paths", {}).items() if k in missed_gt}
        figs_3d.append(fig_3d_trace(
            qi, q3d_map[qi], base3d, graph, qd["trace"],
            bfs_missed,
            qd["gt_results"], qd["hnsw_results"],
            qd["recall"], space, base, queries[qi]))
        figs_edge.append(fig_edge_graph_around_query(
            base3d, graph, qd["trace"], qd["gt_results"],
            qi, q3d_map[qi], base, queries[qi], space))

    # --- stat figures ---
    print("[viz] stat figures ...")
    traces_info = [(qd["trace"], qd["query_idx"]) for qd in sel]
    stats = [
        fig_distance_decay(traces_info),
        fig_degree_hitrate_heatmap(graph, sel, base, queries, space),
        fig_degree_distribution(graph, sel),
        fig_bfs_hop_dist(sel, graph),
        fig_frontier_evolution(sel[0]["trace"], sel[0]["query_idx"]),
        fig_recall_vs_difficulty(sel, base, queries, space),
        fig_local_connectivity(graph, sel, base, space),
    ]

    # --- output ---
    out_dir = args.output_dir or f"results/search_trace_{datetime.now():%Y%m%d_%H%M%S}"
    out_html = str(Path(out_dir) / "dashboard.html")
    params_d = dict(dataset=args.dataset, space=space, n_base=n,
                    n_queries=len(queries), dim=dim,
                    M=args.M, ef_construction=args.ef_construction,
                    ef_search=args.ef_search, K=args.K,
                    avg_recall=float(all_recalls.mean()),
                    projection=args.projection)
    build_html(figs_3d, figs_edge, stats, sel, params_d, out_html)

    # --- save trace JSON ---
    jpath = Path(out_dir) / "trace_data.json"
    export = []
    for qd in sel:
        e = {k: v for k, v in qd.items() if k not in ("trace", "_e2i")}
        bfs_e = {str(k): [int(x) for x in v] for k, v in qd.get("bfs_paths", {}).items()}
        e["bfs_paths"] = bfs_e
        export.append(e)
    jpath.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Trace JSON → {jpath}")
    print(f"\n完成! 在浏览器中打开 dashboard.html 查看交互式可视化结果。")


if __name__ == "__main__":
    main()
