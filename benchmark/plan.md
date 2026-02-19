# Coding Plan: HotpotQA → 向量 RAG 数据集 + hnswlib Benchmark

## 项目目标

将 HotpotQA Training Set 转化为标准向量 RAG 评测数据集，使用 ONNX 优化的 `BAAI/bge-large-en-v1.5` 生成 embedding，并用当前目录的 hnswlib C++ 项目进行完整的搜索性能评测（延迟、吞吐量、内存占用、召回率），最终汇总结果并生成可视化图表，所有输出按时间戳保存。

---

## Phase 0: 环境准备

### 0.1 Python 依赖

```bash
pip install transformers sentence-transformers optimum[onnxruntime] onnxruntime numpy tqdm matplotlib
```

### 0.2 数据下载

```bash
mkdir -p data/
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/hotpot_train_v1.1.json
```

### 0.3 hnswlib 编译确认

确认当前目录 hnswlib 项目可正常编译：

```bash
cd hnswlib/   # 你的 hnswlib 源码目录
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

---

## Phase 1: HotpotQA 数据解析与语料库构建

### 1.1 解析原始 JSON

读取 `hotpot_train_v1.1.json`，结构如下：

```python
# 每条数据结构:
{
  "_id": "...",
  "question": "...",
  "answer": "...",
  "type": "bridge" | "comparison",
  "level": "easy" | "medium" | "hard",
  "supporting_facts": [["title", sent_idx], ...],
  "context": [["title", ["sent0", "sent1", ...]], ...]
}
```

### 1.2 构建去重语料库

**输出文件：`data/corpus.jsonl`**

```python
"""
遍历所有 90,564 个问题的 context 字段。
每个 context 条目是 [title, [sentences...]] 形式。
以 title 为 key 做去重（同一 Wikipedia 文章可能出现在多个问题中）。
每篇文章的所有 sentences 拼接为一个段落作为一个 document。

输出格式（每行一个 JSON）：
{
  "doc_id": 0,
  "title": "Wikipedia Article Title",
  "text": "Sentence 0. Sentence 1. Sentence 2. ..."
}

预计去重后约 50-60 万个唯一文档。
"""
```

**关键决策**：HotpotQA 原始段落大多在 100-300 tokens，bge-large-en-v1.5 支持最大 512 tokens，因此**不需要额外 chunking**，直接以每个 title 对应的段落为单位作为一个 document。如果个别段落超过 512 tokens，截断即可（极少数情况）。

### 1.3 构建 Query 集

**输出文件：`data/queries.jsonl`**

```python
"""
提取所有问题及其 ground truth 信息。

输出格式（每行一个 JSON）：
{
  "query_id": 0,
  "question": "Which team does the player named ...",
  "answer": "...",
  "type": "bridge",
  "level": "hard",
  "supporting_titles": ["Title A", "Title B"],
  "supporting_facts": [["Title A", 1], ["Title B", 0]]
}
"""
```

### 1.4 构建 Ground Truth 映射

**输出文件：`data/ground_truth.jsonl`**

```python
"""
将每个 query 的 supporting_titles 映射到 corpus 中的 doc_id。

输出格式（每行一个 JSON）：
{
  "query_id": 0,
  "relevant_doc_ids": [1234, 5678]
}

同时输出 data/title_to_docid.json 供后续使用。
"""
```

### 1.5 数据统计校验

输出以下统计信息用于验证：

- 总 query 数（应为 90,564）
- 去重后总 document 数
- 平均每篇文档 token 数（用 tokenizer 统计）
- 超过 512 token 的文档数及占比
- 每个 query 平均关联 document 数（应接近 2）
- type 分布（bridge vs comparison）
- level 分布（easy/medium/hard）

---

## Phase 2: ONNX 优化的 Embedding 生成

### 2.1 模型转换为 ONNX

```python
"""
使用 optimum 将 bge-large-en-v1.5 导出为 ONNX 格式。

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "BAAI/bge-large-en-v1.5"

model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained("models/bge-large-en-v1.5-onnx")
tokenizer.save_pretrained("models/bge-large-en-v1.5-onnx")
"""
```

### 2.2 多进程并行 Embedding 生成

**核心策略**：将语料分片，每个子进程加载一个 ONNX 模型副本独立编码，最后合并。

```python
"""
配置参数：
- NUM_WORKERS: CPU总核数 // 4（每个 worker 分配 4 个线程）
  例如 64 核 → 16 workers
- BATCH_SIZE: 64
- MAX_LENGTH: 512

对 document 编码：
  - 不需要加 instruction prefix
  - normalize_embeddings=True
  - 输出 float32，维度 1024

对 query 编码：
  - 需要加 instruction prefix:
    "Represent this sentence for searching relevant passages: "
  - normalize_embeddings=True
  - 输出 float32，维度 1024

每个 worker 流程：
  1. 设置 ORT session options:
     session_options = ort.SessionOptions()
     session_options.intra_op_num_threads = 4
     session_options.inter_op_num_threads = 1
  2. 加载 ONNX 模型
  3. 对分配到的文本分片，按 batch 编码
  4. 将结果保存为 .npy 分片文件

最终合并所有分片：
  - data/corpus_embeddings.npy  — shape: (num_docs, 1024), dtype: float32
  - data/query_embeddings.npy   — shape: (90564, 1024), dtype: float32

同时生成 data/embedding_meta.json 记录：
  - 模型名称, 推理后端, 维度, 是否归一化, 距离度量
  - 总文档数, 总 query 数, 编码总耗时, workers 数量
"""
```

### 2.3 生成精确 KNN Ground Truth

```python
"""
用 numpy 暴力计算每个 query 的精确 top-100 最近邻（内积距离，向量已归一化）。

必须分 batch 计算（每次 1000 个 query），避免 90k × 600k 矩阵撑爆内存。

  for batch in query_batches:
      scores = batch @ corpus_embeddings.T   # (1000, num_docs)
      top100_indices = np.argpartition(-scores, 100, axis=1)[:, :100]
      # 对 top100 再排序

输出：
  - data/knn_gt_indices.npy   — shape: (num_queries, 100), dtype: int32
  - data/knn_gt_distances.npy — shape: (num_queries, 100), dtype: float32
"""
```

### 2.4 导出为标准 ANN Benchmark 二进制格式

**输出文件**：

```
data/
├── corpus_vectors.fvecs      # 文档向量，fvecs 格式
├── query_vectors.fvecs       # 查询向量，fvecs 格式
├── ground_truth.ivecs        # 精确 KNN top-100，ivecs 格式
├── corpus_embeddings.npy     # numpy 格式备用
├── query_embeddings.npy
└── embedding_meta.json
```

**fvecs/ivecs 格式**（供 C++ 读取）：

```
fvecs: 每条记录 = [dim (int32)] [v1 (float32)] ... [v_dim (float32)]
ivecs: 每条记录 = [K (int32)] [id1 (int32)] ... [id_K (int32)]
```

```python
"""
def write_fvecs(filename, vectors):
    n, dim = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(np.array([dim], dtype=np.int32).tobytes())
            f.write(vectors[i].tobytes())

def write_ivecs(filename, indices):
    n, k = indices.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(np.array([k], dtype=np.int32).tobytes())
            f.write(indices[i].tobytes())
"""
```

---

## Phase 3: C++ Benchmark 程序（基于 hnswlib）

### 3.1 文件结构

```
hnswlib/
├── benchmark/
│   ├── CMakeLists.txt
│   ├── io_utils.h          # fvecs/ivecs 读写
│   ├── timer.h             # 高精度计时 + percentile 统计
│   ├── memory_utils.h      # /proc/self/status 内存监测
│   ├── metrics.h           # recall@K 计算
│   └── bench_hotpotqa.cpp  # 主 benchmark 程序
```

### 3.2 io_utils.h — 数据读取

```cpp
/**
 * float* read_fvecs_flat(const std::string& filename, int& num, int& dim);
 *   - 读取 fvecs，返回连续内存 float 数组
 *
 * int* read_ivecs_flat(const std::string& filename, int& num, int& k);
 *   - 读取 ivecs，返回连续内存 int 数组
 */
```

### 3.3 timer.h — 计时工具

```cpp
/**
 * 使用 std::chrono::high_resolution_clock
 *
 * class Timer {
 *   void start();
 *   double elapsed_ms();
 *   double elapsed_us();
 * };
 *
 * struct LatencyStats {
 *   double mean_us, p50_us, p95_us, p99_us, max_us;
 * };
 * LatencyStats compute_latency_stats(const std::vector<double>& latencies);
 */
```

### 3.4 memory_utils.h — 内存监测

```cpp
/**
 * 从 /proc/self/status 读取：
 *
 * struct MemoryInfo {
 *   size_t vm_rss_kb;   // 物理内存
 *   size_t vm_size_kb;  // 虚拟内存
 * };
 * MemoryInfo get_memory_info();
 *
 * 采集时间点：
 *   1. 程序启动后（baseline）
 *   2. 数据加载完成后
 *   3. 索引构建完成后 ← 核心指标
 *   4. 查询完成后
 */
```

### 3.5 metrics.h — Recall 计算

```cpp
/**
 * float compute_recall(
 *     const int* ann_results,  // ANN 返回的 K 个 id
 *     const int* gt_results,   // ground truth 的 K 个 id
 *     int K
 * );
 * // = |intersection| / K
 *
 * 计算所有 query 的平均 recall@K。
 */
```

### 3.6 bench_hotpotqa.cpp — 主 Benchmark 程序

```cpp
/**
 * 命令行参数：
 *   --base_path       : corpus_vectors.fvecs 路径
 *   --query_path      : query_vectors.fvecs 路径
 *   --gt_path         : ground_truth.ivecs 路径
 *   --M               : HNSW M 参数（默认 16）
 *   --ef_construction  : 构建 ef（默认 200）
 *   --ef_search       : 查询 ef，逗号分隔（默认 "10,20,50,100,200,500"）
 *   --K               : top-K，逗号分隔（默认 "1,5,10,20,50,100"）
 *   --num_threads     : 查询线程数（默认 1）
 *   --index_path      : 索引保存/加载路径（可选）
 *   --metric          : "ip" 或 "l2"（默认 "ip"）
 *   --output          : 结果 JSON 输出路径
 *
 * 主流程：
 *
 * 1. 加载数据
 *    - 读取 base vectors, query vectors, ground truth
 *    - 打印数据统计
 *    - 记录内存 baseline
 *
 * 2. 构建索引（或从文件加载）
 *    - hnswlib::HierarchicalNSW<float>(space, num_base, M, ef_construction)
 *    - 多线程插入
 *    - 记录：构建时间(sec)、构建吞吐(points/sec)、索引内存(MB)
 *    - 可选保存索引
 *
 * 3. 查询评测 — 对每个 (ef_search, K) 组合：
 *
 *    a) 设置 index->setEf(ef_search)
 *
 *    b) Warmup: 先跑 1000 次查询不计时
 *
 *    c) 单线程延迟测试：
 *       - 逐条查询所有 query
 *       - 记录每条延迟
 *       - 计算 LatencyStats: mean, p50, p95, p99, max (微秒)
 *
 *    d) 多线程吞吐量测试：
 *       - 线程数从 1 递增到 num_threads（1,2,4,8,16,...）
 *       - 每个线程数下用 OpenMP 并行查询所有 query
 *       - 记录 QPS
 *
 *    e) Recall 计算：
 *       - 计算所有 query 的平均 recall@K
 *
 * 4. 输出结果 JSON
 *
 *    {
 *      "dataset": "HotpotQA-Train",
 *      "embedding_model": "bge-large-en-v1.5",
 *      "dim": 1024,
 *      "num_base": 590000,
 *      "num_query": 90564,
 *      "index": {
 *        "type": "HNSW",
 *        "M": 16,
 *        "ef_construction": 200,
 *        "build_time_sec": 45.2,
 *        "build_throughput_pts_per_sec": 13053,
 *        "index_memory_mb": 3412,
 *        "data_memory_mb": 2300
 *      },
 *      "results": [
 *        {
 *          "ef_search": 10,
 *          "K": 10,
 *          "recall": 0.923,
 *          "latency": {
 *            "mean_us": 45.2, "p50_us": 42.1,
 *            "p95_us": 72.3, "p99_us": 89.1, "max_us": 234.5
 *          },
 *          "qps": {
 *            "1": 22124, "4": 78456, "16": 156789, "64": 312345
 *          }
 *        },
 *        ...
 *      ]
 *    }
 */
```

### 3.7 CMakeLists.txt

```cmake
# benchmark/CMakeLists.txt
#
# cmake_minimum_required(VERSION 3.14)
# project(hotpotqa_bench)
#
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
#
# include_directories(${CMAKE_SOURCE_DIR}/../hnswlib)
#
# add_executable(bench_hotpotqa bench_hotpotqa.cpp)
# target_link_libraries(bench_hotpotqa pthread)
```

---

## Phase 4: 结果汇总、可视化与归档

### 4.1 输出目录结构

所有结果保存到以时间戳命名的目录下，格式 `results/YYYYMMDD_HHMMSS/`：

```
results/
└── 20260214_153042/
    ├── raw/                          # C++ 输出的原始 JSON
    │   ├── results_M8_efc100.json
    │   ├── results_M8_efc200.json
    │   ├── results_M16_efc100.json
    │   ├── results_M16_efc200.json
    │   ├── results_M32_efc200.json
    │   └── ...
    ├── summary.json                  # 汇总的全量结果
    ├── summary.csv                   # 扁平化 CSV，方便 pandas 分析
    ├── plots/
    │   ├── recall_vs_qps.png         # 核心: recall-QPS 帕累托曲线
    │   ├── recall_vs_latency_p99.png # recall vs p99 延迟
    │   ├── ef_search_vs_recall.png   # 不同 M 下 ef_search 与 recall 关系
    │   ├── ef_search_vs_latency.png  # ef_search 与延迟关系
    │   ├── thread_scalability.png    # 多线程 QPS 扩展性
    │   ├── memory_comparison.png     # 不同 M 的内存占用对比
    │   └── build_time_comparison.png # 构建时间对比
    ├── best_configs.json             # 按不同目标筛选的最优配置
    └── run_meta.json                 # 运行环境元数据
```

### 4.2 汇总脚本：`scripts/common/aggregate_results.py`

```python
"""
功能：读取 raw/ 下所有 JSON，合并为统一的 summary。

输入：results/<timestamp>/raw/*.json
输出：summary.json, summary.csv, best_configs.json

summary.csv 扁平化为每行一个 (M, ef_construction, ef_search, K) 组合：
列: M, ef_construction, ef_search, K, recall, mean_us, p50_us, p95_us, p99_us,
    max_us, qps_1t, qps_4t, qps_16t, qps_64t, index_memory_mb, build_time_sec

best_configs.json 按以下目标各给出最优配置：
  - "highest_recall_at_k10": recall@10 最高的配置
  - "lowest_p99_above_95_recall": recall@10 >= 0.95 前提下 p99 最低
  - "highest_qps_above_95_recall": recall@10 >= 0.95 前提下 QPS 最高
  - "best_memory_efficiency": recall@10 >= 0.95 前提下内存最小
"""
```

### 4.3 可视化脚本：`scripts/common/plot_results.py`

```python
"""
输入：results/<timestamp>/summary.csv
输出：results/<timestamp>/plots/*.png

使用 matplotlib，所有图统一风格：白底网格、150 dpi、12pt 字体。

--- 图 1: recall_vs_qps.png (核心帕累托曲线) ---
  X 轴: Recall@10
  Y 轴: QPS (单线程, log scale)
  每条线: 一个 (M, ef_construction) 组合
    不同点对应不同 ef_search，点旁标注 ef_search 值
  不同颜色区分 M，不同线型区分 ef_construction
  标注帕累托前沿（最优配置连线）

--- 图 2: recall_vs_latency_p99.png ---
  X 轴: Recall@10
  Y 轴: P99 latency (us, log scale)
  分组同图 1，越左下越好

--- 图 3: ef_search_vs_recall.png ---
  X 轴: ef_search (log scale)
  Y 轴: Recall@10
  每条线: 一个 M 值（固定 ef_construction=200）
  展示 recall 随 ef_search 的收敛趋势

--- 图 4: ef_search_vs_latency.png ---
  X 轴: ef_search (log scale)
  Y 轴: mean latency (us)
  每条线: 一个 M 值（固定 ef_construction=200）
  用半透明区域标出 p50 到 p99 范围

--- 图 5: thread_scalability.png ---
  X 轴: 线程数 (1,2,4,8,16,32,64)
  Y 轴: QPS
  每条线: 一个 ef_search 值（固定 M=16, ef_construction=200, K=10）
  虚线: 理想线性扩展参考线
  右侧 Y 轴: 加速比 (speedup)

--- 图 6: memory_comparison.png ---
  分组柱状图
  X 轴: M 值 (8, 16, 32, 48)
  Y 轴: 索引内存 (MB)
  每组内并排放不同 ef_construction
  柱顶标注具体数值

--- 图 7: build_time_comparison.png ---
  分组柱状图
  X 轴: M 值
  Y 轴: 构建时间 (sec)
  同图 6 的分组方式
  柱顶标注具体数值

通用设置：
  plt.style.use('seaborn-v0_8-whitegrid')
  fig, ax = plt.subplots(figsize=(10, 6))
  plt.savefig(path, dpi=150, bbox_inches='tight')
"""
```

### 4.4 运行环境元数据：`scripts/common/collect_meta.py`

```python
"""
输出 run_meta.json，自动采集：

{
  "timestamp": "2026-02-14T15:30:42",
  "hostname": "<hostname>",
  "cpu_model": "<lscpu Model name>",
  "cpu_cores": <nproc>,
  "total_memory_gb": <free -g total>,
  "os": "<lsb_release -d>",
  "compiler": "<g++ --version first line>",
  "hnswlib_commit": "<git rev-parse --short HEAD>",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "onnx_runtime_version": "<ort.__version__>",
  "dataset": {
    "name": "HotpotQA-Train",
    "num_docs": <from embedding_meta.json>,
    "num_queries": <from embedding_meta.json>,
    "dim": 1024
  }
}
"""
```

---

## Phase 5: 完整执行脚本

```bash
#!/bin/bash
# run_all.sh — 一键执行全流程

set -e

DATA_DIR="./data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="./results/${TIMESTAMP}"
RAW_DIR="${RESULT_DIR}/raw"
PLOT_DIR="${RESULT_DIR}/plots"

mkdir -p "$DATA_DIR" "$RAW_DIR" "$PLOT_DIR"

echo "=========================================="
echo " Run ID: ${TIMESTAMP}"
echo " Results: ${RESULT_DIR}"
echo "=========================================="

# ---- Phase 1: 数据准备 ----
echo "[Phase 1] Parsing HotpotQA..."
python scripts/hotpotqa/parse_data.py \
    --input $DATA_DIR/hotpot_train_v1.1.json \
    --output_dir $DATA_DIR

# ---- Phase 2: Embedding 生成 ----
echo "[Phase 2] Generating embeddings (ONNX)..."
python scripts/hotpotqa/generate_embeddings.py \
    --corpus $DATA_DIR/corpus.jsonl \
    --queries $DATA_DIR/queries.jsonl \
    --model_dir models/bge-large-en-v1.5-onnx \
    --output_dir $DATA_DIR \
    --num_workers 16 \
    --threads_per_worker 4 \
    --batch_size 64

echo "[Phase 2.5] Computing exact KNN ground truth..."
python scripts/hotpotqa/compute_knn_gt.py \
    --corpus_emb $DATA_DIR/corpus_embeddings.npy \
    --query_emb $DATA_DIR/query_embeddings.npy \
    --output_dir $DATA_DIR \
    --top_k 100 \
    --batch_size 1000

echo "[Phase 2.6] Exporting fvecs/ivecs..."
python scripts/hotpotqa/export_vecs.py \
    --corpus_emb $DATA_DIR/corpus_embeddings.npy \
    --query_emb $DATA_DIR/query_embeddings.npy \
    --gt $DATA_DIR/knn_gt_indices.npy \
    --output_dir $DATA_DIR

# ---- Phase 3: C++ Benchmark ----
echo "[Phase 3] Running C++ benchmark..."
BENCH_BIN="./hnswlib/benchmark/build/bench_hotpotqa"

for M in 8 16 32 48; do
  for EFC in 100 200 400; do
    echo "  M=${M}, ef_construction=${EFC}"
    $BENCH_BIN \
      --base_path $DATA_DIR/corpus_vectors.fvecs \
      --query_path $DATA_DIR/query_vectors.fvecs \
      --gt_path $DATA_DIR/ground_truth.ivecs \
      --M $M \
      --ef_construction $EFC \
      --ef_search "10,20,50,100,200,500,1000" \
      --K "1,5,10,20,50,100" \
      --num_threads 64 \
      --metric ip \
      --output ${RAW_DIR}/results_M${M}_efc${EFC}.json
  done
done

# ---- Phase 4: 汇总与可视化 ----
echo "[Phase 4] Aggregating results..."
python scripts/common/aggregate_results.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$RESULT_DIR"

echo "[Phase 4] Generating plots..."
python scripts/common/plot_results.py \
    --summary_csv "${RESULT_DIR}/summary.csv" \
    --output_dir "$PLOT_DIR"

echo "[Phase 4] Collecting run metadata..."
python scripts/common/collect_meta.py \
    --output "${RESULT_DIR}/run_meta.json" \
    --data_dir "$DATA_DIR" \
    --hnswlib_dir "./hnswlib"

echo "=========================================="
echo " Done! Results saved to: ${RESULT_DIR}"
echo "=========================================="

# 打印最优配置摘要
python -c "
import json
with open('${RESULT_DIR}/best_configs.json') as f:
    best = json.load(f)
print('\nBest Configurations:')
for target, cfg in best.items():
    print(f'  {target}:')
    print(f'    M={cfg[\"M\"]}, efc={cfg[\"ef_construction\"]}, efs={cfg[\"ef_search\"]}')
    print(f'    recall@10={cfg[\"recall\"]:.4f}, p99={cfg[\"p99_us\"]:.1f}us, qps={cfg[\"qps_1t\"]}')
"
```

---

## 文件清单总览

### Python 脚本（scripts/）

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `01_parse_hotpotqa.py` | 解析 JSON，构建语料库/query/GT | `hotpot_train_v1.1.json` | `corpus.jsonl`, `queries.jsonl`, `ground_truth.jsonl`, `title_to_docid.json`, `stats.json` |
| `02_generate_embeddings.py` | 多进程 ONNX embedding | `corpus.jsonl`, `queries.jsonl` | `corpus_embeddings.npy`, `query_embeddings.npy`, `embedding_meta.json` |
| `03_compute_knn_gt.py` | 暴力精确 KNN | `.npy` 文件 | `knn_gt_indices.npy`, `knn_gt_distances.npy` |
| `04_export_vecs.py` | 导出 fvecs/ivecs | `.npy` 文件 | `corpus_vectors.fvecs`, `query_vectors.fvecs`, `ground_truth.ivecs` |
| `06_aggregate_results.py` | 汇总 C++ 结果 | `raw/*.json` | `summary.json`, `summary.csv`, `best_configs.json` |
| `07_plot_results.py` | 生成 7 张可视化图表 | `summary.csv` | `plots/*.png` |
| `08_collect_meta.py` | 采集运行环境信息 | 系统信息 | `run_meta.json` |

### C++ 源码（hnswlib/benchmark/）

| 文件 | 功能 |
|------|------|
| `io_utils.h` | fvecs/ivecs 读取 |
| `timer.h` | 高精度计时 + percentile |
| `memory_utils.h` | /proc/self/status 内存监测 |
| `metrics.h` | recall@K 计算 |
| `bench_hotpotqa.cpp` | 主 benchmark 程序 |
| `CMakeLists.txt` | 构建配置 |

---

## 注意事项

1. **距离度量**：bge-large-en-v1.5 输出归一化向量，使用内积（ip）等价于余弦。hnswlib 的 ip space 返回 `1 - ip`（越小越好），构建 ground truth 时要一致。

2. **内存管理**：精确 KNN 计算必须分 batch（每次 1000 个 query），不要构造完整 score 矩阵。

3. **ONNX 线程配置**：`intra_op_num_threads × 进程数 ≤ CPU 总核数`，否则线程争抢严重。

4. **Query instruction**：编码 query 时前缀 `"Represent this sentence for searching relevant passages: "`，document 不加。遗漏会严重影响检索质量。

5. **Warmup**：延迟测试前先跑 1000 次查询，消除缓存冷启动。

6. **可复现性**：HNSW 构建有随机性，建议固定随机种子或多次构建取平均。

7. **时间戳隔离**：每次运行的所有输出都在独立的 `results/YYYYMMDD_HHMMSS/` 目录下，不会互相覆盖，便于对比不同运行。
