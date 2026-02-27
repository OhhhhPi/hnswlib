# HNSW Benchmark

使用 hnswlib 评估向量检索性能，支持多个标准数据集。

## 支持的数据集

| 数据集 | 维度 | 基向量数 | 查询数 | 距离度量 | 说明 |
|--------|------|----------|--------|----------|------|
| **HotpotQA** | 1024 | 可变 | 可变 | IP | RAG场景，BGE-large-en-v1.5嵌入 |
| **LoCoMo** | 1024 | 可变 | 可变 | IP | 长对话记忆检索，BGE-large-en-v1.5嵌入 |
| **SIFT1M** | 128 | 1,000,000 | 10,000 | L2 | 标准ANN benchmark |

---

# HotpotQA Benchmark

使用 HotpotQA 数据集和 BGE-large-en-v1.5 向量模型评估 hnswlib 检索性能。

## 目录结构

```
├── benchmark/                          # C++ 基准测试程序
│   ├── bench_hotpotqa.cpp              # HotpotQA 主程序
│   ├── bench_sift1m.cpp                # SIFT1M 主程序
│   ├── io_utils.h                      # fvecs/ivecs 读写
│   ├── timer.h                         # 高精度计时
│   ├── memory_utils.h                  # 内存统计
│   ├── metrics.h                       # Recall 计算
│   ├── CMakeLists.txt                  # 构建配置
│   └── build/                          # 编译输出
├── scripts/                            # 所有脚本 (按流水线分目录)
│   ├── common/                         # 共享工具
│   │   ├── aggregate_results.py        # 汇总结果
│   │   ├── plot_results.py             # 生成可视化图表
│   │   └── collect_meta.py             # 收集系统元数据
│   ├── hotpotqa/                       # HotpotQA 流水线
│   │   ├── parse_data.py               # 解析 HotpotQA 数据
│   │   ├── generate_embeddings.py      # 多进程向量嵌入
│   │   ├── generate_embeddings_simple.py # 单进程嵌入 (调试用)
│   │   ├── compute_knn_gt.py           # 计算精确 KNN Ground Truth
│   │   ├── export_vecs.py              # 导出 fvecs/ivecs 格式
│   │   ├── run_all.sh                  # 完整流水线一键运行
│   │   ├── run_benchmark.sh            # 仅运行 C++ benchmark
│   │   └── run_latency_ablation.sh     # 延迟测量消融实验
│   ├── locomo/                         # LoCoMo 流水线
│   │   ├── download_data.py            # 下载 LoCoMo 数据集
│   │   ├── parse_data.py               # 解析对话数据
│   │   ├── run_all.sh                  # 完整流水线一键运行
│   │   └── run_benchmark.sh            # 仅运行 C++ benchmark
│   ├── sift1m/                         # SIFT1M 流水线
│   │   ├── download_data.py            # 下载 SIFT1M 数据集
│   │   └── run_benchmark.sh            # 运行 SIFT1M benchmark
│   └── dim_ablation/                   # 维度消融实验
│       ├── prepare_data.py             # 零填充 + PCA 分析
│       ├── plot_results.py             # 消融实验可视化
│       └── run_ablation.sh             # 一键运行消融实验
├── data/                               # 数据文件 (gitignore)
│   ├── hotpotqa/                       # HotpotQA 数据
│   │   ├── hotpot_train_v1.1.json      # 原始数据
│   │   ├── corpus.jsonl                # 文档语料
│   │   ├── queries.jsonl               # 查询集合
│   │   ├── corpus_embeddings.npy       # 文档向量
│   │   ├── query_embeddings.npy        # 查询向量
│   │   ├── corpus_vectors.fvecs        # C++ 格式文档向量
│   │   ├── query_vectors.fvecs         # C++ 格式查询向量
│   │   └── ground_truth.ivecs          # 精确 KNN 结果
│   ├── locomo/                         # LoCoMo 数据
│   │   ├── locomo10.json               # 原始数据
│   │   ├── corpus.jsonl                # 对话轮次语料
│   │   ├── queries.jsonl               # QA 查询集合
│   │   └── ground_truth.jsonl          # 证据映射
│   └── sift1m/                         # SIFT1M 数据
├── models/                             # 模型文件
│   └── Xorbits/bge-large-en-v1.5/
└── results/                            # 输出结果
    └── YYYYMMDD_HHMMSS/
        ├── raw/*.json                  # 原始结果
        ├── summary.json                # 汇总 JSON
        ├── summary.csv                 # 汇总 CSV
        ├── best_configs.json           # 最佳配置
        ├── run_meta.json               # 运行元数据
        └── plots/*.png                 # 可视化图表
```

## 依赖要求

### Python
- Python 3.10+
- numpy, tqdm, pandas, matplotlib, seaborn
- onnxruntime, optimum, transformers

### C++
- CMake 3.14+
- GCC with C++17 支持
- OpenMP

## 快速开始

### 1. 构建 C++ Benchmark

```bash
cd benchmark
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 2. 完整运行流程 (Subset 示例)

以下是在 10K 文档 + 5K 查询的子集上运行完整 benchmark 的步骤：

```bash
# Step 0: 构建 C++ Benchmark (首次运行)
cd benchmark && mkdir -p build && cd build && cmake .. && make -j$(nproc)
cd ../..

# Step 1: 解析 HotpotQA 数据 (创建子集)
python scripts/hotpotqa/parse_data.py \
    --input data/hotpotqa/hotpot_train_v1.1.json \
    --output_dir data/hotpotqa \
    --subset_docs 10000 \
    --subset_queries 5000

# Step 2: 生成向量嵌入 (耗时较长，约 20 分钟)
python scripts/hotpotqa/generate_embeddings_simple.py \
    --corpus data/hotpotqa/corpus_subset.jsonl \
    --queries data/hotpotqa/queries_subset.jsonl \
    --output_dir data/hotpotqa \
    --model_name models/Xorbits/bge-large-en-v1.5

# Step 3: 计算精确 KNN Ground Truth (约 2 秒)
python scripts/hotpotqa/compute_knn_gt.py \
    --corpus_emb data/hotpotqa/corpus_embeddings.npy \
    --query_emb data/hotpotqa/query_embeddings.npy \
    --output_dir data/hotpotqa \
    --top_k 100

# Step 4: 导出为 C++ fvecs/ivecs 格式 (约 1 秒)
python scripts/hotpotqa/export_vecs.py \
    --corpus_emb data/hotpotqa/corpus_embeddings.npy \
    --query_emb data/hotpotqa/query_embeddings.npy \
    --gt data/hotpotqa/knn_gt_indices.npy \
    --output_dir data/hotpotqa

# Step 5: 创建结果目录
RUN_DIR=results/$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN_DIR/raw

# Step 6: 运行 C++ Benchmark (约 30 秒)
./benchmark/build/bench_hotpotqa \
    --base_path data/hotpotqa/corpus_vectors.fvecs \
    --query_path data/hotpotqa/query_vectors.fvecs \
    --gt_path data/hotpotqa/ground_truth.ivecs \
    --output $RUN_DIR/raw/benchmark.json \
    --metric ip \
    --M 16 \
    --ef_construction 200 \
    --ef_search 10,50,100,200 \
    --K 1,10,100 \
    --num_threads 16

# Step 7: 汇总结果
python scripts/common/aggregate_results.py \
    --input_dir $RUN_DIR/raw \
    --output_dir $RUN_DIR

# Step 8: 生成可视化图表
python scripts/common/plot_results.py \
    --summary_csv $RUN_DIR/summary.csv \
    --output_dir $RUN_DIR/plots

# Step 9: 收集系统元数据
python scripts/common/collect_meta.py \
    --output $RUN_DIR/run_meta.json \
    --data_dir data/hotpotqa \
    --hnswlib_dir .

echo "Results saved to $RUN_DIR"
```

### 3. 示例输出

```
=== Building Index ===
Build time: 0.40 sec
Build throughput: 25143 points/sec
Index memory: 31 MB

=== Running Queries ===
ef_search=200
  K=  1 recall=0.9998 p99=660us qps(1t)=1840
  K= 10 recall=0.9967 p99=660us qps(1t)=1840
  K=100 recall=0.9828 p99=660us qps(1t)=1840
```

### 4. 一键运行 (可选)

```bash
bash scripts/hotpotqa/run_all.sh
```

该脚本会依次执行完整流程

### 5. 仅运行 Benchmark（推荐复用数据时使用）

当你已经准备好 `data/hotpotqa/` 下的向量与 Ground Truth（`corpus_vectors.fvecs`、`query_vectors.fvecs`、`ground_truth.ivecs`），并且只想快速对比 HNSW 参数组合时：

```bash
bash scripts/hotpotqa/run_benchmark.sh
```

#### 适用场景

- 已完成数据解析、嵌入生成和 Ground Truth 计算，不需要重复前处理
- 需要批量对比多组 `M` 与 `ef_construction` 参数
- 只关注检索性能评估与绘图产物（而非端到端数据构建流程）

#### 脚本做了什么

1. 自动检查并构建 `benchmark/build/bench_hotpotqa`（若不存在）
2. 遍历多组参数运行 C++ benchmark（默认 `M=8,16,32,48`；`ef_construction=100,200,400`）
3. 将每组原始结果写入 `results/<timestamp>/raw/*.json`
4. 自动执行：
   - `scripts/common/aggregate_results.py`
   - `scripts/common/plot_results.py`
   - `scripts/common/collect_meta.py`
5. 在 `results/<timestamp>/` 下输出 `summary.csv`、`best_configs.json`、图表和元数据

#### 可配置项（编辑脚本顶部）

`scripts/hotpotqa/run_benchmark.sh` 中可直接调整：

- `M_VALUES`：例如 `"8 16 32 48"`
- `EFC_VALUES`：例如 `"100 200 400"`
- `EF_SEARCH_VALUES`：例如 `"10,20,50,100,200,500"`
- `K_VALUES`：例如 `"1,10,100"`
- `NUM_THREADS`：并发线程数

#### 断点续跑特性

脚本会跳过已存在的输出文件（`results_M*_efc*.json`），因此在同一结果目录下可避免重复计算；若希望全新实验，直接重新执行脚本即可生成新的时间戳目录。

## 参数说明

### C++ Benchmark 参数 (`bench_hotpotqa`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base_path` | 文档向量文件 (fvecs) | 必填 |
| `--query_path` | 查询向量文件 (fvecs) | 必填 |
| `--gt_path` | Ground Truth 文件 (ivecs) | 必填 |
| `--output` | 结果输出 JSON 文件 | 必填 |
| `--metric` | 距离度量: `ip` 或 `l2` | `ip` |
| `--M` | HNSW M 参数 | 16 |
| `--ef_construction` | 构建 ef 参数 | 200 |
| `--ef_search` | 搜索 ef 值列表 (逗号分隔) | 10,20,50,100,200,500,1000 |
| `--K` | 返回结果数列表 (逗号分隔) | 1,5,10,20,50,100 |
| `--num_threads` | 最大线程数 | 1 |
| `--warmup` | 预热查询数 | 1000 |
| `--seed` | 随机种子 | 42 |
| `--defer_qps` | 分离延迟/QPS 测量 (`1`=开启) | 开启 |
| `--save_index` | 保存索引到文件 (`1`=开启) | 关闭 |
| `--load_index` | 从文件加载索引 (`1`=开启) | 关闭 |
| `--index_path` | 索引文件路径 (保存/加载用) | 无 |

#### 测量模式说明

**默认模式 (interleaved)**：对每个 `(ef_search, K)` 组合依次执行 warmup → 单线程延迟测量 → 多线程 QPS 测量。

**`--defer_qps 1` 模式 (推荐)**：分两阶段执行：
- Phase 1：所有 `(ef_search, K)` 组合的 warmup + 单线程延迟测量（无多线程干扰）
- Phase 2：所有 `(ef_search, K)` 组合的多线程 QPS 测量

> **为什么推荐 `defer_qps`？**
>
> 默认模式下，多线程 QPS 测试（如 16 线程）会大量驱逐 L3 缓存中的索引数据。
> 当下一个 K 值的延迟测量开始时，如果 warmup 不够充分，尾部查询会因 LLC cache miss 而延迟剧增。
> 在 M=8、ef_construction=400 等稀疏图 + 分散内存布局的配置中尤为严重，
> 可导致 p99 延迟膨胀 4-7 倍（如从 55μs → 230μs）。
> `defer_qps` 模式将所有延迟测量集中在单线程阶段，完全消除此干扰。

#### 索引保存/加载

通过 `--save_index` 和 `--load_index` 可复用同一索引进行对比实验，消除多线程构建的不确定性：

```bash
# 构建并保存索引
./benchmark/build/bench_hotpotqa \
    --base_path data/hotpotqa/corpus_vectors.fvecs \
    --query_path data/hotpotqa/query_vectors.fvecs \
    --gt_path data/hotpotqa/ground_truth.ivecs \
    --M 16 --ef_construction 200 \
    --save_index 1 --index_path /tmp/my_index.bin \
    --output results/build_run.json

# 加载索引并运行（跳过构建，索引完全相同）
./benchmark/build/bench_hotpotqa \
    --base_path data/hotpotqa/corpus_vectors.fvecs \
    --query_path data/hotpotqa/query_vectors.fvecs \
    --gt_path data/hotpotqa/ground_truth.ivecs \
    --load_index 1 --index_path /tmp/my_index.bin \
    --defer_qps 1 \
    --output results/query_run.json
```

> **注意**：多线程 `schedule(dynamic)` 构建导致每次索引拓扑略有不同。
> 需要严格控制变量时（如消融实验），务必使用索引保存/加载。

### HNSW 参数调优建议

- **M**: 邻居数，越大召回率越高但内存越大。推荐 16-64
- **ef_construction**: 构建时的 ef，越大索引质量越好但构建越慢。推荐 200-400
- **ef_search**: 搜索时的 ef，越大召回率越高但延迟越高。推荐 K 的 2-10 倍

## 输出指标

| 指标 | 说明 |
|------|------|
| `recall` | Recall@K，与 Ground Truth 的重合率 |
| `mean_us` | 平均延迟 (微秒) |
| `p50_us` / `p95_us` / `p99_us` | 延迟百分位 |
| `qps_1t` ~ `qps_16t` | 不同线程数下的 QPS |
| `build_time_sec` | 索引构建时间 |
| `build_throughput_pts_per_sec` | 构建吞吐量 |
| `index_memory_mb` | 索引内存占用 |

## 示例结果

以下是在 10K 文档 + 5K 查询子集上的 benchmark 结果：

| ef_search | K | recall | p99 (us) | QPS (1t) |
|-----------|---|--------|----------|----------|
| 10 | 1 | 0.9940 | 423 | 3130 |
| 10 | 10 | 0.9887 | 423 | 3130 |
| 10 | 100 | 0.9412 | 423 | 3130 |
| 100 | 1 | 0.9940 | 370 | 3311 |
| 100 | 10 | 0.9887 | 370 | 3311 |
| 100 | 100 | 0.9412 | 370 | 3311 |
| 200 | 1 | **0.9998** | 660 | 1840 |
| 200 | 10 | **0.9967** | 660 | 1840 |
| 200 | 100 | **0.9828** | 660 | 1840 |

**推荐配置**: M=16, ef_construction=200, ef_search=200 可达到 99.67% recall@10

## 可视化

运行 `scripts/common/plot_results.py` 生成以下图表：

1. **recall_vs_qps.png** - 召回率 vs QPS (Pareto 曲线)
2. **recall_vs_latency_p99.png** - 召回率 vs P99 延迟
3. **ef_search_vs_recall.png** - ef_search vs 召回率
4. **ef_search_vs_latency.png** - ef_search vs 延迟
5. **thread_scalability.png** - 线程扩展性 (QPS vs 线程数)
6. **memory_comparison.png** - 内存占用对比
7. **build_time_comparison.png** - 构建时间对比

## 延迟测量消融实验 (`run_latency_ablation.sh`)

该脚本用于验证不同测量策略对延迟指标（尤其是 p99 尾部延迟）的影响，针对多线程 QPS 测试引起的 L3 缓存污染问题。

### 使用方法

```bash
# 需先准备好 data/ 下的 fvecs/ivecs 文件
bash scripts/hotpotqa/run_latency_ablation.sh
```

### 实验内容

脚本固定 M=8, ef_construction=400（最易触发缓存污染的配置），运行 4 组对比：

| 编号 | 方案 | 说明 |
|------|------|------|
| 0 | **Baseline** | warmup=1000，交替执行延迟/QPS（原始方式） |
| 1 | **Fix 1** | warmup=5000（等于查询总数），增大缓存预热覆盖 |
| 2 | **Fix 2** | `--defer_qps 1`，分离延迟与 QPS 测量 |
| 3 | **Fix 1+2** | 两者结合 |

### 实验输出

结果保存在 `results/ablation_<timestamp>/` 下：

```
results/ablation_YYYYMMDD_HHMMSS/
├── baseline.json          # 原始方式
├── fix1_full_warmup.json  # 增大 warmup
├── fix2_defer_qps.json    # 分离延迟/QPS
├── fix1_2_combined.json   # 两者结合
└── analysis.txt           # 自动对比分析表
```

脚本结束时自动打印 `ef_search=10, K=10` 的 p50/p95/p99/max 对比表。

### 典型结果

在 10K 文档 + 5K 查询子集上：

| 方案 | K=10 p99 (μs) | p99/p50 | 说明 |
|------|--------------|---------|------|
| Baseline | ~230-250 | 6-7x | 多线程 QPS 缓存污染导致尾部膨胀 |
| Fix 1 (warmup=5000) | ~55-63 | 1.7x | 充分预热恢复缓存 |
| Fix 2 (defer_qps) | ~54-57 | 1.7x | 消除污染源，效果最佳 |

### 背景

在默认交替模式下，K=1 的 16 线程 QPS 测试会驱逐 L3 缓存中的索引数据。随后 K=10 的延迟测量（仅 1000 次 warmup）不足以恢复 M=8+efc=400 这种内存布局分散的图结构的缓存状态，导致约 1-5% 的查询因 LLC cache miss 而延迟飙升。

该问题经 `sudo perf stat` 验证：baseline 比 defer_qps 多出 **22.7% 的 LLC-load-misses**（约 259 万次），集中在 K=10 延迟测量阶段。

---

# LoCoMo Benchmark

使用 LoCoMo (Long Conversation Memory) 数据集评估 hnswlib 在长对话记忆检索场景下的性能。

## 数据集特点

- **来源**: [snap-research/locomo](https://github.com/snap-research/locomo) — 长对话记忆 QA 数据集
- **文档粒度**: 每个对话轮次 (dialog turn) 为一个 document
- **查询**: 每个 QA pair 为一个 query
- **Ground Truth**: 通过 QA 中的 evidence 字段 (dia_id) 映射到 doc_id
- **向量模型**: BGE-large-en-v1.5 (1024 维)
- **距离度量**: Inner Product

## 快速开始

### 一键运行

```bash
bash scripts/locomo/run_all.sh
```

该脚本会依次执行：
1. 下载 locomo10.json
2. 解析为标准格式 (corpus.jsonl, queries.jsonl, ground_truth.jsonl)
3. 生成向量嵌入 (复用 hotpotqa 的 generate_embeddings.py)
4. 计算精确 KNN Ground Truth
5. 导出 fvecs/ivecs 格式
6. 运行 C++ Benchmark
7. 汇总结果并生成图表

### 仅运行 Benchmark

```bash
bash scripts/locomo/run_benchmark.sh
```

### 分步运行

```bash
# Step 0: 下载数据
python scripts/locomo/download_data.py --output_dir data/locomo

# Step 1: 解析数据
python scripts/locomo/parse_data.py \
    --input data/locomo/locomo10.json \
    --output_dir data/locomo

# Step 2+: 同 HotpotQA 流程，将路径改为 data/locomo/
```

## 目录结构

```
data/locomo/
├── locomo10.json               # 原始数据
├── corpus.jsonl                # 对话轮次语料
├── queries.jsonl               # QA 查询集合
├── ground_truth.jsonl          # 证据映射
├── stats.json                  # 数据统计
├── corpus_embeddings.npy       # 文档向量
├── query_embeddings.npy        # 查询向量
├── corpus_vectors.fvecs        # C++ 格式文档向量
├── query_vectors.fvecs         # C++ 格式查询向量
└── ground_truth.ivecs          # 精确 KNN 结果
```

---

# SIFT1M Benchmark

SIFT1M 是标准的 ANN (Approximate Nearest Neighbor) benchmark 数据集，常用于评估向量检索算法性能。

## 数据集特点

- **维度**: 128 (SIFT 特征描述符)
- **基向量数**: 1,000,000
- **查询向量数**: 10,000
- **Ground Truth**: 每个查询的 100 个最近邻
- **距离度量**: L2 (欧氏距离)

## 快速开始

### 1. 下载 SIFT1M 数据

```bash
python scripts/sift1m/download_data.py --output_dir data/sift1m
```

这会下载并解压 SIFT1M 数据集到 `data/sift1m/` 目录：
- `sift1m_base.fvecs` - 1M 基向量
- `sift1m_query.fvecs` - 10K 查询向量
- `sift1m_groundtruth.ivecs` - Ground Truth

### 2. 构建 C++ Benchmark

```bash
cd benchmark && mkdir -p build && cd build && cmake .. && make -j$(nproc) bench_sift1m
cd ../..
```

### 3. 运行 Benchmark

```bash
bash scripts/sift1m/run_benchmark.sh
```

该脚本会：
1. 自动下载数据（如果不存在）
2. 编译 benchmark（如果未编译）
3. 遍历多组 HNSW 参数运行测试
4. 汇总结果并生成可视化图表

### 4. 手动运行单次测试

```bash
./benchmark/build/bench_sift1m \
    --base_path data/sift1m/sift1m_base.fvecs \
    --query_path data/sift1m/sift1m_query.fvecs \
    --gt_path data/sift1m/sift1m_groundtruth.ivecs \
    --output results/sift1m_test.json \
    --metric l2 \
    --M 16 \
    --ef_construction 200 \
    --ef_search 10,20,50,100,200,500 \
    --K 1,10,100 \
    --num_threads 64 \
    --defer_qps true
```

## 输出目录结构

```
results/sift1m_YYYYMMDD_HHMMSS/
├── raw/
│   ├── M16_efc100.json
│   ├── M16_efc200.json
│   ├── M16_efc400.json
│   ├── M32_efc100.json
│   └── ...
├── summary.json
├── summary.csv
├── best_configs.json
├── run_meta.json
└── plots/
    ├── recall_vs_qps.png
    ├── recall_vs_latency_p99.png
    ├── ef_search_vs_recall.png
    ├── ef_search_vs_latency.png
    ├── thread_scalability.png
    ├── memory_comparison.png
    └── build_time_comparison.png
```

## SIFT1M vs HotpotQA 对比

| 特性 | SIFT1M | HotpotQA |
|------|--------|----------|
| 维度 | 128 | 1024 |
| 基向量数 | 1,000,000 | 可变 (子集10K) |
| 查询数 | 10,000 | 可变 (子集5K) |
| 距离度量 | L2 | Inner Product |
| 数据来源 | SIFT 特征 | BGE 文本嵌入 |
| 数据预处理 | 无需 | 需要解析 + 嵌入生成 |
| 典型索引大小 | ~600MB (M=16) | ~30MB (10K docs) |

## 预期性能

在 128 核服务器上的典型结果：

| M | ef_construction | Build Time | Index Memory | Recall@10 (ef=100) |
|---|-----------------|------------|--------------|-------------------|
| 16 | 100 | ~30s | ~580MB | ~0.97 |
| 16 | 200 | ~45s | ~580MB | ~0.98 |
| 32 | 200 | ~70s | ~1.1GB | ~0.99 |
| 32 | 400 | ~120s | ~1.1GB | ~0.995 |

---

# 维度消融实验 (Dimensionality Ablation)

该实验通过对 SIFT1M 向量进行零填充 (zero-padding)，将 128 维向量扩展到 {256, 512, 1024} 维，用于隔离 **向量维度** 对 HNSW 性能的影响，排除数据分布差异的干扰。

## 实验原理

**核心思路**：零填充不改变向量间的 L2 距离。

```
L2(pad(x), pad(y)) = sqrt(Σ(xi-yi)² + Σ0²) = L2(x, y)
```

因此：
- **最近邻关系完全不变** → Ground Truth 可直接复用
- **搜索问题难度完全不变** → 图结构遍历行为相同
- **唯一变化的是每次距离计算的 FLOP** → 纯粹隔离计算开销

如果 QPS 与 1/dim 线性相关，则证明 SIFT1M 和 HotpotQA 之间的性能差异主要来自维度计算开销，而非数据分布差异。

## 辅助分析：PCA 本征维度

脚本同时对 SIFT1M 和 HotpotQA 进行 PCA 分析，计算累积方差解释率，揭示两个数据集的本征维度差异。这有助于理解：
- HotpotQA 的 1024 维中有多少是"有效维度"
- 数据分布的"内在复杂度"对搜索难度的贡献

## 目录结构

```
data/sift1m_dim_ablation/
├── dim_256/
│   ├── base.fvecs          # 零填充到 256 维
│   ├── query.fvecs
│   └── groundtruth.ivecs   # 与原始 SIFT1M 相同
├── dim_512/
│   ├── base.fvecs
│   ├── query.fvecs
│   └── groundtruth.ivecs
├── dim_1024/
│   ├── base.fvecs
│   ├── query.fvecs
│   └── groundtruth.ivecs
├── pca_analysis.json       # PCA 本征维度分析
└── ablation_meta.json      # 实验元数据

results/dim_ablation_YYYYMMDD_HHMMSS/
├── raw/*.json              # 每组参数的原始结果
├── summary.csv             # 汇总 (含不同 dim 的行)
├── summary.json
├── best_configs.json
├── scaling_analysis.json   # 观测 vs 理论缩放因子
├── pca_analysis.json       # PCA 分析副本
├── run_meta.json
└── plots/
    ├── qps_vs_dim.png          # QPS 随维度变化
    ├── latency_vs_dim.png      # 延迟随维度变化
    ├── recall_vs_dim.png       # Recall 稳定性 (应为常数)
    ├── build_vs_dim.png        # 构建时间/吞吐随维度变化
    ├── memory_vs_dim.png       # 内存随维度变化
    ├── scaling_analysis.png    # 观测 vs 理论缩放因子
    ├── thread_scalability.png  # 不同维度下的线程扩展性
    └── pca_explained_var.png   # PCA 累积方差解释率
```

## 快速开始

```bash
# 一键运行 (需先准备好 data/sift1m/ 下的 SIFT1M 数据)
bash scripts/dim_ablation/run_ablation.sh
```

## 可配置项（编辑脚本顶部）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TARGET_DIMS` | `256 512 1024` | 零填充目标维度 |
| `M_VALUES` | `16 32` | HNSW M 参数 |
| `EFC_VALUES` | `200` | ef_construction (固定以减少运行时间) |
| `EF_SEARCH_VALUES` | `10,50,100,200,500` | ef_search 列表 |
| `NUM_THREADS` | `64` | 最大线程数 |

## 相关脚本

| 脚本 | 说明 |
|------|------|
| `scripts/dim_ablation/prepare_data.py` | 数据准备：零填充 + PCA 分析 |
| `scripts/dim_ablation/plot_results.py` | 消融实验可视化 |
| `scripts/dim_ablation/run_ablation.sh` | 一键驱动脚本 |

## 预期结果

| 维度 | QPS 缩放 (理论) | 延迟缩放 (理论) | Recall 变化 |
|------|-----------------|-----------------|-------------|
| 128 (基准) | 1.0x | 1.0x | 基准值 |
| 256 | 0.5x | 2.0x | 不变 |
| 512 | 0.25x | 4.0x | 不变 |
| 1024 | 0.125x | 8.0x | 不变 |

> 如果观测到的缩放因子接近理论值，则证明 HotpotQA 与 SIFT1M 的性能差异主要来自维度计算开销。如果观测值偏离理论值，差异部分可能来自 SIMD 效率、缓存行为等微架构因素。

---

## 注意事项

1. **向量归一化**: BGE 模型输出已归一化，使用 Inner Product 等价于余弦相似度
2. **内存需求**: 10K 文档 × 1024 维 × float32 ≈ 40MB，加上索引约 80MB
3. **Ground Truth**: 使用批量矩阵乘法计算，确保 recall 计算准确
4. **多线程**: 使用 OpenMP 并行，确保线程安全的 HNSW 只读操作
5. **延迟测量准确性**: 使用 `--defer_qps 1` 可避免多线程 QPS 测试对延迟指标的缓存污染干扰，建议在需要精确延迟数据时启用
6. **索引可复现性**: 多线程 `schedule(dynamic)` 构建导致每次索引拓扑不同；需要严格控制变量时使用 `--save_index` / `--load_index` 复用同一索引
