# Goal
Build a complete benchmark pipeline to evaluate hnswlib vector search performance using the HotpotQA dataset. The pipeline should:
1. Parse HotpotQA training data into a corpus/query format
2. Generate embeddings using BAAI/bge-large-en-v1.5 (1024 dimensions)
3. Run hnswlib C++ benchmark to measure latency, throughput, memory, and recall
4. Generate visualization plots and aggregate results

# Instructions
- The original detailed plan is in the first message of the conversation - it specifies a 5-phase implementation
- Files should be organized inside the hnswlib repository at /sharenvme/usershome/lqa/hnswlib/
- Use 32 ONNX workers with 4 threads each for embedding generation (on a 128-core, 1.5TB RAM server)
- Due to network proxy issues, the BGE model was downloaded from ModelScope (China) instead of HuggingFace
- User chose to use a subset (10K docs, 5K queries) for faster testing instead of the full dataset

# Discoveries
1. Network issues: HuggingFace is blocked by proxy; use ModelScope (Xorbits/bge-large-en-v1.5) instead
2. CPU-only embedding is slow: ~20-30s per batch of 128 texts; full corpus would take 30+ hours
3. Multiprocessing issues: The original multi-worker embedding script had issues; created a simpler sequential version
4. Matplotlib style: seaborn-v0_8-whitegrid not available; use seaborn-whitegrid
5. Pandas/matplotlib compatibility: Need to use .values when passing pandas Series to matplotlib

# Accomplished
## Completed:
- ✅ Created directory structure (scripts/, benchmark/, data/, models/, results/)
- ✅ Phase 1: 01_parse_hotpotqa.py - Parse HotpotQA JSON to corpus/queries/ground_truth
- ✅ Phase 2: 02_generate_embeddings_simple.py - Sequential embedding generation (simpler version)
- ✅ Phase 2: 03_compute_knn_gt.py - Compute exact KNN ground truth
- ✅ Phase 2: 04_export_vecs.py - Export to fvecs/ivecs format
- ✅ Phase 3: C++ benchmark (io_utils.h, timer.h, memory_utils.h, metrics.h, bench_hotpotqa.cpp, CMakeLists.txt)
- ✅ Phase 4: 06_aggregate_results.py - Aggregate JSON results
- ✅ Phase 4: 07_plot_results.py - Generate 7 visualization plots
- ✅ Phase 4: 08_collect_meta.py - Collect system metadata
- ✅ run_all.sh - Master script
- ✅ Downloaded BGE model from ModelScope to models/Xorbits/bge-large-en-v1.5/
- ✅ Successfully ran benchmark on subset (10K docs, 5K queries)

## Benchmark Results (subset):
- Build time: 0.39 sec, 25,588 pts/sec, 32 MB memory
- Best config: M=16, ef_construction=200, ef_search=200 → 98.3% recall@100
- QPS: 3,290 (1t) to 45,590 (16t) with 13.3x speedup

# Relevant files / directories
/sharenvme/usershome/lqa/hnswlib/
├── scripts/
│   ├── 01_parse_hotpotqa.py      # Parse HotpotQA JSON
│   ├── 02_generate_embeddings.py # Multi-process version (has issues)
│   ├── 02_generate_embeddings_simple.py  # Working sequential version
│   ├── 03_compute_knn_gt.py      # Compute exact KNN ground truth
│   ├── 04_export_vecs.py         # Export to fvecs/ivecs
│   ├── 06_aggregate_results.py   # Aggregate benchmark JSONs
│   ├── 07_plot_results.py        # Generate 7 plots
│   └── 08_collect_meta.py        # Collect system metadata
├── benchmark/
│   ├── io_utils.h                # fvecs/ivecs reader
│   ├── timer.h                   # High-precision timing
│   ├── memory_utils.h            # /proc/self/status memory
│   ├── metrics.h                 # Recall computation
│   ├── bench_hotpotqa.cpp        # Main benchmark program
│   ├── CMakeLists.txt            # Build config
│   └── build/bench_hotpotqa      # Compiled binary
├── data/
│   ├── hotpot_train_v1.1.json    # Original data (541MB)
│   ├── corpus.jsonl              # Full corpus (482K docs)
│   ├── queries.jsonl             # Full queries (90K)
│   ├── corpus_subset.jsonl       # Subset (10K docs)
│   ├── queries_subset.jsonl      # Subset (5K queries)
│   ├── corpus_embeddings.npy     # Generated embeddings
│   ├── query_embeddings.npy
│   ├── corpus_vectors.fvecs      # C++ format
│   ├── query_vectors.fvecs
│   └── ground_truth.ivecs
├── models/
│   └── Xorbits/bge-large-en-v1.5/  # Downloaded BGE model
├── results/
│   └── 20260215_074324/          # Benchmark results
│       ├── raw/*.json
│       ├── summary.json / summary.csv
│       ├── best_configs.json
│       ├── run_meta.json
│       └── plots/*.png           # 7 visualization plots
└── run_all.sh                    # Master run script
