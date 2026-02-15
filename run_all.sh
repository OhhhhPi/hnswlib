#!/bin/bash
# run_all.sh - Complete HotpotQA RAG Benchmark Pipeline
#
# Usage: ./run_all.sh [OPTIONS]
#
# Options:
#   --skip-download    Skip data download step
#   --skip-embedding   Skip embedding generation
#   --skip-build       Skip C++ benchmark build
#   --quick            Run with minimal configurations for testing
#   --help             Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
DATA_DIR="./data"
MODELS_DIR="./models"
RESULTS_BASE="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${RESULTS_BASE}/${TIMESTAMP}"
RAW_DIR="${RESULT_DIR}/raw"
PLOT_DIR="${RESULT_DIR}/plots"

# Default parameters
NUM_WORKERS=32
THREADS_PER_WORKER=4
BATCH_SIZE=64
NUM_THREADS=64
MAX_EF_SEARCH=1000

# Quick mode settings
QUICK_MODE=false
SKIP_DOWNLOAD=false
SKIP_EMBEDDING=false
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-embedding)
            SKIP_EMBEDDING=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            NUM_WORKERS=8
            NUM_THREADS=16
            MAX_EF_SEARCH=200
            shift
            ;;
        --help)
            head -20 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Quick mode configurations
if [ "$QUICK_MODE" = true ]; then
    M_VALUES="16"
    EFC_VALUES="200"
    EF_SEARCH_VALUES="10,50,100,200"
    K_VALUES="1,10"
else
    M_VALUES="8 16 32 48"
    EFC_VALUES="100 200 400"
    EF_SEARCH_VALUES="10,20,50,100,200,500,1000"
    K_VALUES="1,5,10,20,50,100"
fi

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RAW_DIR" "$PLOT_DIR"

echo "=========================================="
echo " HotpotQA → HNSW Benchmark Pipeline"
echo "=========================================="
echo " Run ID: ${TIMESTAMP}"
echo " Results: ${RESULT_DIR}"
echo " Quick Mode: ${QUICK_MODE}"
echo " Workers: ${NUM_WORKERS}, Threads/Worker: ${THREADS_PER_WORKER}"
echo "=========================================="
echo ""

# ---- Phase 0: Environment Setup ----
echo "[Phase 0] Environment Setup..."

if [ "$SKIP_DOWNLOAD" = false ]; then
    if [ ! -f "$DATA_DIR/hotpot_train_v1.1.json" ]; then
        echo "Downloading HotpotQA training set..."
        wget -q --show-progress \
            http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json \
            -O "$DATA_DIR/hotpot_train_v1.1.json"
    else
        echo "Data file already exists: $DATA_DIR/hotpot_train_v1.1.json"
    fi
fi

# ---- Phase 1: Data Parsing ----
echo ""
echo "[Phase 1] Parsing HotpotQA..."
python scripts/01_parse_hotpotqa.py \
    --input "$DATA_DIR/hotpot_train_v1.1.json" \
    --output_dir "$DATA_DIR"

# ---- Phase 2: Embedding Generation ----
if [ "$SKIP_EMBEDDING" = false ]; then
    echo ""
    echo "[Phase 2] Generating embeddings..."
    
    python scripts/02_generate_embeddings.py \
        --corpus "$DATA_DIR/corpus.jsonl" \
        --queries "$DATA_DIR/queries.jsonl" \
        --model_name "BAAI/bge-large-en-v1.5" \
        --output_dir "$DATA_DIR" \
        --num_workers "$NUM_WORKERS" \
        --threads_per_worker "$THREADS_PER_WORKER" \
        --batch_size "$BATCH_SIZE"
    
    echo ""
    echo "[Phase 2.5] Computing exact KNN ground truth..."
    python scripts/03_compute_knn_gt.py \
        --corpus_emb "$DATA_DIR/corpus_embeddings.npy" \
        --query_emb "$DATA_DIR/query_embeddings.npy" \
        --output_dir "$DATA_DIR" \
        --top_k 100 \
        --batch_size 1000
    
    echo ""
    echo "[Phase 2.6] Exporting fvecs/ivecs..."
    python scripts/04_export_vecs.py \
        --corpus_emb "$DATA_DIR/corpus_embeddings.npy" \
        --query_emb "$DATA_DIR/query_embeddings.npy" \
        --gt "$DATA_DIR/knn_gt_indices.npy" \
        --output_dir "$DATA_DIR"
else
    echo "[Phase 2] Skipping embedding generation (--skip-embedding)"
fi

# ---- Phase 3: C++ Benchmark ----
echo ""
echo "[Phase 3] C++ Benchmark..."

BENCH_BIN="./benchmark/build/bench_hotpotqa"

if [ "$SKIP_BUILD" = false ]; then
    echo "Building C++ benchmark..."
    mkdir -p benchmark/build
    cd benchmark/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "$SCRIPT_DIR"
fi

if [ ! -f "$BENCH_BIN" ]; then
    echo "Error: Benchmark binary not found at $BENCH_BIN"
    exit 1
fi

echo "Running benchmarks..."
for M in $M_VALUES; do
    for EFC in $EFC_VALUES; do
        OUTPUT_FILE="${RAW_DIR}/results_M${M}_efc${EFC}.json"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping M=${M}, efc=${EFC} (already exists)"
            continue
        fi
        
        echo "  Running M=${M}, ef_construction=${EFC}..."
        
        # Limit ef_search values based on MAX_EF_SEARCH
        if [ "$MAX_EF_SEARCH" -lt 1000 ]; then
            EF_SEARCH_ARG=$(echo "$EF_SEARCH_VALUES" | tr ',' '\n' | awk -v max="$MAX_EF_SEARCH" '$1 <= max {printf "%s%s", sep, $1; sep=","}')
        else
            EF_SEARCH_ARG="$EF_SEARCH_VALUES"
        fi
        
        $BENCH_BIN \
            --base_path "$DATA_DIR/corpus_vectors.fvecs" \
            --query_path "$DATA_DIR/query_vectors.fvecs" \
            --gt_path "$DATA_DIR/ground_truth.ivecs" \
            --M "$M" \
            --ef_construction "$EFC" \
            --ef_search "$EF_SEARCH_ARG" \
            --K "$K_VALUES" \
            --num_threads "$NUM_THREADS" \
            --metric ip \
            --output "$OUTPUT_FILE"
    done
done

# ---- Phase 4: Results Aggregation ----
echo ""
echo "[Phase 4] Aggregating results..."
python scripts/06_aggregate_results.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$RESULT_DIR"

echo ""
echo "[Phase 4] Generating plots..."
python scripts/07_plot_results.py \
    --summary_csv "${RESULT_DIR}/summary.csv" \
    --output_dir "$PLOT_DIR"

echo ""
echo "[Phase 4] Collecting run metadata..."
python scripts/08_collect_meta.py \
    --output "${RESULT_DIR}/run_meta.json" \
    --data_dir "$DATA_DIR" \
    --hnswlib_dir "."

# ---- Summary ----
echo ""
echo "=========================================="
echo " Benchmark Complete!"
echo "=========================================="
echo " Results saved to: ${RESULT_DIR}"
echo ""
echo " Files generated:"
echo "   - ${RESULT_DIR}/summary.json"
echo "   - ${RESULT_DIR}/summary.csv"
echo "   - ${RESULT_DIR}/best_configs.json"
echo "   - ${RESULT_DIR}/run_meta.json"
echo "   - ${PLOT_DIR}/*.png"
echo "=========================================="

# Print best configurations
if [ -f "${RESULT_DIR}/best_configs.json" ]; then
    echo ""
    echo "=== Best Configurations ==="
    python -c "
import json
with open('${RESULT_DIR}/best_configs.json') as f:
    best = json.load(f)
for target, cfg in best.items():
    print(f'\n{target}:')
    print(f'  M={cfg.get(\"M\", \"?\")}, efc={cfg.get(\"ef_construction\", \"?\")}, efs={cfg.get(\"ef_search\", \"?\")}')
    print(f'  recall@{cfg.get(\"K\", \"?\")}={cfg.get(\"recall\", 0):.4f}')
    print(f'  p99={cfg.get(\"p99_us\", 0):.1f}us')
    qps = cfg.get('qps_1t', 0)
    if qps:
        print(f'  QPS(1t)={qps:.0f}')
"
fi

echo ""
echo "To view plots: ls ${PLOT_DIR}/"
echo "To analyze: python -c \"import pandas as pd; df = pd.read_csv('${RESULT_DIR}/summary.csv'); print(df.head())\""
