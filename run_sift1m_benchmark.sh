#!/bin/bash
# Run SIFT1M benchmark with multiple HNSW configurations.
# This script assumes SIFT1M data is already downloaded to data/sift1m/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

M_VALUES="16 32 48"
EFC_VALUES="100 200 400"
EF_SEARCH_VALUES="10,20,50,100,200,500"
K_VALUES="1,10,100"
NUM_THREADS=64

DATA_DIR="data/sift1m"
BASE_PATH="$DATA_DIR/sift1m_base.fvecs"
QUERY_PATH="$DATA_DIR/sift1m_query.fvecs"
GT_PATH="$DATA_DIR/sift1m_groundtruth.ivecs"

SIFT1M_URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

check_data() {
    if [[ ! -f "$BASE_PATH" || ! -f "$QUERY_PATH" || ! -f "$GT_PATH" ]]; then
        echo "SIFT1M data not found. Downloading..."
        python scripts/00_download_sift1m.py --output_dir "$DATA_DIR"
    fi
}

build_benchmark() {
    if [[ ! -f "benchmark/build/bench_sift1m" ]]; then
        echo "Building bench_sift1m..."
        mkdir -p benchmark/build
        cd benchmark/build
        cmake ..
        make -j$(nproc) bench_sift1m
        cd ../..
    fi
}

run_benchmark() {
    local M=$1
    local EFC=$2
    local OUTPUT_FILE=$3
    
    echo "Running: M=$M, ef_construction=$EFC"
    
    ./benchmark/build/bench_sift1m \
        --base_path "$BASE_PATH" \
        --query_path "$QUERY_PATH" \
        --gt_path "$GT_PATH" \
        --output "$OUTPUT_FILE" \
        --metric l2 \
        --M "$M" \
        --ef_construction "$EFC" \
        --ef_search "$EF_SEARCH_VALUES" \
        --K "$K_VALUES" \
        --num_threads "$NUM_THREADS" \
        --defer_qps true
}

main() {
    echo "=== SIFT1M HNSW Benchmark ==="
    echo ""
    
    check_data
    build_benchmark
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_DIR="results/sift1m_${TIMESTAMP}"
    mkdir -p "$RUN_DIR/raw"
    
    echo "Results will be saved to: $RUN_DIR"
    echo ""
    
    for M in $M_VALUES; do
        for EFC in $EFC_VALUES; do
            OUTPUT_FILE="$RUN_DIR/raw/M${M}_efc${EFC}.json"
            run_benchmark "$M" "$EFC" "$OUTPUT_FILE"
            echo ""
        done
    done
    
    echo "=== Aggregating Results ==="
    python scripts/06_aggregate_results.py \
        --input_dir "$RUN_DIR/raw" \
        --output_dir "$RUN_DIR"
    
    echo "=== Generating Plots ==="
    python scripts/07_plot_results.py \
        --summary_csv "$RUN_DIR/summary.csv" \
        --output_dir "$RUN_DIR/plots"
    
    echo "=== Collecting Metadata ==="
    python scripts/08_collect_meta.py \
        --output "$RUN_DIR/run_meta.json" \
        --data_dir "$DATA_DIR" \
        --hnswlib_dir .
    
    echo ""
    echo "=== Benchmark Complete ==="
    echo "Results saved to: $RUN_DIR"
    echo ""
    echo "Summary:"
    head -20 "$RUN_DIR/summary.csv" 2>/dev/null || true
    
    if [[ -f "$RUN_DIR/best_configs.json" ]]; then
        echo ""
        echo "Best configurations:"
        python -c "import json; data=json.load(open('$RUN_DIR/best_configs.json')); [print(f'  {k}: M={v[\"M\"]}, efc={v[\"ef_construction\"]}, efs={v[\"ef_search\"]}, recall={v[\"recall\"]:.4f}') for k,v in data.items() if v]" 2>/dev/null || true
    fi
}

main "$@"
