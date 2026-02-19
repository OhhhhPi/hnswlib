#!/bin/bash
# Run only the C++ benchmark part with multiple M configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="./data"
RESULTS_BASE="./results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${RESULTS_BASE}/${TIMESTAMP}"
RAW_DIR="${RESULT_DIR}/raw"
PLOT_DIR="${RESULT_DIR}/plots"

# Configuration - adjust as needed
M_VALUES="8 16 32 48"
EFC_VALUES="100 200 400"
EF_SEARCH_VALUES="50,100,200,500"
K_VALUES="1,10,100"
NUM_THREADS=64

# ---------------------------------------------------------------------------
# NUMA policy — reduces cross-node memory latency on multi-socket systems.
#   --interleave=all: stripe index memory across all NUMA nodes so that
#   threads on any node have roughly equal average access latency.
# OMP_PROC_BIND=spread + OMP_PLACES=cores: evenly distribute OpenMP threads
#   across all cores/sockets to avoid oversubscribing a single node.
# ---------------------------------------------------------------------------
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

NUMA_CMD=""
if command -v numactl &>/dev/null; then
    NUMA_CMD="numactl --interleave=all"
    echo "[NUMA] Using: $NUMA_CMD"
    echo "[NUMA] Detected $(numactl --hardware 2>/dev/null | head -1)"
else
    echo "[NUMA] numactl not found — running without NUMA control"
fi

mkdir -p "$RAW_DIR" "$PLOT_DIR"

echo "=========================================="
echo " HNSW Benchmark - Multiple M Configurations"
echo "=========================================="
echo " Run ID: ${TIMESTAMP}"
echo " Results: ${RESULT_DIR}"
echo " M values: ${M_VALUES}"
echo " ef_construction values: ${EFC_VALUES}"
echo "=========================================="
echo ""

BENCH_BIN="./benchmark/build/bench_hotpotqa"

if [ ! -f "$BENCH_BIN" ]; then
    echo "Building C++ benchmark first..."
    mkdir -p benchmark/build
    cd benchmark/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "$PROJECT_ROOT"
fi

echo "Running benchmarks..."
for M in $M_VALUES; do
    for EFC in $EFC_VALUES; do
        OUTPUT_FILE="${RAW_DIR}/results_M${M}_efc${EFC}.json"
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo "  Skipping M=${M}, efc=${EFC} (already exists)"
            continue
        fi
        
        echo ""
        echo "  ======================================"
        echo "  Running: M=${M}, ef_construction=${EFC}"
        echo "  ======================================"
        
        $NUMA_CMD $BENCH_BIN \
            --base_path "$DATA_DIR/corpus_vectors.fvecs" \
            --query_path "$DATA_DIR/query_vectors.fvecs" \
            --gt_path "$DATA_DIR/ground_truth.ivecs" \
            --M "$M" \
            --ef_construction "$EFC" \
            --ef_search "$EF_SEARCH_VALUES" \
            --K "$K_VALUES" \
            --num_threads "$NUM_THREADS" \
            --metric ip \
            --output "$OUTPUT_FILE"
    done
done

echo ""
echo "=========================================="
echo " Aggregating results..."
echo "=========================================="
python scripts/common/aggregate_results.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$RESULT_DIR"

echo ""
echo "=========================================="
echo " Generating plots..."
echo "=========================================="
python scripts/common/plot_results.py \
    --summary_csv "${RESULT_DIR}/summary.csv" \
    --output_dir "$PLOT_DIR"

echo ""
echo "=========================================="
echo " Collecting metadata..."
echo "=========================================="
python scripts/common/collect_meta.py \
    --output "${RESULT_DIR}/run_meta.json" \
    --data_dir "$DATA_DIR" \
    --hnswlib_dir "."

echo ""
echo "=========================================="
echo " Benchmark Complete!"
echo "=========================================="
echo " Results saved to: ${RESULT_DIR}"
echo ""
echo " Generated files:"
echo "   - ${RESULT_DIR}/summary.json"
echo "   - ${RESULT_DIR}/summary.csv"
echo "   - ${RESULT_DIR}/best_configs.json"
echo "   - ${RESULT_DIR}/run_meta.json"
echo "   - ${PLOT_DIR}/*.png"
echo "=========================================="
