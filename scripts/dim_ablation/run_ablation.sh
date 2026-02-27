#!/bin/bash
# ==========================================================================
# Dimensionality Ablation Experiment
# ==========================================================================
# Zero-pads SIFT1M (128-dim) to {256, 512, 1024} dims and benchmarks each
# to isolate the effect of vector dimensionality on HNSW performance.
#
# Key insight: zero-padding preserves L2 distances exactly, so the ANN
# search problem is identical across all dimensions — only the per-distance
# FLOP cost changes.  This cleanly separates "compute cost of higher dims"
# from "search difficulty of different data distributions".
#
# Usage:
#   bash scripts/dim_ablation/run_ablation.sh
#
# Output:
#   results/dim_ablation_<timestamp>/
#     ├── raw/*.json         - per-config benchmark results
#     ├── summary.csv        - aggregated results
#     ├── scaling_analysis.json
#     ├── pca_analysis.json  - PCA intrinsic dimensionality
#     ├── run_meta.json      - system metadata
#     └── plots/*.png        - visualization
# ==========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Configuration — edit these to adjust the experiment
# ---------------------------------------------------------------------------
TARGET_DIMS="256 512 1024"          # Dimensions to zero-pad to
ORIGINAL_DIM=128                    # SIFT1M native dimension

M_VALUES="16 32"                    # HNSW M parameter
EFC_VALUES="200"                    # ef_construction (fixed to reduce runtime)
EF_SEARCH_VALUES="10,50,100,200,500"
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
else
    echo "[NUMA] numactl not found — running without NUMA control"
fi

SIFT1M_DIR="data/sift1m"
ABLATION_DATA_DIR="data/sift1m_dim_ablation"
HOTPOTQA_DIR="data/hotpotqa"

RESULTS_BASE="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="${RESULTS_BASE}/dim_ablation_${TIMESTAMP}"
RAW_DIR="${RESULT_DIR}/raw"
PLOT_DIR="${RESULT_DIR}/plots"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo "================================================================"
echo " Dimensionality Ablation Experiment"
echo "================================================================"
echo " Run ID:      ${TIMESTAMP}"
echo " Dimensions:  ${ORIGINAL_DIM} (original) ${TARGET_DIMS} (zero-padded)"
echo " M values:    ${M_VALUES}"
echo " efc values:  ${EFC_VALUES}"
echo " Results:     ${RESULT_DIR}"
echo " NUMA:        ${NUMA_CMD:-disabled}"
echo " OMP:         OMP_PROC_BIND=spread, OMP_PLACES=cores"
echo "================================================================"
echo ""

mkdir -p "$RAW_DIR" "$PLOT_DIR"

# ---------------------------------------------------------------------------
# Step 0: Build bench_sift1m if needed
# ---------------------------------------------------------------------------
BENCH_BIN="./benchmark/build/bench_sift1m"

if [ ! -f "$BENCH_BIN" ]; then
    echo "=== Building bench_sift1m ==="
    mkdir -p benchmark/build
    cd benchmark/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc) bench_sift1m
    cd "$PROJECT_ROOT"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Prepare zero-padded data + PCA analysis
# ---------------------------------------------------------------------------
TARGET_DIMS_CSV=$(echo "$TARGET_DIMS" | tr ' ' ',')

echo "=== Step 1: Preparing zero-padded data ==="
python scripts/dim_ablation/prepare_data.py \
    --sift1m_dir "$SIFT1M_DIR" \
    --hotpotqa_dir "$HOTPOTQA_DIR" \
    --output_dir "$ABLATION_DATA_DIR" \
    --target_dims "$TARGET_DIMS_CSV"
echo ""

# Copy PCA analysis to results dir
if [ -f "${ABLATION_DATA_DIR}/pca_analysis.json" ]; then
    cp "${ABLATION_DATA_DIR}/pca_analysis.json" "${RESULT_DIR}/pca_analysis.json"
fi

# ---------------------------------------------------------------------------
# Step 2: Run benchmark for each dimension
# ---------------------------------------------------------------------------
echo "=== Step 2: Running benchmarks ==="

ALL_DIMS="${ORIGINAL_DIM} ${TARGET_DIMS}"

for DIM in $ALL_DIMS; do
    # Resolve data paths
    if [ "$DIM" -eq "$ORIGINAL_DIM" ]; then
        BASE_PATH="${SIFT1M_DIR}/sift1m_base.fvecs"
        QUERY_PATH="${SIFT1M_DIR}/sift1m_query.fvecs"
        GT_PATH="${SIFT1M_DIR}/sift1m_groundtruth.ivecs"
    else
        BASE_PATH="${ABLATION_DATA_DIR}/dim_${DIM}/base.fvecs"
        QUERY_PATH="${ABLATION_DATA_DIR}/dim_${DIM}/query.fvecs"
        GT_PATH="${ABLATION_DATA_DIR}/dim_${DIM}/groundtruth.ivecs"
    fi

    # Verify data exists
    if [ ! -f "$BASE_PATH" ]; then
        echo "  ERROR: $BASE_PATH not found, skipping dim=${DIM}"
        continue
    fi

    for M in $M_VALUES; do
        for EFC in $EFC_VALUES; do
            OUTPUT_FILE="${RAW_DIR}/dim${DIM}_M${M}_efc${EFC}.json"

            if [ -f "$OUTPUT_FILE" ]; then
                echo "  Skipping dim=${DIM}, M=${M}, efc=${EFC} (already exists)"
                continue
            fi

            echo ""
            echo "  ======================================"
            echo "  dim=${DIM}, M=${M}, efc=${EFC}"
            echo "  ======================================"

            $NUMA_CMD $BENCH_BIN \
                --base_path "$BASE_PATH" \
                --query_path "$QUERY_PATH" \
                --gt_path "$GT_PATH" \
                --M "$M" \
                --ef_construction "$EFC" \
                --ef_search "$EF_SEARCH_VALUES" \
                --K "$K_VALUES" \
                --num_threads "$NUM_THREADS" \
                --metric l2 \
                --defer_qps true \
                --output "$OUTPUT_FILE"
        done
    done
done

# ---------------------------------------------------------------------------
# Step 3: Aggregate results
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Aggregating results ==="
python scripts/common/aggregate_results.py \
    --input_dir "$RAW_DIR" \
    --output_dir "$RESULT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Generate ablation-specific plots
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Generating plots ==="

PCA_ARG=""
if [ -f "${RESULT_DIR}/pca_analysis.json" ]; then
    PCA_ARG="--pca_json ${RESULT_DIR}/pca_analysis.json"
fi

python scripts/dim_ablation/plot_results.py \
    --summary_csv "${RESULT_DIR}/summary.csv" \
    --output_dir "$PLOT_DIR" \
    $PCA_ARG

# ---------------------------------------------------------------------------
# Step 5: Collect metadata
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Collecting metadata ==="
python scripts/common/collect_meta.py \
    --output "${RESULT_DIR}/run_meta.json" \
    --data_dir "$SIFT1M_DIR" \
    --hnswlib_dir "."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Dimensionality Ablation Complete!"
echo "================================================================"
echo " Results: ${RESULT_DIR}"
echo ""
echo " Generated files:"
echo "   ${RESULT_DIR}/summary.csv"
echo "   ${RESULT_DIR}/summary.json"
echo "   ${RESULT_DIR}/best_configs.json"
echo "   ${RESULT_DIR}/scaling_analysis.json"
echo "   ${RESULT_DIR}/pca_analysis.json"
echo "   ${RESULT_DIR}/run_meta.json"
echo "   ${PLOT_DIR}/*.png"
echo ""
echo " Key plots:"
echo "   scaling_analysis.png  - Observed vs theoretical dim scaling"
echo "   recall_vs_dim.png     - Recall stability (should be flat)"
echo "   qps_vs_dim.png        - QPS degradation with dimension"
echo "   pca_explained_var.png - Intrinsic dimensionality comparison"
echo "================================================================"

# Print quick summary
if [ -f "${RESULT_DIR}/scaling_analysis.json" ]; then
    echo ""
    echo " Quick scaling summary (K=10, ef=100):"
    python3 -c "
import json, sys
data = json.load(open('${RESULT_DIR}/scaling_analysis.json'))
configs = [c for c in data['configs'] if c['ef_search'] == 100]
seen = set()
print(f\"  {'dim':>5} {'M':>3} {'recall':>8} {'QPS_1t':>8} {'observed':>10} {'theoretical':>12}\")
print(f\"  {'---':>5} {'---':>3} {'------':>8} {'------':>8} {'--------':>10} {'-----------':>12}\")
for c in configs:
    key = (c['dim'], c['M'])
    if key in seen: continue
    seen.add(key)
    obs = c.get('observed_qps_factor', '?')
    theo = c.get('theoretical_factor', '?')
    print(f\"  {c['dim']:>5} {c['M']:>3} {c['recall']:>8.4f} {c['qps_1t']:>8.0f} {obs:>10} {theo:>12}\")
" 2>/dev/null || true
fi
