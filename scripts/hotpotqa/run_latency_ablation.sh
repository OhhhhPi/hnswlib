#!/bin/bash
# Ablation test: verify 3 fixes for M=8,efc=400 p99 outlier
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="./data/hotpotqa"
BENCH_BIN="./benchmark/build/bench_hotpotqa"
RESULT_DIR="./results/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

COMMON_ARGS="--base_path $DATA_DIR/corpus_vectors.fvecs \
  --query_path $DATA_DIR/query_vectors.fvecs \
  --gt_path $DATA_DIR/ground_truth.ivecs \
  --M 8 --ef_construction 400 \
  --ef_search 10,20,50 \
  --K 1,10,100 \
  --num_threads 16 \
  --metric ip"

echo "=============================================="
echo " Ablation Test: M=8, efc=400 p99 outlier fix"
echo " Results: $RESULT_DIR"
echo "=============================================="

# --- Test 0: Baseline (original: warmup=1000, interleaved QPS) ---
echo ""
echo ">>> [0/3] BASELINE: warmup=1000, interleaved QPS"
$BENCH_BIN $COMMON_ARGS \
  --warmup 1000 \
  --output "$RESULT_DIR/baseline.json"

# --- Test 1: Fix 1 - Full warmup (warmup=5000 = num_query) ---
echo ""
echo ">>> [1/3] FIX 1: warmup=5000 (full query set)"
$BENCH_BIN $COMMON_ARGS \
  --warmup 5000 \
  --output "$RESULT_DIR/fix1_full_warmup.json"

# --- Test 2: Fix 2 - Deferred QPS (latency first, then QPS) ---
echo ""
echo ">>> [2/3] FIX 2: defer_qps (latency-first, no cache pollution)"
$BENCH_BIN $COMMON_ARGS \
  --warmup 1000 \
  --defer_qps 1 \
  --output "$RESULT_DIR/fix2_defer_qps.json"

# --- Test 3: Fix 1+2 combined ---
echo ""
echo ">>> [3/3] FIX 1+2: warmup=5000 + defer_qps"
$BENCH_BIN $COMMON_ARGS \
  --warmup 5000 \
  --defer_qps 1 \
  --output "$RESULT_DIR/fix1_2_combined.json"

echo ""
echo "=============================================="
echo " All tests complete. Analyzing results..."
echo "=============================================="

# --- Analysis ---
python3 -c "
import json, sys

files = {
    'baseline':       '$RESULT_DIR/baseline.json',
    'fix1_warmup':    '$RESULT_DIR/fix1_full_warmup.json',
    'fix2_defer_qps': '$RESULT_DIR/fix2_defer_qps.json',
    'fix1+2_combined':'$RESULT_DIR/fix1_2_combined.json',
}

print()
print('=' * 100)
print('COMPARISON: ef_search=10, K=10 (the outlier point)')
print('=' * 100)
print(f'{\"Variant\":>20} | {\"p50\":>8} {\"p95\":>8} {\"p99\":>8} {\"max\":>10} | {\"p99/p50\":>8} {\"mean\":>8} {\"recall\":>8}')
print('-' * 100)

for name, fpath in files.items():
    with open(fpath) as f:
        data = json.load(f)
    for r in data['results']:
        if int(r['ef_search']) == 10 and int(r['K']) == 10:
            p50 = float(r['p50_us'])
            p95 = float(r['p95_us'])
            p99 = float(r['p99_us'])
            mx  = float(r['max_us'])
            mean = float(r['mean_us'])
            recall = float(r['recall'])
            ratio = p99 / p50
            print(f'{name:>20} | {p50:>8.1f} {p95:>8.1f} {p99:>8.1f} {mx:>10.1f} | {ratio:>8.1f} {mean:>8.1f} {recall:>8.4f}')

print()
print('=' * 100)
print('FULL TABLE: ef_search=10, all K values')
print('=' * 100)
for name, fpath in files.items():
    with open(fpath) as f:
        data = json.load(f)
    print(f'\n--- {name} ---')
    print(f'{\"ef_s\":>5} {\"K\":>4} | {\"p50\":>8} {\"p95\":>8} {\"p99\":>8} {\"max\":>10} | {\"p99/p50\":>8} {\"mean\":>8}')
    for r in data['results']:
        if int(r['ef_search']) == 10:
            K = int(r['K'])
            p50 = float(r['p50_us'])
            p95 = float(r['p95_us'])
            p99 = float(r['p99_us'])
            mx  = float(r['max_us'])
            mean = float(r['mean_us'])
            ratio = p99 / p50
            print(f'{10:>5} {K:>4} | {p50:>8.1f} {p95:>8.1f} {p99:>8.1f} {mx:>10.1f} | {ratio:>8.1f} {mean:>8.1f}')
print()
" | tee "$RESULT_DIR/analysis.txt"

echo "Results saved to $RESULT_DIR/"
