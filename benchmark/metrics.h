/**
 * @file metrics.h
 * @brief Recall and other evaluation metrics for ANN search.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

namespace bench {

/**
 * @brief Compute recall@K for a single query.
 * 
 * @param ann_results Array of K ANN result IDs
 * @param gt_results Array of K ground truth IDs
 * @param k Number of results to compare
 * @return float Recall = |intersection| / K
 */
inline float compute_recall_single(const int* ann_results, const int* gt_results, int k) {
    std::set<int> gt_set(gt_results, gt_results + k);
    
    int hits = 0;
    for (int i = 0; i < k; i++) {
        if (gt_set.count(ann_results[i]) > 0) {
            hits++;
        }
    }
    
    return static_cast<float>(hits) / k;
}

/**
 * @brief Compute recall@K for a single query with variable gt_k.
 * 
 * @param ann_results Array of K ANN result IDs
 * @param gt_results Array of gt_k ground truth IDs (gt_k >= K)
 * @param k Number of ANN results
 * @param gt_k Number of ground truth results (should be >= k)
 * @return float Recall = |intersection| / K
 */
inline float compute_recall_single(
    const int* ann_results, 
    const int* gt_results, 
    int k, 
    int gt_k
) {
    std::set<int> gt_set(gt_results, gt_results + std::min(k, gt_k));
    
    int hits = 0;
    for (int i = 0; i < k; i++) {
        if (gt_set.count(ann_results[i]) > 0) {
            hits++;
        }
    }
    
    return static_cast<float>(hits) / k;
}

/**
 * @brief Compute average recall@K over all queries.
 * 
 * @param ann_results Flat array of ANN results [num_queries * k]
 * @param gt_results Flat array of ground truth [num_queries * gt_k]
 * @param num_queries Number of queries
 * @param k Number of results per query (for ANN)
 * @param gt_k Number of ground truth results per query (should be >= k)
 * @return float Average recall
 */
inline float compute_average_recall(
    const int* ann_results,
    const int* gt_results,
    int num_queries,
    int k,
    int gt_k,
    int ann_stride = -1
) {
    int stride = (ann_stride < 0) ? k : ann_stride;
    float total_recall = 0.0f;
    
    for (int i = 0; i < num_queries; i++) {
        const int* ann = ann_results + i * stride;
        const int* gt = gt_results + i * gt_k;
        total_recall += compute_recall_single(ann, gt, k, gt_k);
    }
    
    return total_recall / num_queries;
}

/**
 * @brief Compute recall@K for each K in a list.
 * 
 * @param ann_results Flat array of ANN results [num_queries * max_k]
 * @param gt_results Flat array of ground truth [num_queries * gt_k]
 * @param num_queries Number of queries
 * @param max_k Maximum K to compute
 * @param gt_k Number of ground truth results per query
 * @param k_values Vector of K values to compute recall for
 * @return std::vector<float> Recall values for each K
 */
inline std::vector<float> compute_recall_at_k(
    const int* ann_results,
    const int* gt_results,
    int num_queries,
    int max_k,
    int gt_k,
    const std::vector<int>& k_values
) {
    std::vector<float> recalls;
    recalls.reserve(k_values.size());
    
    for (int k : k_values) {
        if (k > max_k) {
            recalls.push_back(-1.0f);
            continue;
        }
        
        float recall = compute_average_recall(ann_results, gt_results, num_queries, k, gt_k);
        recalls.push_back(recall);
    }
    
    return recalls;
}

/**
 * @brief Metrics structure for a single benchmark run.
 */
struct BenchmarkMetrics {
    float recall;
    double mean_latency_us;
    double p50_latency_us;
    double p95_latency_us;
    double p99_latency_us;
    double max_latency_us;
    double qps;
    
    void print() const {
        std::cout << "Recall: " << recall 
                  << ", Mean latency: " << mean_latency_us << " us"
                  << ", P99: " << p99_latency_us << " us"
                  << ", QPS: " << qps << std::endl;
    }
};

} // namespace bench
