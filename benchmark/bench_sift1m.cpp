/**
 * @file bench_sift1m.cpp
 * @brief Benchmark program for SIFT1M dataset using hnswlib.
 *
 * Measures index construction, search latency, throughput, and recall.
 * SIFT1M: 1M base vectors, 10K queries, 128 dimensions, L2 distance.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../hnswlib/hnswlib.h"
#include "io_utils.h"
#include "memory_utils.h"
#include "metrics.h"
#include "timer.h"

struct Config {
  std::string base_path;
  std::string query_path;
  std::string gt_path;
  std::string index_path;
  std::string output_path;
  std::string metric;
  int M;
  int ef_construction;
  std::vector<int> ef_search_values;
  std::vector<int> K_values;
  int num_threads;
  bool save_index;
  bool load_index;
  int seed;
  int warmup_queries;
  bool defer_qps;
};

std::vector<int> parse_int_list(const std::string &s) {
  std::vector<int> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(std::stoi(item));
  }
  return result;
}

Config parse_args(int argc, char **argv) {
  Config cfg;
  cfg.M = 16;
  cfg.ef_construction = 200;
  cfg.metric = "l2";
  cfg.num_threads = 16;
  cfg.save_index = false;
  cfg.load_index = false;
  cfg.seed = 42;
  cfg.warmup_queries = 1000;
  cfg.defer_qps = true;

  for (int i = 1; i < argc; i += 2) {
    std::string key = argv[i];
    std::string value = argv[i + 1];

    if (key == "--base_path")
      cfg.base_path = value;
    else if (key == "--query_path")
      cfg.query_path = value;
    else if (key == "--gt_path")
      cfg.gt_path = value;
    else if (key == "--index_path")
      cfg.index_path = value;
    else if (key == "--output")
      cfg.output_path = value;
    else if (key == "--metric")
      cfg.metric = value;
    else if (key == "--M")
      cfg.M = std::stoi(value);
    else if (key == "--ef_construction")
      cfg.ef_construction = std::stoi(value);
    else if (key == "--ef_search")
      cfg.ef_search_values = parse_int_list(value);
    else if (key == "--K")
      cfg.K_values = parse_int_list(value);
    else if (key == "--num_threads")
      cfg.num_threads = std::stoi(value);
    else if (key == "--save_index")
      cfg.save_index = (value == "true" || value == "1");
    else if (key == "--load_index")
      cfg.load_index = (value == "true" || value == "1");
    else if (key == "--seed")
      cfg.seed = std::stoi(value);
    else if (key == "--warmup")
      cfg.warmup_queries = std::stoi(value);
    else if (key == "--defer_qps")
      cfg.defer_qps = (value == "true" || value == "1");
  }

  if (cfg.ef_search_values.empty()) {
    cfg.ef_search_values = {10, 20, 50, 100, 200, 500, 1000};
  }
  if (cfg.K_values.empty()) {
    cfg.K_values = {1, 5, 10, 20, 50, 100};
  }

  return cfg;
}

void write_json_result(
    const std::string &filename, const Config &cfg, int num_base, int num_query,
    int dim, double build_time_sec, double build_throughput,
    size_t index_memory_kb, size_t data_memory_kb,
    const std::vector<std::map<std::string, std::string>> &results) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Error: Cannot write to " << filename << std::endl;
    return;
  }

  out << "{\n";
  out << "  \"dataset\": \"SIFT1M\",\n";
  out << "  \"embedding_model\": \"SIFT-128\",\n";
  out << "  \"dim\": " << dim << ",\n";
  out << "  \"num_base\": " << num_base << ",\n";
  out << "  \"num_query\": " << num_query << ",\n";
  out << "  \"index\": {\n";
  out << "    \"type\": \"HNSW\",\n";
  out << "    \"M\": " << cfg.M << ",\n";
  out << "    \"ef_construction\": " << cfg.ef_construction << ",\n";
  out << "    \"build_time_sec\": " << std::fixed << std::setprecision(2)
      << build_time_sec << ",\n";
  out << "    \"build_throughput_pts_per_sec\": " << std::fixed
      << std::setprecision(0) << build_throughput << ",\n";
  out << "    \"index_memory_mb\": " << std::fixed << std::setprecision(2)
      << (index_memory_kb / 1024.0) << ",\n";
  out << "    \"data_memory_mb\": " << std::fixed << std::setprecision(2)
      << (data_memory_kb / 1024.0) << "\n";
  out << "  },\n";
  out << "  \"results\": [\n";

  for (size_t i = 0; i < results.size(); i++) {
    out << "    {\n";
    size_t j = 0;
    for (const auto &[k, v] : results[i]) {
      out << "      \"" << k << "\": " << v;
      if (j < results[i].size() - 1)
        out << ",";
      out << "\n";
      j++;
    }
    out << "    }";
    if (i < results.size() - 1)
      out << ",";
    out << "\n";
  }

  out << "  ]\n";
  out << "}\n";

  out.close();
  std::cout << "Results written to " << filename << std::endl;
}

int main(int argc, char **argv) {
  std::cout << "=== SIFT1M HNSW Benchmark ===" << std::endl;

  Config cfg = parse_args(argc, argv);

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Base vectors: " << cfg.base_path << std::endl;
  std::cout << "  Query vectors: " << cfg.query_path << std::endl;
  std::cout << "  Ground truth: " << cfg.gt_path << std::endl;
  std::cout << "  M: " << cfg.M << std::endl;
  std::cout << "  ef_construction: " << cfg.ef_construction << std::endl;
  std::cout << "  ef_search: ";
  for (int ef : cfg.ef_search_values)
    std::cout << ef << " ";
  std::cout << std::endl;
  std::cout << "  K: ";
  for (int k : cfg.K_values)
    std::cout << k << " ";
  std::cout << std::endl;
  std::cout << "  Threads: " << cfg.num_threads << std::endl;
  std::cout << "  Metric: " << cfg.metric << std::endl;

  bench::MemoryTracker mem_tracker;
  mem_tracker.snapshot("start");

  std::cout << "\n=== Loading Data ===" << std::endl;
  int num_base, dim;
  float *base_data = bench::read_fvecs_flat(cfg.base_path, num_base, dim);
  mem_tracker.snapshot("after_load_base");

  int num_query, query_dim;
  float *query_data =
      bench::read_fvecs_flat(cfg.query_path, num_query, query_dim);
  mem_tracker.snapshot("after_load_query");

  if (query_dim != dim) {
    std::cerr << "Error: Dimension mismatch" << std::endl;
    return 1;
  }

  int num_gt, gt_k;
  int *gt_data = bench::read_ivecs_flat(cfg.gt_path, num_gt, gt_k);
  mem_tracker.snapshot("after_load_gt");

  if (num_gt != num_query) {
    std::cerr << "Warning: GT count mismatch with query count" << std::endl;
  }

  std::cout << "\nData summary:" << std::endl;
  std::cout << "  Base vectors: " << num_base << std::endl;
  std::cout << "  Query vectors: " << num_query << std::endl;
  std::cout << "  Dimension: " << dim << std::endl;
  std::cout << "  GT K: " << gt_k << std::endl;

  size_t data_memory_kb =
      (static_cast<size_t>(num_base) * dim * sizeof(float)) / 1024;

  hnswlib::SpaceInterface<float> *space;
  if (cfg.metric == "l2") {
    space = new hnswlib::L2Space(dim);
  } else {
    space = new hnswlib::InnerProductSpace(dim);
  }

  hnswlib::HierarchicalNSW<float> *index;
  double build_time_sec = 0;
  double build_throughput = 0;
  size_t index_memory_kb = 0;

  if (cfg.load_index && !cfg.index_path.empty()) {
    std::cout << "\n=== Loading Index from " << cfg.index_path
              << " ===" << std::endl;
    bench::Timer load_timer;
    index = new hnswlib::HierarchicalNSW<float>(space, cfg.index_path);
    double load_sec = load_timer.elapsed_sec();
    std::cout << "Index loaded in " << load_sec << " sec" << std::endl;
    std::cout << "Index max_elements: " << index->max_elements_ << std::endl;
    std::cout << "Index cur_element_count: " << index->cur_element_count
              << std::endl;
    mem_tracker.snapshot("after_build");
    index_memory_kb = mem_tracker.delta_rss_kb("after_load_gt", "after_build");
    std::cout << "Index memory: " << (index_memory_kb / 1024.0) << " MB"
              << std::endl;
  } else {
    std::cout << "\n=== Building Index ===" << std::endl;
    index = new hnswlib::HierarchicalNSW<float>(space, num_base, cfg.M,
                                                cfg.ef_construction, cfg.seed);

    bench::Timer build_timer;

    index->addPoint(base_data, 0);

#pragma omp parallel for schedule(dynamic) num_threads(cfg.num_threads)
    for (int i = 1; i < num_base; i++) {
      index->addPoint(base_data + static_cast<size_t>(i) * dim, i);
    }

    build_time_sec = build_timer.elapsed_sec();
    build_throughput = num_base / build_time_sec;

    mem_tracker.snapshot("after_build");
    index_memory_kb = mem_tracker.delta_rss_kb("after_load_gt", "after_build");

    std::cout << "Build time: " << build_time_sec << " sec" << std::endl;
    std::cout << "Build throughput: " << build_throughput << " points/sec"
              << std::endl;
    std::cout << "Index memory: " << (index_memory_kb / 1024.0) << " MB"
              << std::endl;

    if (cfg.save_index && !cfg.index_path.empty()) {
      std::cout << "Saving index to " << cfg.index_path << std::endl;
      index->saveIndex(cfg.index_path);
    }
  }

  std::vector<std::map<std::string, std::string>> all_results;

  int max_k = *std::max_element(cfg.K_values.begin(), cfg.K_values.end());
  std::vector<int> ann_results(num_query * max_k);
  std::vector<double> latencies(num_query);

  std::cout << "\n=== Running Queries ===" << std::endl;
  if (cfg.defer_qps) {
    std::cout << "  [Mode: defer_qps - latency first, QPS second]" << std::endl;
  }
  std::cout << "  Warmup queries: " << cfg.warmup_queries << std::endl;

  auto measure_latency = [&](int ef_search,
                             int K) -> std::map<std::string, std::string> {
    index->setEf(ef_search);

    int warmup = std::min(cfg.warmup_queries, num_query);
    std::cout << "\n  ef=" << ef_search << " K=" << std::setw(3) << K
              << " - Warmup (" << warmup << ")..." << std::flush;
    for (int i = 0; i < warmup; i++) {
      auto result =
          index->searchKnn(query_data + static_cast<size_t>(i) * dim, K);
      (void)result;
    }
    std::cout << " done" << std::endl;

    std::fill(latencies.begin(), latencies.end(), 0);

    for (int qid = 0; qid < num_query; qid++) {
      bench::Timer query_timer;
      auto result =
          index->searchKnn(query_data + static_cast<size_t>(qid) * dim, K);
      latencies[qid] = query_timer.elapsed_us();

      int idx = K - 1;
      while (!result.empty() && idx >= 0) {
        ann_results[qid * max_k + idx] = result.top().second;
        result.pop();
        idx--;
      }
      while (idx >= 0) {
        ann_results[qid * max_k + idx] = -1;
        idx--;
      }
    }

    auto lat_stats = bench::compute_latency_stats(latencies);

    double total_time_sec =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / 1000000.0;
    double single_thread_qps = num_query / total_time_sec;

    float recall = bench::compute_average_recall(ann_results.data(), gt_data,
                                                 num_query, K, gt_k, max_k);

    std::map<std::string, std::string> result_entry;
    result_entry["ef_search"] = std::to_string(ef_search);
    result_entry["K"] = std::to_string(K);
    result_entry["recall"] = std::to_string(recall);
    result_entry["mean_us"] = std::to_string(lat_stats.mean_us);
    result_entry["p50_us"] = std::to_string(lat_stats.p50_us);
    result_entry["p95_us"] = std::to_string(lat_stats.p95_us);
    result_entry["p99_us"] = std::to_string(lat_stats.p99_us);
    result_entry["max_us"] = std::to_string(lat_stats.max_us);
    result_entry["qps_1t"] = std::to_string(single_thread_qps);

    std::cout << "  ef=" << ef_search << " K=" << std::setw(3) << K
              << " recall=" << std::fixed << std::setprecision(4) << recall
              << " p99=" << std::setprecision(1) << lat_stats.p99_us << "us"
              << " qps(1t)=" << std::setprecision(0) << single_thread_qps
              << std::endl;

    return result_entry;
  };

  auto measure_qps = [&](int ef_search, int K,
                         std::map<std::string, std::string> &entry) {
    index->setEf(ef_search);
    for (int num_t : {2, 4, 8, 16, 32, 64}) {
      if (num_t > cfg.num_threads)
        break;

      bench::Timer mt_timer;

#pragma omp parallel for schedule(static) num_threads(num_t)
      for (int qid = 0; qid < num_query; qid++) {
        auto result =
            index->searchKnn(query_data + static_cast<size_t>(qid) * dim, K);
        (void)result;
      }

      double mt_time_sec = mt_timer.elapsed_sec();
      double mt_qps = num_query / mt_time_sec;
      entry["qps_" + std::to_string(num_t) + "t"] = std::to_string(mt_qps);
    }
  };

  if (cfg.defer_qps) {
    std::cout << "\n--- Phase 1: Latency Measurements (no multi-thread) ---"
              << std::endl;
    for (int ef_search : cfg.ef_search_values) {
      for (int K : cfg.K_values) {
        if (K > gt_k)
          continue;
        auto entry = measure_latency(ef_search, K);
        all_results.push_back(entry);
      }
    }

    std::cout << "\n--- Phase 2: QPS Measurements (multi-thread) ---"
              << std::endl;
    size_t idx = 0;
    for (int ef_search : cfg.ef_search_values) {
      for (int K : cfg.K_values) {
        if (K > gt_k)
          continue;
        std::cout << "  QPS: ef=" << ef_search << " K=" << K << "..."
                  << std::flush;
        measure_qps(ef_search, K, all_results[idx]);
        std::cout << " done" << std::endl;
        idx++;
      }
    }
  } else {
    for (int ef_search : cfg.ef_search_values) {
      for (int K : cfg.K_values) {
        if (K > gt_k)
          continue;
        auto entry = measure_latency(ef_search, K);
        measure_qps(ef_search, K, entry);
        all_results.push_back(entry);
      }
    }
  }

  if (!cfg.output_path.empty()) {
    write_json_result(cfg.output_path, cfg, num_base, num_query, dim,
                      build_time_sec, build_throughput, index_memory_kb,
                      data_memory_kb, all_results);
  }

  std::cout << "\n=== Memory Timeline ===" << std::endl;
  mem_tracker.print_all();

  delete index;
  delete space;
  delete[] base_data;
  delete[] query_data;
  delete[] gt_data;

  std::cout << "\n=== Benchmark Complete ===" << std::endl;

  return 0;
}
