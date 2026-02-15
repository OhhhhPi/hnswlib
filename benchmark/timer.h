/**
 * @file timer.h
 * @brief High-precision timing utilities and latency statistics.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace bench {

/**
 * @brief High-precision timer using steady_clock.
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_us() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(now - start_).count();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    double elapsed_sec() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Latency statistics structure.
 */
struct LatencyStats {
    double mean_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double max_us;
    double min_us;
    double std_us;
};

/**
 * @brief Compute latency statistics from a vector of latencies (in microseconds).
 * 
 * @param latencies Vector of latency measurements in microseconds
 * @return LatencyStats Computed statistics
 */
inline LatencyStats compute_latency_stats(std::vector<double>& latencies) {
    LatencyStats stats;
    
    if (latencies.empty()) {
        stats.mean_us = stats.p50_us = stats.p95_us = stats.p99_us = stats.max_us = stats.min_us = stats.std_us = 0.0;
        return stats;
    }
    
    size_t n = latencies.size();
    
    std::sort(latencies.begin(), latencies.end());
    
    double sum = 0.0;
    for (const auto& lat : latencies) {
        sum += lat;
    }
    stats.mean_us = sum / n;
    
    stats.min_us = latencies.front();
    stats.max_us = latencies.back();
    
    auto percentile = [&latencies, n](double p) -> double {
        if (n == 1) return latencies[0];
        double idx = p * (n - 1) / 100.0;
        size_t lower = static_cast<size_t>(std::floor(idx));
        size_t upper = static_cast<size_t>(std::ceil(idx));
        if (lower == upper) return latencies[lower];
        double frac = idx - lower;
        return latencies[lower] * (1.0 - frac) + latencies[upper] * frac;
    };
    
    stats.p50_us = percentile(50.0);
    stats.p95_us = percentile(95.0);
    stats.p99_us = percentile(99.0);
    
    double sum_sq = 0.0;
    for (const auto& lat : latencies) {
        double diff = lat - stats.mean_us;
        sum_sq += diff * diff;
    }
    stats.std_us = std::sqrt(sum_sq / n);
    
    return stats;
}

/**
 * @brief Simple stopwatch class for timing code blocks.
 */
class StopWatch {
public:
    StopWatch() : elapsed_us_(0), running_(false) {}
    
    void start() {
        if (!running_) {
            timer_.reset();
            running_ = true;
        }
    }
    
    void stop() {
        if (running_) {
            elapsed_us_ += timer_.elapsed_us();
            running_ = false;
        }
    }
    
    void reset() {
        elapsed_us_ = 0;
        running_ = false;
    }
    
    double get_elapsed_us() const {
        if (running_) {
            return elapsed_us_ + timer_.elapsed_us();
        }
        return elapsed_us_;
    }
    
    double get_elapsed_ms() const {
        return get_elapsed_us() / 1000.0;
    }
    
    double get_elapsed_sec() const {
        return get_elapsed_us() / 1000000.0;
    }
    
private:
    Timer timer_;
    double elapsed_us_;
    bool running_;
};

} // namespace bench
