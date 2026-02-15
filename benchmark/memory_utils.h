/**
 * @file memory_utils.h
 * @brief Memory monitoring utilities using /proc/self/status (Linux only).
 */

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace bench {

/**
 * @brief Memory usage information.
 */
struct MemoryInfo {
    size_t vm_rss_kb;    // Resident set size (physical memory)
    size_t vm_size_kb;   // Virtual memory size
    size_t vm_data_kb;   // Data segment size
    size_t vm_stk_kb;    // Stack size
    
    MemoryInfo() : vm_rss_kb(0), vm_size_kb(0), vm_data_kb(0), vm_stk_kb(0) {}
    
    double vm_rss_mb() const { return vm_rss_kb / 1024.0; }
    double vm_size_mb() const { return vm_size_kb / 1024.0; }
    
    void print() const {
        std::cout << "Memory: RSS=" << vm_rss_mb() << " MB, VM=" << vm_size_mb() << " MB" << std::endl;
    }
};

/**
 * @brief Get current memory usage from /proc/self/status.
 * 
 * @return MemoryInfo Current memory usage
 */
inline MemoryInfo get_memory_info() {
    MemoryInfo info;
    
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        std::cerr << "Warning: Cannot open /proc/self/status" << std::endl;
        return info;
    }
    
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            iss >> info.vm_rss_kb;
        } else if (line.find("VmSize:") == 0) {
            std::istringstream iss(line.substr(7));
            iss >> info.vm_size_kb;
        } else if (line.find("VmData:") == 0) {
            std::istringstream iss(line.substr(7));
            iss >> info.vm_data_kb;
        } else if (line.find("VmStk:") == 0) {
            std::istringstream iss(line.substr(6));
            iss >> info.vm_stk_kb;
        }
    }
    
    status.close();
    return info;
}

/**
 * @brief Memory snapshot for tracking changes.
 */
struct MemorySnapshot {
    MemoryInfo info;
    std::string label;
    
    MemorySnapshot(const std::string& lbl = "") : label(lbl) {
        info = get_memory_info();
    }
    
    void print() const {
        std::cout << "[" << label << "] ";
        info.print();
    }
};

/**
 * @brief Memory tracker for collecting multiple snapshots.
 */
class MemoryTracker {
public:
    void snapshot(const std::string& label) {
        snapshots_.emplace_back(label);
    }
    
    void print_all() const {
        std::cout << "\n=== Memory Usage Timeline ===" << std::endl;
        for (const auto& snap : snapshots_) {
            snap.print();
        }
    }
    
    size_t delta_rss_kb(const std::string& from_label, const std::string& to_label) const {
        size_t from_rss = 0, to_rss = 0;
        for (const auto& snap : snapshots_) {
            if (snap.label == from_label) from_rss = snap.info.vm_rss_kb;
            if (snap.label == to_label) to_rss = snap.info.vm_rss_kb;
        }
        if (to_rss >= from_rss) {
            return to_rss - from_rss;
        }
        return 0;
    }
    
    const std::vector<MemorySnapshot>& get_snapshots() const { return snapshots_; }
    
private:
    std::vector<MemorySnapshot> snapshots_;
};

/**
 * @brief Estimate HNSW index memory usage.
 * 
 * @param num_elements Number of elements in the index
 * @param dim Vector dimension
 * @param M HNSW M parameter
 * @return size_t Estimated memory in bytes
 */
inline size_t estimate_hnsw_memory(size_t num_elements, int dim, int M) {
    size_t data_size = num_elements * dim * sizeof(float);
    size_t link_size_0 = num_elements * (sizeof(int) + M * sizeof(int));
    size_t link_size_upper = num_elements * (sizeof(int) + M * sizeof(int)) / 2;
    size_t label_size = num_elements * sizeof(long long);
    
    return data_size + link_size_0 + link_size_upper + label_size;
}

inline double estimate_hnsw_memory_mb(size_t num_elements, int dim, int M) {
    return estimate_hnsw_memory(num_elements, dim, M) / (1024.0 * 1024.0);
}

} // namespace bench
