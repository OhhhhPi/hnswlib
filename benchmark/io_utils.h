/**
 * @file io_utils.h
 * @brief Utilities for reading fvecs and ivecs file formats.
 * 
 * fvecs format: [dim (int32)] [v1 (float32)] ... [v_dim (float32)] per vector
 * ivecs format: [K (int32)] [id1 (int32)] ... [id_K (int32)] per vector
 */

#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

/**
 * @brief Read fvecs file and return flat float array.
 * 
 * @param filename Path to fvecs file
 * @param num Output: number of vectors
 * @param dim Output: dimension of vectors
 * @return float* Flat array of size num * dim (caller must delete[])
 */
inline float* read_fvecs_flat(const std::string& filename, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    in.seekg(0, std::ios::end);
    size_t filesize = in.tellg();
    in.seekg(0, std::ios::beg);
    
    int first_dim;
    in.read(reinterpret_cast<char*>(&first_dim), sizeof(int));
    
    if (first_dim <= 0 || first_dim > 100000) {
        throw std::runtime_error("Invalid dimension in fvecs file: " + std::to_string(first_dim));
    }
    
    size_t record_size = sizeof(int) + first_dim * sizeof(float);
    size_t num_vectors = filesize / record_size;
    
    if (filesize % record_size != 0) {
        std::cerr << "Warning: fvecs file size not aligned with record size" << std::endl;
    }
    
    float* data = new float[num_vectors * first_dim];
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        int d;
        in.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (d != first_dim) {
            delete[] data;
            throw std::runtime_error("Inconsistent dimension at vector " + std::to_string(i));
        }
        in.read(reinterpret_cast<char*>(data + i * first_dim), first_dim * sizeof(float));
    }
    
    in.close();
    
    num = static_cast<int>(num_vectors);
    dim = first_dim;
    
    std::cout << "Loaded fvecs: " << filename << " (" << num << " vectors, dim=" << dim << ")" << std::endl;
    
    return data;
}

/**
 * @brief Read ivecs file and return flat int array.
 * 
 * @param filename Path to ivecs file
 * @param num Output: number of vectors
 * @param k Output: number of integers per vector
 * @return int* Flat array of size num * k (caller must delete[])
 */
inline int* read_ivecs_flat(const std::string& filename, int& num, int& k) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    in.seekg(0, std::ios::end);
    size_t filesize = in.tellg();
    in.seekg(0, std::ios::beg);
    
    int first_k;
    in.read(reinterpret_cast<char*>(&first_k), sizeof(int));
    
    if (first_k <= 0 || first_k > 10000) {
        throw std::runtime_error("Invalid k in ivecs file: " + std::to_string(first_k));
    }
    
    size_t record_size = sizeof(int) + first_k * sizeof(int);
    size_t num_vectors = filesize / record_size;
    
    if (filesize % record_size != 0) {
        std::cerr << "Warning: ivecs file size not aligned with record size" << std::endl;
    }
    
    int* data = new int[num_vectors * first_k];
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        int cur_k;
        in.read(reinterpret_cast<char*>(&cur_k), sizeof(int));
        if (cur_k != first_k) {
            delete[] data;
            throw std::runtime_error("Inconsistent k at vector " + std::to_string(i));
        }
        in.read(reinterpret_cast<char*>(data + i * first_k), first_k * sizeof(int));
    }
    
    in.close();
    
    num = static_cast<int>(num_vectors);
    k = first_k;
    
    std::cout << "Loaded ivecs: " << filename << " (" << num << " vectors, k=" << k << ")" << std::endl;
    
    return data;
}

/**
 * @brief Write fvecs file from flat float array.
 */
inline void write_fvecs_flat(const std::string& filename, const float* data, int num, int dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    for (int i = 0; i < num; i++) {
        out.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        out.write(reinterpret_cast<const char*>(data + i * dim), dim * sizeof(float));
    }
    
    out.close();
}

/**
 * @brief Write ivecs file from flat int array.
 */
inline void write_ivecs_flat(const std::string& filename, const int* data, int num, int k) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }
    
    for (int i = 0; i < num; i++) {
        out.write(reinterpret_cast<const char*>(&k), sizeof(int));
        out.write(reinterpret_cast<const char*>(data + i * k), k * sizeof(int));
    }
    
    out.close();
}

} // namespace bench
