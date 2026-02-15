cmake_minimum_required(VERSION 3.14)
project(hnsw_bench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fopenmp")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

include_directories(${CMAKE_SOURCE_DIR}/..)

find_package(OpenMP REQUIRED)

add_executable(bench_hotpotqa bench_hotpotqa.cpp)
target_link_libraries(bench_hotpotqa OpenMP::OpenMP_CXX)

add_executable(bench_sift1m bench_sift1m.cpp)
target_link_libraries(bench_sift1m OpenMP::OpenMP_CXX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(bench_hotpotqa asan)
    target_link_libraries(bench_sift1m asan)
endif()
