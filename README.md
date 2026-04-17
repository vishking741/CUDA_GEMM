# CUDA_GEMM
This repository contains multiple implementations of GEMM (General Matrix Multiplication) in CUDA, focusing on analyzing performance improvements through different optimization techniques like tiling and various libs like CUBLASS 

## I am using Google colab free Online Tesla T4 for running the kernels and CUDA: 12.8 (nvcc) complier for Size: 1024 × 1024

### basic_MM 
- CPU Time: 7430.16 ms
- GPU Time (kernel only): 9.33075 ms  
basic implementation: each thread computes one output using direct global memory access (no reuse, memory-bound).

---

### tiling_shared_mem_MM
- CPU Time: 5988.08 ms 
- GPU Time (tiled kernel): 5.93725 ms  
Uses shared memory tiling (16×16) to reuse data across threads, reducing global memory traffic and improving locality.

---

### tiling_float4_MM
- CPU Time: 9356.7 ms
- GPU Time (float4 + tiling only): 5.02541 ms  
Extends tiling with `float4` vectorized loads and per-thread multiple outputs, reducing instr by manual loop unrolling and improving speed

### Note :
- CPU times are having too much variations bcoz of online Google colab usage 
- also GPU will have even much more faster processing , becuase i used CUDA API's for time calc not the profiling tool

