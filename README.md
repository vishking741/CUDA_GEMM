# CUDA_GEMM
This repository contains multiple implementations of GEMM (General Matrix Multiplication) in CUDA, focusing on analyzing performance improvements through different optimization techniques like tiling and various libs like CUBLASS 

## I am using Google colab free Online Tesla T4 for running the kernels 

### basic_MM 
- CPU Time: 7430.16 ms
- GPU Time (kernel only): 9.33075 ms

### tiling_shared_mem_MM
- CPU Time: 5988.08 ms 
- GPU Time (tiled kernel): 5.93725 ms
