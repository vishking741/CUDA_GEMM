#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

#define N 1024
#define TILE_SIZE 16

// GPU kernel for GEMM using tiling (shared memory)
__global__ void tiled_MM(const float* A, const float* B, float* C)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // compute global row and column
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // number of tiles in k dimension
    int numTiles = N / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // load tiles into shared memory
        sA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        // wait for all threads
        __syncthreads();

        // compute partial result
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // sync before next tile
        __syncthreads();
    }

    // store result
    C[row * N + col] = sum;
}

// CPU function for matrix multiplication
void cpu_MM_basic(float* A, float* B, float* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            float sum = 0;

            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

int main()
{
    // host memory allocation
    float *h_A = new float[N*N];
    float *h_B = new float[N*N];
    float *h_C = new float[N*N];
    float *h_C_gpu = new float[N*N];

    // initialize matrices
    for (int i = 0; i < N*N; i++) {
        h_A[i] = static_cast<float>(i % 10);
        h_B[i] = static_cast<float>((i * 2) % 10);
        h_C[i] = 0;
        h_C_gpu[i] = 0;
    }

    // device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    // copy input to gpu
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // cpu timing start
    auto cpu_start = chrono::high_resolution_clock::now();

    cpu_MM_basic(h_A, h_B, h_C);

    // cpu timing end
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    // kernel configuration
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);

    // gpu timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    tiled_MM<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // get gpu time
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // copy result back
    cudaMemcpy(h_C_gpu, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // verify results
    float error = 0;
    for(int i=0;i<N*N;i++){
        error += abs(h_C_gpu[i] - h_C[i]);
    }

    if(error < 1e-3)
        cout<<"results are correct\n";
    else
        cout<<"wrong results\n";

    // print execution times
    cout << "CPU Time: " << cpu_time << " ms\n";
    cout << "GPU Time (tiled kernel): " << gpu_time << " ms\n";

    // cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
