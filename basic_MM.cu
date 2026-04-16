#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
using namespace std;

// GPU kernel for GEMM , basic
__global__ void MM_basic(float*A, float*B , float*C, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // check valid index
    if (row < N && col < N) {

        float sum = 0.0f;

        // compute dot product
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        // store result
        C[row * N + col] = sum;
    } 
}

// CPU function for MM , for comparing execution time
void cpu_MM_basic(float* A, float* B, float* C, int N)
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
    const int N = 1024;

    // host memory
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float* h_C_gpu = new float[N*N];

    // device memory
    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,N*N*sizeof(float));
    cudaMalloc((void**)&d_B,N*N*sizeof(float));
    cudaMalloc((void**)&d_C,N*N*sizeof(float));

    // initialize data
    for(int i=0;i<N*N;i++){
        h_A[i]=rand() % 10;
        h_B[i]=rand() % 10;
        h_C[i]=0;
        h_C_gpu[i]=0;
    }

    // cpu timing start
    auto cpu_start = chrono::high_resolution_clock::now();

    cpu_MM_basic(h_A,h_B,h_C,N);

    // cpu timing end
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    // copy input to gpu
    cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,N*N*sizeof(float),cudaMemcpyHostToDevice);

    int block_size=16;
    dim3 threadsperblock(block_size,block_size);
    dim3 numBlocks((N+block_size-1)/block_size,(N+block_size-1)/block_size);

    // gpu timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    MM_basic<<<numBlocks,threadsperblock>>>(d_A,d_B,d_C,N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // get gpu time
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // copy result back
    cudaMemcpy(h_C_gpu,d_C,N*N*sizeof(float),cudaMemcpyDeviceToHost);

    // verify results
    float error = 0;
    for(int i=0;i<N*N;i++){
        error += abs(h_C_gpu[i]-h_C[i]);
    }

    if(error < 0)
        cout<<"results are correct\n";
    else
        cout<<"wrong results\n";

    // print execution times
    cout << "CPU Time: " << cpu_time << " ms\n";
    cout << "GPU Time (kernel only): " << gpu_time << " ms\n";

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
