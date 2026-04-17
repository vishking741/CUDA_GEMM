#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
#include <cmath>
using namespace std;

#define TILE_WIDTH 16

// In float 4 we do the loop unrolling manually by using float4 data type 
// float4 data type will load 4 float values in 1 cycle rather 4 cycles for 1 float value 

// this kind will be help full in increasing parallesim and reduce compute bound of the Program for larger sizes 

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



__global__ void float4_tiled_MM(const float* A , const float* B , float* C , int n) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    //row index no effect we only need col index change
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    //each thread covers 4 columns , and its base col value is this
    int baseCol = blockIdx.x * TILE_WIDTH + (threadIdx.x * 4);
    // basecol , +1 , +2 , +3 -> will be stored and use as 4 stage loop unroll
    // and also depends on block given

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // will be converted into float4 in storage later

    int numTiles = n / TILE_WIDTH; 

    for (int t = 0; t < numTiles; t++){
       
        // 1. Load sub-tile of A and B into shared memory
        //    Each thread loads a float4 from A, and a float4 from B

        // A
        //sA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        // this is the prev original addr 
        {  
            // global col for A => t*TILE_WIDTH + (threadIdx.x*4 .. +3)
            int globalACol = t*TILE_WIDTH + (threadIdx.x * 4);
            const float* srcA = A + (Row*n + globalACol);
            // this is a pointer to the global address inside A for what we need to store final is same prev addr
            float* dstA = &sA[threadIdx.y][threadIdx.x*4];
            // this is a pointer which will store where the float4 should be stored (base addr)

            float4 vecA = *reinterpret_cast<const float4*>(srcA);
            //loading the float4 from srcA which is original A global base addr
            *reinterpret_cast<float4*>(dstA) = vecA;
            //and reloading the float4 data into the shared mem for usage
        }

        //sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        // this is the prev original addr
        // B
        {
            // global row for B => t*TILE_WIDTH + threadIdx.y
            int globalBRow = t*TILE_WIDTH + threadIdx.y;
            // the row is const in this and no need for *4 bcoz we have the same row value
            // columns => baseCol..baseCol+3
            const float* srcB = B + (globalBRow*n + baseCol);
            // we are accessing 4 coloums 

            // as data in the matrix is row_major , i.e stored row wise 
            // for iterating on the coloums we use basecol inside the thing which is externally linked to x dim

            float* dstB = &sB[threadIdx.y][threadIdx.x*4];
            // pointer to store the address where the data must be in the shared mem

            float4 vecB = *reinterpret_cast<const float4*>(srcB);
            // float 4 operational single cycle store explained before 
            *reinterpret_cast<float4*>(dstB) = vecB;
        }

        __syncthreads();
        // sync threads 

        // 2. Compute partial sums
        for(int k = 0; k < TILE_WIDTH; k++){
            float aVal = sA[threadIdx.y][k];
            float4 bVal4 = *reinterpret_cast<float4*>(&sB[k][threadIdx.x*4]);
            // for using float4 we need cast it ! 
            // float4 bval4 {.x , .y , .z , .w};
            sum[0] += aVal * bVal4.x;
            sum[1] += aVal * bVal4.y;
            sum[2] += aVal * bVal4.z;
            sum[3] += aVal * bVal4.w;
        }

        __syncthreads();
        // again sync all the threads 
    }

    if(Row < n){
        float* outC = C + (Row*n + baseCol);
        outC[0] = sum[0];
        outC[1] = sum[1];
        outC[2] = sum[2];
        outC[3] = sum[3];
    }

}

int main(){

  const int N = 1024;
       
    // 1. Allocate host memory 
    float* h_A = new float[N*N];
    float* h_B = new float[N*N];
    float* h_C = new float[N*N];
    float* h_Cgpu = new float[N*N];

    // device memory
    float *d_A,*d_B,*d_C;
    size_t nBytes = N*N*sizeof(float);

    cudaMalloc((void**)&d_A,nBytes);
    cudaMalloc((void**)&d_B,nBytes);
    cudaMalloc((void**)&d_C,nBytes);

    // initialize data
    for(int i=0;i<N*N;i++){
        h_A[i]=rand() % 10;
        h_B[i]=rand() % 10;
        h_C[i]=0;
        h_Cgpu[i]=0;
    }

    // cpu timing start
    auto cpu_start = chrono::high_resolution_clock::now();

    cpu_MM_basic(h_A,h_B,h_C,N);

    // cpu timing end
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // gpu timing start
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(4,16);
    dim3 grid(N/TILE_WIDTH, N/TILE_WIDTH);

    cudaEventRecord(start);
    float4_tiled_MM<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // get gpu time
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(h_Cgpu,d_C,nBytes,cudaMemcpyDeviceToHost);

    // verify results
    float error = 0;
    for(int i=0;i<N*N;i++){
        error += fabs(h_Cgpu[i]-h_C[i]);
    }

    if(error < 1e-3)
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
    delete[] h_Cgpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
