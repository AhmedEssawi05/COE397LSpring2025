#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16  // Adjust as needed

// Kernel for 2D vector (matrix) addition using shared memory
__global__
void vectorAdd2D(const float *A, const float *B, float *C, int numRows, int numCols)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Load data from global memory into shared memory if within bounds
    if (row < numRows && col < numCols) {
        tileA[threadIdx.y][threadIdx.x] = A[row * numCols + col];
        tileB[threadIdx.y][threadIdx.x] = B[row * numCols + col];
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0f;
        tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Perform the addition and write the result back to global memory
    if (row < numRows && col < numCols) {
        C[row * numCols + col] = tileA[threadIdx.y][threadIdx.x] + tileB[threadIdx.y][threadIdx.x];
    }
}

int main(void)
{
    // Matrix dimensions (example values)
    int numRows = 1024;
    int numCols = 1024;
    size_t size = numRows * numCols * sizeof(float);

    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Initialize matrices with random values
    for (int i = 0; i < numRows * numCols; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((numCols + TILE_WIDTH - 1) / TILE_WIDTH,
                 (numRows + TILE_WIDTH - 1) / TILE_WIDTH,
                 1);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the kernel
    vectorAdd2D<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, numRows, numCols);

    // Record the stop event and synchronize
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Optionally verify results here (omitted for brevity)

    // Clean up device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("2D vector addition completed using CUDA shared memory.\n");
    return 0;
}
