#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Kernel: each thread writes its global linear index (or ID) into d_array.
//
__global__ void fillGlobalIndex2D(int *d_array, int totalX, int totalY)
{
    // Compute each thread's 2D coordinates in the overall grid
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalY = blockIdx.y * blockDim.y + threadIdx.y;

    // Convert (globalY, globalX) into a single linear index
    // in row-major order:
    //    index = row * width + col
    int idx = globalY * totalX + globalX;

    // Make sure we do not go out of bounds
    if (globalX < totalX && globalY < totalY)
    {
        d_array[idx] = idx;
    }
}

//------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Expect 4 command line arguments: blockDimX blockDimY gridDimX gridDimY
    if (argc < 5) {
        fprintf(stderr, "Usage: %s blockDimX blockDimY gridDimX gridDimY\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int blockDimX = atoi(argv[1]);
    int blockDimY = atoi(argv[2]);
    int gridDimX  = atoi(argv[3]);
    int gridDimY  = atoi(argv[4]);

    // Compute total array dimensions
    // (the total number of threads in each dimension)
    int totalX = blockDimX * gridDimX;
    int totalY = blockDimY * gridDimY;
    int totalSize = totalX * totalY;

    printf("blockDim = (%d, %d), gridDim = (%d, %d)\n", 
           blockDimX, blockDimY, gridDimX, gridDimY);
    printf("=> totalX = %d, totalY = %d\n", totalX, totalY);

    // Allocate host memory
    int *h_array = (int*)malloc(totalSize * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Error: host memory allocation failed.\n");
        return 1;
    }

    // Allocate device memory
    int *d_array = NULL;
    cudaError_t err = cudaMalloc((void**)&d_array, totalSize * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: device memory allocation failed: %s\n",
                cudaGetErrorString(err));
        free(h_array);
        return 1;
    }

    // Zero out the device memory (optional, but good practice)
    cudaMemset(d_array, 0, totalSize * sizeof(int));

    // Prepare 2D execution configuration
    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridDimX, gridDimY);

    // Launch the kernel
    fillGlobalIndex2D<<<grid, block>>>(d_array, totalX, totalY);

    // Check for any kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_array, d_array, totalSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy to host failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_array);
        free(h_array);
        return 1;
    }

    // Print the 2D layout of global thread indices
    // h_array[row * totalX + col] should contain the linear index
    printf("\nGlobal thread indices in a 2D layout:\n");
    for (int row = 0; row < totalY; ++row)
    {
        for (int col = 0; col < totalX; ++col)
        {
            int idx = row * totalX + col;
            printf("%4d ", h_array[idx]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_array);
    free(h_array);

    return 0;
}
