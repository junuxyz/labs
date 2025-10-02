#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

#include <cuda_runtime.h>

// Macro for checking CUDA errors. Crucial for debugging.
#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)                                                                  \


// CUDA kernel: Performs vector addition C = A + B on the device
__global__ void addVectors(const float* A, const float* B, float* C, \
    int N)
{
    // Calculate a unique index for the current thread
    // blockIdx.x: the x-coordinate of the block within the grid
    // blockDim.x: The number of threads in each block
    // threadIdx.x: The x-coordinate of the thread within its block
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}
