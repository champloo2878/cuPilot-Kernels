// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>

#define UNROLL_FACTOR_SUM 8
#define UNROLL_FACTOR_NORM 8
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void l1_normalize_kernel(const float* input, float* output, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int step = blockDim.x;

    int index = row * dim + tid;
    int end = (row+1) * dim;

    // Unrolled summation phase with warp organization
    float thread_sum = 0.0f;
    int i = index;
    
    // Main unrolled loop
    for (; i + (UNROLL_FACTOR_SUM - 1) * step < end; i += UNROLL_FACTOR_SUM * step) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR_SUM; j++) {
            int idx = i + j * step;
            thread_sum += fabsf(input[idx]);
        }
    }
    
    // Remainder loop
    for (; i < end; i += step) {
        thread_sum += fabsf(input[i]);
    }

    // Warp-level reduction using shuffle primitives
    float warp_sum = thread_sum;
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }

    // First thread in each warp stores warp sum to shared memory
    extern __shared__ float sdata[];
    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();

    // Final block reduction using first warp
    if (warp_id == 0) {
        float block_sum = (lane_id < blockDim.x / 32) ? sdata[lane_id] : 0.0f;
        
        // Second warp-level reduction for block sum
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }

        // First thread computes mean and stores in shared memory
        if (lane_id == 0) {
            sdata[0] = block_sum / dim;
        }
    }
    __syncthreads();

    float mean = sdata[0];

    // Unrolled normalization phase
    i = index;
    
    // Main unrolled loop
    for (; i + (UNROLL_FACTOR_NORM - 1) * step < end; i += UNROLL_FACTOR_NORM * step) {
        #pragma unroll
        for (int j = 0; j < UNROLL_FACTOR_NORM; j++) {
            int idx = i + j * step;
            output[idx] = input[idx] / mean;
        }
    }
    
    // Remainder loop
    for (; i < end; i += step) {
        output[i] = input[i] / mean;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor l1_normalize_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
    TORCH_CHECK(input.size(1) > 0, "input must have at least one column");

    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim = input.size(1);

    const int block_size = 1024;
    dim3 grid(batch_size);
    dim3 block(block_size);
    // Reduced shared memory: only need space for warp counts (block_size/32)
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    l1_normalize_kernel<<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), dim);
    

    return output;
}
// PART-END