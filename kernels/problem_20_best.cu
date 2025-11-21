// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, float negative_slope, int num_elements) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int warp_base = (blockIdx.x * blockDim.x + warp_id * 32) * 4;
    
    if (warp_base >= num_elements) return;
    
    // Prefetch next warp (warp_id+1) into L2 cache
    if (warp_id < 31) {
        int next_warp_id = warp_id + 1;
        int next_warp_base = (blockIdx.x * blockDim.x + next_warp_id * 32) * 4;
        if (next_warp_base + 127 < num_elements) {
            if (lane_id < 8) {
                const float* prefetch_addr = input + next_warp_base + lane_id * 16;
                asm volatile ("prefetch.global.L2 [%0];" : : "l"(prefetch_addr));
            }
        }
    }
    
    // Prefetch warp two ahead (warp_id+2) into L2 cache
    if (warp_id < 30) {
        int next_warp_id = warp_id + 2;
        int next_warp_base = (blockIdx.x * blockDim.x + next_warp_id * 32) * 4;
        if (next_warp_base + 127 < num_elements) {
            if (lane_id < 8) {
                const float* prefetch_addr = input + next_warp_base + lane_id * 16;
                asm volatile ("prefetch.global.L2 [%0];" : : "l"(prefetch_addr));
            }
        }
    }
    
    if (warp_base + 127 < num_elements) {
        int thread_idx = warp_base + (lane_id << 2);
        float4 in_val = *reinterpret_cast<const float4*>(input + thread_idx);
        float4 out_val;
        
        // Optimized computation using ternary conditional
        out_val.x = (in_val.x > 0.0f) ? in_val.x : (in_val.x * negative_slope);
        out_val.y = (in_val.y > 0.0f) ? in_val.y : (in_val.y * negative_slope);
        out_val.z = (in_val.z > 0.0f) ? in_val.z : (in_val.z * negative_slope);
        out_val.w = (in_val.w > 0.0f) ? in_val.w : (in_val.w * negative_slope);
        
        *reinterpret_cast<float4*>(output + thread_idx) = out_val;
    } else {
        int thread_idx = warp_base + (lane_id << 2);
        for (int i = 0; i < 4; ++i) {
            int index = thread_idx + i;
            if (index < num_elements) {
                float val = input[index];
                // Optimized tail computation
                output[index] = (val > 0.0f) ? val : (val * negative_slope);
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor leaky_relu_cuda(torch::Tensor input, float negative_slope) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int block_size = 1024;
    int grid_size = (num_elements + 4 * block_size - 1) / (4 * block_size);
    
    cudaFuncSetCacheConfig(leaky_relu_kernel, cudaFuncCachePreferL1);
    
    leaky_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        negative_slope,
        num_elements
    );
    
    return output;
}
// PART-END