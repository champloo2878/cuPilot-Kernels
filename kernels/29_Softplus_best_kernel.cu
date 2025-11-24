// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Helper function to determine if value is safe for FP16 computation
__device__ __forceinline__ bool is_fp16_safe(float x) {
    return x >= -65504.0f && x <= 65504.0f;
}

// FP16 softplus implementation
__device__ __forceinline__ half softplus_fp16(half x) {
    half threshold = __float2half(20.0f);
    return __hgt(x, threshold) ? x : hlog(__hadd(__float2half(1.0f), hexp(x)));
}

// FP32 softplus implementation
__device__ __forceinline__ float softplus_fp32(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= size) return;
    
    // Vectorized load with tail handling
    float4 val4;
    if (idx + 3 < size) {
        val4 = *reinterpret_cast<const float4*>(input + idx);
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ((float*)&val4)[i] = (idx + i < size) ? input[idx + i] : 0.0f;
        }
    }

    // Process 4 elements with mixed precision based on range detection
    float4 result;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float val = ((float*)&val4)[i];
        
        // Automatic range detection for mixed precision
        if (is_fp16_safe(val)) {
            // Use FP16 for safe ranges
            half val_half = __float2half_rn(val);
            half result_half = softplus_fp16(val_half);
            ((float*)&result)[i] = __half2float(result_half);
        } else {
            // Use FP32 for critical ranges
            ((float*)&result)[i] = softplus_fp32(val);
        }
    }

    // Vectorized store with tail handling
    if (idx + 3 < size) {
        *reinterpret_cast<float4*>(output + idx) = result;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < size) output[idx + i] = ((float*)&result)[i];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor softplus_cuda(torch::Tensor input) {
    auto input_cont = input.contiguous();
    auto output = torch::empty_like(input_cont);
    int size = input_cont.numel();
    
    if (size == 0) {
        return output;
    }
    
    // Optimized for A100: 512 threads/block for better occupancy
    const int block_size = 512;
    int grid_size = (size + block_size * 4 - 1) / (block_size * 4);
    
    // Configure for memory-bound operation with L1 cache preference
    cudaFuncSetCacheConfig(softplus_kernel, cudaFuncCachePreferL1);
    
    // Launch kernel with optimal grid configuration
    softplus_kernel<<<grid_size, block_size>>>(
        input_cont.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );
    return output;
}
// PART-END