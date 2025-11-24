// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template <int warpSize=32>
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <int warpSize=32>
__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = warpSize/2; offset > 0; offset >>= 1) 
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

// Memory coalescing optimization: restructure data access patterns
template<typename T, int VEC_SIZE>
struct CoalescedAccess {
    static __device__ __forceinline__ void load(const T* input, T vals[VEC_SIZE], int base_idx) {
        if constexpr (VEC_SIZE == 8) {
            // Use float4 for optimal coalesced access (128-bit transactions)
            float4 vec0 = reinterpret_cast<const float4*>(input)[base_idx / 4];
            float4 vec1 = reinterpret_cast<const float4*>(input)[base_idx / 4 + 1];
            
            vals[0] = vec0.x; vals[1] = vec0.y; 
            vals[2] = vec0.z; vals[3] = vec0.w;
            vals[4] = vec1.x; vals[5] = vec1.y; 
            vals[6] = vec1.z; vals[7] = vec1.w;
        } else if constexpr (VEC_SIZE == 4) {
            float4 vec = reinterpret_cast<const float4*>(input)[base_idx / 4];
            vals[0] = vec.x; vals[1] = vec.y; 
            vals[2] = vec.z; vals[3] = vec.w;
        }
    }
    
    static __device__ __forceinline__ void store(T* output, const T vals[VEC_SIZE], int base_idx) {
        if constexpr (VEC_SIZE == 8) {
            float4 vec0, vec1;
            vec0.x = vals[0]; vec0.y = vals[1];
            vec0.z = vals[2]; vec0.w = vals[3];
            vec1.x = vals[4]; vec1.y = vals[5];
            vec1.z = vals[6]; vec1.w = vals[7];
            
            reinterpret_cast<float4*>(output)[base_idx / 4] = vec0;
            reinterpret_cast<float4*>(output)[base_idx / 4 + 1] = vec1;
        } else if constexpr (VEC_SIZE == 4) {
            float4 vec;
            vec.x = vals[0]; vec.y = vals[1];
            vec.z = vals[2]; vec.w = vals[3];
            reinterpret_cast<float4*>(output)[base_idx / 4] = vec;
        }
    }
};
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void sum_scaled_squares_kernel(
    const float* __restrict__ input, 
    float* __restrict__ block_results,
    float* __restrict__ global_scale,
    int num_elements
) {
    constexpr int UNROLL_FACTOR = 32;
    constexpr int VEC_SIZE = 8;
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    
    extern __shared__ __align__(4) char shared_buf[];
    float* warp_sums = reinterpret_cast<float*>(shared_buf);
    float* warp_maxs = warp_sums + WARPS_PER_BLOCK;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Optimized memory access pattern: consecutive threads access consecutive elements
    int elements_per_block = THREADS_PER_BLOCK * VEC_SIZE;
    int grid_stride = gridDim.x * elements_per_block;
    int unrolled_stride = UNROLL_FACTOR * grid_stride;
    
    float local_max = 0.0f;
    float local_sum = 0.0f;
    
    // Base index calculation ensures coalesced access
    int base_idx = (blockIdx.x * THREADS_PER_BLOCK + tid) * VEC_SIZE;

    // First pass: find max absolute value with coalesced access
    for (int base = 0; base < unrolled_stride; base += grid_stride) {
        int vec_idx = base_idx + base;
        if (vec_idx + VEC_SIZE - 1 < num_elements) {
            float vals[VEC_SIZE];
            CoalescedAccess<float, VEC_SIZE>::load(input, vals, vec_idx);
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float abs_val = fabsf(vals[j]);
                local_max = fmaxf(local_max, abs_val);
            }
        }
    }

    // Coalesced tail processing for max
    if (lane_id < 16) {
        int tail_base = base_idx + (lane_id * 2);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int pos = tail_base + j * 32;
            if (pos < num_elements) {
                float val1 = input[pos];
                float val2 = (pos+1 < num_elements) ? input[pos+1] : 0.0f;
                local_max = fmaxf(local_max, fmaxf(fabsf(val1), fabsf(val2)));
            }
        }
    }

    // Reduce max within warp
    float warp_max = warpReduceMax(local_max);
    if (lane_id == 0) {
        warp_maxs[warp_id] = warp_max;
    }
    __syncthreads();

    // Reduce max across warps in block
    if (warp_id == 0) {
        float block_max = (lane_id < WARPS_PER_BLOCK) ? warp_maxs[lane_id] : 0.0f;
        block_max = warpReduceMax(block_max);
        if (lane_id == 0) {
            atomicMax(reinterpret_cast<int*>(global_scale), __float_as_int(block_max));
        }
    }
    __syncthreads();
    
    // Second pass: compute scaled squares with coalesced access
    float scale = *global_scale;
    if (scale < 1e-12f) scale = 1.0f;
    float inv_scale = 1.0f / scale;
    
    for (int base = 0; base < unrolled_stride; base += grid_stride) {
        int vec_idx = base_idx + base;
        if (vec_idx + VEC_SIZE - 1 < num_elements) {
            float vals[VEC_SIZE];
            CoalescedAccess<float, VEC_SIZE>::load(input, vals, vec_idx);
            
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                float scaled_val = vals[j] * inv_scale;
                local_sum += scaled_val * scaled_val;
            }
        }
    }
    
    // Coalesced tail processing for scaled squares
    if (lane_id < 16) {
        int tail_base = base_idx + (lane_id * 2);
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int pos = tail_base + j * 32;
            if (pos < num_elements) {
                float val1 = input[pos] * inv_scale;
                float val2 = (pos+1 < num_elements) ? input[pos+1] * inv_scale : 0.0f;
                local_sum += val1 * val1 + val2 * val2;
            }
        }
    }
    
    // Reduce sum within warp
    float warp_sum = warpReduceSum(local_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Reduce sum across warps in block
    if (warp_id == 0 && lane_id < WARPS_PER_BLOCK) {
        float val = warp_sums[lane_id];
        float block_sum = warpReduceSum(val);
        if (lane_id == 0) {
            block_results[blockIdx.x] = block_sum * scale * scale;
        }
    }
}

__global__ void normalize_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    float norm, 
    int num_elements
) {
    constexpr int UNROLL_FACTOR = 32;
    constexpr int VEC_SIZE = 8;
    constexpr int THREADS_PER_BLOCK = 256;
    
    float scale = 1.0f / norm;
    
    // Optimized memory access pattern
    int elements_per_block = THREADS_PER_BLOCK * VEC_SIZE;
    int grid_stride = gridDim.x * elements_per_block;
    int unrolled_stride = UNROLL_FACTOR * grid_stride;
    
    int tid = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
    int base_idx = tid * VEC_SIZE;
    
    // Main processing with coalesced vector loads/stores
    for (int base = 0; base < unrolled_stride; base += grid_stride) {
        int vec_idx = base_idx + base;
        if (vec_idx + VEC_SIZE - 1 < num_elements) {
            float input_vals[VEC_SIZE];
            CoalescedAccess<float, VEC_SIZE>::load(input, input_vals, vec_idx);
            
            float output_vals[VEC_SIZE];
            #pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                output_vals[j] = input_vals[j] * scale;
            }
            
            CoalescedAccess<float, VEC_SIZE>::store(output, output_vals, vec_idx);
        }
    }
    
    // Optimized tail processing with coalesced access
    if (threadIdx.x < 8) {
        int tail_base = base_idx + (threadIdx.x * 4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int pos = tail_base + j * 32;
            if (pos < num_elements) {
                // Process 4 elements at a time for better coalescing
                float4 vec;
                if (pos + 3 < num_elements) {
                    vec = reinterpret_cast<const float4*>(input)[pos / 4];
                    vec.x *= scale; vec.y *= scale; vec.z *= scale; vec.w *= scale;
                    reinterpret_cast<float4*>(output)[pos / 4] = vec;
                } else {
                    // Handle boundary case
                    for (int k = 0; k < 4 && (pos + k) < num_elements; k++) {
                        output[pos + k] = input[pos + k] * scale;
                    }
                }
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor frobenius_normalize_cuda(torch::Tensor input) {
    input = input.contiguous();
    int num_elements = input.numel();
    if (num_elements == 0) return input.clone();
    
    const int block_size = 256;
    const int vec_size = 8;
    
    // Optimized grid size calculation for A100 (108 SMs)
    int max_blocks = 108 * 4;  // 4 blocks per SM for better occupancy
    int elements_per_block = block_size * vec_size;
    int num_blocks = std::min((num_elements + elements_per_block - 1) / elements_per_block, max_blocks);
    
    auto block_results = torch::zeros({num_blocks}, input.options());
    auto global_scale = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    
    size_t shared_mem = (block_size / 32) * (2 * sizeof(float));
    
    sum_scaled_squares_kernel<<<num_blocks, block_size, shared_mem>>>(
        input.data_ptr<float>(),
        block_results.data_ptr<float>(),
        global_scale.data_ptr<float>(),
        num_elements
    );
    
    auto total_sum_tensor = torch::sum(block_results);
    float total_sum = total_sum_tensor.item<float>();
    
    float norm = std::sqrt(total_sum);
    if (norm == 0.0f) norm = 1.0f;
    
    auto output = torch::empty_like(input);
    int norm_blocks = std::min((num_elements + elements_per_block - 1) / elements_per_block, max_blocks);
    normalize_kernel<<<norm_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        norm,
        num_elements
    );
    
    return output;
}
// PART-END