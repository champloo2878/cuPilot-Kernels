// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Optimized warp reduction using cooperative groups with warp shuffle
template<int warpSize = 32>
__device__ __forceinline__ float warpReduceSum(float val) {
    cg::coalesced_group warp = cg::coalesced_threads();
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }
    return val;
}

// Vectorized load with explicit cache hint for A100
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    float4 result;
    asm volatile("ld.global.ca.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(result.x), "=f"(result.y), "=f"(result.z), "=f"(result.w)
                : "l"(ptr)
                : "memory");
    return result;
}

// Optimized sqrt approximation using fused multiply-add
__device__ __forceinline__ float fast_sqrtf(float x) {
    // Initial approximation using rsqrt
    float y = __frsqrt_rn(x);
    // Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
    // Using FMA for better precision and performance
    y = y * __fmaf_rn(-0.5f * x, y * y, 1.5f);
    // Convert rsqrt to sqrt: sqrt(x) = x * rsqrt(x)
    return x * y;
}

// Prefetch function using ld.global.ca hint
__device__ __forceinline__ void prefetch_float4(const float* ptr) {
    asm volatile("ld.global.ca.v4.f32 {%0, %0, %0, %0}, [%1];"
                : 
                : "f"(0.0f), "l"(ptr)
                : "memory");
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void triplet_margin_loss_kernel(
    const float* __restrict__ anchor, 
    const float* __restrict__ positive, 
    const float* __restrict__ negative, 
    float* __restrict__ output, 
    const float margin, 
    const int batch_size, 
    const int feature_size) {
    
    // Optimized configuration: 8 warps per block (256 threads)
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    
    // Grid-stride loop with optimal load balancing
    for (int sample_idx = blockIdx.x * warps_per_block + warp_id; 
         sample_idx < batch_size; 
         sample_idx += gridDim.x * warps_per_block) {
        
        // Precompute offset to avoid repeated multiplication
        const int sample_offset = sample_idx * feature_size;
        const float* anchor_sample = anchor + sample_offset;
        const float* positive_sample = positive + sample_offset;
        const float* negative_sample = negative + sample_offset;
        
        // Vectorized processing with float4 (16 bytes per load)
        const int vectorized_size = feature_size / 4;
        const int total_threads = blockDim.x;  // 256 threads
        
        // Dual accumulators to break dependency chains
        float dist_ap0 = 0.0f, dist_ap1 = 0.0f;
        float dist_an0 = 0.0f, dist_an1 = 0.0f;
        
        // Main vectorized loop with software pipelining
        int vec_idx = lane_id + warp_id * 32;
        #pragma unroll(4)
        for (int i = 0; i < vectorized_size; i += total_threads) {
            // Prefetch next iteration's data to L1 cache
            if (vec_idx + total_threads < vectorized_size) {
                prefetch_float4(anchor_sample + (vec_idx + total_threads) * 4);
                prefetch_float4(positive_sample + (vec_idx + total_threads) * 4);
                prefetch_float4(negative_sample + (vec_idx + total_threads) * 4);
            }
            
            if (vec_idx < vectorized_size) {
                const int base_offset = vec_idx * 4;
                
                // Coalesced float4 loads with cache hint
                const float4 a_vec = load_float4(anchor_sample + base_offset);
                const float4 p_vec = load_float4(positive_sample + base_offset);
                const float4 n_vec = load_float4(negative_sample + base_offset);
                
                // Process 4 elements with unrolled dual accumulation
                // First 2 elements
                {
                    const float a0 = a_vec.x;
                    const float p0 = p_vec.x;
                    const float n0 = n_vec.x;
                    const float diff_ap0 = a0 - p0;
                    const float diff_an0 = a0 - n0;
                    dist_ap0 = __fmaf_rn(diff_ap0, diff_ap0, dist_ap0);
                    dist_an0 = __fmaf_rn(diff_an0, diff_an0, dist_an0);
                }
                {
                    const float a1 = a_vec.y;
                    const float p1 = p_vec.y;
                    const float n1 = n_vec.y;
                    const float diff_ap1 = a1 - p1;
                    const float diff_an1 = a1 - n1;
                    dist_ap1 = __fmaf_rn(diff_ap1, diff_ap1, dist_ap1);
                    dist_an1 = __fmaf_rn(diff_an1, diff_an1, dist_an1);
                }
                // Last 2 elements
                {
                    const float a2 = a_vec.z;
                    const float p2 = p_vec.z;
                    const float n2 = n_vec.z;
                    const float diff_ap2 = a2 - p2;
                    const float diff_an2 = a2 - n2;
                    dist_ap0 = __fmaf_rn(diff_ap2, diff_ap2, dist_ap0);
                    dist_an0 = __fmaf_rn(diff_an2, diff_an2, dist_an0);
                }
                {
                    const float a3 = a_vec.w;
                    const float p3 = p_vec.w;
                    const float n3 = n_vec.w;
                    const float diff_ap3 = a3 - p3;
                    const float diff_an3 = a3 - n3;
                    dist_ap1 = __fmaf_rn(diff_ap3, diff_ap3, dist_ap1);
                    dist_an1 = __fmaf_rn(diff_an3, diff_an3, dist_an1);
                }
            }
            vec_idx += total_threads;
        }
        
        // Combine dual accumulators
        float total_dist_ap = dist_ap0 + dist_ap1;
        float total_dist_an = dist_an0 + dist_an1;
        
        // Warp reduction for squared distances
        total_dist_ap = warpReduceSum(total_dist_ap);
        total_dist_an = warpReduceSum(total_dist_an);
        
        // Final computation in lane 0 only
        if (lane_id == 0) {
            // Convert squared L2 to L2 distance using optimized sqrt
            total_dist_ap = fast_sqrtf(total_dist_ap);
            total_dist_an = fast_sqrtf(total_dist_an);
            
            // Triplet margin loss with ReLU
            const float loss = __fmaf_rn(1.0f, total_dist_ap, __fmaf_rn(-1.0f, total_dist_an, margin));
            output[sample_idx] = fmaxf(loss, 0.0f);
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor, 
    torch::Tensor positive, 
    torch::Tensor negative, 
    float margin) {
    
    const int batch_size = anchor.size(0);
    const int feature_size = anchor.size(1);
    
    auto output = torch::empty({batch_size}, anchor.options());
    
    // Optimized launch configuration for A100 (108 SMs)
    // Using 256 threads per block (8 warps) for optimal occupancy
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    
    // Calculate optimal grid size based on A100 SMs and occupancy
    const int sm_count = 108;
    // Target 8 blocks per SM for maximum occupancy (2048 threads/SM / 256 = 8)
    const int target_blocks_per_sm = 8;
    const int max_blocks = sm_count * target_blocks_per_sm;
    
    // Minimum blocks needed to cover all samples
    const int min_blocks = (batch_size + warps_per_block - 1) / warps_per_block;
    
    // Use smaller grid for better cache locality while maintaining occupancy
    const int grid_size = (min_blocks < max_blocks) ? min_blocks : max_blocks;
    
    // Launch kernel with optimal configuration
    triplet_margin_loss_kernel<<<grid_size, threads_per_block>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(), 
        negative.data_ptr<float>(),
        output.data_ptr<float>(),
        margin,
        batch_size,
        feature_size
    );
    
    // Return mean of all individual losses
    return torch::mean(output);
}
// PART-END