// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// PART-END

// PART-START
namespace cg = cooperative_groups;

__device__ __forceinline__ void elu_vector(float4& out_vec, const float4 in_vec, float alpha) {
    out_vec.x = (in_vec.x > 0) ? in_vec.x : alpha * (expf(in_vec.x) - 1);
    out_vec.y = (in_vec.y > 0) ? in_vec.y : alpha * (expf(in_vec.y) - 1);
    out_vec.z = (in_vec.z > 0) ? in_vec.z : alpha * (expf(in_vec.z) - 1);
    out_vec.w = (in_vec.w > 0) ? in_vec.w : alpha * (expf(in_vec.w) - 1);
}

__global__ void elu_kernel(const float* __restrict__ input, float* output, float alpha, int num_elements) {
    constexpr int VEC = 4;
    constexpr int ELEMENTS_PER_THREAD = VEC;
    constexpr int THREADS_PER_WARP = 32;
    
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<THREADS_PER_WARP> warp = cg::tiled_partition<THREADS_PER_WARP>(tb);
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / THREADS_PER_WARP;
    const int lane_id = threadIdx.x % THREADS_PER_WARP;
    const int warps_per_block = blockDim.x / THREADS_PER_WARP;
    const int total_warps = warps_per_block * gridDim.x;
    
    // Each warp processes contiguous chunks of data
    const int elements_per_warp = THREADS_PER_WARP * ELEMENTS_PER_THREAD; // 128 elements per warp
    const int warp_start_idx = warp_id * elements_per_warp;
    
    // Warp specialization: main warps handle full vectors, boundary warps handle remainder
    if (warp_start_idx < num_elements) {
        const int warp_end_idx = min(warp_start_idx + elements_per_warp, num_elements);
        const int elements_this_warp = warp_end_idx - warp_start_idx;
        
        // Check if this warp handles boundary conditions
        bool is_boundary_warp = (warp_end_idx == num_elements) && 
                               (elements_this_warp % elements_per_warp != 0);
        
        if (!is_boundary_warp) {
            // Main warps: process full vector operations
            #pragma unroll
            for (int base_idx = warp_start_idx + lane_id * ELEMENTS_PER_THREAD; 
                 base_idx < warp_end_idx; 
                 base_idx += THREADS_PER_WARP * ELEMENTS_PER_THREAD) {
                
                if (base_idx + ELEMENTS_PER_THREAD <= num_elements) {
                    // Full vector load/store
                    float4 in_vec = *reinterpret_cast<const float4*>(input + base_idx);
                    float4 out_vec;
                    elu_vector(out_vec, in_vec, alpha);
                    *reinterpret_cast<float4*>(output + base_idx) = out_vec;
                }
            }
        } else {
            // Boundary warps: handle the remainder with individual element processing
            #pragma unroll
            for (int base_idx = warp_start_idx + lane_id; 
                 base_idx < warp_end_idx; 
                 base_idx += THREADS_PER_WARP) {
                
                float x = __ldg(input + base_idx);
                output[base_idx] = (x > 0) ? x : alpha * (expf(x) - 1);
            }
        }
        
        // Warp synchronization to ensure all threads complete
        warp.sync();
    }
}
// PART-END

// PART-START
torch::Tensor elu_forward_cuda(torch::Tensor input, float alpha) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    constexpr int VEC_SIZE = 4;
    constexpr int ELEMENTS_PER_THREAD = VEC_SIZE;
    constexpr int THREADS_PER_WARP = 32;
    const int block_size = 1024;  // Max threads per block for A100
    const int warps_per_block = block_size / THREADS_PER_WARP;
    
    // Calculate optimal grid size: each warp handles 32 * 4 = 128 elements
    const int elements_per_warp = THREADS_PER_WARP * ELEMENTS_PER_THREAD;
    int total_warps_needed = (num_elements + elements_per_warp - 1) / elements_per_warp;
    int grid_size = (total_warps_needed + warps_per_block - 1) / warps_per_block;
    
    // Ensure grid_size doesn't exceed A100 limits and is sufficient for full utilization
    grid_size = min(grid_size, 2147483647);
    grid_size = max(grid_size, 108);  // At least one block per SM for A100 (108 SMs)
    
    if (grid_size > 0) {
        elu_kernel<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            alpha,
            num_elements
        );
    }
    
    return output;
}
// PART-END