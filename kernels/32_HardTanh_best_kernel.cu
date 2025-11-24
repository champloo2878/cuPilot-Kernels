// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Swizzle function to permute thread indices for better memory coalescing
template<const int kElementsPerThread = 4>
static __device__ __forceinline__ int swizzle_thread_idx(int thread_idx, int block_dim) {
    int warp_id = thread_idx / 32;
    int lane_id = thread_idx % 32;
    
    // Permute lanes within warp to ensure consecutive threads access consecutive memory
    int permuted_lane_id = (lane_id ^ (warp_id & 1)) & 31;
    
    return warp_id * 32 + permuted_lane_id;
}

// Calculate the actual memory index after swizzle
template<const int kElementsPerThread = 4>
static __device__ __forceinline__ int swizzle_memory_index(int base_idx, int thread_idx, int block_dim) {
    int swizzled_thread = swizzle_thread_idx<kElementsPerThread>(thread_idx, block_dim);
    return base_idx + swizzled_thread * kElementsPerThread;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void hardtanh_kernel(const float* __restrict__ input, float* __restrict__ output, int num_elements, float min_val, float max_val) {
    extern __shared__ int s_block_data[];
    int* s_block_start = s_block_data;
    int* s_block_end = s_block_data + 1;
    
    if (threadIdx.x == 0) {
        const int elements_per_block = 4 * blockDim.x;
        s_block_start[0] = blockIdx.x * elements_per_block;
        s_block_end[0] = min(s_block_start[0] + elements_per_block, num_elements);
    }
    __syncthreads();
    
    int block_start = s_block_start[0];
    int block_end = s_block_end[0];
    
    // Early exit for empty blocks
    if (block_start >= block_end) return;
    
    // Apply swizzle to thread index for better memory access pattern
    int swizzled_thread_idx_val = swizzle_thread_idx<4>(threadIdx.x, blockDim.x);
    
    // Calculate starting index using swizzled thread index
    int idx = block_start + swizzled_thread_idx_val * 4;
    
    // Check if this warp can process full vector loads
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    int warp_full = 0;
    if (lane_id == 0) {
        int warp_start = block_start + warp_id * 128;
        int warp_end = min(warp_start + 128, block_end);
        warp_full = (warp_end - warp_start) == 128;
    }
    warp_full = __shfl_sync(0xFFFFFFFF, warp_full, 0);
    
    if (warp_full) {
        // Full warp processing with coalesced memory access
        if (idx + 3 < block_end) {
            float4 in = __ldg(reinterpret_cast<const float4*>(input + idx));
            float4 out;
            out.x = fminf(fmaxf(in.x, min_val), max_val);
            out.y = fminf(fmaxf(in.y, min_val), max_val);
            out.z = fminf(fmaxf(in.z, min_val), max_val);
            out.w = fminf(fmaxf(in.w, min_val), max_val);
            *reinterpret_cast<float4*>(output + idx) = out;
        }
    } else {
        // Boundary handling with swizzled access
        if (idx + 3 < block_end) {
            float4 in = __ldg(reinterpret_cast<const float4*>(input + idx));
            float4 out;
            out.x = fminf(fmaxf(in.x, min_val), max_val);
            out.y = fminf(fmaxf(in.y, min_val), max_val);
            out.z = fminf(fmaxf(in.z, min_val), max_val);
            out.w = fminf(fmaxf(in.w, min_val), max_val);
            *reinterpret_cast<float4*>(output + idx) = out;
        } else {
            // Handle remaining elements with scalar operations
            for (int i = 0; i < 4; i++) {
                int element_idx = idx + i;
                if (element_idx < block_end) {
                    output[element_idx] = fminf(fmaxf(__ldg(input + element_idx), min_val), max_val);
                }
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int threads = 1024;
    int blocks = (num_elements + 4 * threads - 1) / (4 * threads);
    size_t shared_mem_size = 2 * sizeof(int);
    
    hardtanh_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        min_val,
        max_val
    );
    
    return output;
}
// PART-END