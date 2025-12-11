// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>

// Warp-level inclusive scan with mask condition optimized for A100
template<typename T, int WARP_SIZE = 32>
__device__ __forceinline__ T warp_scan_inclusive_masked(T val, bool mask, unsigned int lane_id) {
    T result = mask ? val : T(0);
    
    // Optimized warp shuffle sequence for A100 (faster than initial version)
    T shfl_val = result;
    
    // Use fewer shuffle steps with wider offsets for A100's warp shuffle latency
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        T n = __shfl_up_sync(0xffffffff, shfl_val, offset);
        if (lane_id >= offset) shfl_val += n;
    }
    
    return shfl_val;
}

// Block-level scan with warp-level cooperation optimized for A100
template<typename T, int BLOCK_SIZE = 256>
__device__ __forceinline__ T block_scan_inclusive_masked(T val, bool mask, T* shared_scan, 
                                                         unsigned int lane_id, unsigned int warp_id) {
    // First do warp-level scan
    T warp_result = warp_scan_inclusive_masked<T>(val, mask, lane_id);
    
    // Get warp sum for inter-warp accumulation
    T warp_sum = __shfl_sync(0xffffffff, warp_result, 31);
    
    // Store warp sum in shared memory for block-level scan
    if (lane_id == 31) {
        shared_scan[warp_id] = warp_sum;
    }
    
    __syncthreads();
    
    // First warp performs block-level scan on warp sums
    if (warp_id == 0) {
        T warp_val = lane_id < (BLOCK_SIZE / 32) ? shared_scan[lane_id] : T(0);
        T scanned = warp_scan_inclusive_masked<T>(warp_val, true, lane_id);
        if (lane_id < (BLOCK_SIZE / 32)) {
            shared_scan[lane_id] = scanned;
        }
    }
    
    __syncthreads();
    
    // Add warp prefix to individual results
    T warp_prefix = warp_id > 0 ? shared_scan[warp_id - 1] : T(0);
    return warp_result + warp_prefix;
}

// Block reduction sum for computing chunk totals optimized for A100
template<typename T, int BLOCK_SIZE = 256>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[BLOCK_SIZE / 32];
    
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;
    
    // Warp-level reduction optimized for A100
    T warp_sum = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    
    // Store warp sum in shared memory
    if (lane_id == 0) {
        shared[warp_id] = warp_sum;
    }
    
    __syncthreads();
    
    // First warp reduces warp sums
    T block_sum = T(0);
    if (warp_id == 0) {
        T val2 = lane_id < (BLOCK_SIZE / 32) ? shared[lane_id] : T(0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val2 += __shfl_down_sync(0xffffffff, val2, offset);
        }
        if (lane_id == 0) {
            block_sum = val2;
        }
    }
    
    __syncthreads();
    return block_sum;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T>
__global__ void masked_cumsum_kernel(const T* __restrict__ input, 
                                   const bool* __restrict__ mask, 
                                   T* __restrict__ output,
                                   int64_t dim_size, 
                                   int64_t inner_size, 
                                   int64_t outer_size) {
    // Using 256 threads per block (8 warps) for optimal occupancy on A100
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    
    // Shared memory for block-level scan - aligned for A100's 128-byte cache line
    __shared__ __align__(128) T shared_scan[WARPS_PER_BLOCK];
    __shared__ T chunk_total_shared;
    
    // Calculate flattened outer index
    int64_t total_outer = outer_size * inner_size;
    int64_t outer_flat = blockIdx.x;
    
    if (outer_flat >= total_outer) return;
    
    // Calculate base index for this outer position
    int64_t base_idx = outer_flat * dim_size;
    
    // Thread index within block
    unsigned int thread_id = threadIdx.x;
    unsigned int lane_id = thread_id % WARP_SIZE;
    unsigned int warp_id = thread_id / WARP_SIZE;
    
    // Process the entire dimension for this outer position
    T carry = T(0);
    
    // Prefetch first chunk to hide memory latency
    int64_t prefetch_idx = base_idx + thread_id;
    T prefetch_val = T(0);
    bool prefetch_mask = false;
    if (thread_id < dim_size) {
        prefetch_val = input[prefetch_idx];
        prefetch_mask = mask[prefetch_idx];
    }
    
    for (int64_t chunk_start = 0; chunk_start < dim_size; chunk_start += BLOCK_SIZE) {
        // Current chunk values
        T val = prefetch_val;
        bool m = prefetch_mask;
        
        // Prefetch next chunk if not at the end
        int64_t next_chunk_start = chunk_start + BLOCK_SIZE;
        if (next_chunk_start < dim_size) {
            int64_t next_idx = base_idx + next_chunk_start + thread_id;
            if (next_chunk_start + thread_id < dim_size) {
                prefetch_val = input[next_idx];
                prefetch_mask = mask[next_idx];
            } else {
                prefetch_val = T(0);
                prefetch_mask = false;
            }
        } else {
            prefetch_val = T(0);
            prefetch_mask = false;
        }
        
        // Perform block-level scan for this chunk
        T scanned = block_scan_inclusive_masked<T, BLOCK_SIZE>(val, m, shared_scan, lane_id, warp_id);
        
        // Add carry from previous chunks
        T result = scanned + carry;
        
        // Write result to output
        if (chunk_start + thread_id < dim_size) {
            output[base_idx + chunk_start + thread_id] = result;
        }
        
        // Compute total sum of this chunk for next iteration's carry
        // Find the last valid thread in this chunk
        int64_t chunk_end = min(chunk_start + BLOCK_SIZE, dim_size);
        bool is_last_in_chunk = (chunk_start + thread_id == chunk_end - 1);
        
        // Store the scanned value of the last element in the chunk
        if (is_last_in_chunk) {
            chunk_total_shared = scanned;
        }
        
        __syncthreads();
        
        // Update carry for next chunk
        if (thread_id == 0) {
            carry += chunk_total_shared;
        }
        
        __syncthreads();
        
        // Broadcast carry to all threads using shared memory
        if (thread_id == 0) {
            shared_scan[0] = carry;
        }
        
        __syncthreads();
        
        // All threads read the updated carry
        carry = shared_scan[0];
        
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor masked_cumsum_cuda(torch::Tensor input, torch::Tensor mask, int64_t dim) {
    auto input_sizes = input.sizes();
    int64_t dim_size = input_sizes[dim];
    
    // Calculate inner and outer dimensions
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < input_sizes.size(); i++) {
        inner_size *= input_sizes[i];
    }
    
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= input_sizes[i];
    }
    
    auto output = torch::zeros_like(input);
    
    // Optimized block and grid configuration for A100
    constexpr int BLOCK_SIZE = 256;
    
    int64_t total_outer = outer_size * inner_size;
    
    // Calculate grid size - each block handles one outer position
    // Use exactly total_outer blocks to ensure all positions are covered
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((total_outer + 1) / 1);  // Simple ceil division
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "masked_cumsum_cuda", ([&] {
        masked_cumsum_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            output.data_ptr<scalar_t>(),
            dim_size,
            inner_size,
            outer_size
        );
    }));
    
    return output;
}
// PART-END