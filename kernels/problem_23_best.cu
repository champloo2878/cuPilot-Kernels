#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// PART-END

// Part 1: (top-level header files and preprocessing functions)
// PART-START
__device__ __inline__ float fast_exp(float x) {
    const float max_x = 10.0f;
    const float min_x = -10.0f;
    x = fmaxf(fminf(x, max_x), min_x);

    const float log2e = 1.4426950408889635f;
    const float ln2 = 0.6931471805599453f;
    float m = floorf(x * log2e + 0.5f);
    float r = x - m * ln2;

    float p = ((((1.8775767e-3f) * r +
                (8.9893397e-3f)) * r +
                (5.5824188e-2f)) * r +
                (2.4015361e-1f)) * r +
                (6.9315308e-1f);
    p = p * r + 1.0f;

    int m_int = static_cast<int>(m);
    int w = __float_as_int(p) + (m_int << 23);
    return __int_as_float(w);
}

template<int WarpSize = 32>
__device__ __inline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<int WarpSize = 32>
__device__ __inline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Hierarchical atomic counter for reduced global memory contention
__device__ __inline__ int hierarchical_atomic_add(int* global_counter, int* sm_counter, int sm_id, int step) {
    __shared__ int local_counter;
    
    if (threadIdx.x == 0) {
        if (sm_counter[sm_id] <= 0) {
            // Refill SM counter from global with larger step size
            sm_counter[sm_id] = atomicAdd(global_counter, 108 * step); // 108 SMs on A100
            if (sm_counter[sm_id] < 0) sm_counter[sm_id] = 0;
        }
        local_counter = atomicAdd(&sm_counter[sm_id], -step);
    }
    __syncthreads();
    
    return local_counter;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ __launch_bounds__(1024, 2) 
void softmax_kernel(float* __restrict__ input, float* __restrict__ output, 
                    int num_rows, int num_cols, int* __restrict__ global_row_counter,
                    int* __restrict__ sm_counters) {
    // Persistent thread pool with hierarchical counter system
    __shared__ int row_index;
    __shared__ float row_max;
    __shared__ float row_sum;
    const int warps_per_block = blockDim.x / 32;
    __shared__ float max_shared[32];
    __shared__ float sum_shared[32];
    
    // Optimized shared memory layout with linear indexing to avoid bank conflicts
    extern __shared__ float exp_cache[];
    const int cache_entries = blockDim.x * 4;
    float* exp_cache_ptr = exp_cache;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int sm_id = blockIdx.x % 108; // SM-aware work distribution for A100
    
    // Precompute constants
    const int stride = blockDim.x * 4;
    const float neg_inf = -FLT_MAX;
    
    // Hierarchical work distribution with reduced global atomic contention
    for (unsigned int base_row = hierarchical_atomic_add(global_row_counter, sm_counters, sm_id, gridDim.x); 
         base_row < num_rows; 
         base_row = hierarchical_atomic_add(global_row_counter, sm_counters, sm_id, gridDim.x)) {
        
        // Claim a row to process with minimal synchronization
        if (threadIdx.x == 0) {
            row_index = base_row;
        }
        __syncthreads();
        
        if (row_index >= num_rows) break;
        
        float* row_input = input + row_index * num_cols;
        float* row_output = output + row_index * num_cols;
        
        // Step 1: Optimized max reduction with vectorized loads
        float local_max = neg_inf;
        
        // Use aligned vectorized loads for better memory coalescing
        int start_idx = threadIdx.x * 4;
        for (int k = 0; start_idx + k * stride < num_cols; k++) {
            int idx = start_idx + k * stride;
            if (idx < num_cols) {
                const float4 vec = *reinterpret_cast<const float4*>(row_input + idx);
                local_max = fmaxf(local_max, vec.x);
                local_max = fmaxf(local_max, vec.y);
                local_max = fmaxf(local_max, vec.z);
                local_max = fmaxf(local_max, vec.w);
            }
        }
        
        // Optimized warp-level max reduction
        float warp_max = warpReduceMax<32>(local_max);
        
        // Store warp results with bank conflict avoidance
        if (lane_id == 0) {
            max_shared[warp_id] = warp_max;
        }
        __syncthreads();
        
        // Final reduction with warp 0 only to reduce synchronization overhead
        if (warp_id == 0) {
            float val = (lane_id < warps_per_block) ? max_shared[lane_id] : neg_inf;
            float block_max = warpReduceMax<32>(val);
            if (lane_id == 0) {
                row_max = block_max;
            }
        }
        __syncthreads();
        
        // Step 2: Optimized exp computation and caching with linear indexing
        float local_sum = 0.0f;
        const float row_max_val = row_max;  // Cache in register
        
        start_idx = threadIdx.x * 4;
        for (int k = 0; start_idx + k * stride < num_cols; k++) {
            int idx = start_idx + k * stride;
            if (idx < num_cols) {
                const float4 vec = *reinterpret_cast<const float4*>(row_input + idx);
                
                // Optimized exp computation with register caching
                float exp0 = fast_exp(vec.x - row_max_val);
                float exp1 = fast_exp(vec.y - row_max_val);
                float exp2 = fast_exp(vec.z - row_max_val);
                float exp3 = fast_exp(vec.w - row_max_val);
                
                // Linear indexing with modulo to avoid bank conflicts and reuse cache
                int cache_index = (threadIdx.x * 4 + k * 4) % cache_entries;
                exp_cache_ptr[cache_index] = exp0;
                exp_cache_ptr[cache_index + 1] = exp1;
                exp_cache_ptr[cache_index + 2] = exp2;
                exp_cache_ptr[cache_index + 3] = exp3;
                
                // Accumulate sum with fused operations
                local_sum += exp0 + exp1 + exp2 + exp3;
            }
        }
        
        // Optimized sum reduction
        float warp_sum = warpReduceSum<32>(local_sum);
        
        if (lane_id == 0) {
            sum_shared[warp_id] = warp_sum;
        }
        __syncthreads();
        
        // Final sum reduction with warp 0 only
        if (warp_id == 0) {
            float val = (lane_id < warps_per_block) ? sum_shared[lane_id] : 0.0f;
            float block_sum = warpReduceSum<32>(val);
            if (lane_id == 0) {
                row_sum = block_sum;
            }
        }
        __syncthreads();
        
        // Step 3: Optimized normalization with precomputed inverse
        const float inv_sum = __fdividef(1.0f, row_sum);  // Use faster division
        
        start_idx = threadIdx.x * 4;
        for (int k = 0; start_idx + k * stride < num_cols; k++) {
            int idx = start_idx + k * stride;
            if (idx < num_cols) {
                int cache_index = (threadIdx.x * 4 + k * 4) % cache_entries;
                
                float4 result;
                result.x = exp_cache_ptr[cache_index] * inv_sum;
                result.y = exp_cache_ptr[cache_index + 1] * inv_sum;
                result.z = exp_cache_ptr[cache_index + 2] * inv_sum;
                result.w = exp_cache_ptr[cache_index + 3] * inv_sum;
                
                // Use aligned vectorized stores
                *reinterpret_cast<float4*>(row_output + idx) = result;
            }
        }
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    
    // Optimized grid configuration for A100 with hierarchical counter system
    const int num_blocks = 216;  // 2 blocks/SM for 108 SMs
    const int threads_per_block = 1024;
    
    // Create global counter and SM counters for hierarchical work distribution
    auto counter_options = torch::TensorOptions().dtype(torch::kInt32).device(input.device());
    torch::Tensor global_counter = torch::zeros({1}, counter_options);
    
    // Create SM counters (108 SMs on A100) for reduced global atomic contention
    torch::Tensor sm_counters = torch::zeros({108}, counter_options);
    
    // Optimized shared memory calculation
    size_t static_shared = sizeof(int) + 2*sizeof(float) + 2*32*sizeof(float);
    size_t dynamic_shared = threads_per_block * 4 * sizeof(float);
    
    // Launch kernel with hierarchical counter system
    softmax_kernel<<<num_blocks, threads_per_block, static_shared + dynamic_shared>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        num_cols,
        global_counter.data_ptr<int>(),
        sm_counters.data_ptr<int>()
    );
    
    return output;
}
// PART-END