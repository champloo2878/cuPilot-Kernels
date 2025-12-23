// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

constexpr int WARP_SIZE = 32;

// Struct to hold running max and sum for online softmax
struct ReduceData {
    float m; // max value
    float s; // sum of exponentials
};

// Vectorized streaming load helper using inline PTX to bypass L1 cache
__device__ __forceinline__ float4 load_cs_v4(const float* ptr) {
    float4 val;
    asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];" 
        : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w) 
        : "l"(ptr));
    return val;
}

// Reduction operator for merging two (max, sum) pairs
// Optimized to be concise and handle -FLT_MAX implicitly with ternary operators
__device__ __forceinline__ ReduceData reduce_op(ReduceData a, ReduceData b) {
    bool a_bigger = (a.m >= b.m);
    float max_val = a_bigger ? a.m : b.m;
    float diff = a_bigger ? (b.m - a.m) : (a.m - b.m);
    // If a is bigger: a.s + b.s * exp(b.m - a.m)
    // If b is bigger: b.s + a.s * exp(a.m - b.m)
    float scale = expf(diff);
    float sum_val = a_bigger ? (a.s + b.s * scale) : (b.s + a.s * scale);
    return {max_val, sum_val};
}

// Warp-level reduction
__device__ __forceinline__ ReduceData warpReduceMaxSum(ReduceData val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        float other_m = __shfl_down_sync(0xFFFFFFFF, val.m, offset);
        float other_s = __shfl_down_sync(0xFFFFFFFF, val.s, offset);
        val = reduce_op(val, {other_m, other_s});
    }
    return val;
}

// Block-level reduction
__device__ __forceinline__ ReduceData blockReduceMaxSum(ReduceData val) {
    __shared__ float shared_m[32]; 
    __shared__ float shared_s[32];
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    val = warpReduceMaxSum(val);
    
    if (lane_id == 0) {
        shared_m[warp_id] = val.m;
        shared_s[warp_id] = val.s;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        ReduceData warp_val = {-FLT_MAX, 0.0f};
        int num_warps = blockDim.x / WARP_SIZE;
        if (lane_id < num_warps) {
            warp_val.m = shared_m[lane_id];
            warp_val.s = shared_s[lane_id];
        }
        val = warpReduceMaxSum(warp_val);
    }
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void __launch_bounds__(256) cross_entropy_kernel(
    const float* __restrict__ predictions,
    const int64_t* __restrict__ targets,
    float* per_row_loss,
    int num_classes,
    int batch_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= batch_size) return;
    
    // Row pointer
    const float* row_preds = predictions + row * num_classes;
    
    // Initialize thread local state
    ReduceData thread_data = {-FLT_MAX, 0.0f};
    
    // Main loop: Process 4 elements at a time using float4
    int i = tid * 4;
    int stride = blockDim.x * 4;
    int limit = (num_classes / 4) * 4;
    
    // Unroll loop to improve instruction scheduling and latency hiding
    #pragma unroll 4
    for (; i < limit; i += stride) {
        float4 vals = load_cs_v4(row_preds + i);
        
        float v0 = vals.x;
        float v1 = vals.y;
        float v2 = vals.z;
        float v3 = vals.w;
        
        // Local reduction within the vector
        float local_max = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
        float local_sum = expf(v0 - local_max) + expf(v1 - local_max) + 
                          expf(v2 - local_max) + expf(v3 - local_max);
        
        // Merge local result into thread accumulator using optimized online update
        if (local_max > thread_data.m) {
            float scale = expf(thread_data.m - local_max);
            thread_data.s = thread_data.s * scale + local_sum;
            thread_data.m = local_max;
        } else {
            float scale = expf(local_max - thread_data.m);
            thread_data.s += local_sum * scale;
        }
    }
    
    // Handle remaining elements if num_classes is not a multiple of 4
    for (int j = limit + tid; j < num_classes; j += blockDim.x) {
        float val = row_preds[j];
        if (val > thread_data.m) {
            float scale = expf(thread_data.m - val);
            thread_data.s = thread_data.s * scale + 1.0f;
            thread_data.m = val;
        } else {
            thread_data.s += expf(val - thread_data.m);
        }
    }
    
    // Block-wide reduction
    ReduceData result = blockReduceMaxSum(thread_data);
    
    if (tid == 0) {
        // Fetch target value directly from global memory
        int64_t target_idx = targets[row];
        float target_val = row_preds[target_idx];
        
        // Calculate Cross Entropy Loss
        // Loss = log(sum_exp) + max - target
        float log_sum_exp = logf(result.s) + result.m;
        per_row_loss[row] = log_sum_exp - target_val;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor cross_entropy_cuda(torch::Tensor predictions, torch::Tensor targets) {
    int batch_size = predictions.size(0);
    int num_classes = predictions.size(1);
    
    // Use empty for performance as every element will be written
    auto per_row_loss = torch::empty({batch_size}, predictions.options());
    
    // Use block size 256 for optimal occupancy and register usage
    dim3 grid(batch_size);
    dim3 block(256);
    
    cross_entropy_kernel<<<grid, block>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<int64_t>(),
        per_row_loss.data_ptr<float>(),
        num_classes,
        batch_size
    );
    
    return per_row_loss;
}
// PART-END