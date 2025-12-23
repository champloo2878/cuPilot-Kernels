// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// PART-START
// Helper for warp reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void hinge_loss_kernel(
    const float* __restrict__ predictions, 
    const float* __restrict__ targets, 
    float* __restrict__ partial_sums, 
    int total_elements, 
    int num_features
) {
    // Optimization: Vectorized Loads (float4) + ILP Unrolling (4x) + Fast Indexing + Block/Warp Reduction
    
    // Grid-stride loop setup
    int tid = threadIdx.x;
    int block_stride = blockDim.x;
    int grid_stride = gridDim.x * block_stride;
    
    // We unroll by 4, so the effective stride is 4 * grid_stride
    // Each thread processes 4 * float4 elements per iteration
    int idx = blockIdx.x * block_stride * 4 + tid;
    int stride = grid_stride * 4;

    float sum = 0.0f;
    
    // Vectorized pointer
    const float4* preds_vec = reinterpret_cast<const float4*>(predictions);
    int vec_limit = total_elements / 4;
    
    // Calculate shift for fast division: num_features is 32768 (2^15)
    // float4 index divisor = 32768 / 4 = 8192 (2^13)
    // We compute shift dynamically to be safe but it will be 13 for the task
    // __ffs returns 1-based index of first set bit. 
    int shift = __ffs(num_features / 4) - 1;

    // Manual unrolling loop
    for (int i = idx; i < vec_limit; i += stride) {
        float4 p[4];
        bool mask[4];

        // Load 4 float4 vectors
        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            int curr_i = i + k * block_stride;
            mask[k] = (curr_i < vec_limit);
            if (mask[k]) {
                p[k] = preds_vec[curr_i];
            }
        }

        // Process loaded values
        #pragma unroll
        for(int k = 0; k < 4; ++k) {
            if (mask[k]) {
                int curr_i = i + k * block_stride;
                
                // Fast row index calculation using shift (replaces division)
                int target_idx = curr_i >> shift;
                
                // Load target (likely cached in L1/L2)
                float t = targets[target_idx];
                
                float4 val = p[k];
                // Hinge loss: max(0, 1 - p*t)
                // Use fmaf and fmaxf for performance
                float v1 = fmaxf(0.0f, 1.0f - val.x * t);
                float v2 = fmaxf(0.0f, 1.0f - val.y * t);
                float v3 = fmaxf(0.0f, 1.0f - val.z * t);
                float v4 = fmaxf(0.0f, 1.0f - val.w * t);
                
                sum += (v1 + v2 + v3 + v4);
            }
        }
    }
    
    // Block Reduction
    // 1. Warp Reduce
    sum = warpReduceSum(sum);
    
    // 2. Shared Memory Reduce (Inter-warp)
    static __shared__ float shared_sums[32]; 
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    if (lane == 0) {
        shared_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // 3. Final reduction by first warp
    int num_warps = blockDim.x / 32;
    sum = (tid < num_warps) ? shared_sums[tid] : 0.0f;
    
    if (warp_id == 0) {
        sum = warpReduceSum(sum);
        // Thread 0 now has the block sum
        if (tid == 0) {
            atomicAdd(partial_sums, sum);
        }
    }
}

void hinge_loss_launch(
    const float* predictions, 
    const float* targets, 
    float* partial_sums, 
    int total_elements, 
    int num_features, 
    int grid_size, 
    int block_size
) {
    hinge_loss_kernel<<<grid_size, block_size>>>(
        predictions, targets, partial_sums, total_elements, num_features
    );
}
// PART-END

// PART-START
void hinge_loss_forward(
    at::Tensor predictions, 
    at::Tensor targets, 
    at::Tensor partial_sums, 
    int total_elements, 
    int num_features, 
    int grid_size, 
    int block_size
) {
    // Ensure the accumulator is zeroed out before accumulation
    partial_sums.zero_();

    // Optimize launch configuration for A100 (108 SMs)
    // Use 256 threads per block for high occupancy.
    // Set grid size to saturate SMs but limit atomic contention on the single global scalar.
    // 108 SMs * 8 blocks/SM = 864 blocks.
    int new_block_size = 256;
    int new_grid_size = 108 * 8; 

    const float* pred_ptr = predictions.data_ptr<float>();
    const float* targ_ptr = targets.data_ptr<float>();
    float* part_ptr = partial_sums.data_ptr<float>();
    
    hinge_loss_launch(pred_ptr, targ_ptr, part_ptr, total_elements, num_features, new_grid_size, new_block_size);
}
// PART-END