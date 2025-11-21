// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized swizzle permutation for A100
template<const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    return (((j >> 3) ^ (i >> 2)) % (kMmaAtomK >> 3)) << 3;
}

// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void kl_div_kernel(const float* __restrict__ predictions, 
                              const float* __restrict__ targets, 
                              float* __restrict__ per_sample_loss, 
                              int batch_size, 
                              int feature_size) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int sample_idx = blockIdx.x * blockDim.y + warp_id;
    if (sample_idx >= batch_size) return;

    constexpr int kElementsPerThread = 32;
    constexpr int kElementsPerChunk = 32 * kElementsPerThread;
    const int chunks_per_sample = (feature_size + kElementsPerChunk - 1) / kElementsPerChunk;
    float local_sum = 0.0f;

    for (int chunk = 0; chunk < chunks_per_sample; ++chunk) {
        const int base_index = chunk * kElementsPerChunk + swizzle_permuted_A_j(warp_id, lane_id * kElementsPerThread);
        const float* target_ptr = targets + sample_idx * feature_size + base_index;
        const float* pred_ptr = predictions + sample_idx * feature_size + base_index;

        if (base_index + kElementsPerThread <= feature_size) {
            #pragma unroll
            for (int vec = 0; vec < 8; ++vec) {
                // Use read-only cache and aligned loads
                const float4 t = __ldg(reinterpret_cast<const float4*>(target_ptr + vec * 16));
                const float4 p = __ldg(reinterpret_cast<const float4*>(pred_ptr + vec * 16));
                
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float t_val = (&t.x)[j];
                    const float p_val = (&p.x)[j];
                    
                    // Optimized branchless computation
                    const float log_t = t_val != 0.0f ? __logf(t_val) : 0.0f;
                    const float log_p = (t_val != 0.0f && p_val != 0.0f) ? __logf(p_val) : 
                                        (t_val != 0.0f ? -100.0f : 0.0f);
                    local_sum += t_val * (log_t - log_p);
                }
            }
        } else {
            for (int j = 0; j < kElementsPerThread; ++j) {
                const int idx = base_index + j;
                if (idx >= feature_size) break;
                
                // Use read-only cache
                const float t_val = __ldg(target_ptr + j);
                const float p_val = __ldg(pred_ptr + j);
                
                // Optimized branchless computation
                const float log_t = t_val != 0.0f ? __logf(t_val) : 0.0f;
                const float log_p = (t_val != 0.0f && p_val != 0.0f) ? __logf(p_val) : 
                                    (t_val != 0.0f ? -100.0f : 0.0f);
                local_sum += t_val * (log_t - log_p);
            }
        }
    }

    // Fully unrolled warp-level reduction
    float warp_sum = local_sum;
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 16);
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 8);
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 4);
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 2);
    warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, 1);
    
    if (lane_id == 0) {
        atomicAdd(&per_sample_loss[sample_idx], warp_sum);
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor kl_div_forward_cuda(torch::Tensor predictions, 
                                  torch::Tensor targets) {
    const int batch_size = predictions.size(0);
    const int feature_size = predictions.size(1);
    auto per_sample_loss = torch::zeros({batch_size}, predictions.options());
    
    constexpr int warps_per_block = 32;
    constexpr int threads_per_warps = 32;
    constexpr int threads_per_block = warps_per_block * threads_per_warps;
    const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;
    
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(blocks);
    
    // Ensure inputs are contiguous and 16-byte aligned
    kl_div_kernel<<<grid_dim, block_dim, 0>>>(
        predictions.contiguous().data_ptr<float>(),
        targets.contiguous().data_ptr<float>(),
        per_sample_loss.data_ptr<float>(),
        batch_size,
        feature_size
    );
    
    return per_sample_loss.mean();
}
// PART-END