// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    static_assert(kColStride <= 16, "kColStride must <= 16");
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
    static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
    if constexpr (kStep == 8) {
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        static_assert(kStep == 4, "kStep must be 4 if not 8");
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

template<const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__device__ __inline__ float warpReduceSum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float beta,
    float* partial_sums,
    int total_elements
) {
    extern __shared__ float warp_sums[];
    int tid = threadIdx.x;
    int lane_id = tid & 0x1F;
    int warp_id = tid >> 5;
    int swizzled_lane = swizzle_permuted_A_j(warp_id, lane_id);
    int swizzled_tid = (warp_id << 5) | swizzled_lane;
    constexpr int ELEMENTS_PER_THREAD = 128;
    int base_idx = (blockIdx.x * blockDim.x + swizzled_tid) * ELEMENTS_PER_THREAD;
    
    float local_sum = 0.0f;
    float half_inv_beta = 0.5f / beta;
    float half_beta = 0.5f * beta;
    int remaining = total_elements - base_idx;
    int num_valid = min(ELEMENTS_PER_THREAD, remaining);
    
    if (num_valid <= 0) {
        // Skip processing if no valid elements
    } else if (num_valid == ELEMENTS_PER_THREAD) {
        #pragma unroll
        for (int offset = 0; offset < ELEMENTS_PER_THREAD; offset += 16) {
            // Load 16 elements per iteration (4 float4 vectors)
            const float4 pred4_1 = *reinterpret_cast<const float4*>(predictions + base_idx + offset);
            const float4 pred4_2 = *reinterpret_cast<const float4*>(predictions + base_idx + offset + 4);
            const float4 pred4_3 = *reinterpret_cast<const float4*>(predictions + base_idx + offset + 8);
            const float4 pred4_4 = *reinterpret_cast<const float4*>(predictions + base_idx + offset + 12);
            
            const float4 targ4_1 = *reinterpret_cast<const float4*>(targets + base_idx + offset);
            const float4 targ4_2 = *reinterpret_cast<const float4*>(targets + base_idx + offset + 4);
            const float4 targ4_3 = *reinterpret_cast<const float4*>(targets + base_idx + offset + 8);
            const float4 targ4_4 = *reinterpret_cast<const float4*>(targets + base_idx + offset + 12);
            
            // Process 16 elements per iteration
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float diff = reinterpret_cast<const float*>(&pred4_1)[i] - reinterpret_cast<const float*>(&targ4_1)[i];
                float abs_diff = fabsf(diff);
                local_sum += (abs_diff < beta) ? (diff * diff * half_inv_beta) : (abs_diff - half_beta);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float diff = reinterpret_cast<const float*>(&pred4_2)[i] - reinterpret_cast<const float*>(&targ4_2)[i];
                float abs_diff = fabsf(diff);
                local_sum += (abs_diff < beta) ? (diff * diff * half_inv_beta) : (abs_diff - half_beta);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float diff = reinterpret_cast<const float*>(&pred4_3)[i] - reinterpret_cast<const float*>(&targ4_3)[i];
                float abs_diff = fabsf(diff);
                local_sum += (abs_diff < beta) ? (diff * diff * half_inv_beta) : (abs_diff - half_beta);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float diff = reinterpret_cast<const float*>(&pred4_4)[i] - reinterpret_cast<const float*>(&targ4_4)[i];
                float abs_diff = fabsf(diff);
                local_sum += (abs_diff < beta) ? (diff * diff * half_inv_beta) : (abs_diff - half_beta);
            }
        }
    } else {
        // Process boundary elements individually
        for (int i = 0; i < num_valid; ++i) {
            int idx = base_idx + i;
            float diff = predictions[idx] - targets[idx];
            float abs_diff = fabsf(diff);
            local_sum += (abs_diff < beta) ? (diff * diff * half_inv_beta) : (abs_diff - half_beta);
        }
    }
    
    local_sum = warpReduceSum(local_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        float val = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[threadIdx.x] : 0.0f;
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            partial_sums[blockIdx.x] = val;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor smooth_l1_loss_cuda(torch::Tensor predictions, torch::Tensor targets, float beta) {
    TORCH_CHECK(predictions.device().is_cuda(), "predictions must be CUDA tensor");
    TORCH_CHECK(targets.device().is_cuda(), "targets must be CUDA tensor");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input sizes must match");
    
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    int total_elements = predictions.numel();
    if (total_elements == 0) {
        return torch::tensor(0.0f, torch::TensorOptions().device(predictions.device()));
    }
    
    // Dynamic block size optimization
    constexpr int kNumCandidates = 3;
    int candidate_block_sizes[kNumCandidates] = {256, 512, 1024};
    int best_block_size = 1024;
    int max_active_warps = 0;
    int device_id = predictions.device().index();
    cudaSetDevice(device_id);
    
    for (int i = 0; i < kNumCandidates; ++i) {
        int block_size = candidate_block_sizes[i];
        int shared_mem = (block_size >> 5) * sizeof(float);
        
        int active_blocks;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, smooth_l1_loss_kernel, block_size, shared_mem) == cudaSuccess) {
            int active_warps = active_blocks * (block_size / 32);
            if (active_warps > max_active_warps) {
                max_active_warps = active_warps;
                best_block_size = block_size;
            }
        }
    }
    
    // Calculate grid size based on selected block size
    constexpr int ELEMENTS_PER_THREAD = 128;
    int grid_size = (total_elements + best_block_size * ELEMENTS_PER_THREAD - 1) / (best_block_size * ELEMENTS_PER_THREAD);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(predictions.device());
        
    torch::Tensor partial_sums = torch::zeros({grid_size}, options);
    
    int shared_mem_size = (best_block_size / 32) * sizeof(float);
    smooth_l1_loss_kernel<<<grid_size, best_block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        beta,
        partial_sums.data_ptr<float>(),
        total_elements
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "Kernel failed: ", cudaGetErrorString(err));
    }
    
    float total_sum = partial_sums.sum().item<float>();
    float mean_loss = total_sum / total_elements;
    
    return torch::tensor(mean_loss, options);
}
// PART-END