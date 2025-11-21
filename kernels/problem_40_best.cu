// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector>

namespace cg = cooperative_groups;
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void layer_norm_forward_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    const float epsilon,
    const int num_elements,
    const int normalized_size
) {
    const int tid = threadIdx.x;
    const int batch_idx = blockIdx.x;
    const int offset = batch_idx * normalized_size;
    
    float thread_sum = 0.0f;
    float thread_sqsum = 0.0f;

    // Vectorized path
    if (normalized_size % 4 == 0) {
        int num_vec = normalized_size / 4;
        int vec_per_thread = (num_vec + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < vec_per_thread; i++) {
            int vec_index = tid + i * blockDim.x;
            if (vec_index < num_vec) {
                int base_idx = offset + 4 * vec_index;
                float4 in_vec = *reinterpret_cast<const float4*>(input + base_idx);
                thread_sum += in_vec.x + in_vec.y + in_vec.z + in_vec.w;
                thread_sqsum += in_vec.x * in_vec.x + in_vec.y * in_vec.y + in_vec.z * in_vec.z + in_vec.w * in_vec.w;
            }
        }
    } 
    // Scalar path
    else {
        for (int i = tid; i < normalized_size; i += blockDim.x) {
            float val = input[offset + i];
            thread_sum += val;
            thread_sqsum += val * val;
        }
    }

    // Warp-level reduction using cooperative groups
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    float warp_sqsum = cg::reduce(warp, thread_sqsum, cg::plus<float>());

    // Use warp leader to store results
    extern __shared__ float shared_sums[];
    if (warp.thread_rank() == 0) {
        int warp_id = warp.meta_group_rank();
        shared_sums[2 * warp_id] = warp_sum;
        shared_sums[2 * warp_id + 1] = warp_sqsum;
    }
    
    cg::sync(cg::this_thread_block());

    // Final reduction using first warp
    if (warp.meta_group_rank() == 0) {
        float final_sum = warp.thread_rank() < (blockDim.x / 32) ? shared_sums[2 * warp.thread_rank()] : 0.0f;
        float final_sqsum = warp.thread_rank() < (blockDim.x / 32) ? shared_sums[2 * warp.thread_rank() + 1] : 0.0f;
        
        final_sum = cg::reduce(warp, final_sum, cg::plus<float>());
        final_sqsum = cg::reduce(warp, final_sqsum, cg::plus<float>());
        
        if (warp.thread_rank() == 0) {
            shared_sums[0] = final_sum;
            shared_sums[1] = final_sqsum;
        }
    }
    
    cg::sync(cg::this_thread_block());

    // Compute statistics
    float total_sum = shared_sums[0];
    float total_sqsum = shared_sums[1];
    float mean = total_sum / normalized_size;
    float variance = total_sqsum / normalized_size - mean * mean;
    float inv_std = rsqrtf(variance + epsilon);

    // Write results (vectorized)
    if (normalized_size % 4 == 0) {
        int num_vec = normalized_size / 4;
        int vec_per_thread = (num_vec + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < vec_per_thread; i++) {
            int vec_index = tid + i * blockDim.x;
            if (vec_index < num_vec) {
                int base_idx = offset + 4 * vec_index;
                float4 in_vec = *reinterpret_cast<const float4*>(input + base_idx);
                float4 gamma_vec = *reinterpret_cast<const float4*>(gamma + 4 * vec_index);
                float4 beta_vec = *reinterpret_cast<const float4*>(beta + 4 * vec_index);
                float4 out_vec;
                out_vec.x = (in_vec.x - mean) * inv_std * gamma_vec.x + beta_vec.x;
                out_vec.y = (in_vec.y - mean) * inv_std * gamma_vec.y + beta_vec.y;
                out_vec.z = (in_vec.z - mean) * inv_std * gamma_vec.z + beta_vec.z;
                out_vec.w = (in_vec.w - mean) * inv_std * gamma_vec.w + beta_vec.w;
                *reinterpret_cast<float4*>(output + base_idx) = out_vec;
            }
        }
    } 
    // Write results (scalar)
    else {
        for (int i = tid; i < normalized_size; i += blockDim.x) {
            float val = input[offset + i];
            float normalized = (val - mean) * inv_std;
            output[offset + i] = normalized * gamma[i] + beta[i];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon
) {
    input = input.contiguous();
    auto output = torch::empty_like(input);
    const int normalized_size = gamma.numel();
    const int num_batches = input.numel() / normalized_size;

    int block_size;
    if (normalized_size % 4 == 0) {
        int num_vec = normalized_size / 4;
        block_size = std::min(1024, (num_vec + 31) / 32 * 32);
    } else {
        block_size = std::min(256, normalized_size);
        block_size = (block_size + 31) / 32 * 32;
    }
    const int num_warps = (block_size + 31) / 32;
    const int shared_mem_size = 2 * num_warps * sizeof(float);

    layer_norm_forward_kernel<<<num_batches, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        epsilon,
        input.numel(),
        normalized_size
    );

    return output;
}
// PART-END