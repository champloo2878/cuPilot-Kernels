// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Warp reduction helper
static __inline__ __device__ float warpReduceSum(float val) {
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FULL_MASK, val, mask);
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void instance_norm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_elements,
    const int channels,
    const int height,
    const int width,
    const float epsilon
) {
    const int n = blockIdx.y;
    const int c = blockIdx.x;
    const int plane_size = height * width;
    const int base_idx = n * channels * plane_size + c * plane_size;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int chunk_size = (plane_size + num_warps - 1) / num_warps;
    const int start = warp_id * chunk_size;
    const int end = min((warp_id + 1) * chunk_size, plane_size);

    float warp_sum = 0.0f;
    float warp_sqsum = 0.0f;

    // Vectorized reduction with stride 128 (32 threads * 4 elements)
    #pragma unroll 4
    for (int idx = start + lane_id*4; idx < end; idx += 128) {
        float4 data = *reinterpret_cast<const float4*>(input + base_idx + idx);
        warp_sum += data.x + data.y + data.z + data.w;
        warp_sqsum += data.x*data.x + data.y*data.y + data.z*data.z + data.w*data.w;
    }

    // Use cooperative groups for warp reduction
    warp_sum = cg::reduce(warp, warp_sum, cg::plus<float>());
    warp_sqsum = cg::reduce(warp, warp_sqsum, cg::plus<float>());

    // Bank conflict-free shared memory layout with padding
    extern __shared__ float shared_data[];
    // Pad each warp's data to avoid bank conflicts (32 banks on A100)
    float* warp_sums = shared_data;
    float* warp_sqsums = warp_sums + (num_warps + 1); // +1 for padding

    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
        warp_sqsums[warp_id] = warp_sqsum;
    }
    block.sync();

    // Consolidated reduction using cooperative groups
    cg::thread_block_tile<32> first_warp = cg::tiled_partition<32>(block);
    if (warp_id == 0) {
        float block_sum = 0.0f;
        float block_sqsum = 0.0f;
        
        if (lane_id < num_warps) {
            block_sum = warp_sums[lane_id];
            block_sqsum = warp_sqsums[lane_id];
        }
        
        block_sum = cg::reduce(first_warp, block_sum, cg::plus<float>());
        block_sqsum = cg::reduce(first_warp, block_sqsum, cg::plus<float>());

        if (lane_id == 0) {
            float mean_val = block_sum / plane_size;
            float variance = block_sqsum / plane_size - mean_val * mean_val;
            warp_sums[0] = mean_val;  // Store mean at shared_data[0]
            warp_sqsums[0] = rsqrtf(variance + epsilon);  // Store inv_std at shared_data[num_warps + 1]
        }
    }
    block.sync();

    const float mean = warp_sums[0];
    const float inv_std = warp_sqsums[0];
    const float bias = -mean * inv_std;  // Precompute bias term

    // Optimized vectorized normalization with cooperative groups
    const int vectorized_size = plane_size / 4;
    const int threads_per_instance = blockDim.x;
    
    #pragma unroll 4
    for (int idx = tid; idx < vectorized_size; idx += threads_per_instance) {
        int offset = base_idx + idx * 4;
        float4 in4 = *reinterpret_cast<const float4*>(input + offset);
        float4 out4;
        out4.x = in4.x * inv_std + bias;
        out4.y = in4.y * inv_std + bias;
        out4.z = in4.z * inv_std + bias;
        out4.w = in4.w * inv_std + bias;
        *reinterpret_cast<float4*>(output + offset) = out4;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor instance_norm_forward(
    torch::Tensor input,
    float epsilon
) {
    auto output = torch::empty_like(input);
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int plane_size = height * width;

    // Ensure tensors are 16-byte aligned
    TORCH_CHECK(plane_size % 4 == 0, 
                "plane_size must be divisible by 4 for vectorized operations");

    const int threads = 1024;  // Max threads for A100
    dim3 grid(channels, batch_size);
    const int num_warps = threads / 32;
    // Updated shared memory calculation with padding to avoid bank conflicts
    size_t shared_mem_size = (2 * num_warps + 2) * sizeof(float);  // warp_sums + warp_sqsums + padding

    instance_norm_forward_kernel<<<grid, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        channels,
        height,
        width,
        epsilon
    );

    return output;
}
// PART-END