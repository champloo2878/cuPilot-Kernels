// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(mask, val, offset);
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void group_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W,
    const int num_groups,
    const float eps
) {
    extern __shared__ float s_buffer[];
    const int group_size = C / num_groups;
    const int group_elements = group_size * H * W;
    const int n = blockIdx.x / num_groups;
    const int g = blockIdx.x % num_groups;
    const int c_start = g * group_size;
    const int group_offset = n * C * H * W + c_start * H * W;
    const float* group_input = input + group_offset;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    float* warp_sums = s_buffer;
    float* warp_sq_sums = warp_sums + 32;
    float* s_weight = warp_sq_sums + 32;
    float* s_bias = s_weight + group_size;
    
    const int n_vectors = group_elements / 4;
    const int vectors_per_warp = (n_vectors + num_warps - 1) / num_warps;
    const int warp_base = warp_id * vectors_per_warp;
    const int warp_end = min(warp_base + vectors_per_warp, n_vectors);
    
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    for (int i = warp_base + lane_id; i < warp_end; i += 32) {
        float4 vals = *reinterpret_cast<const float4*>(&group_input[i * 4]);
        thread_sum += vals.x + vals.y + vals.z + vals.w;
        thread_sq_sum += vals.x*vals.x + vals.y*vals.y + vals.z*vals.z + vals.w*vals.w;
    }
    
    float warp_sum = warpReduceSum(thread_sum);
    float warp_sq_sum = warpReduceSum(thread_sq_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
        warp_sq_sums[warp_id] = warp_sq_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        float v_sq_sum = (lane_id < num_warps) ? warp_sq_sums[lane_id] : 0.0f;
        v_sum = warpReduceSum(v_sum);
        v_sq_sum = warpReduceSum(v_sq_sum);
        
        if (lane_id == 0) {
            warp_sums[0] = v_sum;
            warp_sq_sums[0] = v_sq_sum;
        }
    }
    __syncthreads();
    
    if (tid < group_size) {
        s_weight[tid] = weight[c_start + tid];
        s_bias[tid] = bias[c_start + tid];
    }
    __syncthreads();
    
    const float total_sum = warp_sums[0];
    const float total_sq_sum = warp_sq_sums[0];
    const float mean = total_sum / group_elements;
    const float variance = total_sq_sum / group_elements - mean * mean;
    const float inv_std = rsqrtf(variance + eps);
    
    const int spatial_vectors = H * W / 4;
    const int vec_per_warp_per_channel = (spatial_vectors + num_warps - 1) / num_warps;
    
    // Double-buffering implementation for normalization phase
    const int prefetch_distance = 1;
    const int iterations_per_channel = (vec_per_warp_per_channel + 31) / 32;
    
    for (int c = c_start; c < c_start + group_size; ++c) {
        const int channel_offset = n * C * H * W + c * H * W;
        const float* channel_input = input + channel_offset;
        float* channel_output = output + channel_offset;
        const int local_c = c - c_start;
        const float wgt = s_weight[local_c];
        const float bia = s_bias[local_c];
        
        const int warp_base_spatial = warp_id * vec_per_warp_per_channel;
        const int warp_end_spatial = min(warp_base_spatial + vec_per_warp_per_channel, spatial_vectors);
        
        // Double-buffering registers
        float4 in4_buffer[2];
        float4 out4_buffer[2];
        int buffer_idx = 0;
        
        // Prefetch first element
        int current_idx = warp_base_spatial + lane_id;
        if (current_idx < warp_end_spatial) {
            in4_buffer[buffer_idx] = *reinterpret_cast<const float4*>(&channel_input[current_idx * 4]);
        }
        
        // Main loop with double-buffering
        for (int i = 0; i < iterations_per_channel; ++i) {
            // Prefetch next element if available
            int next_idx = current_idx + 32;
            if (i < iterations_per_channel - 1 && next_idx < warp_end_spatial) {
                in4_buffer[(buffer_idx + 1) % 2] = *reinterpret_cast<const float4*>(&channel_input[next_idx * 4]);
            }
            
            // Process current element
            if (current_idx < warp_end_spatial) {
                float4 in4 = in4_buffer[buffer_idx];
                out4_buffer[buffer_idx].x = (in4.x - mean) * inv_std * wgt + bia;
                out4_buffer[buffer_idx].y = (in4.y - mean) * inv_std * wgt + bia;
                out4_buffer[buffer_idx].z = (in4.z - mean) * inv_std * wgt + bia;
                out4_buffer[buffer_idx].w = (in4.w - mean) * inv_std * wgt + bia;
                
                // Write back processed element
                *reinterpret_cast<float4*>(&channel_output[current_idx * 4]) = out4_buffer[buffer_idx];
            }
            
            // Switch buffers
            buffer_idx = (buffer_idx + 1) % 2;
            current_idx = next_idx;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor group_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.defined() && bias.defined(), "Weight and bias must be defined");
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    TORCH_CHECK((H * W) % 4 == 0, "Spatial dimensions must be divisible by 4 for vectorization");
    
    auto output = torch::empty_like(input);
    const int num_blocks = N * num_groups;
    const int group_size = C / num_groups;
    const int spatial_size = H * W;
    
    // Optimized thread block configuration based on spatial dimensions and group size
    int block_size;
    if (spatial_size >= 262144) { // 512x512 = 262144
        // For large spatial dimensions and small group sizes, use smaller blocks for better occupancy
        if (group_size <= 16) {
            block_size = 512;  // Better occupancy for small groups
        } else {
            block_size = 1024; // Default large block size
        }
    } else if (spatial_size >= 65536) {
        block_size = 512;
    } else {
        block_size = 256;
    }
    
    // Ensure block_size is a multiple of warp size (32) and doesn't exceed max threads per block
    block_size = min(1024, (block_size + 31) / 32 * 32);
    
    const size_t shared_mem = (64 + 2 * group_size) * sizeof(float);
    
    group_norm_kernel<<<num_blocks, block_size, shared_mem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W, num_groups, eps
    );
    
    return output;
}
// PART-END