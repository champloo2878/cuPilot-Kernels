// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ __launch_bounds__(256, 8) void conv_transpose2d_kernel(
    const float* input, 
    const at::Half* weight_reordered,
    float* output,
    int batch_size, int out_channels, int H_out, int W_out,
    int in_channels, int H_in, int W_in,
    int kernel_size0, int kernel_size1,
    int stride0, int stride1,
    int padding0, int padding1,
    int dilation0, int dilation1,
    int groups,
    int in_channels_per_group,
    int out_channels_per_group,
    int total_elements
) {
    // Compute padded input channels (round to multiple of 4)
    const int padded_ic = (in_channels_per_group + 3) & ~3;
    extern __shared__ float shared_weights[];
    const int weight_block_size = kernel_size0 * kernel_size1 * padded_ic;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Decompose blockIdx.z for batch, group, and channel segments
    int channel_blocks_per_group = (out_channels_per_group + 7) / 8;
    int total_channel_blocks = groups * channel_blocks_per_group;
    int batch_idx = blockIdx.z / total_channel_blocks;
    int group_block_idx = blockIdx.z % total_channel_blocks;
    int group_idx = group_block_idx / channel_blocks_per_group;
    int channel_block_idx = group_block_idx % channel_blocks_per_group;
    int channel_start = channel_block_idx * 8;
    
    // Preload weights using float4 for vectorized access
    for (int idx = tid; idx < weight_block_size; idx += blockDim.x * blockDim.y) {
        int c = idx % padded_ic;
        int dx = (idx / padded_ic) % kernel_size1;
        int dy = idx / (padded_ic * kernel_size1);
        
        if (dy < kernel_size0 && dx < kernel_size1 && c < in_channels_per_group) {
            int shared_base = (dy * (kernel_size1 * padded_ic) + dx * padded_ic + c) * 8;
            float4* shared_ptr = reinterpret_cast<float4*>(shared_weights + shared_base);
            float values[8] = {0.0f};
            
            // Load 8 output channels with bounds checking and FP16->FP32 conversion
            #pragma unroll
            for (int oc = 0; oc < 8; oc++) {
                if (channel_start + oc < out_channels_per_group) {
                    int global_channel = group_idx * out_channels_per_group + channel_start + oc;
                    int global_idx = global_channel * (kernel_size0 * kernel_size1 * in_channels_per_group)
                                   + dy * (kernel_size1 * in_channels_per_group)
                                   + dx * in_channels_per_group
                                   + c;
                    values[oc] = __half2float(weight_reordered[global_idx]);
                }
            }
            // Store as two float4 vectors
            shared_ptr[0] = *reinterpret_cast<float4*>(&values[0]);
            shared_ptr[1] = *reinterpret_cast<float4*>(&values[4]);
        }
    }
    __syncthreads();

    // Process spatial elements
    int block_spatial_w = blockIdx.x * blockDim.x + threadIdx.x;
    int block_spatial_h = blockIdx.y * blockDim.y + threadIdx.y;

    if (block_spatial_w >= W_out || block_spatial_h >= H_out) return;

    // Validate output channels
    bool valid_channels[8] = {
        channel_start + 0 < out_channels_per_group,
        channel_start + 1 < out_channels_per_group,
        channel_start + 2 < out_channels_per_group,
        channel_start + 3 < out_channels_per_group,
        channel_start + 4 < out_channels_per_group,
        channel_start + 5 < out_channels_per_group,
        channel_start + 6 < out_channels_per_group,
        channel_start + 7 < out_channels_per_group
    };

    float results[8] = {0.0f};
    int group_channel_offset = group_idx * in_channels_per_group;
    int batch_offset = batch_idx * in_channels * H_in * W_in;

    for (int dy = 0; dy < kernel_size0; dy++) {
        for (int dx = 0; dx < kernel_size1; dx++) {
            int h_in = block_spatial_h + padding0 - dy * dilation0;
            int w_in = block_spatial_w + padding1 - dx * dilation1;
            
            if ((h_in % stride0 == 0) && (w_in % stride1 == 0)) {
                h_in /= stride0;
                w_in /= stride1;
                
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const float* input_ptr = input + batch_offset
                                          + group_channel_offset * (H_in * W_in)
                                          + h_in * W_in
                                          + w_in;
                    
                    int weight_base = (dy * (kernel_size1 * padded_ic) + dx * padded_ic) * 8;
                    
                    // Process input channels in groups
                    #pragma unroll
                    for (int c = 0; c < in_channels_per_group; c++) {
                        float input_val = input_ptr[c * (H_in * W_in)];
                        float* weight_ptr = shared_weights + weight_base + c * 8;
                        
                        // Load two float4 vectors for 8 channels
                        float4 weights0 = *reinterpret_cast<float4*>(weight_ptr);
                        float4 weights1 = *reinterpret_cast<float4*>(weight_ptr + 4);
                        
                        // Process 8 channels using FMA
                        results[0] = fmaf(input_val, weights0.x, results[0]);
                        results[1] = fmaf(input_val, weights0.y, results[1]);
                        results[2] = fmaf(input_val, weights0.z, results[2]);
                        results[3] = fmaf(input_val, weights0.w, results[3]);
                        results[4] = fmaf(input_val, weights1.x, results[4]);
                        results[5] = fmaf(input_val, weights1.y, results[5]);
                        results[6] = fmaf(input_val, weights1.z, results[6]);
                        results[7] = fmaf(input_val, weights1.w, results[7]);
                    }
                }
            }
        }
    }
    
    // Store results with boundary checks
    int output_offset_base = batch_idx * (out_channels * H_out * W_out)
                          + group_idx * out_channels_per_group * (H_out * W_out)
                          + block_spatial_h * W_out
                          + block_spatial_w;
    
    #pragma unroll
    for (int oc = 0; oc < 8; oc++) {
        if (valid_channels[oc]) {
            output[output_offset_base + (channel_start + oc) * (H_out * W_out)] = results[oc];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight_reordered,
    int out_channels,
    int kernel_size0, int kernel_size1,
    int stride0, int stride1,
    int padding0, int padding1,
    int output_padding0, int output_padding1,
    int dilation0, int dilation1,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int H_out = (H_in - 1) * stride0 - 2 * padding0 + dilation0 * (kernel_size0 - 1) + output_padding0 + 1;
    int W_out = (W_in - 1) * stride1 - 2 * padding1 + dilation1 * (kernel_size1 - 1) + output_padding1 + 1;
    
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;
    
    auto output = torch::zeros({batch_size, out_channels, H_out, W_out}, input.options());
    int total_elements = batch_size * out_channels * H_out * W_out;
    
    // Convert weights to FP16 for memory efficiency
    torch::Tensor weight_reordered_half = weight_reordered.to(torch::kFloat16);

    // Optimized block size (256 threads)
    dim3 block_dim(32, 8);
    int grid_x = (W_out + block_dim.x - 1) / block_dim.x;
    int grid_y = (H_out + block_dim.y - 1) / block_dim.y;
    int channel_blocks = (out_channels_per_group + 7) / 8;
    int grid_z = batch_size * groups * channel_blocks;
    dim3 grid_dim(grid_x, grid_y, grid_z);
    
    // Calculate shared memory size (padded to 8 output channels)
    int padded_ic = (in_channels_per_group + 3) & ~3;
    int shared_mem_size = kernel_size0 * kernel_size1 * padded_ic * 8 * sizeof(float);
    
    conv_transpose2d_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight_reordered_half.data_ptr<at::Half>(),
        output.data_ptr<float>(),
        batch_size, out_channels, H_out, W_out,
        in_channels, H_in, W_in,
        kernel_size0, kernel_size1,
        stride0, stride1,
        padding0, padding1,
        dilation0, dilation1,
        groups,
        in_channels_per_group,
        out_channels_per_group,
        total_elements
    );
    
    return output;
}
// PART-END