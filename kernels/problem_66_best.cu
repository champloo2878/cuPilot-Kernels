// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int depth,
    const int height,
    const int width,
    const int out_channels,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int padding_d,
    const int padding_h,
    const int padding_w,
    const int dilation_d,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int depth_out,
    const int height_out,
    const int width_out,
    const int in_channels_per_group,
    const int n
) {
    extern __shared__ float shared_mem[];
    // Precomputed constants
    const int InC = 3;
    const int KD = 3;
    const int KH = 5;
    const int KW = 7;
    const int block_channels = 16;
    const int tile_h = 16;
    const int tile_w = 32;
    const int height_tile = tile_h + KH - 1;
    const int width_tile = tile_w + KW - 1;
    
    const int input_tile_size = InC * KD * height_tile * width_tile;
    const int input_tile_size_half = (input_tile_size + 1) / 2;
    const int weight_tile_size = InC * KD * KH * KW * block_channels;
    float* shared_input = shared_mem;
    float* shared_weights = shared_mem + input_tile_size;
    float* shared_bias = shared_weights + weight_tile_size;

    // Optimized block indexing
    const int out_channel_blocks = (out_channels + block_channels - 1) / block_channels;
    const int block_out_c = blockIdx.z % out_channel_blocks;
    const int batch_idx = blockIdx.z / (out_channel_blocks * depth_out);
    const int d_out = (blockIdx.z / out_channel_blocks) % depth_out;
    
    const int block_start_h = blockIdx.y * tile_h;
    const int block_start_w = blockIdx.x * tile_w;

    // Cooperative vectorized loading of input tile with float2
    for (int load_idx = threadIdx.y * blockDim.x + threadIdx.x; 
         load_idx < input_tile_size_half; 
         load_idx += blockDim.x * blockDim.y) {
        int linear_index = load_idx * 2;
        if (linear_index >= input_tile_size) break;
        
        int in_c = linear_index / (KD * height_tile * width_tile);
        int remainder = linear_index % (KD * height_tile * width_tile);
        int kd = remainder / (height_tile * width_tile);
        int rest = remainder % (height_tile * width_tile);
        int h_load = rest / width_tile;
        int w_load = rest % width_tile;
        
        int d_global = d_out + kd;
        int h_global = block_start_h + h_load;
        int w_global = block_start_w + w_load;
        int global_idx_base = ((batch_idx * in_channels + in_c) * depth + d_global) * 
                            (height * width) + h_global * width + w_global;
        
        // Vectorized load for two consecutive elements
        if (d_global < depth && h_global < height && w_global < width) {
            if (w_global + 1 < width) {
                float2 val = __ldg(reinterpret_cast<const float2*>(input + global_idx_base));
                shared_input[linear_index] = val.x;
                shared_input[linear_index+1] = val.y;
            } else {
                shared_input[linear_index] = input[global_idx_base];
                shared_input[linear_index+1] = (linear_index+1 < input_tile_size) ? 0.0f : 0.0f;
            }
        } else {
            shared_input[linear_index] = 0.0f;
            if (linear_index+1 < input_tile_size) {
                shared_input[linear_index+1] = 0.0f;
            }
        }
    }
    
    // Cooperative scalar loading of weights
    for (int load_idx = threadIdx.y * blockDim.x + threadIdx.x; 
         load_idx < weight_tile_size; 
         load_idx += blockDim.x * blockDim.y) {
        int in_c = load_idx / (KD * KH * KW * block_channels);
        int remainder = load_idx % (KD * KH * KW * block_channels);
        int kd = remainder / (KH * KW * block_channels);
        remainder %= (KH * KW * block_channels);
        int kh = remainder / (KW * block_channels);
        remainder %= (KW * block_channels);
        int kw = remainder / block_channels;
        int local_c = remainder % block_channels;
        
        if (block_out_c * block_channels + local_c < out_channels) {
            int weight_idx = ((block_out_c * block_channels + local_c) * in_channels_per_group + in_c) * 
                            KD * KH * KW + kd * (KH * KW) + kh * KW + kw;
            shared_weights[load_idx] = __ldg(weight + weight_idx);
        } else {
            shared_weights[load_idx] = 0.0f;
        }
    }
    
    // Vectorized bias loading with float4
    if (threadIdx.y == 0 && threadIdx.x < 4) {
        int local_c_base = threadIdx.x * 4;
        if (bias) {
            if (block_out_c * block_channels + local_c_base + 3 < out_channels) {
                float4 val = __ldg(reinterpret_cast<const float4*>(bias + block_out_c * block_channels + local_c_base));
                *reinterpret_cast<float4*>(shared_bias + local_c_base) = val;
            } else {
                for (int i = 0; i < 4; i++) {
                    int c = local_c_base + i;
                    if (c < block_channels && (block_out_c * block_channels + c) < out_channels) {
                        shared_bias[c] = __ldg(bias + block_out_c * block_channels + c);
                    } else if (c < block_channels) {
                        shared_bias[c] = 0.0f;
                    }
                }
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (local_c_base + i < block_channels) {
                    shared_bias[local_c_base + i] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Thread-local computation with vectorization
    const int h_thread = threadIdx.y;
    const int w_thread = threadIdx.x;
    const int h_out = block_start_h + h_thread;
    const int w_out = block_start_w + w_thread;
    float results[16] = {0.0f};
    
    if (h_out < height_out && w_out < width_out) {
        #pragma unroll
        for (int in_c = 0; in_c < InC; in_c++) {
            #pragma unroll
            for (int kd = 0; kd < KD; kd++) {
                #pragma unroll
                for (int kh = 0; kh < KH; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < KW; kw++) {
                        int input_idx = in_c * (KD * height_tile * width_tile) + 
                                      kd * (height_tile * width_tile) +
                                      (h_thread + kh) * width_tile + 
                                      (w_thread + kw);
                        float input_val = shared_input[input_idx];
                        
                        // Vectorized weight access with float4
                        int weight_base = (in_c * KD * KH * KW + 
                                         kd * KH * KW + 
                                         kh * KW + kw) * block_channels;
                        float4* weight_vec = reinterpret_cast<float4*>(&shared_weights[weight_base]);
                        float4 w0 = weight_vec[0];
                        float4 w1 = weight_vec[1];
                        float4 w2 = weight_vec[2];
                        float4 w3 = weight_vec[3];
                        
                        // Manually unrolled accumulation
                        results[0] += input_val * w0.x;
                        results[1] += input_val * w0.y;
                        results[2] += input_val * w0.z;
                        results[3] += input_val * w0.w;
                        results[4] += input_val * w1.x;
                        results[5] += input_val * w1.y;
                        results[6] += input_val * w1.z;
                        results[7] += input_val * w1.w;
                        results[8] += input_val * w2.x;
                        results[9] += input_val * w2.y;
                        results[10] += input_val * w2.z;
                        results[11] += input_val * w2.w;
                        results[12] += input_val * w3.x;
                        results[13] += input_val * w3.y;
                        results[14] += input_val * w3.z;
                        results[15] += input_val * w3.w;
                    }
                }
            }
        }
        
        // Precompute base output index and valid channels
        int base_channel = block_out_c * block_channels;
        int base_out_idx = batch_idx * (out_channels * depth_out * height_out * width_out) +
                         base_channel * (depth_out * height_out * width_out) +
                         d_out * (height_out * width_out) +
                         h_out * width_out +
                         w_out;

        // Fused bias addition and output store
        #pragma unroll
        for (int local_c = 0; local_c < block_channels; local_c++) {
            output[base_out_idx + local_c * (depth_out * height_out * width_out)] = 
                results[local_c] + shared_bias[local_c];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    int64_t depth_out,
    int64_t height_out,
    int64_t width_out
) {
    // Get tensor dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, input.options());
    
    // Get pointers
    const float* input_ptr = input.contiguous().data_ptr<float>();
    const float* weight_ptr = weight.contiguous().data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.contiguous().data_ptr<float>() : nullptr;
    float* output_ptr = output.data_ptr<float>();
    
    // Optimized kernel launch parameters
    const int block_channels = 16;
    const int tile_h = 16;
    const int tile_w = 32;
    const int out_channel_blocks = (out_channels + block_channels - 1) / block_channels;
    const int total_z = batch_size * depth_out * out_channel_blocks;
    
    dim3 gridDim(
        (width_out + tile_w - 1) / tile_w,
        (height_out + tile_h - 1) / tile_h,
        total_z
    );
    dim3 blockDim(tile_w, tile_h, 1);

    // Precomputed shared memory configuration
    const int InC = 3;
    const int KD = 3;
    const int KH = 5;
    const int KW = 7;
    const int height_tile = tile_h + KH - 1;
    const int width_tile = tile_w + KW - 1;
    const size_t input_shared = InC * KD * height_tile * width_tile * sizeof(float);
    const size_t weight_shared = InC * KD * KH * KW * block_channels * sizeof(float);
    const size_t bias_shared = block_channels * sizeof(float);
    const size_t total_shared = input_shared + weight_shared + bias_shared;
    
    // Launch optimized kernel
    conv3d_kernel<<<gridDim, blockDim, total_shared>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        depth,
        height,
        width,
        out_channels,
        kernel_d,
        kernel_h,
        kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        depth_out,
        height_out,
        width_out,
        in_channels / groups,
        output.numel()
    );
    
    return output;
}
// PART-END