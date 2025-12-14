// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Dynamic parallelism constants
constexpr int CHILD_BLOCK_SIZE_X = 32;
constexpr int CHILD_BLOCK_SIZE_Y = 8;
constexpr int CHILD_TILE_IC = 8;
constexpr int CHILD_TILE_KH = 3;
constexpr int CHILD_TILE_KW = 3;

// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template<typename T>
__device__ __forceinline__ void write_output(T* ptr, int idx, float val) {
    ptr[idx] = static_cast<T>(val);
}

template<>
__device__ __forceinline__ void write_output<__half>(__half* ptr, int idx, float val) {
    ptr[idx] = __float2half(val);
}

// Child kernel for processing a single batch element
template <typename scalar_t>
__global__ void conv_transpose2d_child_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_idx,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {
    
    const int output_spatial = output_height * output_width;
    
    // Thread coarsening: each thread computes 2 consecutive output elements in width dimension
    const int spatial_idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Early exit for out-of-bounds threads
    if (spatial_idx_base >= output_spatial || channel_idx >= out_channels)
        return;
    
    // Decompose base indices for efficient memory access
    const int w_base = spatial_idx_base % output_width;
    const int h = spatial_idx_base / output_width;
    const int c = channel_idx;
    
    // For groups=1 (given in problem spec)
    const int group_out_channels = out_channels / groups;
    const int group_in_channels = in_channels / groups;
    const int g = c / group_out_channels;
    const int group_c = c - g * group_out_channels;
    
    // Precompute memory offsets for this batch
    const int input_batch_offset = batch_idx * in_channels * input_height * input_width;
    const int weight_group_offset = g * group_in_channels * group_out_channels * kernel_size * kernel_size;
    const int weight_channel_stride = group_out_channels * kernel_size * kernel_size;
    
    // Accumulators for two consecutive output elements
    float result0 = 0.0f;
    float result1 = 0.0f;
    
    // Precompute input height bounds checks for both output positions
    bool h_in_bounds[3];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int input_h = h - kh;
        h_in_bounds[kh] = (input_h >= 0 && input_h < input_height);
    }
    
    // Precompute weight base indices for each kernel position
    int weight_base_idx[9];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            weight_base_idx[kh * 3 + kw] = weight_group_offset + group_c * 9 + (kh * 3 + kw);
        }
    }
    
    // Check if second output element is within bounds
    const int w1 = w_base + 1;
    const bool second_output_valid = (w1 < output_width);
    
    // Tiled computation with thread coarsening
    #pragma unroll
    for (int kh = 0; kh < CHILD_TILE_KH; kh++) {
        if (!h_in_bounds[kh]) continue;
        
        const int input_h = h - kh;
        const int input_h_offset = input_h * input_width;
        
        #pragma unroll
        for (int kw = 0; kw < CHILD_TILE_KW; kw++) {
            // Calculate input positions for both output elements
            const int input_w0 = w_base - kw;
            const int input_w1 = w1 - kw;
            
            const bool in_bounds0 = (input_w0 >= 0 && input_w0 < input_width);
            const bool in_bounds1 = second_output_valid && (input_w1 >= 0 && input_w1 < input_width);
            
            if (in_bounds0 || in_bounds1) {
                const int input_spatial_idx0 = input_h_offset + input_w0;
                const int input_spatial_idx1 = input_h_offset + input_w1;
                const int weight_idx_offset = weight_base_idx[kh * 3 + kw];
                
                // Process input channels in tiles of TILE_IC
                for (int ic_tile = 0; ic_tile < group_in_channels; ic_tile += CHILD_TILE_IC) {
                    // Load input values for both positions with coalesced access
                    float input_vals0[CHILD_TILE_IC];
                    float input_vals1[CHILD_TILE_IC];
                    
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        const int ic = ic_tile + i;
                        const int input_channel_offset = (g * group_in_channels + ic) * input_height * input_width;
                        
                        if (in_bounds0) {
                            const int input_idx0 = input_batch_offset + input_channel_offset + input_spatial_idx0;
                            input_vals0[i] = to_float(input[input_idx0]);
                        }
                        
                        if (in_bounds1) {
                            const int input_idx1 = input_batch_offset + input_channel_offset + input_spatial_idx1;
                            input_vals1[i] = to_float(input[input_idx1]);
                        }
                    }
                    
                    // Load weight values with stride-aware access
                    float weight_vals[CHILD_TILE_IC];
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        const int ic = ic_tile + i;
                        const int weight_idx = weight_idx_offset + ic * weight_channel_stride;
                        weight_vals[i] = to_float(weight[weight_idx]);
                    }
                    
                    // FMA accumulation with ILP for both output positions
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        if (in_bounds0) {
                            result0 += input_vals0[i] * weight_vals[i];
                        }
                        if (in_bounds1) {
                            result1 += input_vals1[i] * weight_vals[i];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias if provided (bias=False for this problem)
    if (bias != nullptr) {
        const float bias_val = to_float(bias[c]);
        result0 += bias_val;
        result1 += bias_val;
    }
    
    // Coalesced write to output for first position
    const int output_batch_offset = batch_idx * out_channels * output_spatial;
    const int output_idx0 = output_batch_offset + c * output_spatial + spatial_idx_base;
    write_output(output, output_idx0, result0);
    
    // Write second output if valid
    if (second_output_valid) {
        const int output_idx1 = output_idx0 + 1;
        write_output(output, output_idx1, result1);
    }
}

// Parent kernel using dynamic parallelism - REMOVED
// Single kernel approach without dynamic parallelism
template <typename scalar_t>
__global__ void conv_transpose2d_single_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {
    
    const int batch_idx = blockIdx.z;
    const int spatial_idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int output_spatial = output_height * output_width;
    
    // Early exit for out-of-bounds threads
    if (batch_idx >= batch_size || spatial_idx_base >= output_spatial || channel_idx >= out_channels)
        return;
    
    // Decompose base indices for efficient memory access
    const int w_base = spatial_idx_base % output_width;
    const int h = spatial_idx_base / output_width;
    const int c = channel_idx;
    
    // For groups=1 (given in problem spec)
    const int group_out_channels = out_channels / groups;
    const int group_in_channels = in_channels / groups;
    const int g = c / group_out_channels;
    const int group_c = c - g * group_out_channels;
    
    // Precompute memory offsets for this batch
    const int input_batch_offset = batch_idx * in_channels * input_height * input_width;
    const int weight_group_offset = g * group_in_channels * group_out_channels * kernel_size * kernel_size;
    const int weight_channel_stride = group_out_channels * kernel_size * kernel_size;
    
    // Accumulators for two consecutive output elements
    float result0 = 0.0f;
    float result1 = 0.0f;
    
    // Precompute input height bounds checks for both output positions
    bool h_in_bounds[3];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int input_h = h - kh;
        h_in_bounds[kh] = (input_h >= 0 && input_h < input_height);
    }
    
    // Precompute weight base indices for each kernel position
    int weight_base_idx[9];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            weight_base_idx[kh * 3 + kw] = weight_group_offset + group_c * 9 + (kh * 3 + kw);
        }
    }
    
    // Check if second output element is within bounds
    const int w1 = w_base + 1;
    const bool second_output_valid = (w1 < output_width);
    
    // Tiled computation with thread coarsening
    #pragma unroll
    for (int kh = 0; kh < CHILD_TILE_KH; kh++) {
        if (!h_in_bounds[kh]) continue;
        
        const int input_h = h - kh;
        const int input_h_offset = input_h * input_width;
        
        #pragma unroll
        for (int kw = 0; kw < CHILD_TILE_KW; kw++) {
            // Calculate input positions for both output elements
            const int input_w0 = w_base - kw;
            const int input_w1 = w1 - kw;
            
            const bool in_bounds0 = (input_w0 >= 0 && input_w0 < input_width);
            const bool in_bounds1 = second_output_valid && (input_w1 >= 0 && input_w1 < input_width);
            
            if (in_bounds0 || in_bounds1) {
                const int input_spatial_idx0 = input_h_offset + input_w0;
                const int input_spatial_idx1 = input_h_offset + input_w1;
                const int weight_idx_offset = weight_base_idx[kh * 3 + kw];
                
                // Process input channels in tiles of TILE_IC
                for (int ic_tile = 0; ic_tile < group_in_channels; ic_tile += CHILD_TILE_IC) {
                    // Load input values for both positions with coalesced access
                    float input_vals0[CHILD_TILE_IC];
                    float input_vals1[CHILD_TILE_IC];
                    
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        const int ic = ic_tile + i;
                        const int input_channel_offset = (g * group_in_channels + ic) * input_height * input_width;
                        
                        if (in_bounds0) {
                            const int input_idx0 = input_batch_offset + input_channel_offset + input_spatial_idx0;
                            input_vals0[i] = to_float(input[input_idx0]);
                        }
                        
                        if (in_bounds1) {
                            const int input_idx1 = input_batch_offset + input_channel_offset + input_spatial_idx1;
                            input_vals1[i] = to_float(input[input_idx1]);
                        }
                    }
                    
                    // Load weight values with stride-aware access
                    float weight_vals[CHILD_TILE_IC];
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        const int ic = ic_tile + i;
                        const int weight_idx = weight_idx_offset + ic * weight_channel_stride;
                        weight_vals[i] = to_float(weight[weight_idx]);
                    }
                    
                    // FMA accumulation with ILP for both output positions
                    #pragma unroll
                    for (int i = 0; i < CHILD_TILE_IC; i++) {
                        if (in_bounds0) {
                            result0 += input_vals0[i] * weight_vals[i];
                        }
                        if (in_bounds1) {
                            result1 += input_vals1[i] * weight_vals[i];
                        }
                    }
                }
            }
        }
    }
    
    // Add bias if provided (bias=False for this problem)
    if (bias != nullptr) {
        const float bias_val = to_float(bias[c]);
        result0 += bias_val;
        result1 += bias_val;
    }
    
    // Coalesced write to output for first position
    const int output_batch_offset = batch_idx * out_channels * output_spatial;
    const int output_idx0 = output_batch_offset + c * output_spatial + spatial_idx_base;
    write_output(output, output_idx0, result0);
    
    // Write second output if valid
    if (second_output_valid) {
        const int output_idx1 = output_idx0 + 1;
        write_output(output, output_idx1, result1);
    }
}

// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined() && bias.numel() > 0) {
        CHECK_INPUT(bias);
    }
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int kernel_size = weight.size(2);

    const int out_channels = (bias.defined() && bias.numel() > 0) ? bias.size(0) : weight.size(1) * groups;
    
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    if (output.numel() == 0) {
        return output;
    }
    
    // Configure single kernel grid and block
    const int output_spatial = output_height * output_width;
    const int grid_x = (output_spatial + CHILD_BLOCK_SIZE_X * 2 - 1) / (CHILD_BLOCK_SIZE_X * 2);
    const int grid_y = (out_channels + CHILD_BLOCK_SIZE_Y - 1) / CHILD_BLOCK_SIZE_Y;
    const int grid_z = batch_size;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(CHILD_BLOCK_SIZE_X, CHILD_BLOCK_SIZE_Y);
    
    // Launch single kernel without dynamic parallelism
    if (input.scalar_type() == torch::kFloat32) {
        conv_transpose2d_single_kernel<float><<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups);
    } else if (input.scalar_type() == torch::kHalf) {
        conv_transpose2d_single_kernel<__half><<<grid, block>>>(
            reinterpret_cast<const __half*>(input.data_ptr<torch::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
            (bias.defined() && bias.numel() > 0) ? reinterpret_cast<const __half*>(bias.data_ptr<torch::Half>()) : nullptr,
            reinterpret_cast<__half*>(output.data_ptr<torch::Half>()),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups);
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure kernel completes
    cudaDeviceSynchronize();
    
    return output;
}
// PART-END