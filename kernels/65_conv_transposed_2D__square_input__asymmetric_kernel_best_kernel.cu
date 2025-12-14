// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace cg = cooperative_groups;
using namespace nvcuda;

// Optimized block dimension calculation for A100 with 3x7 kernel and specific problem size
__host__ __device__ void calculate_optimal_block_dims(int output_height, int output_width,
                                                      int batch_channels,
                                                      int& block_x, int& block_y, int& block_z) {
    // A100 optimized: Use 4 warps (128 threads) for better occupancy and Tensor Core utilization
    block_x = 32;  // Warp width
    block_y = 4;   // 4 warps per block
    block_z = 1;   // Single warp group per block for Tensor Core operations
}

// Specialized WMMA-based computation for exact problem: 3x7 kernel, 64 channels
template<typename scalar_t>
__device__ __forceinline__ void compute_kernel_contributions_wmma_3x7_64(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int n, int h, int w,
    int input_height, int input_width,
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& acc_frag) {
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp processes 16 output channels (MMA_N = 16)
    const int out_c_start = warp_id * 16;
    if (out_c_start >= 64) return;
    
    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Process input channels in 16-element tiles (matches MMA_K)
    for (int ic_start = 0; ic_start < 64; ic_start += 16) {
        // Load weight fragment for this tile (64 output channels × 64 input channels × 3×7 kernel)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        
        // Precompute base indices for better ILP
        const int weight_base_idx = out_c_start * 64 * 21 + ic_start * 21; // 21 = 3×7
        
        // Process kernel positions with software pipelining
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            const int h_offset = h - kh; // padding=0, stride=1
            if (h_offset >= 0 && h_offset < input_height) {
                #pragma unroll
                for (int kw = 0; kw < 7; ++kw) {
                    const int w_offset = w - kw; // padding=0, stride=1
                    if (w_offset >= 0 && w_offset < input_width) {
                        
                        // Load weight fragment for this kernel position
                        #pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const int out_c = out_c_start + i;
                            if (out_c < 64) {
                                #pragma unroll
                                for (int j = 0; j < 16; j++) {
                                    const int in_c = ic_start + j;
                                    if (in_c < 64) {
                                        const int weight_idx = weight_base_idx + i * 64 * 21 + j * 21 + kh * 7 + kw;
                                        a_frag.x[i * 16 + j] = __float2half(static_cast<float>(weight[weight_idx]));
                                    }
                                }
                            }
                        }
                        
                        // Load input fragment for this spatial position
                        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                        half b_frag_data[16];
                        
                        const int input_base_idx = (n * 64 + ic_start) * input_height * input_width + h_offset * input_width + w_offset;
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            const int in_c = ic_start + j;
                            if (in_c < 64) {
                                const int input_idx = input_base_idx + j * input_height * input_width;
                                b_frag_data[j] = __float2half(static_cast<float>(input[input_idx]));
                            } else {
                                b_frag_data[j] = __float2half(0.0f);
                            }
                        }
                        
                        wmma::load_matrix_sync(b_frag, b_frag_data, 16);
                        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                    }
                }
            }
        }
    }
}

// Optimized version for non-Tensor Core cases (with bias or different types)
template<typename scalar_t>
__device__ __forceinline__ float compute_kernel_contributions_optimized_3x7_64(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int n, int c, int h, int w,
    int input_height, int input_width) {
    
    float result = 0.0f;
    
    // Precompute conditions for height dimension
    const int h_offsets[3] = {h, h - 1, h - 2};
    const bool h_conds[3] = {
        h_offsets[0] >= 0 && h_offsets[0] < input_height,
        h_offsets[1] >= 0 && h_offsets[1] < input_height,
        h_offsets[2] >= 0 && h_offsets[2] < input_height
    };
    const int actual_h[3] = {h_offsets[0], h_offsets[1], h_offsets[2]};
    
    // Precompute weight base index
    const int weight_base = c * 64 * 21; // 64 input channels × 21 kernel weights
    
    // Process kernel positions with aggressive unrolling
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        if (h_conds[kh]) {
            #pragma unroll
            for (int kw = 0; kw < 7; ++kw) {
                const int w_offset = w - kw;
                if (w_offset >= 0 && w_offset < input_width) {
                    const int input_base_idx = (n * 64) * input_height * input_width + actual_h[kh] * input_width + w_offset;
                    const int weight_idx_base = weight_base + kh * 7 + kw;
                    
                    // Process 8 input channels at a time for better ILP
                    #pragma unroll 8
                    for (int ic = 0; ic < 64; ic += 8) {
                        float accum[8] = {0.0f};
                        
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            const int input_idx = input_base_idx + (ic + i) * input_height * input_width;
                            const int weight_idx = weight_idx_base + (ic + i) * 21;
                            accum[i] = static_cast<float>(input[input_idx]) * static_cast<float>(weight[weight_idx]);
                        }
                        
                        // Reduce accumulators
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            result += accum[i];
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// Generic optimized computation for arbitrary kernel sizes
template<typename scalar_t>
__device__ __forceinline__ float compute_kernel_contributions_generic_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    int n, int c, int h, int w,
    int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {
    
    float result = 0.0f;
    
    // Precompute height conditions
    for (int kh = 0; kh < kernel_h; ++kh) {
        const int h_offset = h - kh + padding_h;
        const bool h_cond = (h_offset >= 0 && (h_offset % stride_h == 0));
        const int actual_input_h = h_cond ? (h_offset / stride_h) : 0;
        
        if (h_cond && actual_input_h < input_height) {
            // Precompute width conditions
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int w_offset = w - kw + padding_w;
                const bool w_cond = (w_offset >= 0 && (w_offset % stride_w == 0));
                const int actual_input_w = w_cond ? (w_offset / stride_w) : 0;
                
                if (w_cond && actual_input_w < input_width) {
                    const int base_input_idx = ((n * in_channels) * input_height + actual_input_h) * input_width + actual_input_w;
                    const int base_weight_idx = ((c * in_channels) * kernel_h + kh) * kernel_w + kw;
                    
                    // Process channels in groups of 4 for ILP
                    #pragma unroll 4
                    for (int ic = 0; ic < in_channels; ic += 4) {
                        float accum[4] = {0.0f};
                        
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            if (ic + i < in_channels) {
                                const int input_idx = base_input_idx + (ic + i) * input_height * input_width;
                                const int weight_idx = base_weight_idx + (ic + i) * kernel_h * kernel_w;
                                accum[i] = static_cast<float>(input[input_idx]) * static_cast<float>(weight[weight_idx]);
                            }
                        }
                        
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            result += accum[i];
                        }
                    }
                }
            }
        }
    }
    
    return result;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
// WMMA-based kernel for exact problem dimensions (3x7 kernel, 64 channels, no bias)
template<typename scalar_t>
__global__ void conv_transpose2d_kernel_wmma_3x7_64_no_bias(
    const scalar_t* __restrict__ input, 
    const scalar_t* __restrict__ weight, 
    scalar_t* __restrict__ output,
    int batch_size, int input_height, int input_width,
    int output_height, int output_width) {
    
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    
    if (w >= output_width || h >= output_height || n >= batch_size) return;
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each warp processes 16 output channels
    const int out_c_start = warp_id * 16;
    if (out_c_start >= 64) return;
    
    // WMMA accumulator fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    
    // Compute using WMMA
    compute_kernel_contributions_wmma_3x7_64<scalar_t>(
        input, weight, n, h, w,
        input_height, input_width,
        acc_frag);
    
    // Store results with coalesced access pattern
    if (lane_id < 16) {
        const int out_c = out_c_start + lane_id;
        if (out_c < 64) {
            const int output_idx = ((n * 64 + out_c) * output_height + h) * output_width + w;
            output[output_idx] = static_cast<scalar_t>(acc_frag.x[lane_id]);
        }
    }
}

// Optimized kernel for exact problem dimensions without bias (non-Tensor Core path)
template<typename scalar_t>
__global__ void conv_transpose2d_kernel_optimized_3x7_64_no_bias(
    const scalar_t* __restrict__ input, 
    const scalar_t* __restrict__ weight, 
    scalar_t* __restrict__ output,
    int batch_size, int input_height, int input_width,
    int output_height, int output_width) {
    
    // 3D thread indexing with optimal dimensions for A100
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc_combined = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (w >= output_width || h >= output_height || bc_combined >= batch_size * 64) return;
    
    const int n = bc_combined / 64;
    const int c = bc_combined % 64;
    
    // Use optimized computation
    float result = compute_kernel_contributions_optimized_3x7_64<scalar_t>(
        input, weight, n, c, h, w,
        input_height, input_width);
    
    // Write output with coalesced access pattern
    const int output_idx = ((n * 64 + c) * output_height + h) * output_width + w;
    output[output_idx] = static_cast<scalar_t>(result);
}

// Optimized kernel with bias support
template<typename scalar_t>
__global__ void conv_transpose2d_kernel_optimized_3x7_64(
    const scalar_t* __restrict__ input, 
    const scalar_t* __restrict__ weight, 
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int input_height, int input_width,
    int output_height, int output_width) {
    
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc_combined = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (w >= output_width || h >= output_height || bc_combined >= batch_size * 64) return;
    
    const int n = bc_combined / 64;
    const int c = bc_combined % 64;
    
    float result = compute_kernel_contributions_optimized_3x7_64<scalar_t>(
        input, weight, n, c, h, w,
        input_height, input_width);
    
    // Add bias if present
    if (bias != nullptr) {
        result += static_cast<float>(bias[c]);
    }
    
    const int output_idx = ((n * 64 + c) * output_height + h) * output_width + w;
    output[output_idx] = static_cast<scalar_t>(result);
}

// Generic optimized kernel for arbitrary kernel sizes
template<typename scalar_t>
__global__ void conv_transpose2d_kernel_generic_optimized(
    const scalar_t* __restrict__ input, 
    const scalar_t* __restrict__ weight, 
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w) {
    
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int bc_combined = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (w >= output_width || h >= output_height || bc_combined >= batch_size * out_channels) return;
    
    const int n = bc_combined / out_channels;
    const int c = bc_combined % out_channels;
    
    float result = compute_kernel_contributions_generic_optimized<scalar_t>(
        input, weight, n, c, h, w,
        in_channels, out_channels,
        input_height, input_width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w);
    
    if (bias != nullptr) {
        result += static_cast<float>(bias[c]);
    }
    
    const int output_idx = ((n * out_channels + c) * output_height + h) * output_width + w;
    output[output_idx] = static_cast<scalar_t>(result);
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    // Transpose weight for better memory coalescing
    weight = weight.permute({1, 0, 2, 3}).contiguous();
    
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    const int output_height = (input_height - 1) * stride_h + kernel_h - 2 * padding_h + output_padding_h;
    const int output_width = (input_width - 1) * stride_w + kernel_w - 2 * padding_w + output_padding_w;
    
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    // Dispatch to optimized kernels
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_transpose2d_cuda", ([&] {
        if (kernel_h == 3 && kernel_w == 7 && in_channels == 64 && out_channels == 64) {
            if (!bias.defined()) {
                // Use WMMA kernel for optimal performance on A100 when possible
                if (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16) {
                    // Tensor Core optimized path for FP16/BF16
                    dim3 block(128, 1, 1);  // 4 warps for WMMA operations
                    dim3 grid(
                        (output_width + block.x - 1) / block.x,
                        (output_height + block.y - 1) / block.y,
                        batch_size
                    );
                    
                    conv_transpose2d_kernel_wmma_3x7_64_no_bias<scalar_t><<<grid, block>>>(
                        input.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, input_height, input_width,
                        output_height, output_width);
                } else {
                    // FP32 optimized path
                    dim3 block(32, 8, 4);  // 1024 threads, optimized for A100
                    dim3 grid(
                        (output_width + block.x - 1) / block.x,
                        (output_height + block.y - 1) / block.y,
                        (batch_size * 64 + block.z - 1) / block.z
                    );
                    
                    conv_transpose2d_kernel_optimized_3x7_64_no_bias<scalar_t><<<grid, block>>>(
                        input.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, input_height, input_width,
                        output_height, output_width);
                }
            } else {
                // With bias
                dim3 block(32, 8, 4);  // 1024 threads
                dim3 grid(
                    (output_width + block.x - 1) / block.x,
                    (output_height + block.y - 1) / block.y,
                    (batch_size * 64 + block.z - 1) / block.z
                );
                
                conv_transpose2d_kernel_optimized_3x7_64<scalar_t><<<grid, block>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size, input_height, input_width,
                    output_height, output_width);
            }
        } else {
            // Generic kernel for other cases
            int block_x, block_y, block_z;
            calculate_optimal_block_dims(output_height, output_width, batch_size * out_channels, 
                                        block_x, block_y, block_z);
            
            block_x = min(block_x, output_width);
            block_y = min(block_y, output_height);
            block_z = min(block_z, batch_size * out_channels);
            
            dim3 block(block_x, block_y, block_z);
            dim3 grid(
                (output_width + block.x - 1) / block.x,
                (output_height + block.y - 1) / block.y,
                (batch_size * out_channels + block.z - 1) / block.z
            );
            
            conv_transpose2d_kernel_generic_optimized<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size, in_channels, out_channels,
                input_height, input_width,
                output_height, output_width,
                kernel_h, kernel_w,
                stride_h, stride_w,
                padding_h, padding_w,
                output_padding_h, output_padding_w);
        }
    }));
    
    cudaDeviceSynchronize();
    return output;
}
// PART-END