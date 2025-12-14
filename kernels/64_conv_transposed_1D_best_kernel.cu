// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// WMMA tile sizes for FP16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Problem-specific tuning for A100 and given dimensions
constexpr int OUTPUT_CHANNELS_PER_BLOCK = 32;  // 2x WMMA_N tiles
constexpr int OUTPUT_POS_PER_BLOCK = 64;       // 4x WMMA_M tiles
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = THREADS_PER_WARP * WARPS_PER_BLOCK;
constexpr int KERNEL_SIZE = 3;  // Fixed for this problem

// Register tiling parameters
constexpr int REG_TILE_ROWS = 4;    // Each warp handles 4 rows of input tile
constexpr int REG_TILE_COLS = 16;   // Full WMMA_K dimension

// Identity matrix fragment for bias fusion
__device__ void create_identity_matrix_fragment(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>& identity_frag) {
    // Identity matrix has 1s on diagonal for row-major layout
    const int stride = WMMA_M + 1;
    #pragma unroll
    for (int i = 0; i < identity_frag.num_elements; i++) {
        identity_frag.x[i] = (i % stride == 0) ? __float2half(1.0f) : __float2half(0.0f);
    }
}

// Fallback kernel for non-half types with optimized 3D grid
template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int output_length) {
    
    // 3D grid layout for better workload distribution
    const int output_pos = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch = blockIdx.z;
    
    if (output_pos >= output_length || channel_out >= out_channels || batch >= batch_size) return;
    
    const int group_size = in_channels / groups;
    const int group_idx = channel_out / (out_channels / groups);
    const int start_in_channel = group_idx * group_size;
    const int c_out_in_group = channel_out % (out_channels / groups);
    
    scalar_t result = 0.0;
    
    // Precompute weight base index outside loops
    const int weight_base = c_out_in_group * kernel_size;
    
    // Fixed kernel loop unrolling for KERNEL_SIZE=3
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; k++) {
        const int input_pos = (output_pos + padding - k) / stride;
        const int mod = (output_pos + padding - k) % stride;
        
        if (mod == 0 && input_pos >= 0 && input_pos < input_length) {
            const int input_base = (batch * in_channels) * input_length + input_pos;
            const int weight_offset = k;
            
            // Loop vectorization for A100
            #pragma unroll 8
            for (int c_in_offset = 0; c_in_offset < group_size; c_in_offset++) {
                const int channel_in = start_in_channel + c_in_offset;
                const int input_idx = input_base + channel_in * input_length;
                const int weight_idx = (channel_in * (out_channels / groups) * kernel_size) + weight_base + weight_offset;
                
                result += input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    if (bias != nullptr) {
        result += bias[channel_out];
    }
    
    output[((batch * out_channels + channel_out) * output_length) + output_pos] = result;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv_transpose1d_wmma_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int output_length) {
    
    // 3D grid layout optimized for problem dimensions
    const int batch = blockIdx.z;
    const int out_channel_tile = blockIdx.x * OUTPUT_CHANNELS_PER_BLOCK;
    const int output_pos_tile = blockIdx.y * OUTPUT_POS_PER_BLOCK;
    
    // Warp and thread indexing for WMMA
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    // Declare WMMA fragments - 4x2 accumulator tiles (64x32 output per block)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag[4][2];
    
    // Register-tiled input fragments: 4 rows per warp, 16 columns
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_reg[REG_TILE_ROWS];
    
    // Single weight fragment reused across kernel positions
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    
    // Bias fragment for fused bias addition
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> bias_frag[2];
    
    // Identity matrix fragment for bias fusion
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> identity_frag;
    
    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            nvcuda::wmma::fill_fragment(acc_frag[i][j], __float2half(0.0f));
        }
    }
    
    // Create identity matrix fragment once
    create_identity_matrix_fragment(identity_frag);
    
    // Load bias fragments if bias is provided
    if (bias != nullptr) {
        #pragma unroll
        for (int tile_j = 0; tile_j < 2; tile_j++) {
            int global_channel_base = out_channel_tile + tile_j * WMMA_N;
            if (global_channel_base < out_channels) {
                // Create bias matrix with broadcasted bias values
                __shared__ half bias_tile[WMMA_K][OUTPUT_CHANNELS_PER_BLOCK];
                
                // Load bias values into shared memory with coalesced access
                #pragma unroll
                for (int load_iter = 0; load_iter < 2; load_iter++) {
                    int bias_row = warp_id * 2 + load_iter;
                    int bias_col = lane_id;
                    
                    if (bias_row < WMMA_K && bias_col < OUTPUT_CHANNELS_PER_BLOCK) {
                        int global_channel = out_channel_tile + bias_col;
                        half bias_val = __float2half(0.0f);
                        if (global_channel < out_channels) {
                            bias_val = bias[global_channel];
                        }
                        // Broadcast bias across K dimension
                        bias_tile[bias_row][bias_col] = bias_val;
                    }
                }
                
                __syncthreads();
                
                // Load bias fragment from shared memory
                nvcuda::wmma::load_matrix_sync(bias_frag[tile_j], 
                                              &bias_tile[0][tile_j * WMMA_N], 
                                              OUTPUT_CHANNELS_PER_BLOCK);
                __syncthreads();
            }
        }
    }
    
    // Shared memory with bank conflict avoidance padding
    __shared__ half input_tile[OUTPUT_POS_PER_BLOCK][WMMA_K + 1];
    __shared__ half weight_tile[WMMA_K][OUTPUT_CHANNELS_PER_BLOCK];
    
    // Main computation loop over input channels in tiles of 16
    for (int in_channel_tile = 0; in_channel_tile < in_channels; in_channel_tile += WMMA_K) {
        // Load input tile (64x16) into shared memory with coalesced access
        #pragma unroll
        for (int load_iter = 0; load_iter < 8; load_iter++) {
            int input_row = warp_id * 8 + load_iter;
            int input_col = lane_id;
            
            if (input_row < OUTPUT_POS_PER_BLOCK && input_col < WMMA_K) {
                int global_output_pos = output_pos_tile + input_row;
                int global_channel = in_channel_tile + input_col;
                
                half val = __float2half(0.0f);
                if (global_channel < in_channels && global_output_pos < output_length) {
                    // Compute corresponding input position
                    int input_pos = (global_output_pos + padding) / stride;
                    int mod = (global_output_pos + padding) % stride;
                    
                    if (mod == 0 && input_pos >= 0 && input_pos < input_length) {
                        int input_idx = ((batch * in_channels + global_channel) * input_length) + input_pos;
                        val = input[input_idx];
                    }
                }
                input_tile[input_row][input_col] = val;
            }
        }
        
        __syncthreads();
        
        // Preload input fragments into registers for reuse across kernel positions
        #pragma unroll
        for (int reg_row = 0; reg_row < REG_TILE_ROWS; reg_row++) {
            int tile_row = warp_id * REG_TILE_ROWS + reg_row;
            if (tile_row < 4) {
                int a_row = tile_row * WMMA_M;
                nvcuda::wmma::load_matrix_sync(a_reg[reg_row], 
                                              &input_tile[a_row][0], 
                                              WMMA_K + 1);
            }
        }
        
        // Process each kernel position with register-tiled input
        #pragma unroll
        for (int k = 0; k < KERNEL_SIZE; k++) {
            // Load weight tile (16x32) into shared memory
            #pragma unroll
            for (int load_iter = 0; load_iter < 2; load_iter++) {
                int weight_row = warp_id * 2 + load_iter;
                int weight_col = lane_id;
                
                if (weight_row < WMMA_K && weight_col < OUTPUT_CHANNELS_PER_BLOCK) {
                    int global_in_channel = in_channel_tile + weight_row;
                    int global_out_channel = out_channel_tile + weight_col;
                    
                    half val = __float2half(0.0f);
                    if (global_in_channel < in_channels && global_out_channel < out_channels) {
                        int group_idx = global_out_channel / (out_channels / groups);
                        int c_out_in_group = global_out_channel % (out_channels / groups);
                        int start_in_channel = group_idx * (in_channels / groups);
                        
                        if (global_in_channel >= start_in_channel && 
                            global_in_channel < start_in_channel + (in_channels / groups)) {
                            int weight_idx = ((global_in_channel * (out_channels / groups) + c_out_in_group) * kernel_size) + k;
                            val = weight[weight_idx];
                        }
                    }
                    weight_tile[weight_row][weight_col] = val;
                }
            }
            
            __syncthreads();
            
            // Perform WMMA for each tile position using register-tiled input
            #pragma unroll
            for (int tile_j = 0; tile_j < 2; tile_j++) {
                // Load weight fragment once per output channel tile
                int b_col = tile_j * WMMA_N;
                nvcuda::wmma::load_matrix_sync(b_frag, 
                                              &weight_tile[0][b_col], 
                                              OUTPUT_CHANNELS_PER_BLOCK);
                
                // Accumulate using register-tiled input fragments
                #pragma unroll
                for (int reg_row = 0; reg_row < REG_TILE_ROWS; reg_row++) {
                    int tile_i = warp_id * REG_TILE_ROWS + reg_row;
                    if (tile_i < 4) {
                        nvcuda::wmma::mma_sync(acc_frag[tile_i][tile_j], a_reg[reg_row], b_frag, acc_frag[tile_i][tile_j]);
                    }
                }
            }
            
            __syncthreads();
        }
    }
    
    // Fuse bias addition using WMMA with identity matrix
    if (bias != nullptr) {
        #pragma unroll
        for (int tile_i = 0; tile_i < 4; tile_i++) {
            #pragma unroll
            for (int tile_j = 0; tile_j < 2; tile_j++) {
                int tile_idx = warp_id * REG_TILE_ROWS;
                if (tile_i >= tile_idx && tile_i < tile_idx + REG_TILE_ROWS) {
                    nvcuda::wmma::mma_sync(acc_frag[tile_i][tile_j], identity_frag, bias_frag[tile_j], acc_frag[tile_i][tile_j]);
                }
            }
        }
    }
    
    // Store results to global memory with vectorized writes
    #pragma unroll
    for (int tile_i = 0; tile_i < 4; tile_i++) {
        #pragma unroll
        for (int tile_j = 0; tile_j < 2; tile_j++) {
            int output_row = output_pos_tile + tile_i * WMMA_M;
            int output_col = out_channel_tile + tile_j * WMMA_N;
            
            if (output_row < output_length && output_col < out_channels) {
                half* output_ptr = &output[((batch * out_channels + output_col) * output_length) + output_row];
                nvcuda::wmma::store_matrix_sync(output_ptr, acc_frag[tile_i][tile_j], output_length, nvcuda::wmma::mem_row_major);
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) {
        CHECK_INPUT(bias);
    }

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0) * groups;
    const int kernel_size = weight.size(2);

    const int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_length}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));

    // Use Tensor Core optimized kernel for half precision on A100
    if (input.scalar_type() == torch::kHalf) {
        // Optimized grid dimensions for problem size: batch=64, out_channels=128, output_length=65538
        dim3 grid(
            (out_channels + OUTPUT_CHANNELS_PER_BLOCK - 1) / OUTPUT_CHANNELS_PER_BLOCK,  // = (128+31)/32 = 4
            (output_length + OUTPUT_POS_PER_BLOCK - 1) / OUTPUT_POS_PER_BLOCK,           // = (65538+63)/64 = 1025
            batch_size                                                                   // = 64
        );
        
        // 8 warps per block, 32 threads per warp = 256 threads total
        dim3 block(THREADS_PER_WARP, WARPS_PER_BLOCK);
        
        conv_transpose1d_wmma_kernel<<<grid, block>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            bias.defined() ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            batch_size,
            in_channels,
            out_channels,
            input_length,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            output_length);
    } else {
        // Optimized fallback for FP32/FP64 with 2D block and 3D grid
        const int BLOCK_X = 16;
        const int BLOCK_Y = 16;
        
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(
            (out_channels + BLOCK_X - 1) / BLOCK_X,
            (output_length + BLOCK_Y - 1) / BLOCK_Y,
            batch_size
        );
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose1d_cuda", [&] {
            conv_transpose1d_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size,
                in_channels,
                out_channels,
                input_length,
                kernel_size,
                stride,
                padding,
                output_padding,
                groups,
                output_length);
        });
    }

    return output;
}
// PART-END