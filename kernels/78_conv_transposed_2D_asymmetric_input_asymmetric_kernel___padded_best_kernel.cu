// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void custom_conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int H,
    int W
) {
    // Optimized tiling parameters
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 32;
    constexpr int CHANNELS_PER_BLOCK_OUT = 32;
    constexpr int CHANNELS_PER_BLOCK_IN = 4;
    
    // Thread coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    // Compute grid dimensions
    int grid_x = (W + TILE_W - 1) / TILE_W;
    int grid_y = (H + TILE_H - 1) / TILE_H;
    int grid_xy = grid_x * grid_y;
    int num_oc_blocks = (out_channels + CHANNELS_PER_BLOCK_OUT - 1) / CHANNELS_PER_BLOCK_OUT;
    
    // Decode block index using Morton order
    int block_id = blockIdx.x;
    int z_index = block_id / grid_xy;
    int morton = block_id % grid_xy;
    
    // Decode Morton to x/y coordinates
    unsigned int x = 0;
    unsigned int y = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        x |= ((morton >> (2*i)) & 1) << i;
        y |= ((morton >> (2*i+1)) & 1) << i;
    }
    
    // Batch and channel grouping
    int group_out = z_index % num_oc_blocks;
    int n = z_index / num_oc_blocks;
    int c_out_base = group_out * CHANNELS_PER_BLOCK_OUT;
    
    // Shared memory - input double buffered, weight single buffer
    __shared__ float in_smem0[4][18][38];
    __shared__ float in_smem1[4][18][38];
    __shared__ float w_smem[3][7][4][32];
    
    // Double buffer pointers for input
    float (*current_in)[18][38] = in_smem0;
    float (*next_in)[18][38] = in_smem1;

    // Accumulators for 32 output channels
    float acc[32] = {0.0f};
    
    // Calculate input tile start position
    int input_start_i = y * TILE_H - 1;
    int input_start_j = x * TILE_W - 3;
    
    // Preload first input group and weights
    #pragma unroll
    for (int idx = tid; idx < 18*38; idx += blockDim.x * blockDim.y) {
        int h = idx / 38;
        int w = idx % 38;
        int gi = input_start_i + h;
        int gj = input_start_j + w;
        for (int c = 0; c < 4; c++) {
            float val = 0.0f;
            if (gi >= 0 && gi < H && gj >= 0 && gj < W) {
                int input_idx = ((n * in_channels + c) * H + gi) * W + gj;
                val = input[input_idx];
            }
            current_in[c][h][w] = val;
        }
    }
    
    #pragma unroll
    for (int idx = tid; idx < 3*7*4*32; idx += blockDim.x * blockDim.y) {
        int di = idx / (7*4*32);
        int dj = (idx % (7*4*32)) / (4*32);
        int c_in = (idx % (4*32)) / 32;
        int oc = idx % 32;
        float w_val = 0.0f;
        if (c_in < in_channels && (c_out_base + oc) < out_channels) {
            int weight_idx = (c_in * out_channels + (c_out_base + oc)) * 21 + di * 7 + dj;
            w_val = weight[weight_idx];
        }
        w_smem[di][dj][c_in][oc] = w_val;
    }
    __syncthreads();

    // Process all groups with overlapping prefetch
    for (int g_in = 0; g_in < 8; ++g_in) {
        // Prefetch next input group if not last
        if (g_in < 7) {
            int next_c_base = (g_in+1) * CHANNELS_PER_BLOCK_IN;
            #pragma unroll
            for (int idx = tid; idx < 18*38; idx += blockDim.x * blockDim.y) {
                int h = idx / 38;
                int w = idx % 38;
                int gi = input_start_i + h;
                int gj = input_start_j + w;
                for (int c = 0; c < 4; c++) {
                    float val = 0.0f;
                    if (gi >= 0 && gi < H && gj >= 0 && gj < W) {
                        int input_idx = ((n * in_channels + (next_c_base + c)) * H + gi) * W + gj;
                        val = input[input_idx];
                    }
                    next_in[c][h][w] = val;
                }
            }
        }

        // Compute current group
        #pragma unroll
        for (int c_in = 0; c_in < 4; ++c_in) {
            #pragma unroll
            for (int di = 0; di < 3; ++di) {
                #pragma unroll
                for (int dj = 0; dj < 7; ++dj) {
                    int h_in_tile = ty + 2 - di;
                    int w_in_tile = tx + 6 - dj;
                    float in_val = current_in[c_in][h_in_tile][w_in_tile];
                    #pragma unroll
                    for (int oc = 0; oc < 32; ++oc) {
                        acc[oc] = __fmaf_rn(in_val, w_smem[di][dj][c_in][oc], acc[oc]);
                    }
                }
            }
        }

        // Prepare next group
        if (g_in < 7) {
            __syncthreads();
            
            // Load next weights
            int next_c_base = (g_in+1) * CHANNELS_PER_BLOCK_IN;
            #pragma unroll
            for (int idx = tid; idx < 3*7*4*32; idx += blockDim.x * blockDim.y) {
                int di = idx / (7*4*32);
                int dj = (idx % (7*4*32)) / (4*32);
                int c_in = (idx % (4*32)) / 32;
                int oc = idx % 32;
                float w_val = 0.0f;
                if ((next_c_base + c_in) < in_channels && (c_out_base + oc) < out_channels) {
                    int weight_idx = ((next_c_base + c_in) * out_channels + (c_out_base + oc)) * 21 + di * 7 + dj;
                    w_val = weight[weight_idx];
                }
                w_smem[di][dj][c_in][oc] = w_val;
            }
            __syncthreads();
            
            // Swap input buffers
            float (*temp_in)[18][38] = current_in;
            current_in = next_in;
            next_in = temp_in;
        }
    }
    
    // Write results
    int i_global = y * TILE_H + ty;
    int j_global = x * TILE_W + tx;
    
    if (i_global < H && j_global < W) {
        for (int oc = 0; oc < min(32, out_channels - c_out_base); ++oc) {
            int output_idx = ((n * out_channels + (c_out_base + oc)) * H + i_global) * W + j_global;
            output[output_idx] = acc[oc];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor custom_conv_transpose(
    torch::Tensor input,
    torch::Tensor weight,
    int batch_size,
    int in_channels,
    int out_channels,
    int H,
    int W
) {
    auto output = torch::zeros({batch_size, out_channels, H, W}, input.options());
    
    // Optimized grid configuration
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 32;
    constexpr int CHANNELS_PER_BLOCK_OUT = 32;
    
    int grid_x = (W + TILE_W - 1) / TILE_W;
    int grid_y = (H + TILE_H - 1) / TILE_H;
    int grid_xy = grid_x * grid_y;
    int grid_oc = (out_channels + CHANNELS_PER_BLOCK_OUT - 1) / CHANNELS_PER_BLOCK_OUT;
    int total_blocks = batch_size * grid_oc * grid_xy;
    
    // Launch kernel
    dim3 block(32, 16);
    custom_conv_transpose_kernel<<<total_blocks, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        H,
        W
    );
    
    return output;
}
// PART-END