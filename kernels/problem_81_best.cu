// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#define KERNEL_SIZE 3
#define TILE_IN_DIM 16
#define THREADS_PER_WARP 32
#define TILE_OUT_DIM 64
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, float* __restrict__ output,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kernel_size, int stride, int padding, int dilation
) {
    __shared__ float s_input[32][TILE_IN_DIM][TILE_IN_DIM];
    __shared__ float s_weight[KERNEL_SIZE][KERNEL_SIZE][32][8];
    __shared__ bool s_h_active[KERNEL_SIZE][TILE_OUT_DIM];
    __shared__ bool s_w_active[KERNEL_SIZE][TILE_OUT_DIM];
    __shared__ int s_h_in_eff[KERNEL_SIZE][TILE_OUT_DIM];
    __shared__ int s_w_in_eff[KERNEL_SIZE][TILE_OUT_DIM];
    
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int total_threads = blockDim.x * blockDim.y;
    
    int w_tile = blockIdx.x;
    int h_tile = blockIdx.y;
    int n_c_tile = blockIdx.z;
    int tile_c = 8;
    int n = n_c_tile / ((C_out + tile_c - 1) / tile_c);
    int c_out_tile = n_c_tile % ((C_out + tile_c - 1) / tile_c);
    int c_out_base = c_out_tile * tile_c;
    
    int w_start = w_tile * TILE_OUT_DIM;
    int w_end = min(w_start + TILE_OUT_DIM, W_out);
    int h_start = h_tile * TILE_OUT_DIM;
    int h_end = min(h_start + TILE_OUT_DIM, H_out);
    
    int in_start_w = w_start - dilation * (KERNEL_SIZE - 1) + padding;
    int in_start_h = h_start - dilation * (KERNEL_SIZE - 1) + padding;
    int w_in_start = (in_start_w >= 0) ? in_start_w / stride : (in_start_w - stride + 1) / stride;
    int h_in_start = (in_start_h >= 0) ? in_start_h / stride : (in_start_h - stride + 1) / stride;
    int w_in_end = (w_end - 1 + padding) / stride + 1;
    int h_in_end = (h_end - 1 + padding) / stride + 1;
    
    int w_in_low = max(0, w_in_start);
    int w_in_high = min(W_in, w_in_end);
    int h_in_low = max(0, h_in_start);
    int h_in_high = min(H_in, h_in_end);
    
    int w_in_count = w_in_high - w_in_low;
    int h_in_count = h_in_high - h_in_low;
    w_in_count = min(w_in_count, TILE_IN_DIM);
    h_in_count = min(h_in_count, TILE_IN_DIM);
    
    // Precompute output base and strides in registers
    int H_out_W_out = H_out * W_out;
    int base_output_channel = n * (C_out * H_out_W_out) + c_out_base * H_out_W_out;
    
    // Precomputation using all threads
    if (tid < 2 * KERNEL_SIZE * TILE_OUT_DIM) {
        int idx = tid;
        if (idx < KERNEL_SIZE * TILE_OUT_DIM) {
            int i = idx / TILE_OUT_DIM;
            int h_local = idx % TILE_OUT_DIM;
            int h = h_start + h_local;
            int residue = (h + padding - dilation * i) % stride;
            if (residue < 0) residue += stride;
            s_h_active[i][h_local] = (residue == 0);
            s_h_in_eff[i][h_local] = (h + padding - dilation * i) / stride;
        } else {
            idx -= KERNEL_SIZE * TILE_OUT_DIM;
            int j = idx / TILE_OUT_DIM;
            int w_local = idx % TILE_OUT_DIM;
            int w = w_start + w_local;
            int residue = (w + padding - dilation * j) % stride;
            if (residue < 0) residue += stride;
            s_w_active[j][w_local] = (residue == 0);
            s_w_in_eff[j][w_local] = (w + padding - dilation * j) / stride;
        }
    }
    
    // Vectorized weight loading with all threads
    int weight_float4_count = (KERNEL_SIZE * KERNEL_SIZE * 32 * 8 + 3) / 4;
    for (int idx4 = tid; idx4 < weight_float4_count; idx4 += total_threads) {
        int j = idx4 % KERNEL_SIZE;
        int i = (idx4 / KERNEL_SIZE) % KERNEL_SIZE;
        int kc = (idx4 / (KERNEL_SIZE * KERNEL_SIZE)) % 32;
        int c_base = (idx4 / (KERNEL_SIZE * KERNEL_SIZE * 32)) * 4;
        
        float4 weight_val4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (kc < C_in) {
            for (int v = 0; v < 4; v++) {
                int c = c_base + v;
                if (c_out_base + c < C_out && c < 8) {
                    int weight_idx = kc * C_out * KERNEL_SIZE * KERNEL_SIZE 
                                  + (c_out_base + c) * KERNEL_SIZE * KERNEL_SIZE 
                                  + i * KERNEL_SIZE + j;
                    switch(v) {
                        case 0: weight_val4.x = weight[weight_idx]; break;
                        case 1: weight_val4.y = weight[weight_idx]; break;
                        case 2: weight_val4.z = weight[weight_idx]; break;
                        case 3: weight_val4.w = weight[weight_idx]; break;
                    }
                }
            }
        }
        
        // Scalar store to avoid alignment issues
        s_weight[i][j][kc][c_base] = weight_val4.x;
        if (c_base+1 < 8) s_weight[i][j][kc][c_base+1] = weight_val4.y;
        if (c_base+2 < 8) s_weight[i][j][kc][c_base+2] = weight_val4.z;
        if (c_base+3 < 8) s_weight[i][j][kc][c_base+3] = weight_val4.w;
    }
    
    // Vectorized input loading with all threads
    int w_in_count_aligned = (w_in_count + 3) & ~3;
    int spatial_vec_count = h_in_count * (w_in_count_aligned / 4);
    int total_input_load = 32 * spatial_vec_count;
    for (int idx4 = tid; idx4 < total_input_load; idx4 += total_threads) {
        int kc = idx4 / spatial_vec_count;
        int spatial4_index = idx4 % spatial_vec_count;
        int h_in_local = spatial4_index / (w_in_count_aligned / 4);
        int w_in_local_base = (spatial4_index % (w_in_count_aligned / 4)) * 4;
        
        float4 in_val4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (kc < C_in) {
            for (int v = 0; v < 4; v++) {
                int w_in_local = w_in_local_base + v;
                if (w_in_local < w_in_count) {
                    int w_in_global = w_in_low + w_in_local;
                    int h_in_global = h_in_low + h_in_local;
                    if (h_in_global < H_in && w_in_global < W_in) {
                        int input_idx = n * (C_in * H_in * W_in) 
                                     + kc * (H_in * W_in) 
                                     + h_in_global * W_in 
                                     + w_in_global;
                        switch(v) {
                            case 0: in_val4.x = input[input_idx]; break;
                            case 1: in_val4.y = input[input_idx]; break;
                            case 2: in_val4.z = input[input_idx]; break;
                            case 3: in_val4.w = input[input_idx]; break;
                        }
                    }
                }
            }
        }
        
        // Scalar store with boundary checks
        s_input[kc][h_in_local][w_in_local_base] = in_val4.x;
        if (w_in_local_base+1 < w_in_count) 
            s_input[kc][h_in_local][w_in_local_base+1] = in_val4.y;
        if (w_in_local_base+2 < w_in_count) 
            s_input[kc][h_in_local][w_in_local_base+2] = in_val4.z;
        if (w_in_local_base+3 < w_in_count) 
            s_input[kc][h_in_local][w_in_local_base+3] = in_val4.w;
    }
    
    __syncthreads();
    
    // Unified computation phase with all threads
    for (int idx = tid; idx < TILE_OUT_DIM * TILE_OUT_DIM; idx += total_threads) {
        int h_local = idx / TILE_OUT_DIM;
        int w_local = idx % TILE_OUT_DIM;
        int h = h_start + h_local;
        int w = w_start + w_local;
        
        if (w < w_end && h < h_end) {
            float out_val[8] = {0.0f};
            
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    if (s_h_active[i][h_local] && s_w_active[j][w_local]) {
                        int h_in_eff = s_h_in_eff[i][h_local];
                        int w_in_eff = s_w_in_eff[j][w_local];
                        int h_in_local_idx = h_in_eff - h_in_low;
                        int w_in_local_idx = w_in_eff - w_in_low;
                        
                        if (h_in_local_idx >= 0 && h_in_local_idx < h_in_count && 
                            w_in_local_idx >= 0 && w_in_local_idx < w_in_count) {
                            
                            for (int kc = 0; kc < 32; kc++) {
                                float in_val = s_input[kc][h_in_local_idx][w_in_local_idx];
                                #pragma unroll
                                for (int c = 0; c < 8; c++) {
                                    out_val[c] += in_val * s_weight[i][j][kc][c];
                                }
                            }
                        }
                    }
                }
            }
            
            // Optimized output write with precomputed offsets
            int base_output_pixel = base_output_channel + h * W_out + w;
            for (int c = 0; c < 8; c++) {
                if (c_out_base + c < C_out) {
                    output[base_output_pixel + c * H_out_W_out] = out_val[c];
                }
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(kernel_size == KERNEL_SIZE, "Kernel size must be 3");

    int H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    int W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());
    int total_elements = N * C_out * H_out * W_out;
    
    if (total_elements == 0) {
        return output;
    }

    const int tile_c = 8;
    dim3 block_size(16, 32);  
    dim3 grid_size(
        (W_out + TILE_OUT_DIM - 1) / TILE_OUT_DIM,
        (H_out + TILE_OUT_DIM - 1) / TILE_OUT_DIM,
        N * ((C_out + tile_c - 1) / tile_c)
    );
    
    conv_transpose2d_kernel<<<grid_size, block_size>>>(
        input_contig.data_ptr<float>(), 
        weight_contig.data_ptr<float>(), 
        output.data_ptr<float>(),
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kernel_size, stride, padding, dilation
    );
    
    return output;
}
// PART-END