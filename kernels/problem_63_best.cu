// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector_types.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int h_out,
    const int w_out
) {
    const int BLOCK_X = 32;
    const int BLOCK_Y = 8;
    const int GROUP_SIZE = 16;
    const int TILE_Y = 16;
    const int TILE_X = 32;
    const int INPUT_TILE_Y = TILE_Y + 2;
    const int INPUT_TILE_X = TILE_X + 2;
    const int K_SQ = 9;
    const int NUM_F4_PER_ROW = GROUP_SIZE / 4; // Vectorized groups
    
    extern __shared__ float shared_mem[];
    float* shared_input = shared_mem;
    float4* shared_weights4 = reinterpret_cast<float4*>(&shared_mem[16 * INPUT_TILE_Y * INPUT_TILE_X]);
    float* shared_bias = reinterpret_cast<float*>(&shared_weights4[16 * K_SQ * NUM_F4_PER_ROW]);
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BLOCK_X + tx;
    
    int num_output_groups = (out_channels + GROUP_SIZE - 1) / GROUP_SIZE;
    int n = blockIdx.z / num_output_groups;
    int oc_group = (blockIdx.z % num_output_groups) * GROUP_SIZE;
    
    // Vectorized input tile loading
    int chunks_per_row = (INPUT_TILE_X + 3) / 4;
    int total_chunks = 16 * INPUT_TILE_Y * chunks_per_row;
    for (int idx = tid; idx < total_chunks; idx += BLOCK_X * BLOCK_Y) {
        int ic = idx / (INPUT_TILE_Y * chunks_per_row);
        int residual = idx % (INPUT_TILE_Y * chunks_per_row);
        int y = residual / chunks_per_row;
        int chunk_x = residual % chunks_per_row;
        int x = chunk_x * 4;
        
        int in_y = blockIdx.y * TILE_Y + y - padding;
        float4 val4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (in_y >= 0 && in_y < height) {
            int in_x0 = blockIdx.x * TILE_X + x - padding;
            const float* base_ptr = &input[((n * in_channels + ic) * height + in_y) * width];
            if (in_x0 >= 0 && in_x0 + 3 < width) {
                val4 = *reinterpret_cast<const float4*>(base_ptr + in_x0);
            } else {
                if (in_x0 >= 0 && in_x0 < width) val4.x = base_ptr[in_x0];
                if (in_x0+1 >= 0 && in_x0+1 < width) val4.y = base_ptr[in_x0+1];
                if (in_x0+2 >= 0 && in_x0+2 < width) val4.z = base_ptr[in_x0+2];
                if (in_x0+3 >= 0 && in_x0+3 < width) val4.w = base_ptr[in_x0+3];
            }
        }
        if (x < INPUT_TILE_X) {
            shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + y * INPUT_TILE_X + x] = val4.x;
            if (x+1 < INPUT_TILE_X) {
                shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + y * INPUT_TILE_X + x+1] = val4.y;
                if (x+2 < INPUT_TILE_X) {
                    shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + y * INPUT_TILE_X + x+2] = val4.z;
                    if (x+3 < INPUT_TILE_X) {
                        shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + y * INPUT_TILE_X + x+3] = val4.w;
                    }
                }
            }
        }
    }
    
    // Vectorized weight loading using float4
    const int total_weight_elements = 16 * K_SQ * NUM_F4_PER_ROW;
    for (int idx = tid; idx < total_weight_elements; idx += BLOCK_X * BLOCK_Y) {
        int ic = idx / (K_SQ * NUM_F4_PER_ROW);
        int residual = idx % (K_SQ * NUM_F4_PER_ROW);
        int pos = residual / NUM_F4_PER_ROW;
        int chunk = residual % NUM_F4_PER_ROW;
        
        if (ic < in_channels && chunk < NUM_F4_PER_ROW) {
            int oc_base = chunk * 4;
            int weight_idx0 = ((oc_group + oc_base) * in_channels + ic) * K_SQ + pos;
            
            float4 w4;
            w4.x = weight[weight_idx0];
            w4.y = (oc_base+1 < GROUP_SIZE) ? weight[weight_idx0 + in_channels * K_SQ] : 0.0f;
            w4.z = (oc_base+2 < GROUP_SIZE) ? weight[weight_idx0 + 2 * in_channels * K_SQ] : 0.0f;
            w4.w = (oc_base+3 < GROUP_SIZE) ? weight[weight_idx0 + 3 * in_channels * K_SQ] : 0.0f;
            
            shared_weights4[ic * K_SQ * NUM_F4_PER_ROW + pos * NUM_F4_PER_ROW + chunk] = w4;
        }
    }
    
    // Load bias
    if (tid < GROUP_SIZE) {
        shared_bias[tid] = bias ? bias[oc_group + tid] : 0.0f;
    }
    
    __syncthreads();
    
    // Local registers for results
    float results0[GROUP_SIZE] = {0.0f};  // For first pixel
    float results1[GROUP_SIZE] = {0.0f};  // For second pixel
    
    int y_local0 = ty * 2;
    int y_local1 = y_local0 + 1;
    int x_local = tx;
    
    int h0 = blockIdx.y * TILE_Y + y_local0;
    int h1 = h0 + 1;
    int w = blockIdx.x * TILE_X + x_local;
    
    // Only proceed if at least one pixel is in bounds
    if ((h0 < h_out && w < w_out) || (h1 < h_out && w < w_out)) {
        for (int ic = 0; ic < 16; ic++) {
            #pragma unroll
            for (int ky = 0; ky < 3; ky++) {
                #pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    int weight_base = ic * K_SQ * NUM_F4_PER_ROW + (ky * 3 + kx) * NUM_F4_PER_ROW;
                    
                    // Vectorized weight access using float4
                    float4 w0 = shared_weights4[weight_base];
                    float4 w1 = shared_weights4[weight_base + 1];
                    float4 w2 = shared_weights4[weight_base + 2];
                    float4 w3 = shared_weights4[weight_base + 3];
                    
                    // Compute for first pixel if in bounds
                    if (h0 < h_out && w < w_out) {
                        float in_val0 = shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + 
                                                   (y_local0 + ky) * INPUT_TILE_X + 
                                                   (x_local + kx)];
                        results0[0] += in_val0 * w0.x;
                        results0[1] += in_val0 * w0.y;
                        results0[2] += in_val0 * w0.z;
                        results0[3] += in_val0 * w0.w;
                        results0[4] += in_val0 * w1.x;
                        results0[5] += in_val0 * w1.y;
                        results0[6] += in_val0 * w1.z;
                        results0[7] += in_val0 * w1.w;
                        results0[8] += in_val0 * w2.x;
                        results0[9] += in_val0 * w2.y;
                        results0[10] += in_val0 * w2.z;
                        results0[11] += in_val0 * w2.w;
                        results0[12] += in_val0 * w3.x;
                        results0[13] += in_val0 * w3.y;
                        results0[14] += in_val0 * w3.z;
                        results0[15] += in_val0 * w3.w;
                    }
                    
                    // Compute for second pixel if in bounds
                    if (h1 < h_out && w < w_out) {
                        float in_val1 = shared_input[ic * INPUT_TILE_Y * INPUT_TILE_X + 
                                                   (y_local1 + ky) * INPUT_TILE_X + 
                                                   (x_local + kx)];
                        results1[0] += in_val1 * w0.x;
                        results1[1] += in_val1 * w0.y;
                        results1[2] += in_val1 * w0.z;
                        results1[3] += in_val1 * w0.w;
                        results1[4] += in_val1 * w1.x;
                        results1[5] += in_val1 * w1.y;
                        results1[6] += in_val1 * w1.z;
                        results1[7] += in_val1 * w1.w;
                        results1[8] += in_val1 * w2.x;
                        results1[9] += in_val1 * w2.y;
                        results1[10] += in_val1 * w2.z;
                        results1[11] += in_val1 * w2.w;
                        results1[12] += in_val1 * w3.x;
                        results1[13] += in_val1 * w3.y;
                        results1[14] += in_val1 * w3.z;
                        results1[15] += in_val1 * w3.w;
                    }
                }
            }
        }
        
        // Precompute valid output channels in group
        int num_channels_in_group = min(GROUP_SIZE, out_channels - oc_group);
        
        // Store results with bias for first pixel
        if (h0 < h_out && w < w_out) {
            for (int i = 0; i < num_channels_in_group; i++) {
                output[((n * out_channels + (oc_group + i)) * h_out + h0) * w_out + w] = 
                    results0[i] + shared_bias[i];
            }
        }
        
        // Store results with bias for second pixel
        if (h1 < h_out && w < w_out) {
            for (int i = 0; i < num_channels_in_group; i++) {
                output[((n * out_channels + (oc_group + i)) * h_out + h1) * w_out + w] = 
                    results1[i] + shared_bias[i];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    int kernel_size
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    
    const int h_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int w_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, h_out, w_out}, input.options());
    
    const int BLOCK_X = 32;
    const int BLOCK_Y = 8;
    const int GROUP_SIZE = 16;
    const int TILE_Y = 16;
    const int TILE_X = 32;
    const int INPUT_TILE_Y = TILE_Y + 2;
    const int INPUT_TILE_X = TILE_X + 2;
    const int K_SQ = 9;
    const int NUM_F4_PER_ROW = GROUP_SIZE / 4; // Vectorized groups
    
    int num_output_groups = (out_channels + GROUP_SIZE - 1) / GROUP_SIZE;
    
    dim3 grid_dim(
        (w_out + TILE_X - 1) / TILE_X,
        (h_out + TILE_Y - 1) / TILE_Y,
        batch_size * num_output_groups
    );
    
    const int input_tile_size = 16 * INPUT_TILE_Y * INPUT_TILE_X;
    const int weight_tile_size = 16 * K_SQ * GROUP_SIZE;  // Same byte size
    const int bias_tile_size = GROUP_SIZE;
    
    const int shared_mem_size = 
        (input_tile_size + weight_tile_size + bias_tile_size) * sizeof(float);
    
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    
    conv2d_forward_kernel<<<grid_dim, dim3(BLOCK_X, BLOCK_Y), shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        h_out,
        w_out
    );
    
    return output;
}
// PART-END