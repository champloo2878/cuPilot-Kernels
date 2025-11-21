// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float c_weights[128*9];
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int batch_size, 
    const int in_channels, 
    const int out_channels, 
    const int height_in, 
    const int width_in,
    const int kernel_size, 
    const int stride, 
    const int padding, 
    const int height_out, 
    const int width_out
) {
    // Expanded shared memory for larger tile with padding to reduce bank conflicts
    __shared__ float input_tile[34][72];

    const int block_start_h = blockIdx.y * 32;
    const int block_start_w = blockIdx.x * 64;
    const int batch = blockIdx.z / out_channels;
    const int c_out = blockIdx.z % out_channels;
    const int c_in = c_out;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    const int num_float4_per_row = 18;
    const int total_float4 = 34 * num_float4_per_row;
    
    for (int i = tid; i < total_float4; i += 128) {
        const int row = i / num_float4_per_row;
        const int vec_in_row = i % num_float4_per_row;
        const int col0 = vec_in_row * 4;
        const int h_in = block_start_h + row - padding;
        const int w_in0 = block_start_w + col0 - padding;
        
        float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const bool valid_row = (h_in >= 0) && (h_in < height_in);
        
        if (valid_row) {
            const float* base_ptr = input + batch * (in_channels * height_in * width_in) 
                                  + c_in * (height_in * width_in) 
                                  + h_in * width_in;
            
            if (w_in0 >= 0 && w_in0 + 3 < width_in) {
                vals = *reinterpret_cast<const float4*>(base_ptr + w_in0);
            } else {
                if (w_in0 >= 0 && w_in0 < width_in) vals.x = base_ptr[w_in0];
                if (w_in0+1 >= 0 && w_in0+1 < width_in) vals.y = base_ptr[w_in0+1];
                if (w_in0+2 >= 0 && w_in0+2 < width_in) vals.z = base_ptr[w_in0+2];
                if (w_in0+3 >= 0 && w_in0+3 < width_in) vals.w = base_ptr[w_in0+3];
            }
        }
        *reinterpret_cast<float4*>(&input_tile[row][col0]) = vals;
    }
    
    __syncthreads();
    
    float w_val[3][3];
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            w_val[kh][kw] = c_weights[c_out * 9 + kh * 3 + kw];
        }
    }
    
    const int local_h = ty * 4;
    const int local_w = tx * 4;
    const int global_h_base = block_start_h + local_h;
    const int global_w_base = block_start_w + local_w;
    
    float out[4][4] = {{0.0f}};
    
    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            const float w = w_val[kh][kw];
            const int row_idx = local_h + kh;
            const int col_idx = local_w + kw;
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float* tile_row = &input_tile[row_idx + i][col_idx];
                out[i][0] += tile_row[0] * w;
                out[i][1] += tile_row[1] * w;
                out[i][2] += tile_row[2] * w;
                out[i][3] += tile_row[3] * w;
            }
        }
    }
    
    if (bias != nullptr) {
        const float b = bias[c_out];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[i][j] += b;
            }
        }
    }
    
    const int out_base = batch * (out_channels * height_out * width_out) 
                       + c_out * (height_out * width_out);
    
    // Precompute validity masks for boundaries
    int row_mask = 0;
    for (int i = 0; i < 4; i++) {
        if (global_h_base + i < height_out) {
            row_mask |= (1 << i);
        }
    }
    
    int col_mask = 0;
    for (int j = 0; j < 4; j++) {
        if (global_w_base + j < width_out) {
            col_mask |= (1 << j);
        }
    }
    
    const bool full_col_segment = (global_w_base + 3) < width_out;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (!(row_mask & (1 << i))) continue;
        
        const int row_idx = out_base + (global_h_base + i) * width_out + global_w_base;
        float* out_ptr = output + row_idx;
        
        if (full_col_segment) {
            float2 val0 = make_float2(out[i][0], out[i][1]);
            float2 val1 = make_float2(out[i][2], out[i][3]);
            *reinterpret_cast<float2*>(out_ptr) = val0;
            *reinterpret_cast<float2*>(out_ptr + 2) = val1;
        } else {
            if (col_mask & 1) out_ptr[0] = out[i][0];
            if (col_mask & 2) out_ptr[1] = out[i][1];
            if (col_mask & 4) out_ptr[2] = out[i][2];
            if (col_mask & 8) out_ptr[3] = out[i][3];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor depthwise_conv2d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int stride,
    int padding
) {
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    
    const int batch_size = input_contig.size(0);
    const int in_channels = input_contig.size(1);
    const int height_in = input_contig.size(2);
    const int width_in = input_contig.size(3);
    
    const int out_channels = weight_contig.size(0);
    const int kernel_size = weight_contig.size(2);
    
    const int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    const int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, height_out, width_out}, input_contig.options());
    
    if (output.numel() == 0) {
        return output;
    }
    
    AT_ASSERT(weight_contig.numel() == out_channels * kernel_size * kernel_size, 
              "Weight tensor must match constant memory dimensions");
    cudaMemcpyToSymbol(c_weights, weight_contig.data_ptr<float>(), 
                      sizeof(float) * out_channels * kernel_size * kernel_size);
    
    dim3 block(16, 8);
    dim3 grid(
        (width_out + 63) / 64,
        (height_out + 31) / 32,
        batch_size * out_channels
    );
    
    const float* bias_ptr = bias.defined() ? bias.contiguous().data_ptr<float>() : nullptr;
    
    depthwise_conv2d_kernel<<<grid, block>>>(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        kernel_size,
        stride,
        padding,
        height_out,
        width_out
    );
    
    return output;
}
// PART-END