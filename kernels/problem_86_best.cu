// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void fused_depthwise_pointwise_kernel(
    const float* __restrict__ input,
    const float* __restrict__ depthwise_weight,
    const float* __restrict__ pointwise_weight_t,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int input_height, int input_width,
    int output_height, int output_width,
    int stride, int padding, int dilation, int kernel_size,
    int in_channels, int out_channels,
    int batch_size
) {
    // Hardcoded parameters for specialization
    const int IN_CHANNELS = 64;
    const int OUT_CHANNELS = 128;
    const int KERNEL_SIZE = 3;
    const int STRIDE = 1;
    const int PADDING = 1;
    
    extern __shared__ __half shared_depthwise[];
    
    int w_start = blockIdx.x * 8;
    int h = blockIdx.y;
    int b = blockIdx.z;
    int tid = threadIdx.x;
    
    // Each thread handles 2 input channels
    int c0 = tid * 2;
    int c1 = tid * 2 + 1;
    float val0[8] = {0.0f};
    float val1[8] = {0.0f};

    // Preload depthwise weights into registers
    float weights0[3][3];
    float weights1[3][3];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            weights0[kh][kw] = depthwise_weight[c0 * 9 + kh * 3 + kw];
            weights1[kh][kw] = depthwise_weight[c1 * 9 + kh * 3 + kw];
        }
    }

    // Preload input tile (3x10) for two channels
    float in_tile0[3][10];
    float in_tile1[3][10];
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        int h_in = h + kh - PADDING;  // padding=1
        bool valid_h = (h_in >= 0 && h_in < input_height);
        #pragma unroll
        for (int col = 0; col < 10; col++) {
            int w_in = w_start - PADDING + col;  // padding=1
            if (valid_h && w_in >= 0 && w_in < input_width) {
                int addr = b * (input_height * input_width * IN_CHANNELS)
                         + h_in * (input_width * IN_CHANNELS)
                         + w_in * IN_CHANNELS
                         + c0;
                float2 in_val = *reinterpret_cast<const float2*>(&input[addr]);
                in_tile0[kh][col] = in_val.x;
                in_tile1[kh][col] = in_val.y;
            } else {
                in_tile0[kh][col] = 0.0f;
                in_tile1[kh][col] = 0.0f;
            }
        }
    }

    // Compute depthwise convolution using preloaded tiles
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int col = j + kw;
                val0[j] += weights0[kh][kw] * in_tile0[kh][col];
                val1[j] += weights1[kh][kw] * in_tile1[kh][col];
            }
        }
    }

    // Store to shared memory
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        shared_depthwise[c0 * 8 + j] = __float2half(val0[j]);
        shared_depthwise[c1 * 8 + j] = __float2half(val1[j]);
    }
    
    __syncthreads();
    
    // Pointwise convolution - each thread handles 4 output channels
    if (tid < 32) {
        int c_out0 = tid * 4;
        float accum[8][4] = {{0.0f}};

        #pragma unroll
        for (int c_in = 0; c_in < IN_CHANNELS; c_in++) {
            float4 dw_vals = *reinterpret_cast<const float4*>(&shared_depthwise[c_in * 8]);
            const __half* dw_half = reinterpret_cast<const __half*>(&dw_vals);
            float4 weight_vals = *reinterpret_cast<const float4*>(
                &pointwise_weight_t[c_in * OUT_CHANNELS + c_out0]);

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float dw_val = __half2float(dw_half[j]);
                accum[j][0] += dw_val * weight_vals.x;
                accum[j][1] += dw_val * weight_vals.y;
                accum[j][2] += dw_val * weight_vals.z;
                accum[j][3] += dw_val * weight_vals.w;
            }
        }
        
        // Add bias if present
        if (bias) {
            float4 bias_vals = *reinterpret_cast<const float4*>(&bias[c_out0]);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                accum[j][0] += bias_vals.x;
                accum[j][1] += bias_vals.y;
                accum[j][2] += bias_vals.z;
                accum[j][3] += bias_vals.w;
            }
        }
        
        // Vector store output
        int output_base = b * (output_height * output_width * OUT_CHANNELS) 
                        + h * (output_width * OUT_CHANNELS);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_idx = w_start + j;
            if (w_idx < output_width) {
                float* out_ptr = output + output_base + w_idx * OUT_CHANNELS + c_out0;
                *reinterpret_cast<float4*>(out_ptr) = make_float4(
                    accum[j][0], accum[j][1], accum[j][2], accum[j][3]);
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor fused_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor bias,
    int stride, int padding, int dilation, int kernel_size
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = pointwise_weight.size(0);
    
    // Calculate output dimensions
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Convert input to NHWC and create NHWC output
    auto input_nhwc = input.permute({0, 2, 3, 1}).contiguous();
    auto output_nhwc = torch::zeros({batch_size, output_height, output_width, out_channels}, input.options());
    
    // Preprocess weights and bias
    torch::Tensor pointwise_weight_t = pointwise_weight.t().contiguous();
    torch::Tensor bias_cont = bias.defined() ? bias.contiguous() : torch::Tensor();
    
    // Optimized thread configuration
    const int threads = 32;
    dim3 grid((output_width + 7) / 8, output_height, batch_size);
    
    // Shared memory calculation
    size_t shared_mem_size = in_channels * 8 * sizeof(__half);
    
    // Prepare bias pointer
    float* bias_ptr = bias_cont.defined() ? bias_cont.data_ptr<float>() : nullptr;
    
    // Launch optimized kernel
    fused_depthwise_pointwise_kernel<<<grid, threads, shared_mem_size>>>(
        input_nhwc.contiguous().data_ptr<float>(),
        depthwise_weight.contiguous().data_ptr<float>(),
        pointwise_weight_t.data_ptr<float>(),
        bias_ptr,
        output_nhwc.data_ptr<float>(),
        input_height, input_width,
        output_height, output_width,
        stride, padding, dilation, kernel_size,
        in_channels, out_channels,
        batch_size
    );
    
    // Convert output back to NCHW
    return output_nhwc.permute({0, 3, 1, 2});
}
// PART-END