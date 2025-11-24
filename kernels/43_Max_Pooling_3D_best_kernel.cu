// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template <int KERNEL_D, int KERNEL_H, int KERNEL_W, 
          int DILATION_D, int DILATION_H, int DILATION_W>
__global__ void max_pool3d_kernel_specific(
    const float* input,
    float* output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int plane_size, int channel_size,
    int step_d, int step_h, int step_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_elements) return;

    int n = idx;
    int w_out = n % out_w;
    n /= out_w;
    int h_out = n % out_h;
    n /= out_h;
    int d_out = n % out_d;
    n /= out_d;
    int c = n % channels;
    n /= channels;
    int b = n;

    int start_d = d_out * stride_d - padding_d;
    int start_h = h_out * stride_h - padding_h;
    int start_w = w_out * stride_w - padding_w;
    
    float max_val = -FLT_MAX;
    
    int base1 = (b * channels + c) * channel_size;
    int base_offset = start_d * plane_size + start_h * in_w + start_w;
    int base3 = base1 + base_offset;
    
    #pragma unroll
    for (int kd = 0; kd < KERNEL_D; kd++) {
        int d_in = start_d + kd * DILATION_D;
        bool d_valid = (d_in >= 0) && (d_in < in_d);
        int offset_kd = base3 + kd * step_d;
        #pragma unroll
        for (int kh = 0; kh < KERNEL_H; kh++) {
            int h_in = start_h + kh * DILATION_H;
            bool h_valid = (h_in >= 0) && (h_in < in_h);
            int offset_kh = offset_kd + kh * step_h;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_W; kw++) {
                int w_in = start_w + kw * DILATION_W;
                bool w_valid = (w_in >= 0) && (w_in < in_w);
                int offset = offset_kh + kw * step_w;
                float candidate = (d_valid && h_valid && w_valid) ? input[offset] : -FLT_MAX;
                max_val = fmaxf(max_val, candidate);
            }
        }
    }
    
    output[idx] = max_val;
}

__global__ void max_pool3d_kernel(
    const float* input,
    float* output,
    int batch_size, int channels,
    int in_d, int in_h, int in_w,
    int out_d, int out_h, int out_w,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    int plane_size, int channel_size,
    int step_d, int step_h, int step_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    if (idx >= total_elements) return;

    int n = idx;
    int w_out = n % out_w;
    n /= out_w;
    int h_out = n % out_h;
    n /= out_h;
    int d_out = n % out_d;
    n /= out_d;
    int c = n % channels;
    n /= channels;
    int b = n;

    int start_d = d_out * stride_d - padding_d;
    int start_h = h_out * stride_h - padding_h;
    int start_w = w_out * stride_w - padding_w;
    
    float max_val = -FLT_MAX;
    
    int base1 = (b * channels + c) * channel_size;
    int base_offset = start_d * plane_size + start_h * in_w + start_w;
    int base3 = base1 + base_offset;
    
    for (int kd = 0; kd < kernel_d; kd++) {
        int d_in = start_d + kd * dilation_d;
        if (d_in < 0 || d_in >= in_d) continue;
        int offset_kd = base3 + kd * step_d;
        for (int kh = 0; kh < kernel_h; kh++) {
            int h_in = start_h + kh * dilation_h;
            if (h_in < 0 || h_in >= in_h) continue;
            int offset_kh = offset_kd + kh * step_h;
            for (int kw = 0; kw < kernel_w; kw++) {
                int w_in = start_w + kw * dilation_w;
                if (w_in < 0 || w_in >= in_w) continue;
                int offset = offset_kh + kw * step_w;
                max_val = fmaxf(max_val, input[offset]);
            }
        }
    }
    
    output[idx] = max_val;
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor max_pool3d_cuda(
    torch::Tensor input,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    
    int out_d = (in_d + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, channels, out_d, out_h, out_w}, input.options());
    int total_elements = batch_size * channels * out_d * out_h * out_w;
    
    const int threads_per_block = 512;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    int plane_size = in_h * in_w;
    int channel_size = in_d * plane_size;
    int step_d = dilation_d * plane_size;
    int step_h = dilation_h * in_w;
    int step_w = dilation_w;
    
    if (kernel_d == 3 && kernel_h == 3 && kernel_w == 3 &&
        dilation_d == 3 && dilation_h == 3 && dilation_w == 3) {
        max_pool3d_kernel_specific<3,3,3,3,3,3><<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            plane_size, channel_size,
            step_d, step_h, step_w
        );
    } else {
        max_pool3d_kernel<<<blocks, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            in_d, in_h, in_w,
            out_d, out_h, out_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            padding_d, padding_h, padding_w,
            dilation_d, dilation_h, dilation_w,
            plane_size, channel_size,
            step_d, step_h, step_w
        );
    }
    
    return output;
}
// PART-END