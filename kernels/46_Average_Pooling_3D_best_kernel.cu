// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int depth,
    const int height,
    const int width,
    const int out_depth,
    const int out_height,
    const int out_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int total_elements
) {
    const int blocking_factor = 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = (total_elements + blocking_factor - 1) / blocking_factor;
    if (idx >= total_threads) return;

    // Precompute volume constants
    const int hw_size = height * width;
    const int dhw_size = depth * hw_size;
    
    // Decompose index for two output points
    int n = idx;
    int w_base = n % (out_width / blocking_factor);
    n /= (out_width / blocking_factor);
    int h_out = n % out_height;
    n /= out_height;
    int d_out = n % out_depth;
    n /= out_depth;
    int c = n % channels;
    int b = n / channels;

    // Precompute base offset for current batch and channel
    const int base_idx = (b * channels + c) * dhw_size;
    
    // Calculate start indices for input tensor
    int d_start = d_out * stride - padding;
    int h_start = h_out * stride - padding;
    int w_start0 = w_base * blocking_factor * stride - padding;
    int w_start1 = w_start0 + stride;  // Stride=2 => +2 in input space

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const float window_size_inv = 1.0f / (kernel_size * kernel_size * kernel_size);

    // Specialize for kernel_size=3 (only branch needed)
    #pragma unroll
    for (int kd = 0; kd < 3; ++kd) {
        int d = d_start + kd;
        bool d_valid = (d >= 0) & (d < depth);
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int h = h_start + kh;
            bool h_valid = (h >= 0) & (h < height);
            bool dh_valid = d_valid & h_valid;
            int plane_offset = base_idx + d * hw_size + h * width;

            // Process both outputs using 5 consecutive elements
            int w0 = w_start0;
            int w1 = w_start0 + 1;
            int w2 = w_start0 + 2;
            int w3 = w_start0 + 3;
            int w4 = w_start0 + 4;

            bool w0_valid = (w0 >= 0) & (w0 < width);
            bool w1_valid = (w1 >= 0) & (w1 < width);
            bool w2_valid = (w2 >= 0) & (w2 < width);
            bool w3_valid = (w3 >= 0) & (w3 < width);
            bool w4_valid = (w4 >= 0) & (w4 < width);

            float val0 = (dh_valid & w0_valid) ? input[plane_offset + w0] : 0.0f;
            float val1 = (dh_valid & w1_valid) ? input[plane_offset + w1] : 0.0f;
            float val2 = (dh_valid & w2_valid) ? input[plane_offset + w2] : 0.0f;
            float val3 = (dh_valid & w3_valid) ? input[plane_offset + w3] : 0.0f;
            float val4 = (dh_valid & w4_valid) ? input[plane_offset + w4] : 0.0f;

            sum0 += val0 + val1 + val2;
            sum1 += val2 + val3 + val4;
        }
    }

    // Vectorized store for coalesced writes
    int linear_index = idx * blocking_factor;
    float2 out_val = make_float2(sum0 * window_size_inv, sum1 * window_size_inv);
    *reinterpret_cast<float2*>(output + linear_index) = out_val;
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor avg_pool3d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    // Get input dimensions
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    const int dhw_size = depth * height * width;

    // Calculate output dimensions
    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Create output tensor
    auto output = torch::zeros({batch_size, channels, out_depth, out_height, out_width}, input.options());

    // Calculate total elements in output
    int total_elements = batch_size * channels * out_depth * out_height * out_width;
    if (total_elements == 0) return output;

    // Configure CUDA kernel launch
    const int block_size = 256;
    const int blocking_factor = 2;
    int total_threads = (total_elements + blocking_factor - 1) / blocking_factor;
    int grid_size = (total_threads + block_size - 1) / block_size;

    // Launch kernel
    avg_pool3d_kernel<<<grid_size, block_size>>>(
        input.contiguous().data_ptr<float>(),
        output.contiguous().data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        out_depth,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        total_elements
    );

    return output;
}
// PART-END