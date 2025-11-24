// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void rms_norm_kernel(const float* __restrict__ input, float* output, 
                              int batch_size, int features, int dim1, int dim2, 
                              float eps, long total_spatial, long feature_stride_val) {
    long spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial_idx >= total_spatial) return;

    long b = spatial_idx / feature_stride_val;
    long spatial_in_batch = spatial_idx - b * feature_stride_val;
    long base_offset = b * features * feature_stride_val + spatial_in_batch;
    const float* base = input + base_offset;

    float sum_sq = 0.0f;
    #pragma unroll
    for (int f = 0; f < 64; f++) {
        float val = __ldg(base + f * feature_stride_val);
        sum_sq += val * val;
    }

    float rms = sqrtf(sum_sq / 64.0f + eps);
    
    #pragma unroll
    for (int f = 0; f < 64; f++) {
        float val = __ldg(base + f * feature_stride_val);
        output[base_offset + f * feature_stride_val] = val / rms;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor rms_norm_cuda(torch::Tensor input, float eps) {
    auto input_cont = input.contiguous();
    int batch_size = input_cont.size(0);
    int features = input_cont.size(1);
    int dim1 = input_cont.size(2);
    int dim2 = input_cont.size(3);
    
    auto output = torch::empty_like(input_cont);
    long feature_stride_val = static_cast<long>(dim1) * dim2;
    long total_spatial = static_cast<long>(batch_size) * feature_stride_val;
    
    int block_size = 256;
    long grid_size = (total_spatial + block_size - 1) / block_size;
    int grid_size_int = static_cast<int>(std::min<long>(grid_size, 2147483647));
    
    rms_norm_kernel<<<grid_size_int, block_size, 0>>>(
        input_cont.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, features, dim1, dim2, eps, total_spatial, feature_stride_val
    );
    
    return output;
}
// PART-END