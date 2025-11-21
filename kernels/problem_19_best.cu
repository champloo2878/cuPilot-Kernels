// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;          // Warp index
    int lane_id = tid % 32;           // Thread index in warp
    
    // Calculate total segments needed to cover input
    int total_segments = (n + 127) / 128;
    int permuted_warp_id;
    if (warp_id < total_segments) {
        // Apply XOR permutation to distribute segment accesses
        permuted_warp_id = warp_id ^ ((warp_id & 0x1F) << 5);
    } else {
        permuted_warp_id = total_segments; // Ensure out-of-bounds if warp_id exceeds segments
    }
    int segment_start = permuted_warp_id * 128; // Adjusted segment start

    // Early exit for warps beyond input size
    if (segment_start >= n) {
        return;
    }

    int warp_remaining = n - segment_start;
    if (warp_remaining >= 128) {
        // Full segment processing with coalesced access
        int idx = segment_start + lane_id * 4;
        float4 in_val = *reinterpret_cast<const float4*>(input + idx);
        float4 out_val;
        out_val.x = fmaxf(in_val.x, 0.0f);
        out_val.y = fmaxf(in_val.y, 0.0f);
        out_val.z = fmaxf(in_val.z, 0.0f);
        out_val.w = fmaxf(in_val.w, 0.0f);
        *reinterpret_cast<float4*>(output + idx) = out_val;
    } else {
        // Remainder handling
        for (int i = 0; i < 4; ++i) {
            int pos = segment_start + lane_id * 4 + i;
            if (pos < n) {
                output[pos] = fmaxf(input[pos], 0.0f);
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor custom_relu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    
    if (num_elements == 0) {
        return output;
    }

    const float* input_data = input.data_ptr<float>();
    float* output_data = output.data_ptr<float>();

    // Optimized launch parameters
    int block_size = 1024;
    int grid_size = (num_elements + block_size * 4 - 1) / (block_size * 4);

    cudaFuncSetCacheConfig(relu_kernel, cudaFuncCachePreferL1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    relu_kernel<<<grid_size, block_size, 0, stream>>>(input_data, output_data, num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}
// PART-END