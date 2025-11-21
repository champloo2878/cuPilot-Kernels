// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
// GELU approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
__device__ __forceinline__ float gelu_function(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta = 0.044715f;
    float x3 = x * x * x;
    float inner = kAlpha * (x + kBeta * x3);
    float tanh_inner = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_inner);
}

__global__ void gelu_kernel_fp32(const float* input, float* output, int64_t num_elements) {
    int64_t base_index = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (base_index >= num_elements) {
        return;
    }

    if (base_index + 3 < num_elements) {
        // Load 4 floats at once using float4
        float4 in4 = reinterpret_cast<const float4*>(input)[base_index / 4];
        float4 out4;
        out4.x = gelu_function(in4.x);
        out4.y = gelu_function(in4.y);
        out4.z = gelu_function(in4.z);
        out4.w = gelu_function(in4.w);
        reinterpret_cast<float4*>(output)[base_index / 4] = out4;
    } else {
        // Handle tail elements individually with unrolling
        #pragma unroll 4
        for (int i = 0; i < 4; ++i) {
            int64_t idx = base_index + i;
            if (idx < num_elements) {
                output[idx] = gelu_function(input[idx]);
            }
        }
    }
}

__global__ void gelu_kernel_fp16(const __half* input, __half* output, int64_t num_elements) {
    int64_t base_index = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (base_index >= num_elements) {
        return;
    }

    if (base_index + 1 < num_elements) {
        // Load 2 halfs at once using half2
        half2 in2 = reinterpret_cast<const half2*>(input)[base_index / 2];
        float2 f2;
        f2.x = __half2float(in2.x);
        f2.y = __half2float(in2.y);
        f2.x = gelu_function(f2.x);
        f2.y = gelu_function(f2.y);
        half2 out2 = __float22half2_rn(f2);
        reinterpret_cast<half2*>(output)[base_index / 2] = out2;
    } else {
        // Handle last element individually
        float x = __half2float(input[base_index]);
        float y = gelu_function(x);
        output[base_index] = __float2half(y);
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor gelu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    
    if (num_elements == 0) {
        return output;
    }

    // Experiment with different block sizes for A100 optimization
    // Try 512 threads per block for better occupancy on A100
    const int block_size = 512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (input.dtype() == torch::kFloat32) {
        int64_t grid_size = (num_elements + (4 * block_size) - 1) / (4 * block_size);
        // Ensure grid_size doesn't exceed maximum grid dimension
        if (grid_size > 2147483647) grid_size = 2147483647;
        dim3 grid(grid_size);
        dim3 block(block_size);
        
        gelu_kernel_fp32<<<grid, block, 0, stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements
        );
    } else if (input.dtype() == torch::kFloat16) {
        int64_t grid_size = (num_elements + (2 * block_size) - 1) / (2 * block_size);
        // Ensure grid_size doesn't exceed maximum grid dimension
        if (grid_size > 2147483647) grid_size = 2147483647;
        dim3 grid(grid_size);
        dim3 block(block_size);
        
        gelu_kernel_fp16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_elements
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
// PART-END