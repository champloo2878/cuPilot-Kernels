// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void swish_kernel(const float* input, float* output, int num_elements) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    
    if (idx + 3 < num_elements) {
        // Vectorized load using __ldg for read-only cache
        const float4 in = __ldg(reinterpret_cast<const float4*>(input + idx));
        float4 out;
        // Compute Swish activation with reciprocal approximation
        out.x = in.x * __frcp_rn(1.0f + __expf(-in.x));
        out.y = in.y * __frcp_rn(1.0f + __expf(-in.y));
        out.z = in.z * __frcp_rn(1.0f + __expf(-in.z));
        out.w = in.w * __frcp_rn(1.0f + __expf(-in.w));
        *reinterpret_cast<float4*>(output + idx) = out;
    } else {
        // Scalar processing for tail elements
        for (int i = 0; i < 4; ++i) {
            if (idx + i < num_elements) {
                // Scalar __ldg for read-only access
                const float x = __ldg(input + idx + i);
                output[idx + i] = x * __frcp_rn(1.0f + __expf(-x));
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor swish_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    if (num_elements == 0) {
        return output;
    }

    const int block_size = 1024;
    int grid_size = (num_elements + 4 * block_size - 1) / (4 * block_size);

    swish_kernel<<<grid_size, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    return output;
}
// PART-END