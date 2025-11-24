// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void softsign_kernel(const float* input, float* output, int num_elements) {
    int index = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    bool thread_valid = (index < num_elements);
    unsigned active_mask = __ballot_sync(0xFFFFFFFF, thread_valid);
    
    if (active_mask == 0) return;
    
    float4 in = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 out = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (index + 3 < num_elements) {
        in = *reinterpret_cast<const float4*>(input + index);
    } else {
        for (int i = 0; i < 4; ++i) {
            if (index + i < num_elements) {
                (&in.x)[i] = input[index + i];
            }
        }
    }
    
    out.x = in.x * __frcp_rn(1.0f + fabsf(in.x));
    out.y = in.y * __frcp_rn(1.0f + fabsf(in.y));
    out.z = in.z * __frcp_rn(1.0f + fabsf(in.z));
    out.w = in.w * __frcp_rn(1.0f + fabsf(in.w));
    
    if (index + 3 < num_elements) {
        *reinterpret_cast<float4*>(output + index) = out;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (index + i < num_elements) {
                output[index + i] = (&out.x)[i];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor softsign_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    if (num_elements == 0) {
        return output;
    }
    
    const int threads = 1024;
    int blocks = (num_elements + 4095) / 4096;
    
    softsign_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    return output;
}
// PART-END