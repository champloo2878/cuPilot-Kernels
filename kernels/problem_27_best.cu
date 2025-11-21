// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void selu_kernel(const float* input, float* output, int num_elements) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    constexpr int kElementsPerThread = 4;
    const int elements_per_block = blockDim.x * kElementsPerThread;

    // Prefetch next block into L2 cache if it exists
    if (threadIdx.x < 32) {
        int next_block_idx = blockIdx.x + 1;
        int next_base_idx = next_block_idx * elements_per_block;
        int remaining = num_elements - next_base_idx;
        
        if (remaining > 0) {
            for (int i = 0; i < 4; i++) {
                int cache_line_idx = threadIdx.x + i * 32;
                int element_offset = cache_line_idx * 32;
                
                if (element_offset < remaining) {
                    const float* prefetch_addr = input + next_base_idx + element_offset;
                    // PTX assembly for L2 prefetch
                    asm volatile ("prefetch.L2 [%0];" :: "l"(prefetch_addr));
                }
            }
        }
    }

    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * kElementsPerThread;
    float x[kElementsPerThread];
    float results[kElementsPerThread];
    
    // Vectorized load with float4
    if (base_idx + 3 < num_elements) {
        float4 vec = reinterpret_cast<const float4*>(input + base_idx)[0];
        x[0] = vec.x; x[1] = vec.y; x[2] = vec.z; x[3] = vec.w;
    } else {
        // Scalar fallback for boundary elements
        for (int i = 0; i < kElementsPerThread; ++i) {
            int idx = base_idx + i;
            if (idx < num_elements) x[i] = input[idx];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
        int idx = base_idx + i;
        if (idx < num_elements) {
            float val = x[i];
            results[i] = scale * (val > 0 ? val : alpha * (expf(val) - 1));
        }
    }
    
    // Vectorized store with float4
    if (base_idx + 3 < num_elements) {
        float4 out;
        out.x = results[0]; out.y = results[1]; 
        out.z = results[2]; out.w = results[3];
        reinterpret_cast<float4*>(output + base_idx)[0] = out;
    } else {
        // Scalar fallback for boundary elements
        for (int i = 0; i < kElementsPerThread; ++i) {
            int idx = base_idx + i;
            if (idx < num_elements) output[idx] = results[i];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor selu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    const int threads = 1024;  // Increased block size
    constexpr int kElementsPerThread = 4;
    int grid_size = (num_elements + threads * kElementsPerThread - 1) / (threads * kElementsPerThread);
    
    selu_kernel<<<grid_size, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    return output;
}
// PART-END