// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void tanh_kernel(const float* input, float* output, int num_elements) {
    __shared__ float smem[32 * 4 * 33]; // 32 warps × 4 elements × 33 floats (padded)
    
    int tid = threadIdx.x;
    int warp_id = tid >> 5;       // Warp index [0, 31]
    int lane_id = tid & 31;        // Lane index [0, 31]
    
    int block_offset = blockIdx.x * (blockDim.x << 2);  // blockDim.x*4
    int thread_offset = block_offset + (tid << 2);      // tid*4
    
    // Load data with float4
    float4 in_val;
    if (thread_offset + 3 < num_elements) {
        in_val = *reinterpret_cast<const float4*>(input + thread_offset);
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = thread_offset + i;
            (&in_val.x)[i] = (idx < num_elements) ? input[idx] : 0.0f;
        }
    }

    // Store to shared memory (bank-conflict free)
    int smem_base = warp_id * (4 * 33);  // 132 elements per warp
    smem[smem_base + lane_id] = in_val.x;
    smem[smem_base + 33 + lane_id] = in_val.y;
    smem[smem_base + 66 + lane_id] = in_val.z;
    smem[smem_base + 99 + lane_id] = in_val.w;
    
    __syncthreads();

    // Compute tanh from shared memory
    float4 out_val;
    out_val.x = fast_tanh(smem[smem_base + lane_id]);
    out_val.y = fast_tanh(smem[smem_base + 33 + lane_id]);
    out_val.z = fast_tanh(smem[smem_base + 66 + lane_id]);
    out_val.w = fast_tanh(smem[smem_base + 99 + lane_id]);

    // Write results
    if (thread_offset + 3 < num_elements) {
        *reinterpret_cast<float4*>(output + thread_offset) = out_val;
    } else {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = thread_offset + i;
            if (idx < num_elements) {
                output[idx] = (&out_val.x)[i];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor tanh_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    if (num_elements == 0) {
        return output;
    }
    
    const int block_size = 1024;
    const int elements_per_block = block_size * 4;  // 4096
    const int grid_size = (num_elements + elements_per_block - 1) / elements_per_block;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tanh_kernel<<<grid_size, block_size, 0, stream>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        num_elements
    );
    
    return output;
}
// PART-END