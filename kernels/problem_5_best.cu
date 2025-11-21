// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>

// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void matrix_scalar_mult_kernel(const float* __restrict__ A, float* __restrict__ C, float s, int num_elements) {
    __shared__ float scalar_smem;
    if (threadIdx.x == 0) {
        scalar_smem = s;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start_index = idx * 4;
    
    if (start_index >= num_elements) return;
    
    float s_val = scalar_smem;
    
    if (start_index + 3 < num_elements) {
        float4 a4_val;
        a4_val.x = __ldg(A + start_index);
        a4_val.y = __ldg(A + start_index + 1);
        a4_val.z = __ldg(A + start_index + 2);
        a4_val.w = __ldg(A + start_index + 3);
        
        float4 c4_val = {a4_val.x * s_val, 
                         a4_val.y * s_val,
                         a4_val.z * s_val,
                         a4_val.w * s_val};
        reinterpret_cast<float4*>(C)[idx] = c4_val;
    } else {
        int remaining = num_elements - start_index;
        for (int i = 0; i < remaining; ++i) {
            C[start_index + i] = __ldg(A + start_index + i) * s_val;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor matrix_scalar_mult_cuda(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");

    auto C = torch::empty_like(A);
    int num_elements = A.numel();
    
    if (num_elements == 0) {
        return C;
    }
    
    const int block_size = 1024;
    int num_float4 = (num_elements + 3) >> 2;
    int grid_size = (num_float4 + block_size - 1) / block_size;
    
    matrix_scalar_mult_kernel<<<grid_size, block_size, 0>>>(
        A.data_ptr<float>(), 
        C.data_ptr<float>(), 
        s,
        num_elements
    );
    
    return C;
}
// PART-END