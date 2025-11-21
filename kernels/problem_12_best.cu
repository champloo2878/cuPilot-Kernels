// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void diag_mult_kernel(const float* A, const float* B, float* out, int N, int M) {
    constexpr int VEC = 4;
    int row = blockIdx.x;
    if (row >= N) return;
    
    // Cache diagonal value in register using read-only cache
    float a_val = __ldg(&A[row]);
    
    int tid = threadIdx.x;
    int stride = blockDim.x * VEC;
    
    for (int base_col = tid * VEC; base_col < M; base_col += stride) {
        if (base_col + VEC <= M) {
            // Coalesced vector load
            float4 b4 = *reinterpret_cast<const float4*>(&B[row * M + base_col]);
            float4 out4;
            out4.x = a_val * b4.x;
            out4.y = a_val * b4.y;
            out4.z = a_val * b4.z;
            out4.w = a_val * b4.w;
            // Coalesced vector store
            *reinterpret_cast<float4*>(&out[row * M + base_col]) = out4;
        } else {
            // Handle tail elements
            for (int j = base_col; j < M; j++) {
                out[row * M + j] = a_val * B[row * M + j];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor diag_mult_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 1, "A must be 1-dimensional");
    TORCH_CHECK(B.dim() == 2, "B must be 2-dimensional");
    TORCH_CHECK(A.size(0) == B.size(0), "First dimension of B must match length of A");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    
    int N = A.size(0);
    int M = B.size(1);
    auto out = torch::empty_like(B);
    
    const int block_size = 1024;
    diag_mult_kernel<<<N, block_size>>>(A.data_ptr<float>(), 
                                      B.data_ptr<float>(), 
                                      out.data_ptr<float>(), 
                                      N, M);
    return out;
}
// PART-END