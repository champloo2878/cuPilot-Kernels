#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// PART-START
__global__ void gemv_kernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int K) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    unsigned int lane_id = tid & 31;
    unsigned int warp_id = tid >> 5;
    
    const int num_vectors = (K + 3) / 4;
    const float4* A_row_vec = reinterpret_cast<const float4*>(A + row * K);
    const float4* B_vec = reinterpret_cast<const float4*>(B);
    
    float sum = 0.0f;

    if (tid < num_vectors) {
        float4 a_val = A_row_vec[tid];
        float4 b_val = B_vec[tid];
        
        for (int i = tid; i < num_vectors - blockDim.x*3; i += blockDim.x*4) {
            float4 a_val_next1 = A_row_vec[i+blockDim.x];
            float4 b_val_next1 = B_vec[i+blockDim.x];
            float4 a_val_next2 = A_row_vec[i+blockDim.x*2];
            float4 b_val_next2 = B_vec[i+blockDim.x*2];
            float4 a_val_next3 = A_row_vec[i+blockDim.x*3];
            float4 b_val_next3 = B_vec[i+blockDim.x*3];
            
            sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
            a_val = a_val_next1;
            b_val = b_val_next1;
            sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
            
            a_val = a_val_next2;
            b_val = b_val_next2;
            sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
            
            a_val = a_val_next3;
            b_val = b_val_next3;
            sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
        }
        
        int tail_start = num_vectors - (num_vectors % (blockDim.x*4));
        for (int i = tail_start + tid; i < num_vectors; i += blockDim.x) {
            float4 a_val_tail = A_row_vec[i];
            float4 b_val_tail = B_vec[i];
            sum += a_val_tail.x * b_val_tail.x + a_val_tail.y * b_val_tail.y + 
                   a_val_tail.z * b_val_tail.z + a_val_tail.w * b_val_tail.w;
        }
    }
    
    float tmp = sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float neighbor = __shfl_down_sync(0xFFFFFFFF, tmp, offset);
        if (lane_id < offset) {
            tmp += neighbor;
        }
    }
    
    __shared__ float warp_sums[32];
    if (lane_id == 0) {
        warp_sums[warp_id] = tmp;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x >> 5)) ? warp_sums[lane_id] : 0.0f;
        float tmp_val = val;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float neighbor = __shfl_down_sync(0xFFFFFFFF, tmp_val, offset);
            if (lane_id < offset) {
                tmp_val += neighbor;
            }
        }
        if (lane_id == 0) {
            C[row] = tmp_val;
        }
    }
}
// PART-END

// PART-START
torch::Tensor gemv_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int M = A.size(0);
    int K = A.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B shapes not matched");
    TORCH_CHECK(B.size(1) == 1, "B must be a column vector");

    auto C = torch::zeros({M, 1}, A.options());

    if (A.numel() == 0 || B.numel() == 0) {
        return C;
    }

    dim3 block(1024);
    dim3 grid(M);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    gemv_kernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, M, K);

    return C;
}
// PART-END