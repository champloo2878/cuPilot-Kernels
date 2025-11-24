#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

// Helper function for 2-element vectorized BF16 conversion
struct __align__(4) bf162 { __nv_bfloat16 x, y; };
static __device__ bf162 float2_to_bf162(float2 f) {
    bf162 v;
    v.x = __float2bfloat16(f.x);
    v.y = __float2bfloat16(f.y);
    return v;
}
// PART-END

// PART-START
__global__ void matmul_transposed_kernel(
    const float* A, 
    const float* B, 
    float* C, 
    int M, 
    int K, 
    int N
) {
    const int BLOCK_M = 64;
    const int BLOCK_N = 256;  // Increased output tile size
    const int BLOCK_K = 64;
    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 32;  // Doubled warp parallelism
    
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    const int warp_i = (warpId / 8) * 16;  // Adjusted for larger tile
    const int warp_j = (warpId % 8) * 32;   // 8 warps cover 256 columns
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> b_frag0, b_frag1;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag0, c_frag1;
    
    nvcuda::wmma::fill_fragment(c_frag0, 0.0f);
    nvcuda::wmma::fill_fragment(c_frag1, 0.0f);
    
    __shared__ __nv_bfloat16 As[BLOCK_M][BLOCK_K + 8];
    __shared__ __nv_bfloat16 Bs[BLOCK_K][BLOCK_N + 8];  // Padded for 256 columns
    
    const int global_m = blockIdx.x * BLOCK_M;
    const int global_n = blockIdx.y * BLOCK_N;
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int k_offset = 0; k_offset < K; k_offset += BLOCK_K) {
        bool tile_in_A = (k_offset + BLOCK_K <= K) && (global_m + BLOCK_M <= M);
        bool tile_in_B = (k_offset + BLOCK_K <= K) && (global_n + BLOCK_N <= N);

        // Optimized A tile loading (4 elements/thread)
        {
            int k_index = tid % 64;  // Single index for coalesced access
            int m_group = (tid / 64) * 4;
            int global_k = k_offset + k_index;
            int global_m_base = global_m + m_group;
            
            if (tile_in_A) {
                float4 val = *reinterpret_cast<const float4*>(A + global_k * M + global_m_base);
                bf162 v0 = float2_to_bf162(make_float2(val.x, val.y));
                bf162 v1 = float2_to_bf162(make_float2(val.z, val.w));
                As[m_group][k_index] = v0.x;
                As[m_group+1][k_index] = v0.y;
                As[m_group+2][k_index] = v1.x;
                As[m_group+3][k_index] = v1.y;
            } else {
                for (int i = 0; i < 4; i++) {
                    if (global_m_base + i < M) {
                        if (global_k < K) As[m_group+i][k_index] = __float2bfloat16(A[global_k * M + global_m_base + i]);
                        else As[m_group+i][k_index] = __float2bfloat16(0.0f);
                    } else {
                        As[m_group+i][k_index] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // Vectorized B tile loading (16 elements/thread)
        {
            int k_group0 = tid % 8;
            int k_group1 = k_group0 + 8;
            int n_index = (tid / 8) * 2;
            int global_k_base0 = k_offset + k_group0 * 4;
            int global_k_base1 = k_offset + k_group1 * 4;
            
            for (int n_offset = 0; n_offset < 2; n_offset++) {
                int global_n_idx = global_n + n_index + n_offset;
                if (tile_in_B) {
                    float4 val0 = *reinterpret_cast<const float4*>(B + global_n_idx * K + global_k_base0);
                    float4 val1 = *reinterpret_cast<const float4*>(B + global_n_idx * K + global_k_base1);
                    
                    bf162 v00 = float2_to_bf162(make_float2(val0.x, val0.y));
                    bf162 v01 = float2_to_bf162(make_float2(val0.z, val0.w));
                    Bs[k_group0*4][n_index+n_offset] = v00.x;
                    Bs[k_group0*4+1][n_index+n_offset] = v00.y;
                    Bs[k_group0*4+2][n_index+n_offset] = v01.x;
                    Bs[k_group0*4+3][n_index+n_offset] = v01.y;
                    
                    bf162 v10 = float2_to_bf162(make_float2(val1.x, val1.y));
                    bf162 v11 = float2_to_bf162(make_float2(val1.z, val1.w));
                    Bs[k_group1*4][n_index+n_offset] = v10.x;
                    Bs[k_group1*4+1][n_index+n_offset] = v10.y;
                    Bs[k_group1*4+2][n_index+n_offset] = v11.x;
                    Bs[k_group1*4+3][n_index+n_offset] = v11.y;
                } else {
                    for (int k = 0; k < 4; k++) {
                        int global_k0 = global_k_base0 + k;
                        int global_k1 = global_k_base1 + k;
                        if (global_n_idx < N) {
                            if (global_k0 < K) Bs[k_group0*4 + k][n_index+n_offset] = __float2bfloat16(B[global_n_idx * K + global_k0]);
                            else Bs[k_group0*4 + k][n_index+n_offset] = __float2bfloat16(0.0f);
                            if (global_k1 < K) Bs[k_group1*4 + k][n_index+n_offset] = __float2bfloat16(B[global_n_idx * K + global_k1]);
                            else Bs[k_group1*4 + k][n_index+n_offset] = __float2bfloat16(0.0f);
                        } else {
                            Bs[k_group0*4 + k][n_index+n_offset] = __float2bfloat16(0.0f);
                            Bs[k_group1*4 + k][n_index+n_offset] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Process tile in 16-element chunks
        for (int inner_k = 0; inner_k < BLOCK_K; inner_k += 16) {
            nvcuda::wmma::load_matrix_sync(a_frag, &As[warp_i][inner_k], BLOCK_K + 8);
            nvcuda::wmma::load_matrix_sync(b_frag0, &Bs[inner_k][warp_j], BLOCK_N + 8);
            nvcuda::wmma::load_matrix_sync(b_frag1, &Bs[inner_k][warp_j + 16], BLOCK_N + 8);
            
            nvcuda::wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
            nvcuda::wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        }
        
        __syncthreads();
    }
    
    // Store results to global memory
    float* C_ptr0 = C + (global_m + warp_i) * N + global_n + warp_j;
    float* C_ptr1 = C_ptr0 + 16;
    nvcuda::wmma::store_matrix_sync(C_ptr0, c_frag0, N, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(C_ptr1, c_frag1, N, nvcuda::wmma::mem_row_major);
}
// PART-END

// PART-START
torch::Tensor matmul_transposed_cuda(
    torch::Tensor A, 
    torch::Tensor B
) {
    int M = A.size(1);
    int K = A.size(0);
    int N = B.size(0);
    
    auto C = torch::zeros({M, N}, A.options());

    const int BLOCK_M = 64;
    const int BLOCK_N = 256;  // Larger output tile
    dim3 block(32, 32);       // Max threads (1024) for A100
    dim3 grid(
        (M + BLOCK_M - 1) / BLOCK_M,
        (N + BLOCK_N - 1) / BLOCK_N  // Fewer blocks for same work
    );

    matmul_transposed_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
// PART-END