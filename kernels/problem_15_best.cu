// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

__device__ __constant__ int2 block_list[4096];

__host__ __device__ __forceinline__ int2 morton_to_xy(uint32_t code) {
    uint32_t x = 0, y = 0;
    for (int i = 0; i < 16; ++i) {
        x |= (code & (1u << (2*i))) >> i;
        y |= (code & (1u << (2*i+1))) >> (i+1);
    }
    return make_int2(x, y);
}

__host__ __device__ __forceinline__ uint32_t xy_to_morton(uint32_t x, uint32_t y) {
    uint32_t code = 0;
    for (int i = 0; i < 16; ++i) {
        code |= (x & (1u << i)) << i;
        code |= (y & (1u << i)) << (i+1);
    }
    return code;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void lower_tri_mul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    int bx, by;
    if (gridDim.y == 1) {
        int2 block_coord = block_list[blockIdx.x];
        bx = block_coord.x;
        by = block_coord.y;
    } else {
        bx = blockIdx.x;
        by = blockIdx.y;
        if (by < bx) return;
    }

    __shared__ half A_shared[64][72];
    __shared__ half B_shared[64][72];
    __shared__ float C_shared[64][64];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int warpID = (ty * 16 + tx) >> 5;
    int tile_row = warpID / 2;
    int tile_col_offset = (warpID % 2) * 2;

    // Precompute store masks
    unsigned store_masks[4];
    int global_row_base = by * 64 + ty * 4;
    int global_col_base = bx * 64 + tx * 4;
    for (int i = 0; i < 4; i++) {
        int global_row = global_row_base + i;
        store_masks[i] = 0;
        for (int j = 0; j < 4; j++) {
            int global_col = global_col_base + j;
            if (global_col <= global_row) {
                store_masks[i] |= (1 << j);
            }
        }
    }

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag1, b_frag2;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag1, acc_frag2;

    nvcuda::wmma::fill_fragment(acc_frag1, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag2, 0.0f);

    for (int k_tile = bx; k_tile <= by; k_tile++) {
        // Load A tile without conditional zeroing
        for (int i_off = 0; i_off < 4; i_off++) {
            int row = by * 64 + ty * 4 + i_off;
            int col = k_tile * 64 + tx * 4;
            float4 tmp = *reinterpret_cast<const float4*>(&A[row * N + col]);
            
            A_shared[ty*4+i_off][tx*4]   = __float2half(tmp.x);
            A_shared[ty*4+i_off][tx*4+1] = __float2half(tmp.y);
            A_shared[ty*4+i_off][tx*4+2] = __float2half(tmp.z);
            A_shared[ty*4+i_off][tx*4+3] = __float2half(tmp.w);
        }

        // Load B tile without conditional zeroing
        for (int i_off = 0; i_off < 4; i_off++) {
            int row = k_tile * 64 + ty * 4 + i_off;
            int col = bx * 64 + tx * 4;
            float4 tmp = *reinterpret_cast<const float4*>(&B[row * N + col]);
            
            B_shared[ty*4+i_off][tx*4]   = __float2half(tmp.x);
            B_shared[ty*4+i_off][tx*4+1] = __float2half(tmp.y);
            B_shared[ty*4+i_off][tx*4+2] = __float2half(tmp.z);
            B_shared[ty*4+i_off][tx*4+3] = __float2half(tmp.w);
        }
        __syncthreads();

        #pragma unroll
        for (int inner_step = 0; inner_step < 4; inner_step++) {
            nvcuda::wmma::load_matrix_sync(a_frag, &A_shared[tile_row * 16][inner_step * 16], 72);
            nvcuda::wmma::load_matrix_sync(b_frag1, &B_shared[inner_step * 16][tile_col_offset * 16], 72);
            nvcuda::wmma::load_matrix_sync(b_frag2, &B_shared[inner_step * 16][(tile_col_offset + 1) * 16], 72);
            nvcuda::wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
            nvcuda::wmma::mma_sync(acc_frag2, a_frag, b_frag2, acc_frag2);
        }
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&C_shared[tile_row * 16][tile_col_offset * 16], acc_frag1, 64, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(&C_shared[tile_row * 16][(tile_col_offset + 1) * 16], acc_frag2, 64, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // Store results using precomputed masks
    for (int i = 0; i < 4; i++) {
        int row = global_row_base + i;
        if (row < N) {
            int col_base = global_col_base;
            if (col_base + 3 < N) {
                float4 val = {
                    C_shared[ty*4+i][tx*4],
                    C_shared[ty*4+i][tx*4+1],
                    C_shared[ty*4+i][tx*4+2],
                    C_shared[ty*4+i][tx*4+3]
                };

                float4 result;
                unsigned mask = store_masks[i];
                result.x = (mask & 1) ? val.x : 0.0f;
                result.y = (mask & 2) ? val.y : 0.0f;
                result.z = (mask & 4) ? val.z : 0.0f;
                result.w = (mask & 8) ? val.w : 0.0f;

                *reinterpret_cast<float4*>(&C[row * N + col_base]) = result;
            } else {
                for (int j = 0; j < 4; j++) {
                    int col = col_base + j;
                    if (col < N) {
                        C[row * N + col] = (col <= row) ? C_shared[ty*4+i][tx*4+j] : 0.0f;
                    }
                }
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
#include <vector>
#include <algorithm>

torch::Tensor lower_tri_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int N = A.size(0);
    TORCH_CHECK(A.size(1) == N, "A must be square");
    TORCH_CHECK(B.size(0) == N, "B must have the same size as A");
    TORCH_CHECK(B.size(1) == N, "B must be square");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    auto out = torch::zeros_like(A);

    if (N == 0) {
        return out;
    }

    dim3 block(16, 16);
    int grid_dim = (N + 63) / 64;
    int num_blocks = (grid_dim * (grid_dim + 1)) / 2;
    
    if (num_blocks <= 4096) {
        std::vector<int2> block_vec;
        block_vec.reserve(num_blocks);
        
        for (int by = 0; by < grid_dim; by++) {
            for (int bx = 0; bx <= by; bx++) {
                block_vec.push_back({bx, by});
            }
        }
        
        auto morton_compare = [](const int2& a, const int2& b) {
            uint32_t code_a = xy_to_morton(a.x, a.y);
            uint32_t code_b = xy_to_morton(b.x, b.y);
            return code_a < code_b;
        };
        
        std::sort(block_vec.begin(), block_vec.end(), morton_compare);
        cudaMemcpyToSymbol(block_list, block_vec.data(), num_blocks * sizeof(int2));
        
        dim3 grid(num_blocks, 1);
        cudaFuncSetCacheConfig(lower_tri_mul_kernel, cudaFuncCachePreferShared);
        lower_tri_mul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N);
    } else {
        dim3 grid(grid_dim, grid_dim);
        cudaFuncSetCacheConfig(lower_tri_mul_kernel, cudaFuncCachePreferShared);
        lower_tri_mul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), N);
    }

    return out;
}
// PART-END