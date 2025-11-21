// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void upper_tri_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    extern __shared__ __nv_bfloat16 shared_memory[];
    const int tile_size = 64;
    const int num_warps = 16;
    const int warp_size = 32;
    
    int tiles = (N + tile_size - 1) / tile_size;
    int total_triangular_blocks = (tiles * (tiles + 1)) / 2;
    
    int block_index = blockIdx.x;
    if (block_index >= total_triangular_blocks) return;

    int block_i = 0;
    int block_j = 0;
    int remaining = block_index;
    for (int i = 0; i < tiles; i++) {
        int blocks_in_this_row = tiles - i;
        if (remaining < blocks_in_this_row) {
            block_j = i + remaining;
            block_i = i;
            break;
        }
        remaining -= blocks_in_this_row;
    }
    
    int row_start = block_i * tile_size;
    int col_start = block_j * tile_size;
    
    int warpId = threadIdx.y;
    int laneId = threadIdx.x;
    
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    
    __nv_bfloat16* Asub_bf16 = shared_memory;
    __nv_bfloat16* Bsub_bf16 = Asub_bf16 + (64 * 72);
    float* warp_output = (float*)(Bsub_bf16 + (64 * 72));
    
    int sub_tile_row = warpId / 4;
    int sub_tile_col = warpId % 4;
    int sub_tile_row_start = sub_tile_row * 16;
    int sub_tile_col_start = sub_tile_col * 16;
    
    for (int tile_k = block_i; tile_k <= block_j; tile_k++) {
        int k_start = tile_k * tile_size;
        
        // Vectorized loading with float2 for coalesced access
        #pragma unroll
        for (int row_offset = 0; row_offset < 4; row_offset++) {
            int row_in_tile = warpId * 4 + row_offset;
            int global_row_A = row_start + row_in_tile;
            int global_row_B = k_start + row_in_tile;
            int global_col_A = k_start + laneId * 2;
            int global_col_B = col_start + laneId * 2;
            
            // Load two consecutive elements for A using float2
            float2 a_load = {0.0f, 0.0f};
            if (global_row_A < N && global_col_A < N) {
                a_load = *reinterpret_cast<const float2*>(&A[global_row_A * N + global_col_A]);
            }
            float a_val0 = (global_col_A >= global_row_A) ? a_load.x : 0.0f;
            float a_val1 = (global_col_A+1 < N && global_col_A+1 >= global_row_A) ? a_load.y : 0.0f;
            __nv_bfloat162 a_bf162 = __float22bfloat162_rn(make_float2(a_val0, a_val1));
            *reinterpret_cast<__nv_bfloat162*>(&Asub_bf16[row_in_tile * 72 + laneId*2]) = a_bf162;
            
            // Load two consecutive elements for B using float2
            float2 b_load = {0.0f, 0.0f};
            if (global_row_B < N && global_col_B < N) {
                b_load = *reinterpret_cast<const float2*>(&B[global_row_B * N + global_col_B]);
            }
            float b_val0 = (global_col_B >= global_row_B) ? b_load.x : 0.0f;
            float b_val1 = (global_col_B+1 < N && global_col_B+1 >= global_row_B) ? b_load.y : 0.0f;
            __nv_bfloat162 b_bf162 = __float22bfloat162_rn(make_float2(b_val0, b_val1));
            *reinterpret_cast<__nv_bfloat162*>(&Bsub_bf16[row_in_tile * 72 + laneId*2]) = b_bf162;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int inner_k = 0; inner_k < 4; inner_k++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
            
            wmma::load_matrix_sync(a_frag, Asub_bf16 + (sub_tile_row_start * 72) + (inner_k * 16), 72);
            wmma::load_matrix_sync(b_frag, Bsub_bf16 + (inner_k * 16 * 72) + (sub_tile_col_start), 72);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    wmma::store_matrix_sync(warp_output + warpId * 256, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();
    
    // Write-back with vectorized access and separate conditions
    #pragma unroll
    for (int row_offset = 0; row_offset < 4; row_offset++) {
        int row_in_tile = warpId * 4 + row_offset;
        int global_i = row_start + row_in_tile;
        int col_in_tile = laneId * 2;
        int global_j = col_start + col_in_tile;
        
        int warp_id = (row_in_tile / 16) * 4 + (col_in_tile / 16);
        int in_warp_row = row_in_tile % 16;
        int in_warp_col = col_in_tile % 16;
        float val0 = warp_output[warp_id * 256 + in_warp_row * 16 + in_warp_col];
        float val1 = warp_output[warp_id * 256 + in_warp_row * 16 + in_warp_col+1];
        
        // Apply upper triangle conditions separately
        if (global_i < N && global_j < N && global_i <= global_j) {
            C[global_i * N + global_j] = val0;
        }
        if (global_i < N && global_j+1 < N && global_i <= global_j+1) {
            C[global_i * N + global_j+1] = val1;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor upper_tri_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same size");
    
    int N = A.size(0);
    auto out = torch::zeros({N, N}, A.options());
    if (N == 0) return out;

    const int tile_size = 64;
    const int num_warps = 16;
    dim3 block_dim(32, num_warps);
    
    int tiles = (N + tile_size - 1) / tile_size;
    int total_triangular_blocks = (tiles * (tiles + 1)) / 2;
    dim3 grid_dim(total_triangular_blocks);
    
    size_t smem_size = (64 * 72 * 2) * sizeof(__nv_bfloat16)
                     + (16 * 16 * 16) * sizeof(float);
    
    upper_tri_matmul_kernel<<<grid_dim, block_dim, smem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );
    
    return out;
}
// PART-END