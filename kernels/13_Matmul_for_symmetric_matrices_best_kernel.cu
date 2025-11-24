#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/std/cstdint>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 4
#define BLOCK_WARPS (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
#define THREADS_PER_BLOCK (BLOCK_WARPS * 32)
// PART-END

// PART-START
__global__ void symmetric_matmul_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ __align__(16) half sA[2][128][WMMA_K];
    __shared__ __align__(16) half sB[2][WMMA_K][136];
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_i = threadIdx.y / BLOCK_COL_WARPS;
    const int warp_j = threadIdx.y % BLOCK_COL_WARPS;
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frags[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frags[2];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accum_frags[4];
    
    for (int i = 0; i < 4; ++i) {
        nvcuda::wmma::fill_fragment(accum_frags[i], 0.0f);
    }
    
    const int block_row = blockIdx.y * 128;
    const int block_col = blockIdx.x * 128;
    const int num_tiles = N / WMMA_K;
    
    int buf_idx = 0;
    
    // Preload first tile
    if (tid < 256) {
        int row_in_tile = tid / 2;
        int col_in_tile = (tid % 2) * 8;
        const half* global_ptr = A + (block_row + row_in_tile) * N + col_in_tile;
        half* shared_ptr = &sA[buf_idx][row_in_tile][col_in_tile];
        uint32_t smem_addr = __cvta_generic_to_shared(shared_ptr);
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(global_ptr);
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(smem_addr), "l"(gmem_addr)
        );
    }
    else if (tid < 512) {
        int seg_id_B = tid - 256;
        int row_in_tile_B = seg_id_B / 16;
        int col_in_tile_B = (seg_id_B % 16) * 8;
        const half* global_ptr_B = B + row_in_tile_B * N + (block_col + col_in_tile_B);
        half* shared_ptr_B = &sB[buf_idx][row_in_tile_B][col_in_tile_B];
        uint32_t smem_addr = __cvta_generic_to_shared(shared_ptr_B);
        uint64_t gmem_addr = reinterpret_cast<uint64_t>(global_ptr_B);
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(smem_addr), "l"(gmem_addr)
        );
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group %0;" :: "n"(0));
    __syncthreads();
    
    for (int t = 0; t < num_tiles; ++t) {
        int next_buf = 1 - buf_idx;
        
        // Prefetch next tile
        if (t < num_tiles - 1) {
            int tile_k_next = (t + 1) * WMMA_K;
            if (tid < 256) {
                int row_in_tile = tid / 2;
                int col_in_tile = (tid % 2) * 8;
                const half* global_ptr = A + (block_row + row_in_tile) * N + tile_k_next + col_in_tile;
                half* shared_ptr = &sA[next_buf][row_in_tile][col_in_tile];
                uint32_t smem_addr = __cvta_generic_to_shared(shared_ptr);
                uint64_t gmem_addr = reinterpret_cast<uint64_t>(global_ptr);
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr), "l"(gmem_addr)
                );
            }
            else if (tid < 512) {
                int seg_id_B = tid - 256;
                int row_in_tile_B = seg_id_B / 16;
                int col_in_tile_B = (seg_id_B % 16) * 8;
                const half* global_ptr_B = B + (tile_k_next + row_in_tile_B) * N + (block_col + col_in_tile_B);
                half* shared_ptr_B = &sB[next_buf][row_in_tile_B][col_in_tile_B];
                uint32_t smem_addr = __cvta_generic_to_shared(shared_ptr_B);
                uint64_t gmem_addr = reinterpret_cast<uint64_t>(global_ptr_B);
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr), "l"(gmem_addr)
                );
            }
            asm volatile("cp.async.commit_group;");
        }
        
        // Compute current tile
        nvcuda::wmma::load_matrix_sync(a_frags[0], &sA[buf_idx][warp_i*32][0], 16);
        nvcuda::wmma::load_matrix_sync(a_frags[1], &sA[buf_idx][warp_i*32+16][0], 16);
        nvcuda::wmma::load_matrix_sync(b_frags[0], &sB[buf_idx][0][warp_j*32], 136);
        nvcuda::wmma::load_matrix_sync(b_frags[1], &sB[buf_idx][0][warp_j*32+16], 136);
        
        nvcuda::wmma::mma_sync(accum_frags[0], a_frags[0], b_frags[0], accum_frags[0]);
        nvcuda::wmma::mma_sync(accum_frags[1], a_frags[0], b_frags[1], accum_frags[1]);
        nvcuda::wmma::mma_sync(accum_frags[2], a_frags[1], b_frags[0], accum_frags[2]);
        nvcuda::wmma::mma_sync(accum_frags[3], a_frags[1], b_frags[1], accum_frags[3]);
        
        // Wait for next tile and swap buffers
        if (t < num_tiles - 1) {
            asm volatile("cp.async.wait_group %0;" :: "n"(0));
            __syncthreads();
            buf_idx = next_buf;
        }
    }
    
    // Store results
    for (int i = 0; i < 4; ++i) {
        int row_offset = (i < 2) ? 0 : 16;
        int col_offset = (i % 2) ? 16 : 0;
        float* c_ptr = &C[(block_row + warp_i*32 + row_offset) * N + (block_col + warp_j*32 + col_offset)];
        nvcuda::wmma::store_matrix_sync(c_ptr, accum_frags[i], N, nvcuda::wmma::mem_row_major);
    }
}
// PART-END

// PART-START
torch::Tensor symmetric_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    auto A_half = A.to(torch::kHalf);
    auto B_half = B.to(torch::kHalf);
    
    dim3 gridDim((N + 127) / 128, (N + 127) / 128);
    dim3 blockDim(32, BLOCK_WARPS);
    
    symmetric_matmul_kernel<<<gridDim, blockDim>>>(
        reinterpret_cast<const half*>(A_half.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(B_half.data_ptr<torch::Half>()),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}
// PART-END