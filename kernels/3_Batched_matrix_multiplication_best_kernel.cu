#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>

#define TILE_M 32
#define TILE_N 8
#define TILE_K 32
#define BLOCK_SIZE 512
#define PADDING_A 8
#define PADDING_B 8

using namespace nvcuda;
using namespace cooperative_groups;

__global__ void batched_matmul_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, 
                                      int batch_size, int m, int k, int n) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 4;
    const int warp_n = warp_id % 4;

    const int batch = blockIdx.z;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;

    __shared__ half As[2][128][TILE_K + PADDING_A];
    __shared__ half Bs[2][TILE_K][128 + PADDING_B];

    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, 16, float> frag_C[4];
    for (int i = 0; i < 4; i++) {
        wmma::fill_fragment(frag_C[i], 0.0f);
    }

    const int n_tiles = k / TILE_K;
    int stage = 0;

    {
        int row = tid / 4;
        int col_start = (tid % 4) * 8;
        const half* A_ptr = &A[batch * m * k + (tile_m * 128 + row) * k + col_start];
        half* A_shared = &As[stage][row][col_start];
        __pipeline_memcpy_async(A_shared, A_ptr, 16);

        int b_row = tid / 16;
        int b_col_start = (tid % 16) * 8;
        const half* B_ptr = &B[batch * k * n + b_row * n + (tile_n * 128 + b_col_start)];
        half* B_shared = &Bs[stage][b_row][b_col_start];
        __pipeline_memcpy_async(B_shared, B_ptr, 16);

        __pipeline_commit();
        __pipeline_wait_prior(0);
        block.sync();
    }

    for (int t = 0; t < n_tiles; t++) {
        const int next_stage = stage ^ 1;

        if (t < n_tiles - 1) {
            int row = tid / 4;
            int col_start = (tid % 4) * 8;
            const half* A_ptr = &A[batch * m * k + (tile_m * 128 + row) * k + (t+1)*TILE_K + col_start];
            half* A_shared_next = &As[next_stage][row][col_start];
            __pipeline_memcpy_async(A_shared_next, A_ptr, 16);

            int b_row = tid / 16;
            int b_col_start = (tid % 16) * 8;
            const half* B_ptr = &B[batch * k * n + ((t+1)*TILE_K + b_row) * n + (tile_n * 128 + b_col_start)];
            half* B_shared_next = &Bs[next_stage][b_row][b_col_start];
            __pipeline_memcpy_async(B_shared_next, B_ptr, 16);

            __pipeline_commit();
        }

        for (int inner_tile = 0; inner_tile < 2; inner_tile++) {
            wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, 16, half, wmma::row_major> frag_A;
            wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, 16, half, wmma::row_major> frag_B[4];

            wmma::load_matrix_sync(frag_A, &As[stage][warp_m * TILE_M][inner_tile * 16], TILE_K + PADDING_A);
            
            for (int i = 0; i < 4; i++) {
                wmma::load_matrix_sync(frag_B[i], &Bs[stage][inner_tile * 16][warp_n * 32 + i * TILE_N], 128 + PADDING_B);
            }
            
            for (int i = 0; i < 4; i++) {
                wmma::mma_sync(frag_C[i], frag_A, frag_B[i], frag_C[i]);
            }
        }

        if (t < n_tiles - 1) {
            __pipeline_wait_prior(0);
            block.sync();
            stage = next_stage;
        }
    }

    for (int i = 0; i < 4; i++) {
        int row = tile_m * 128 + warp_m * TILE_M;
        int col = tile_n * 128 + warp_n * 32 + i * TILE_N;
        wmma::store_matrix_sync(&C[batch * m * n + row * n + col], frag_C[i], n, wmma::mem_row_major);
    }
}
//PART-END

//PART-START part3
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 3, "A must be 3D tensor");
    TORCH_CHECK(B.dim() == 3, "B must be 3D tensor");
    
    torch::Tensor A_half = A;
    torch::Tensor B_half = B;
    if (A.scalar_type() != torch::kHalf) {
        A_half = A.to(torch::kHalf);
    }
    if (B.scalar_type() != torch::kHalf) {
        B_half = B.to(torch::kHalf);
    }

    const int batch_size = A_half.size(0);
    const int m = A_half.size(1);
    const int k = A_half.size(2);
    const int n = B_half.size(2);
    
    auto C = torch::zeros({batch_size, m, n}, torch::dtype(torch::kFloat32).device(A.device()));
    
    dim3 grid(
        n / 128,
        m / 128,
        batch_size
    );
    dim3 block(BLOCK_SIZE);
    
    batched_matmul_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A_half.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B_half.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        batch_size, m, k, n
    );
    
    return C;
}