// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void gemm_kernel(const float* A, const float* B, float* out, int M, int N, int K) {
    using namespace nvcuda;
    constexpr int BLOCK_SIZE = 64;
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int TILE_SIZE = 16;
    
    extern __shared__ char shared_mem[];
    half* sA = reinterpret_cast<half*>(shared_mem);
    half* sB = reinterpret_cast<half*>(shared_mem + BLOCK_SIZE * BLOCK_SIZE * sizeof(half));
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int row_in_block = (tid / (BLOCK_SIZE/4)) * 4;
    int col_in_block = (tid % (BLOCK_SIZE/4)) * 4;
    
    // Load A tile (convert FP32 to FP16) - no conditionals needed
    for (int i = 0; i < 4; i++) {
        int row = blockIdx.y * BLOCK_SIZE + row_in_block + i;
        int col = col_in_block;
        float4 vec = *reinterpret_cast<const float4*>(&A[row * K + col]);
        half2* half_vec = reinterpret_cast<half2*>(&sA[(row_in_block + i) * BLOCK_SIZE + col_in_block]);
        half_vec[0] = __float22half2_rn(float2{vec.x, vec.y});
        half_vec[1] = __float22half2_rn(float2{vec.z, vec.w});
    }
    
    // Load B tile (convert FP32 to FP16) - no conditionals needed
    for (int i = 0; i < 4; i++) {
        int row = row_in_block + i;
        int col = blockIdx.x * BLOCK_SIZE + col_in_block;
        float4 vec = *reinterpret_cast<const float4*>(&B[row * N + col]);
        half2* half_vec = reinterpret_cast<half2*>(&sB[row * BLOCK_SIZE + col_in_block]);
        half_vec[0] = __float22half2_rn(float2{vec.x, vec.y});
        half_vec[1] = __float22half2_rn(float2{vec.z, vec.w});
    }
    
    __syncthreads();
    
    // WMMA computation
    int warp_id = threadIdx.y;
    int tile_row = (warp_id / 2) * TILE_SIZE;
    int tile_col = (warp_id % 2) * 32;
    
    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major> b_frag1, b_frag2;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> c_frag1, c_frag2;
    
    wmma::fill_fragment(c_frag1, 0.0f);
    wmma::fill_fragment(c_frag2, 0.0f);
    
    // Unrolled accumulation loop
    #pragma unroll
    for (int k_step = 0; k_step < BLOCK_SIZE; k_step += TILE_SIZE) {
        wmma::load_matrix_sync(a_frag, &sA[tile_row * BLOCK_SIZE + k_step], BLOCK_SIZE);
        wmma::load_matrix_sync(b_frag1, &sB[k_step * BLOCK_SIZE + tile_col], BLOCK_SIZE);
        wmma::load_matrix_sync(b_frag2, &sB[k_step * BLOCK_SIZE + tile_col + TILE_SIZE], BLOCK_SIZE);
        wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        wmma::mma_sync(c_frag2, a_frag, b_frag2, c_frag2);
    }
    
    // Direct store to global memory
    int global_row = blockIdx.y * BLOCK_SIZE + tile_row;
    int global_col = blockIdx.x * BLOCK_SIZE + tile_col;
    wmma::store_matrix_sync(&out[global_row * N + global_col], c_frag1, N, wmma::mem_row_major);
    wmma::store_matrix_sync(&out[global_row * N + global_col + TILE_SIZE], c_frag2, N, wmma::mem_row_major);
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto out = torch::zeros({M, N}, A.options());
    
    constexpr int BLOCK_SIZE = 64;
    dim3 block(32, 8);  // 256 threads per block
    // Exact grid dimensions since sizes are multiples
    dim3 grid(N / BLOCK_SIZE, M / BLOCK_SIZE);
    
    // Shared memory: 
    //   sA: 64x64 half = 8192 bytes
    //   sB: 64x64 half = 8192 bytes
    int shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(half);
    
    gemm_kernel<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        M, N, K
    );
    
    return out;
}
// PART-END