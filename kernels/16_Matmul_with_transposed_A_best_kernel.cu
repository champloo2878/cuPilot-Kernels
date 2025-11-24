// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void matmul_transposed_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Double-buffered shared memory with optimized padding and alignment
    __shared__ __align__(128) half shmem_A[2][64 * 24];
    __shared__ __align__(128) half shmem_B[2][16 * 72];
    
    // Compute grid dimensions
    int gridM = (M + 63) / 64;
    int gridN = (N + 63) / 64;
    
    // Block swizzling parameters
    const int tile_size = 8;
    int num_blocks_per_tile = tile_size * tile_size;
    int num_tiles_M = (gridM + tile_size - 1) / tile_size;
    int num_tiles_N = (gridN + tile_size - 1) / tile_size;
    
    // Compute swizzled block indices
    int linear_index = blockIdx.y * gridM + blockIdx.x;
    int tile_index = linear_index / num_blocks_per_tile;
    int tile_offset = linear_index % num_blocks_per_tile;
    
    int tile_i = tile_index % num_tiles_M;
    int tile_j = tile_index / num_tiles_M;
    
    int i_in_tile = tile_offset / tile_size;
    int j_in_tile = tile_offset % tile_size;
    
    int blockM = tile_i * tile_size + i_in_tile;
    int blockN = tile_j * tile_size + j_in_tile;
    
    // Thread indexing
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    // Warp-centric addressing
    int segment = tid / 32;  // 0-15 segments
    int lane = tid % 32;     // 0-31 within segment
    
    // Warp assignment
    int warpId = tid / 32;
    int local_m = warpId / 4;
    int local_n = warpId % 4;
    
    // Accumulator fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Double-buffering stage
    int stage = 0;
    {
        // Preload first tile (stage=0) with warp-centric coalescing
        int k = segment;
        int m_offset = lane * 2;
        int n_offset = tx * 2;
        
        // Coalesced load for A tile
        float2 a_val = __ldg(reinterpret_cast<const float2*>(&A[k * M + blockM * 64 + m_offset]));
        shmem_A[0][m_offset * 24 + segment] = __float2half(a_val.x);
        shmem_A[0][(m_offset+1) * 24 + segment] = __float2half(a_val.y);
        
        // Coalesced load for B tile
        float2 b_val = __ldg(reinterpret_cast<const float2*>(&B[k * N + blockN * 64 + n_offset]));
        shmem_B[0][segment * 72 + n_offset] = __float2half(b_val.x);
        shmem_B[0][segment * 72 + n_offset+1] = __float2half(b_val.y);
    }
    __syncthreads();

    // K-loop with double buffering and optimized unroll
    #pragma unroll 16
    for (int k_tile = 0; k_tile < K; k_tile += 16) {
        int next_stage = stage ^ 1;
        
        if (k_tile + 16 < K) {
            // Preload next tile with warp-centric coalescing
            int k_next = k_tile + 16 + segment;
            int m_offset = lane * 2;
            int n_offset = tx * 2;
            
            // Coalesced load for next A tile
            float2 a_val = __ldg(reinterpret_cast<const float2*>(&A[k_next * M + blockM * 64 + m_offset]));
            shmem_A[next_stage][m_offset * 24 + segment] = __float2half(a_val.x);
            shmem_A[next_stage][(m_offset+1) * 24 + segment] = __float2half(a_val.y);
            
            // Coalesced load for next B tile
            float2 b_val = __ldg(reinterpret_cast<const float2*>(&B[k_next * N + blockN * 64 + n_offset]));
            shmem_B[next_stage][segment * 72 + n_offset] = __float2half(b_val.x);
            shmem_B[next_stage][segment * 72 + n_offset+1] = __float2half(b_val.y);
        }
        
        // Tensor Core operations using current buffer
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        
        wmma::load_matrix_sync(a_frag, shmem_A[stage] + local_m * 16 * 24, 24);
        wmma::load_matrix_sync(b_frag, shmem_B[stage] + local_n * 16, 72);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncthreads();
        stage = next_stage;
    }
    
    // Store output fragment
    int m_start = blockM * 64 + local_m * 16;
    int n_start = blockN * 64 + local_n * 16;
    float* c_ptr = C + m_start * N + n_start;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor matmul_transposed_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "A and B must have the same number of rows");

    auto C = torch::zeros({M, N}, A.options());
    
    if (M == 0 || N == 0 || K == 0) {
        return C;
    }

    // Optimized block size: 32x16 threads (512 threads/block)
    dim3 block(32, 16);
    // Grid: ceil(M/64), ceil(N/64)
    dim3 grid((M + 63) / 64, (N + 63) / 64);

    matmul_transposed_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}
// PART-END