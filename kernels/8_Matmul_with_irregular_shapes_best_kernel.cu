// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 8
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    static_assert(kColStride <= 16, "kColStride must <= 16");
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
    static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
    if (kStep == 8) {
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    return swizzle_permuted_j<BLOCK_K, 8>(i, j);
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void matmul_kernel(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ out, 
                              int M, int K, int N) {
    // Double buffered shared memory with col-major layout for B
    __shared__ float sA[2][BLOCK_M][BLOCK_K];
    __shared__ float sB[2][(BLOCK_K+1) * BLOCK_N];  // Col-major with padding
    
    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> frag_b;  // Col-major
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
    
    // Thread indexing
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp tiling
    int warp_tile_i = warp_id / (BLOCK_N / WMMA_N);
    int warp_tile_j = warp_id % (BLOCK_N / WMMA_N);
    
    // Apply refined swizzle for bank conflict reduction
    warp_tile_j = (warp_tile_j ^ (warp_tile_i % (BLOCK_N / (WMMA_N / 16)))) % (BLOCK_N / WMMA_N);
    
    // Output tile position
    int tile_m = blockIdx.y * BLOCK_M;
    int tile_n = blockIdx.x * BLOCK_N;
    
    // Initialize accumulator
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);
    
    // Double buffering indices
    int buf_idx = 0;
    int next_buf = 1;
    
    // Pre-load first tile of A
    int a_row = tile_m + tid / BLOCK_K;
    int a_col = tid % BLOCK_K;
    
    if (a_row < M && a_col < K) {
        sA[buf_idx][tid / BLOCK_K][tid % BLOCK_K] = A[a_row * K + a_col];
    } else {
        sA[buf_idx][tid / BLOCK_K][tid % BLOCK_K] = 0.0f;
    }
    
    // Pre-load first tile of B with col-major layout
    int b_row_local = tid / BLOCK_N;   // 0 to BLOCK_K-1
    int b_col_local = tid % BLOCK_N;   // 0 to BLOCK_N-1
    int b_row_global = b_row_local;
    int b_col_global = tile_n + b_col_local;
    int sB_index = b_row_local + b_col_local * (BLOCK_K+1);
    
    if (b_row_global < K && b_col_global < N) {
        sB[buf_idx][sB_index] = B[b_row_global * N + b_col_global];
    } else {
        sB[buf_idx][sB_index] = 0.0f;
    }
    
    __syncthreads();
    
    // Main computation loop
    for (int k_step = BLOCK_K; k_step < K; k_step += BLOCK_K) {
        // Pre-load next A tile
        int a_next_col = k_step + a_col;
        if (a_row < M && a_next_col < K) {
            sA[next_buf][tid / BLOCK_K][tid % BLOCK_K] = A[a_row * K + a_next_col];
        } else {
            sA[next_buf][tid / BLOCK_K][tid % BLOCK_K] = 0.0f;
        }
        
        // Pre-load next B tile with col-major layout
        int b_next_row = k_step + b_row_local;
        if (b_next_row < K && b_col_global < N) {
            sB[next_buf][sB_index] = B[b_next_row * N + b_col_global];
        } else {
            sB[next_buf][sB_index] = 0.0f;
        }
        
        // Tensor Core computation
        nvcuda::wmma::load_matrix_sync(frag_a, &sA[buf_idx][warp_tile_i * WMMA_M][0], BLOCK_K);
        nvcuda::wmma::load_matrix_sync(frag_b, &sB[buf_idx][warp_tile_j * WMMA_N * (BLOCK_K+1)], BLOCK_K+1); // Padded stride
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        
        __syncthreads();
        buf_idx = 1 - buf_idx;
        next_buf = 1 - next_buf;
    }
    
    // Final tile computation
    nvcuda::wmma::load_matrix_sync(frag_a, &sA[buf_idx][warp_tile_i * WMMA_M][0], BLOCK_K);
    nvcuda::wmma::load_matrix_sync(frag_b, &sB[buf_idx][warp_tile_j * WMMA_N * (BLOCK_K+1)], BLOCK_K+1);
    nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    
    // Store results
    __shared__ float sC[BLOCK_M][BLOCK_N];  // No skew needed with optimized layout
    nvcuda::wmma::store_matrix_sync(&sC[warp_tile_i * WMMA_M][warp_tile_j * WMMA_N], frag_c, BLOCK_N, nvcuda::wmma::mem_row_major);
    
    __syncthreads();
    
    // Write back to global memory
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    const int ROWS_PER_THREAD = BLOCK_M / blockDim.y;
    const int COLS_PER_THREAD = BLOCK_N / blockDim.x;
    
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
        for (int j = 0; j < COLS_PER_THREAD; j++) {
            int row_in_sC = thread_row * ROWS_PER_THREAD + i;
            int col_in_sC = thread_col * COLS_PER_THREAD + j;
            int global_row = tile_m + row_in_sC;
            int global_col = tile_n + col_in_sC;
            if (global_row < M && global_col < N) {
                out[global_row * N + global_col] = sC[row_in_sC][col_in_sC];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto out = torch::zeros({M, N}, A.options());
    
    dim3 block(32, 16);  // 512 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), out.data_ptr<float>(), M, K, N);
    
    return out;
}
// PART-END