// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_pipeline.h>

static __device__ __inline__ void load_two_tiles_async(
    const float* A, const float* B, 
    float* s_A, float* s_B, 
    int n, int M, int K, int L,
    int global_m_base, int global_l_base, int k_step
) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Load two consecutive tiles for A (128x16)
    if (tid < 128) {
        int global_row = global_m_base + tid;
        for (int tile = 0; tile < 2; tile++) {
            int global_col = k_step + tile*8;
            if (global_row < M && global_col < K) {
                uint32_t smem_addr_A = __cvta_generic_to_shared(s_A + tile*1024 + tid*8);
                const float* global_ptr_A = A + n*M*K + global_row*K + global_col;
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr_A), "l"(global_ptr_A)
                );
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr_A+16), "l"(global_ptr_A+4)
                );
            }
        }
    }
    
    // Load two consecutive tiles for B (16x128)
    if (tid < 128) {
        int row = tid / 16;
        int col_chunk = tid % 16;
        for (int tile = 0; tile < 2; tile++) {
            int global_row = k_step + tile*8 + row;
            int global_col = global_l_base + col_chunk * 8;
            if (global_row < K && global_col < L) {
                uint32_t smem_addr_B = __cvta_generic_to_shared(s_B + tile*1024 + row*128 + col_chunk*8);
                const float* global_ptr_B = B + global_row*L + global_col;
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr_B), "l"(global_ptr_B)
                );
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(smem_addr_B+16), "l"(global_ptr_B+4)
                );
            }
        }
    }
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void batched_matmul_kernel(const float* A, const float* B, float* out,
                                      int N, int M, int K, int L) {
    // Double-buffered shared memory for two-tile sets
    extern __shared__ float shared_mem[];
    float* s_A0 = shared_mem;
    float* s_B0 = s_A0 + 128*16;
    float* s_A1 = s_B0 + 16*128;
    float* s_B1 = s_A1 + 128*16;
    
    // Thread and warp assignment
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.y;
    const int n = blockIdx.z;
    const int global_m_base = blockIdx.x * 128;
    const int global_l_base = blockIdx.y * 128;
    
    // Determine warp's sub-tile (64x64)
    int row_offset, col_offset;
    if (warp_id < 2) {
        row_offset = 0;
    } else {
        row_offset = 64;
    }
    if (warp_id % 2 == 0) {
        col_offset = 0;
    } else {
        col_offset = 64;
    }
    const int row_start = global_m_base + row_offset;
    const int col_start = global_l_base + col_offset;
    
    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> accum_frag[4][4];
    
    // Initialize accumulators
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            nvcuda::wmma::fill_fragment(accum_frag[i][j], 0.0f);
        }
    }
    
    int stage = 0;
    // Preload first two tiles asynchronously
    if (0 < K) {
        load_two_tiles_async(A, B, s_A0, s_B0, n, M, K, L, global_m_base, global_l_base, 0);
        __pipeline_commit();
    }
    __pipeline_wait_prior(0);
    __syncthreads();
    
    // Main computation loop (step by 16 in K)
    for (int k_step = 0; k_step < K; k_step += 16) {
        int remaining = K - k_step;
        int num_inner_steps = (remaining >= 16) ? 2 : (remaining >= 8) ? 1 : 0;
        
        // Preload next two tiles asynchronously
        if (k_step + 16 < K) {
            float* s_A_next = (stage == 0) ? s_A1 : s_A0;
            float* s_B_next = (stage == 0) ? s_B1 : s_B0;
            load_two_tiles_async(A, B, s_A_next, s_B_next, n, M, K, L, global_m_base, global_l_base, k_step+16);
            __pipeline_commit();
        }
        
        // Wait for current buffer to be ready
        if (k_step > 0) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }
        
        float* s_A_cur = (stage == 0) ? s_A0 : s_A1;
        float* s_B_cur = (stage == 0) ? s_B0 : s_B1;
        
        // Process 1-2 tiles per iteration
        for (int inner = 0; inner < num_inner_steps; inner++) {
            float* s_A_tile = s_A_cur + inner * 1024;
            float* s_B_tile = s_B_cur + inner * 1024;
            
            for (int i = 0; i < 4; i++) {
                // Load A fragment (128x8 tile)
                nvcuda::wmma::load_matrix_sync(a_frag, s_A_tile + (row_offset + i*16)*8, 8);
                
                for (int j = 0; j < 4; j++) {
                    // Load B fragment (8x128 tile)
                    nvcuda::wmma::load_matrix_sync(b_frag, s_B_tile + col_offset + j*16, 128);
                    
                    // Tensor Core operation
                    nvcuda::wmma::mma_sync(accum_frag[i][j], a_frag, b_frag, accum_frag[i][j]);
                }
            }
        }
        
        // Switch buffers
        stage = 1 - stage;
    }
    
    // Store results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int store_row = row_start + i*16;
            int store_col = col_start + j*16;
            if (store_row < M && store_col < L) {
                nvcuda::wmma::store_matrix_sync(
                    out + n*M*L + store_row*L + store_col,
                    accum_frag[i][j],
                    L,
                    nvcuda::wmma::mem_row_major
                );
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    auto out = torch::zeros({N, M, L}, A.options());
    
    // Block and grid dimensions
    constexpr int TILE_SIZE = 128;
    dim3 block(32, 4);  // 128 threads (4 warps)
    dim3 grid(
        (M + TILE_SIZE - 1) / TILE_SIZE,
        (L + TILE_SIZE - 1) / TILE_SIZE,
        N
    );
    
    // Shared memory for double-buffered two-tile sets
    size_t shared_mem_size = 2 * (128*16 + 16*128) * sizeof(float);  // 32768 bytes
    
    // Launch kernel
    batched_matmul_kernel<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        out.data_ptr<float>(),
        N, M, K, L
    );
    
    return out;
}
// PART-END