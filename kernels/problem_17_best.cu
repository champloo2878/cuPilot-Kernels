// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>

#define TILE_SIZE 64
#define WARPS_PER_BLOCK 16
#define THREADS_PER_WARP 32
#define BUFFER_COUNT 2
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
using namespace nvcuda;
namespace cg = cooperative_groups;

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Block index checks
    if (blockIdx.y * TILE_SIZE >= M || blockIdx.x * TILE_SIZE >= N) return;

    // Warp and thread identification
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tile_in_block_x = warp_id % 4;
    const int tile_in_block_y = warp_id / 4;
    const int tile_start_row = blockIdx.y * TILE_SIZE + tile_in_block_y * 16;
    const int tile_start_col = blockIdx.x * TILE_SIZE + tile_in_block_x * 16;

    // Shared memory declaration with double buffering
    __shared__ __nv_bfloat16 As[BUFFER_COUNT][TILE_SIZE][TILE_SIZE + 8];
    __shared__ __nv_bfloat16 Bs[BUFFER_COUNT][TILE_SIZE][TILE_SIZE + 8];
    
    // Double-buffered WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Thread-specific loading indices
    const int tid = warp_id * THREADS_PER_WARP + lane_id;
    const int row_in_tile = tid / (TILE_SIZE/8);
    const int col_in_tile = (tid % (TILE_SIZE/8)) * 8;

    int stage = 0;
    int next_stage = 0;

    // Preload first tile with vectorized stores
    int global_row_A = blockIdx.y * TILE_SIZE + row_in_tile;
    int global_col_A = col_in_tile;
    if (global_row_A < M && global_col_A <= K-8) {
        float4 vec1 = *reinterpret_cast<const float4*>(&A[global_row_A * K + global_col_A]);
        float4 vec2 = *reinterpret_cast<const float4*>(&A[global_row_A * K + global_col_A + 4]);
        
        // Convert to bfloat16 vectors
        __nv_bfloat162 v0 = __floats2bfloat162_rn(vec1.x, vec1.y);
        __nv_bfloat162 v1 = __floats2bfloat162_rn(vec1.z, vec1.w);
        __nv_bfloat162 v2 = __floats2bfloat162_rn(vec2.x, vec2.y);
        __nv_bfloat162 v3 = __floats2bfloat162_rn(vec2.z, vec2.w);
        
        // Vectorized store to shared memory
        float4* shared_ptr = reinterpret_cast<float4*>(&As[stage][row_in_tile][col_in_tile]);
        shared_ptr[0] = make_float4(
            *reinterpret_cast<float*>(&v0),
            *reinterpret_cast<float*>(&v1),
            *reinterpret_cast<float*>(&v2),
            *reinterpret_cast<float*>(&v3)
        );
    } else {
        for (int j = 0; j < 8; j++) {
            if (global_row_A < M && (global_col_A + j) < K) {
                As[stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(A[global_row_A * K + global_col_A + j]);
            } else {
                As[stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(0.0f);
            }
        }
    }

    int global_row_B = blockIdx.x * TILE_SIZE + row_in_tile;
    int global_col_B = col_in_tile;
    if (global_row_B < N && global_col_B <= K-8) {
        float4 vec1 = *reinterpret_cast<const float4*>(&B[global_row_B * K + global_col_B]);
        float4 vec2 = *reinterpret_cast<const float4*>(&B[global_row_B * K + global_col_B + 4]);
        
        // Convert to bfloat16 vectors
        __nv_bfloat162 v0 = __floats2bfloat162_rn(vec1.x, vec1.y);
        __nv_bfloat162 v1 = __floats2bfloat162_rn(vec1.z, vec1.w);
        __nv_bfloat162 v2 = __floats2bfloat162_rn(vec2.x, vec2.y);
        __nv_bfloat162 v3 = __floats2bfloat162_rn(vec2.z, vec2.w);
        
        // Vectorized store to shared memory
        float4* shared_ptr = reinterpret_cast<float4*>(&Bs[stage][row_in_tile][col_in_tile]);
        shared_ptr[0] = make_float4(
            *reinterpret_cast<float*>(&v0),
            *reinterpret_cast<float*>(&v1),
            *reinterpret_cast<float*>(&v2),
            *reinterpret_cast<float*>(&v3)
        );
    } else {
        for (int j = 0; j < 8; j++) {
            if (global_row_B < N && (global_col_B + j) < K) {
                Bs[stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(B[global_row_B * K + global_col_B + j]);
            } else {
                Bs[stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(0.0f);
            }
        }
    }

    __syncthreads();

    // Main computation loop
    for (int t = TILE_SIZE; t < K; t += TILE_SIZE) {
        next_stage = (stage + 1) % BUFFER_COUNT;
        
        // Load next tile with vectorized stores
        global_row_A = blockIdx.y * TILE_SIZE + row_in_tile;
        global_col_A = t + col_in_tile;
        if (global_row_A < M && global_col_A <= K-8) {
            float4 vec1 = *reinterpret_cast<const float4*>(&A[global_row_A * K + global_col_A]);
            float4 vec2 = *reinterpret_cast<const float4*>(&A[global_row_A * K + global_col_A + 4]);
            
            // Convert to bfloat16 vectors
            __nv_bfloat162 v0 = __floats2bfloat162_rn(vec1.x, vec1.y);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(vec1.z, vec1.w);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(vec2.x, vec2.y);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(vec2.z, vec2.w);
            
            // Vectorized store to shared memory
            float4* shared_ptr = reinterpret_cast<float4*>(&As[next_stage][row_in_tile][col_in_tile]);
            shared_ptr[0] = make_float4(
                *reinterpret_cast<float*>(&v0),
                *reinterpret_cast<float*>(&v1),
                *reinterpret_cast<float*>(&v2),
                *reinterpret_cast<float*>(&v3)
            );
        } else {
            for (int j = 0; j < 8; j++) {
                if (global_row_A < M && (global_col_A + j) < K) {
                    As[next_stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(A[global_row_A * K + global_col_A + j]);
                } else {
                    As[next_stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(0.0f);
                }
            }
        }

        global_row_B = blockIdx.x * TILE_SIZE + row_in_tile;
        global_col_B = t + col_in_tile;
        if (global_row_B < N && global_col_B <= K-8) {
            float4 vec1 = *reinterpret_cast<const float4*>(&B[global_row_B * K + global_col_B]);
            float4 vec2 = *reinterpret_cast<const float4*>(&B[global_row_B * K + global_col_B + 4]);
            
            // Convert to bfloat16 vectors
            __nv_bfloat162 v0 = __floats2bfloat162_rn(vec1.x, vec1.y);
            __nv_bfloat162 v1 = __floats2bfloat162_rn(vec1.z, vec1.w);
            __nv_bfloat162 v2 = __floats2bfloat162_rn(vec2.x, vec2.y);
            __nv_bfloat162 v3 = __floats2bfloat162_rn(vec2.z, vec2.w);
            
            // Vectorized store to shared memory
            float4* shared_ptr = reinterpret_cast<float4*>(&Bs[next_stage][row_in_tile][col_in_tile]);
            shared_ptr[0] = make_float4(
                *reinterpret_cast<float*>(&v0),
                *reinterpret_cast<float*>(&v1),
                *reinterpret_cast<float*>(&v2),
                *reinterpret_cast<float*>(&v3)
            );
        } else {
            for (int j = 0; j < 8; j++) {
                if (global_row_B < N && (global_col_B + j) < K) {
                    Bs[next_stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(B[global_row_B * K + global_col_B + j]);
                } else {
                    Bs[next_stage][row_in_tile][col_in_tile+j] = __float2bfloat16_rn(0.0f);
                }
            }
        }

        __syncthreads();

        // Preload first fragment for current tile
        wmma::load_matrix_sync(a_frag[0], &As[stage][tile_in_block_y * 16][0], TILE_SIZE + 8);
        wmma::load_matrix_sync(b_frag[0], &Bs[stage][tile_in_block_x * 16][0], TILE_SIZE + 8);
        int next = 1;  // Next fragment index

        // Compute with pipelined fragments
        #pragma unroll
        for (int k_step = 0; k_step < TILE_SIZE; k_step += 16) {
            // Preload next fragment if not last step
            if (k_step + 16 < TILE_SIZE) {
                wmma::load_matrix_sync(a_frag[next], &As[stage][tile_in_block_y * 16][k_step+16], TILE_SIZE + 8);
                wmma::load_matrix_sync(b_frag[next], &Bs[stage][tile_in_block_x * 16][k_step+16], TILE_SIZE + 8);
            }
            
            // Compute with current fragment
            int current = 1 - next;
            wmma::mma_sync(acc_frag, a_frag[current], b_frag[current], acc_frag);
            
            // Toggle fragment index
            next = 1 - next;
        }

        // Switch buffers
        stage = next_stage;
    }

    // Compute last tile with pipelined fragments
    wmma::load_matrix_sync(a_frag[0], &As[stage][tile_in_block_y * 16][0], TILE_SIZE + 8);
    wmma::load_matrix_sync(b_frag[0], &Bs[stage][tile_in_block_x * 16][0], TILE_SIZE + 8);
    int next = 1;
    
    #pragma unroll
    for (int k_step = 0; k_step < TILE_SIZE; k_step += 16) {
        if (k_step + 16 < TILE_SIZE) {
            wmma::load_matrix_sync(a_frag[next], &As[stage][tile_in_block_y * 16][k_step+16], TILE_SIZE + 8);
            wmma::load_matrix_sync(b_frag[next], &Bs[stage][tile_in_block_x * 16][k_step+16], TILE_SIZE + 8);
        }
        
        int current = 1 - next;
        wmma::mma_sync(acc_frag, a_frag[current], b_frag[current], acc_frag);
        next = 1 - next;
    }

    // Store results
    if (tile_start_row + 16 <= M && tile_start_col + 16 <= N) {
        float* c_ptr = &C[tile_start_row * N + tile_start_col];
        wmma::store_matrix_sync(c_ptr, acc_frag, N, wmma::mem_row_major);
    } else {
        int row_in_group = lane_id / 4;
        int group = lane_id % 4;
        int row0 = tile_start_row + row_in_group * 2;
        int col0 = tile_start_col + group * 4;

        if (row0 < M) {
            if (col0 + 3 < N) {
                float4 vec0 = make_float4(acc_frag.x[0], acc_frag.x[1], 
                                         acc_frag.x[2], acc_frag.x[3]);
                *reinterpret_cast<float4*>(&C[row0 * N + col0]) = vec0;
            } else {
                for (int j = 0; j < 4; j++) {
                    if (col0 + j < N) {
                        C[row0 * N + col0 + j] = acc_frag.x[j];
                    }
                }
            }
        }

        int row1 = row0 + 1;
        if (row1 < M) {
            if (col0 + 3 < N) {
                float4 vec1 = make_float4(acc_frag.x[4], acc_frag.x[5], 
                                         acc_frag.x[6], acc_frag.x[7]);
                *reinterpret_cast<float4*>(&C[row1 * N + col0]) = vec1;
            } else {
                for (int j = 0; j < 4; j++) {
                    if (col0 + j < N) {
                        C[row1 * N + col0 + j] = acc_frag.x[4 + j];
                    }
                }
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
    int N = B.size(0);

    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threads(32, 16);  // 32x16=512 threads
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
// PART-END