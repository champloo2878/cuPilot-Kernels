#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// PART-END

// PART-START
__device__ void convert_tile(float src[][64], __half dst[][64], int ty, int tx) {
    for (int i = 0; i < 4; i++) {
        int row = ty * 4 + i;
        int col = tx * 4;
        float4 val = reinterpret_cast<float4*>(&src[row][col])[0];
        __half2* dst_ptr = reinterpret_cast<__half2*>(&dst[row][col]);
        dst_ptr[0] = __floats2half2_rn(val.x, val.y);
        dst_ptr[1] = __floats2half2_rn(val.z, val.w);
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    const int TILE_SIZE = 64;
    const int BLOCK_DIM = 16;
    const int num_tiles = N / TILE_SIZE;

    __shared__ float As_fp32[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs_fp32[TILE_SIZE][TILE_SIZE];
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bx = blockIdx.x;

    int grid_size_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    int grid_size_y = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    unsigned int by_swizzled = by;
    unsigned int bx_swizzled = bx;
    
    if (grid_size_x == 64 && grid_size_y == 64) {
        unsigned int linear_index = by * grid_size_x + bx;
        unsigned int code = linear_index;
        bx_swizzled = ( (code & 0x0001)       ) |
                      ( (code & 0x0004) >> 1 ) |
                      ( (code & 0x0010) >> 2 ) |
                      ( (code & 0x0040) >> 3 ) |
                      ( (code & 0x0100) >> 4 ) |
                      ( (code & 0x0400) >> 5 );
        by_swizzled = ( (code & 0x0002) >> 1 ) |
                      ( (code & 0x0008) >> 2 ) |
                      ( (code & 0x0020) >> 3 ) |
                      ( (code & 0x0080) >> 4 ) |
                      ( (code & 0x0200) >> 5 ) |
                      ( (code & 0x0800) >> 6 );
    }

    int start_row = by_swizzled * TILE_SIZE;
    int start_col = bx_swizzled * TILE_SIZE;

    float accum[4][4] = {{0.0f}};

    // Preload first tile into FP32 buffers
    for (int i = 0; i < 4; i++) {
        int row = start_row + ty * 4 + i;
        int col = tx * 4;
        const float* src_A = &A[row * N + col];
        float* dst_A = &As_fp32[ty * 4 + i][tx * 4];
        asm volatile (
            "cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_A))), "l"(src_A)
        );
    }
    for (int i = 0; i < 4; i++) {
        int row = ty * 4 + i;
        int col = start_col + tx * 4;
        const float* src_B = &B[row * N + col];
        float* dst_B = &Bs_fp32[ty * 4 + i][tx * 4];
        asm volatile (
            "cp.async.cg.shared.global [%0], [%1], 16;"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_B))), "l"(src_B)
        );
    }
    asm volatile ("cp.async.commit_group;");
    asm volatile ("cp.async.wait_group 0;");
    __syncthreads();
    
    // Convert first tile to FP16
    convert_tile(As_fp32, As, ty, tx);
    convert_tile(Bs_fp32, Bs, ty, tx);
    __syncthreads();

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int k_tile = tile_idx * TILE_SIZE;
        
        if (tile_idx > 0) {
            asm volatile ("cp.async.wait_group 0;");
            __syncthreads();
            convert_tile(As_fp32, As, ty, tx);
            convert_tile(Bs_fp32, Bs, ty, tx);
            __syncthreads();
        }

        // FP16 accumulation for current tile
        __half tile_accum[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                tile_accum[i][j] = __float2half(0.0f);
            }
        }
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 2) {
            __half a_vals0[4], a_vals1[4];
            __half b_vals0[4], b_vals1[4];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int row_idx = ty * 4 + i;
                a_vals0[i] = As[row_idx][k];
                a_vals1[i] = As[row_idx][k+1];
            }
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int col_idx = tx * 4 + j;
                b_vals0[j] = Bs[k][col_idx];
                b_vals1[j] = Bs[k+1][col_idx];
            }
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    tile_accum[i][j] = __hfma(a_vals0[i], b_vals0[j], tile_accum[i][j]);
                    tile_accum[i][j] = __hfma(a_vals1[i], b_vals1[j], tile_accum[i][j]);
                }
            }
        }

        // Convert to FP32 and accumulate
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                accum[i][j] += __half2float(tile_accum[i][j]);
            }
        }

        // Preload next tile
        if (tile_idx < num_tiles - 1) {
            int next_k_tile = (tile_idx + 1) * TILE_SIZE;
            for (int i = 0; i < 4; i++) {
                int row = start_row + ty * 4 + i;
                int col = next_k_tile + tx * 4;
                const float* src_A = &A[row * N + col];
                float* dst_A = &As_fp32[ty * 4 + i][tx * 4];
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_A))), "l"(src_A)
                );
            }
            for (int i = 0; i < 4; i++) {
                int row = next_k_tile + ty * 4 + i;
                int col = start_col + tx * 4;
                const float* src_B = &B[row * N + col];
                float* dst_B = &Bs_fp32[ty * 4 + i][tx * 4];
                asm volatile (
                    "cp.async.cg.shared.global [%0], [%1], 16;"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_B))), "l"(src_B)
                );
            }
            asm volatile ("cp.async.commit_group;");
        }
    }

    // Write results
    for (int i = 0; i < 4; i++) {
        int row = start_row + ty * 4 + i;
        int col_base = start_col + tx * 4;
        float4 val = {accum[i][0], accum[i][1], accum[i][2], accum[i][3]};
        reinterpret_cast<float4*>(&C[row * N + col_base])[0] = val;
    }
}
// PART-END

// PART-START
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    if (N == 0) {
        return torch::zeros({0, 0}, A.options());
    }
    
    auto C = torch::zeros({N, N}, A.options());
    
    const int TILE_SIZE = 64;
    const int BLOCK_DIM = 16;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                   B.data_ptr<float>(), 
                                   C.data_ptr<float>(), 
                                   N);
    
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}
// PART-END