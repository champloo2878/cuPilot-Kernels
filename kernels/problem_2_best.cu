// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

const int TILE_SIZE = 64;
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Compiler hints for fixed-size optimization
    __builtin_assume(M == 2048);
    __builtin_assume(K == 8192);
    __builtin_assume(N == 4096);
    
    // Tensor core declarations
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Thread identifiers
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    
    // Fragment position within block
    int frag_m = warp_id % 4;
    int frag_n = warp_id / 4;
    
    // Precompute invariant offsets
    const int a_offset = lane_id % 8;
    const int b_offset_val = 32768;  // 16 * (4096/2) = 32768
    
    // Shared memory tiles
    __shared__ half As[2][64][24];
    __shared__ half Bs[2][16][72];
    
    // Persistent thread configuration
    const int total_tiles = 2048;  // (2048/64)*(4096/64)=32*64=2048
    const int tiles_per_block = (total_tiles + gridDim.x - 1) / gridDim.x;
    
    for (int tile_idx = 0; tile_idx < tiles_per_block; tile_idx++) {
        int index = blockIdx.x + tile_idx * gridDim.x;
        if (index < total_tiles) {
            // Compute tile coordinates
            int by = index / 64;  // N/TILE_SIZE=4096/64=64
            int bx = index % 64;
            int a_start_row = by * TILE_SIZE;
            int b_start_col = bx * TILE_SIZE;
            
            // Reset accumulator
            wmma::fill_fragment(c_frag, 0.0f);
            
            // Compute base pointers for current tile
            const float2* base_A = reinterpret_cast<const float2*>(A) + 
                                  (a_start_row + warp_id*4 + lane_id/8) * 4096;  // K/2=8192/2=4096
            const float2* base_B = reinterpret_cast<const float2*>(B) + 
                                  warp_id * 2048 + (b_start_col/2 + lane_id);  // N/2=4096/2=2048
            
            // Pre-load tile0
            float2 a_val = __ldg(base_A + a_offset);
            float2 b_val = __ldg(base_B);
            
            half2* a_ptr = reinterpret_cast<half2*>(&As[0][warp_id*4 + lane_id/8][a_offset*2]);
            a_ptr->x = __float2half(a_val.x);
            a_ptr->y = __float2half(a_val.y);
            
            half2* b_ptr = reinterpret_cast<half2*>(&Bs[0][warp_id][lane_id*2]);
            b_ptr->x = __float2half(b_val.x);
            b_ptr->y = __float2half(b_val.y);
            
            __syncthreads();
            
            // Main computation loop with increased unrolling
            #pragma unroll 16
            for (int t = 0; t < 512; t++) {
                int s = t % 2;
                int next_s = 1 - s;
                
                // Load fragments and perform MMA
                wmma::load_matrix_sync(a_frag, &As[s][frag_m*16][0], 24);
                wmma::load_matrix_sync(b_frag, &Bs[s][0][frag_n*16], 72);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                
                if (t < 511) {
                    // Pre-load next tile
                    float2 a_val_next = __ldg(base_A + (t+1)*8 + a_offset);
                    float2 b_val_next = __ldg(base_B + (t+1)*b_offset_val);
                    
                    half2* a_ptr_next = reinterpret_cast<half2*>(&As[next_s][warp_id*4 + lane_id/8][a_offset*2]);
                    a_ptr_next->x = __float2half(a_val_next.x);
                    a_ptr_next->y = __float2half(a_val_next.y);
                    
                    half2* b_ptr_next = reinterpret_cast<half2*>(&Bs[next_s][warp_id][lane_id*2]);
                    b_ptr_next->x = __float2half(b_val_next.x);
                    b_ptr_next->y = __float2half(b_val_next.y);
                }
                __syncthreads();
            }
            
            // Store result with fixed leading dimension
            int c_row = a_start_row + frag_m*16;
            int c_col = b_start_col + frag_n*16;
            wmma::store_matrix_sync(C + c_row*4096 + c_col, c_frag, 4096, wmma::mem_row_major);
        }
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions do not match");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Persistent thread configuration
    const int blocks_per_sm = 4;
    const int sm_count = 108;
    const int total_tiles = (M / 64) * (N / 64);  // 2048/64=32, 4096/64=64 -> 2048 tiles
    const int grid_size = (total_tiles + blocks_per_sm * sm_count - 1) / (blocks_per_sm * sm_count) * blocks_per_sm * sm_count;
    
    dim3 grid(grid_size);
    dim3 block(32, 16);  // 512 threads per block
    
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    return C;
}
// PART-END