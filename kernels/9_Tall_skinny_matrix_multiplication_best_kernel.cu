// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>

#define BLOCK_DIM 64
#define TILE_K 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARPS_PER_BLOCK 16
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define SKEW 8
#define ROW_BLOCKS_PER_BLOCK 8
#define COL_TILES_PER_BLOCK 16
#define VECTOR_SIZE 4

__device__ __forceinline__ int swizzle_block_idx(int block_idx, int grid_dim) {
    const int block_size = 8;
    return (block_idx ^ (block_idx / block_size)) % grid_dim;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void tall_skinny_matmul_kernel(const float* __restrict__ A, 
                                          const float* __restrict__ B, 
                                          float* __restrict__ out, 
                                          int M, int N) {
    extern __shared__ __align__(16) __half smem[];
    __half* sA = smem;
    __half* sB = sA + ROW_BLOCKS_PER_BLOCK * BLOCK_DIM * (TILE_K + SKEW);
    
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int warpRow = warpId / 4;
    int warpCol = warpId % 4;
    
    int swizzled_block_x = swizzle_block_idx(blockIdx.x, gridDim.x);
    int swizzled_block_y = swizzle_block_idx(blockIdx.y, gridDim.y);
    int row_block_start = (ROW_BLOCKS_PER_BLOCK * swizzled_block_x) * BLOCK_DIM;
    int col_block_start = swizzled_block_y * (COL_TILES_PER_BLOCK * BLOCK_DIM);

    // Vectorized A loading with half2 stores
    const int numVectorsA = (ROW_BLOCKS_PER_BLOCK * BLOCK_DIM * TILE_K) / VECTOR_SIZE;
    for (int idx = tid; idx < numVectorsA; idx += THREADS_PER_BLOCK) {
        int block_idx = idx / (BLOCK_DIM * TILE_K / VECTOR_SIZE);
        int local_idx = idx % (BLOCK_DIM * TILE_K / VECTOR_SIZE);
        int row = local_idx / (TILE_K / VECTOR_SIZE);
        int col = (local_idx % (TILE_K / VECTOR_SIZE)) * VECTOR_SIZE;
        int global_row = row_block_start + block_idx * BLOCK_DIM + row;
        
        float4 vecA = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (global_row < M) {
            if (col + VECTOR_SIZE <= N) {
                vecA = *reinterpret_cast<const float4*>(&A[global_row * N + col]);
            } else {
                if (col < N) vecA.x = A[global_row * N + col];
                if (col+1 < N) vecA.y = A[global_row * N + col+1];
                if (col+2 < N) vecA.z = A[global_row * N + col+2];
                if (col+3 < N) vecA.w = A[global_row * N + col+3];
            }
        }
        
        int store_idx = (block_idx * BLOCK_DIM + row) * (TILE_K + SKEW) + col;
        half2* ptr = reinterpret_cast<half2*>(&sA[store_idx]);
        ptr[0] = __floats2half2_rn(vecA.x, vecA.y);
        ptr[1] = __floats2half2_rn(vecA.z, vecA.w);
    }
    __syncthreads();

    for (int tile_j = 0; tile_j < COL_TILES_PER_BLOCK; tile_j++) {
        int col_tile_start = col_block_start + tile_j * BLOCK_DIM;
        
        // Vectorized B loading with half2 stores
        const int numVectorsB = (TILE_K * BLOCK_DIM) / VECTOR_SIZE;
        for (int idx = tid; idx < numVectorsB; idx += THREADS_PER_BLOCK) {
            int row = idx / (BLOCK_DIM / VECTOR_SIZE);
            int col = (idx % (BLOCK_DIM / VECTOR_SIZE)) * VECTOR_SIZE;
            int global_col = col_tile_start + col;
            
            float4 vecB = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row < N) {
                if (global_col + VECTOR_SIZE <= M) {
                    vecB = *reinterpret_cast<const float4*>(&B[row * M + global_col]);
                } else {
                    if (global_col < M) vecB.x = B[row * M + global_col];
                    if (global_col+1 < M) vecB.y = B[row * M + global_col+1];
                    if (global_col+2 < M) vecB.z = B[row * M + global_col+2];
                    if (global_col+3 < M) vecB.w = B[row * M + global_col+3];
                }
            }
            
            int store_idx = row * (BLOCK_DIM + SKEW) + col;
            half2* ptr = reinterpret_cast<half2*>(&sB[store_idx]);
            ptr[0] = __floats2half2_rn(vecB.x, vecB.y);
            ptr[1] = __floats2half2_rn(vecB.z, vecB.w);
        }
        __syncthreads();

        // Preload B fragments for both k-steps
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag0;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> b_frag16;
        nvcuda::wmma::load_matrix_sync(b_frag0, sB + 0 * (BLOCK_DIM + SKEW) + warpCol * WMMA_N, BLOCK_DIM + SKEW);
        nvcuda::wmma::load_matrix_sync(b_frag16, sB + 16 * (BLOCK_DIM + SKEW) + warpCol * WMMA_N, BLOCK_DIM + SKEW);

        // Process row blocks with unrolled k-steps
        #pragma unroll
        for (int r = 0; r < ROW_BLOCKS_PER_BLOCK; r++) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag0;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, nvcuda::wmma::row_major> a_frag16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
            
            // Unrolled k-step processing
            nvcuda::wmma::load_matrix_sync(a_frag0, 
                sA + r * BLOCK_DIM * (TILE_K + SKEW) + warpRow * WMMA_M * (TILE_K + SKEW) + 0, 
                TILE_K + SKEW);
            nvcuda::wmma::mma_sync(acc_frag, a_frag0, b_frag0, acc_frag);
            
            nvcuda::wmma::load_matrix_sync(a_frag16, 
                sA + r * BLOCK_DIM * (TILE_K + SKEW) + warpRow * WMMA_M * (TILE_K + SKEW) + 16, 
                TILE_K + SKEW);
            nvcuda::wmma::mma_sync(acc_frag, a_frag16, b_frag16, acc_frag);
            
            int out_row = row_block_start + r * BLOCK_DIM + warpRow * WMMA_M;
            int out_col = col_tile_start + warpCol * WMMA_N;
            
            if (out_row < M && out_col < M) {
                nvcuda::wmma::store_matrix_sync(out + out_row * M + out_col, 
                    acc_frag, M, nvcuda::wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = A.size(1);
    auto out = torch::zeros({M, M}, A.options());
    
    dim3 block(THREADS_PER_BLOCK);
    int grid_x = (M + (ROW_BLOCKS_PER_BLOCK * BLOCK_DIM) - 1) / (ROW_BLOCKS_PER_BLOCK * BLOCK_DIM);
    int grid_y = (M + (COL_TILES_PER_BLOCK * BLOCK_DIM) - 1) / (COL_TILES_PER_BLOCK * BLOCK_DIM);
    dim3 grid(grid_x, grid_y);
    
    size_t shared_mem_size = 
        (ROW_BLOCKS_PER_BLOCK * BLOCK_DIM * (TILE_K + SKEW) + 
         TILE_K * (BLOCK_DIM + SKEW)) * sizeof(__half);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Configure persistent L2 cache for both A and B
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    
    // Calculate memory regions for both matrices
    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    size_t A_size = static_cast<size_t>(M * N) * sizeof(float);
    size_t B_size = static_cast<size_t>(N * M) * sizeof(float);
    
    // Determine combined memory region
    float* min_ptr = (A_ptr < B_ptr) ? A_ptr : B_ptr;
    float* max_ptr = (A_ptr + A_size/sizeof(float) > B_ptr + B_size/sizeof(float)) 
                     ? A_ptr + A_size/sizeof(float) : B_ptr + B_size/sizeof(float);
    size_t total_size = (max_ptr - min_ptr) * sizeof(float);
    
    attr.accessPolicyWindow.base_ptr = min_ptr;
    attr.accessPolicyWindow.num_bytes = total_size;
    attr.accessPolicyWindow.hitRatio = 1.0;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    
    tall_skinny_matmul_kernel<<<grid, block, shared_mem_size, stream>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        out.data_ptr<float>(), 
        M, 
        N
    );
    
    // Reset stream attributes
    attr.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    
    return out;
}
// PART-END