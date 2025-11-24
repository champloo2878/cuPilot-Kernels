// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

constexpr int WARPS_PER_BLOCK = 32;
constexpr int SHARED_MEM_PER_WARP_PADDED = WMMA_M * (WMMA_N + 1);  // 272 elements with padding
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void matmul_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K, int K_segment_start, int K_segment_length) {
    using namespace nvcuda;
    
    extern __shared__ float shm_frag[];
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SIZE;
    const int laneId = tid % WARP_SIZE;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int global_tile_row = block_row * WMMA_M;
    const int global_tile_col = block_col * WMMA_N;
    
    // Compute number of K-dimension tiles within segment
    const int total_tiles_k = (K_segment_length + WMMA_K - 1) / WMMA_K;
    const int tiles_per_warp = (total_tiles_k + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int start_tile = warpId * tiles_per_warp;
    const int end_tile = min(start_tile + tiles_per_warp, total_tiles_k);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Double buffering fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_prefetch[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_prefetch[2];
    
    // Prefetch first two tiles
    if (start_tile < end_tile) {
        int k_offset0 = K_segment_start + start_tile * WMMA_K;
        wmma::load_matrix_sync(a_prefetch[0], A + global_tile_row * K + k_offset0, K);
        wmma::load_matrix_sync(b_prefetch[0], B + k_offset0 * N + global_tile_col, N);
    }
    if (start_tile + 1 < end_tile) {
        int k_offset1 = K_segment_start + (start_tile + 1) * WMMA_K;
        wmma::load_matrix_sync(a_prefetch[1], A + global_tile_row * K + k_offset1, K);
        wmma::load_matrix_sync(b_prefetch[1], B + k_offset1 * N + global_tile_col, N);
    }
    
    // Main loop with increased unroll factor and double buffering
    const int unroll_factor = 8;
    int t = start_tile;
    int buffer_idx = 0;
    
    while (t <= end_tile - unroll_factor) {
        #pragma unroll
        for (int u = 0; u < unroll_factor; u++) {
            // Compute current tile
            wmma::mma_sync(c_frag, a_prefetch[buffer_idx], b_prefetch[buffer_idx], c_frag);
            
            // Prefetch next tile
            if (u < unroll_factor - 2 && t + u + 2 < end_tile) {
                int next_tile = t + u + 2;
                int next_k_offset = K_segment_start + next_tile * WMMA_K;
                int next_buffer = 1 - buffer_idx;
                
                wmma::load_matrix_sync(a_prefetch[next_buffer], 
                                      A + global_tile_row * K + next_k_offset, K);
                wmma::load_matrix_sync(b_prefetch[next_buffer], 
                                      B + next_k_offset * N + global_tile_col, N);
            }
            
            // Switch buffer for next iteration
            buffer_idx = 1 - buffer_idx;
        }
        t += unroll_factor;
    }
    
    // Process remaining tiles
    for (; t < end_tile; t++) {
        int k_offset = K_segment_start + t * WMMA_K;
        wmma::load_matrix_sync(a_frag, A + global_tile_row * K + k_offset, K);
        wmma::load_matrix_sync(b_frag, B + k_offset * N + global_tile_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store fragment to shared memory with padding to avoid bank conflicts
    wmma::store_matrix_sync(shm_frag + warpId * SHARED_MEM_PER_WARP_PADDED, 
                           c_frag, WMMA_N + 1, wmma::mem_row_major);
    __syncthreads();
    
    // Warp0 reduces partial results
    if (warpId == 0) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        wmma::fill_fragment(acc_frag, 0.0f);
        
        // Accumulate results from all warps
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> tmp_frag;
            wmma::load_matrix_sync(tmp_frag, 
                                  shm_frag + w * SHARED_MEM_PER_WARP_PADDED, 
                                  WMMA_N + 1, wmma::mem_row_major);
            for (int i = 0; i < acc_frag.num_elements; i++) {
                acc_frag.x[i] += tmp_frag.x[i];
            }
        }
        
        // Store final result to global memory
        if (global_tile_row < M && global_tile_col < N) {
            wmma::store_matrix_sync(C + global_tile_row * N + global_tile_col, 
                                   acc_frag, N, wmma::mem_row_major);
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Matrix dimensions mismatch");

    auto options = A.options().dtype(torch::kFloat32);
    if (M == 0 || N == 0 || K == 0) {
        return torch::zeros({M, N}, options);
    }

    auto A_fp16 = A.to(torch::kFloat16).contiguous();
    auto B_fp16 = B.to(torch::kFloat16).contiguous();

    half* A_data = reinterpret_cast<half*>(A_fp16.data_ptr<at::Half>());
    half* B_data = reinterpret_cast<half*>(B_fp16.data_ptr<at::Half>());

    // Break K into 8 segments to maximize concurrency
    constexpr int num_segments = 8;
    const int K_segment = (K + num_segments - 1) / num_segments;

    // Create temporary outputs and streams
    std::vector<torch::Tensor> segment_outputs;
    std::vector<cudaStream_t> streams(num_segments);
    for (int i = 0; i < num_segments; i++) {
        segment_outputs.push_back(torch::zeros({M, N}, options));
        cudaStreamCreate(&streams[i]);
    }

    // Kernel configuration
    const int grid_x = (N + WMMA_N - 1) / WMMA_N;
    const int grid_y = (M + WMMA_M - 1) / WMMA_M;
    dim3 grid(grid_x, grid_y);
    dim3 block(WARP_SIZE * WARPS_PER_BLOCK);
    const size_t shared_mem_size = WARPS_PER_BLOCK * SHARED_MEM_PER_WARP_PADDED * sizeof(float);

    // Set L1 cache preference for kernel
    cudaFuncSetCacheConfig(matmul_kernel, cudaFuncCachePreferL1);

    // Launch kernels for each segment
    for (int i = 0; i < num_segments; i++) {
        const int start_k = i * K_segment;
        if (start_k >= K) break;
        const int segment_length = std::min(K_segment, K - start_k);
        float* C_segment = segment_outputs[i].data_ptr<float>();

        matmul_kernel<<<grid, block, shared_mem_size, streams[i]>>>(
            A_data, B_data, C_segment, M, N, K, start_k, segment_length
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            // Clean up: destroy all streams
            for (int j = 0; j < num_segments; j++) {
                cudaStreamDestroy(streams[j]);
            }
            TORCH_CHECK(false, "Kernel launch failed for segment: ", i, ": ", cudaGetErrorString(err));
        }
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_segments; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Reduce partial results
    torch::Tensor C = segment_outputs[0];
    for (int i = 1; i < num_segments; i++) {
        C.add_(segment_outputs[i]);
    }

    return C;
}
// PART-END