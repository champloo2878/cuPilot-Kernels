#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <ATen/cuda/CUDAContext.h>

//PART-START
__global__ void tensor_matmul_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ out,
                                     int M, int l, int k) {
    // Compute grid dimensions
    const int grid_x = (M + 63) / 64;
    const int grid_y = (k + 255) / 256;
    const int total_blocks = grid_x * grid_y;
    
    // Apply XOR-shuffle remapping
    const int block_id = blockIdx.x;
    const int swizzled_block_id = (total_blocks == 65536) ? 
                                 (block_id ^ (block_id >> 8)) : 
                                 block_id;
    
    // Compute tile indices
    const int row_tile = swizzled_block_id / grid_y;
    const int col_tile = swizzled_block_id % grid_y;

    __shared__ __align__(16) half smem_A[2][64][16];
    __shared__ __align__(16) half smem_B[2][16][264];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    int stage = 0;

    const int row_group = warp_id / 4;
    const int col_group = warp_id % 4;
    const int k_offset = col_group * 64;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag0, b_frag1, b_frag2, b_frag3;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag0, acc_frag1, acc_frag2, acc_frag3;

    nvcuda::wmma::fill_fragment(acc_frag0, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag1, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag2, 0.0f);
    nvcuda::wmma::fill_fragment(acc_frag3, 0.0f);

    auto load_tile = [&](int stage_idx, int t_offset) {
        const int n_chunks_A = 128;
        const int n_chunks_B = 512;
        
        for (int chunk = threadIdx.x; chunk < n_chunks_A; chunk += blockDim.x) {
            int row = chunk / 2;
            int col_chunk = (chunk % 2) * 8;
            int global_idx = (row_tile * 64 + row) * l + t_offset + col_chunk;
            __pipeline_memcpy_async(&smem_A[stage_idx][row][col_chunk], &A[global_idx], 16);
        }

        for (int chunk = threadIdx.x; chunk < n_chunks_B; chunk += blockDim.x) {
            int row = chunk / 32;
            int col_chunk = (chunk % 32) * 8;
            int global_idx = (t_offset + row) * k + col_tile * 256 + col_chunk;
            __pipeline_memcpy_async(&smem_B[stage_idx][row][col_chunk], &B[global_idx], 16);
        }
    };

    if (0 < l) {
        load_tile(0, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }

    for (int t = 0; t < l; t += 16) {
        int current_stage = stage;
        int next_stage = 1 - stage;

        if (t + 16 < l) {
            load_tile(next_stage, t + 16);
            __pipeline_commit();
        }

        nvcuda::wmma::load_matrix_sync(a_frag, &smem_A[current_stage][row_group * 16][0], 16);
        
        nvcuda::wmma::load_matrix_sync(b_frag0, &smem_B[current_stage][0][k_offset], 264);
        nvcuda::wmma::load_matrix_sync(b_frag1, &smem_B[current_stage][0][k_offset + 16], 264);
        nvcuda::wmma::load_matrix_sync(b_frag2, &smem_B[current_stage][0][k_offset + 32], 264);
        nvcuda::wmma::load_matrix_sync(b_frag3, &smem_B[current_stage][0][k_offset + 48], 264);
        
        nvcuda::wmma::mma_sync(acc_frag0, a_frag, b_frag0, acc_frag0);
        nvcuda::wmma::mma_sync(acc_frag1, a_frag, b_frag1, acc_frag1);
        nvcuda::wmma::mma_sync(acc_frag2, a_frag, b_frag2, acc_frag2);
        nvcuda::wmma::mma_sync(acc_frag3, a_frag, b_frag3, acc_frag3);

        if (t + 16 < l) {
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        stage = next_stage;
    }

    float* out_base = out + (row_tile * 64 + row_group * 16) * k + col_tile * 256;
    
    nvcuda::wmma::store_matrix_sync(
        out_base + k_offset,
        acc_frag0,
        k,
        nvcuda::wmma::mem_row_major
    );
    nvcuda::wmma::store_matrix_sync(
        out_base + k_offset + 16,
        acc_frag1,
        k,
        nvcuda::wmma::mem_row_major
    );
    nvcuda::wmma::store_matrix_sync(
        out_base + k_offset + 32,
        acc_frag2,
        k,
        nvcuda::wmma::mem_row_major
    );
    nvcuda::wmma::store_matrix_sync(
        out_base + k_offset + 48,
        acc_frag3,
        k,
        nvcuda::wmma::mem_row_major
    );
}
//PART-END

//PART-START
torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    int b = A_cont.size(0);
    int i = A_cont.size(1);
    int j = A_cont.size(2);
    int l = A_cont.size(3);
    int k_size = B_cont.size(1);
    
    auto A_half = A_cont.to(torch::kFloat16);
    auto B_half = B_cont.to(torch::kFloat16);
    
    int M = b * i * j;
    auto out = torch::zeros({M, k_size}, A_cont.options().dtype(torch::kFloat));
    
    // Compute grid dimensions
    const int grid_x = (M + 63) / 64;
    const int grid_y = (k_size + 255) / 256;
    const int total_blocks = grid_x * grid_y;
    
    // Launch 1D grid with XOR-shuffle remapping
    dim3 grid(total_blocks);
    dim3 block(512, 1);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    tensor_matmul_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(A_half.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(B_half.data_ptr<torch::Half>()),
        out.data_ptr<float>(),
        M, l, k_size
    );
    
    return out.view({b, i, j, k_size});
}
//PART-END