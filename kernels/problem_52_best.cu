// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>

constexpr int TILE_ROWS = 32;
constexpr int NCOLS_PER_BLOCK = 256;
constexpr int UNROLL_STEP = 8;
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void argmin_kernel(const float* input, int64_t* output, 
                              int dim0, int dim1, int dim2) {
    __shared__ float shmem[TILE_ROWS][NCOLS_PER_BLOCK];
    
    int batch = blockIdx.y;
    int col = blockIdx.x * NCOLS_PER_BLOCK + threadIdx.x;
    
    if (col >= dim2) return;

    float local_min_value = INFINITY;
    int local_min_index = -1;

    for (int row_start = 0; row_start < dim1; row_start += TILE_ROWS) {
        int row_end = min(row_start + TILE_ROWS, dim1);
        int rows_in_tile = row_end - row_start;
        int tile_index = row_start / TILE_ROWS;
        
        // Cache-aware memory loading
        if (tile_index < 2) {
            for (int r = 0; r < rows_in_tile; r++) {
                int row = row_start + r;
                int global_idx = batch * dim1 * dim2 + row * dim2 + col;
                shmem[r][threadIdx.x] = __ldg(&input[global_idx]);
            }
        } else {
            for (int r = 0; r < rows_in_tile; r++) {
                int row = row_start + r;
                int global_idx = batch * dim1 * dim2 + row * dim2 + col;
                shmem[r][threadIdx.x] = __ldcg(&input[global_idx]);
            }
        }
        __syncthreads();
        
        // Process tile with loop unrolling
        int r = 0;
        for (; r <= rows_in_tile - UNROLL_STEP; r += UNROLL_STEP) {
            #pragma unroll
            for (int i = 0; i < UNROLL_STEP; i++) {
                float val = shmem[r + i][threadIdx.x];
                int row_idx = row_start + r + i;
                if (val < local_min_value) {
                    local_min_value = val;
                    local_min_index = row_idx;
                }
                else if (val == local_min_value && row_idx < local_min_index) {
                    local_min_index = row_idx;
                }
            }
        }
        // Process remaining rows
        for (; r < rows_in_tile; r++) {
            float val = shmem[r][threadIdx.x];
            int row_idx = row_start + r;
            if (val < local_min_value) {
                local_min_value = val;
                local_min_index = row_idx;
            }
            else if (val == local_min_value && row_idx < local_min_index) {
                local_min_index = row_idx;
            }
        }
        __syncthreads();
    }
    
    output[batch * dim2 + col] = local_min_index;
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    auto sizes = input.sizes();
    int dim0 = sizes[0];
    int dim1 = sizes[1];
    int dim2 = sizes[2];
    auto output = torch::empty({dim0, dim2}, 
                              torch::dtype(torch::kLong).device(input.device()));

    int grid_x = (dim2 + NCOLS_PER_BLOCK - 1) / NCOLS_PER_BLOCK;
    dim3 grid(grid_x, dim0);
    size_t shared_mem_size = 0;  // Static shared memory allocation

    argmin_kernel<<<grid, NCOLS_PER_BLOCK, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        dim0, dim1, dim2
    );

    return output;
}
// PART-END