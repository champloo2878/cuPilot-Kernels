// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>

__device__ __inline__ float warpReduceSum(float val) {
    int lane = threadIdx.x & 0x1f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane < offset) {
            val += other;
        }
    }
    if (lane == 0) {
        return val;
    }
    return 0.0f;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void mean_reduce_kernel(const float* input, float* output, 
                                  int64_t outer_dims, int64_t reduction_size, int64_t inner_dims) {
    constexpr int VEC = 4;
    constexpr int TILE_SIZE = 512;
    
    extern __shared__ float shared_arr[];
    float* shared = shared_arr; // 1D shared memory: [BLOCK_Y * TILE_SIZE]
    
    int tile_idx = blockIdx.x;
    int outer_idx = blockIdx.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    int inner_base = tile_idx * TILE_SIZE;
    int inner_start = inner_base + tid_x * VEC;
    
    if (inner_base >= inner_dims) return;
    
    const float* outer_base = input + outer_idx * reduction_size * inner_dims;
    const float* thread_base = outer_base + inner_start;
    
    float sum[VEC] = {0.0f};
    
    int segment_size = (reduction_size + blockDim.y - 1) / blockDim.y;
    int r_start = tid_y * segment_size;
    int r_end = (r_start + segment_size) < reduction_size ? (r_start + segment_size) : reduction_size;
    
    for (int r = r_start; r < r_end; r++) {
        const float* base_ptr = thread_base + r * inner_dims;
        
        for (int v = 0; v < VEC; v++) {
            int inner_idx = inner_start + v;
            if (inner_idx < inner_dims) {
                sum[v] += base_ptr[v];
            }
        }
    }
    
    for (int v = 0; v < VEC; v++) {
        int inner_idx_in_tile = tid_x * VEC + v;
        if (inner_idx_in_tile < TILE_SIZE) {
            shared[tid_y * TILE_SIZE + inner_idx_in_tile] = sum[v];
        }
    }
    
    __syncthreads();
    
    if (tid_y == 0) {
        for (int v = 0; v < VEC; v++) {
            int inner_idx = inner_start + v;
            if (inner_idx < inner_dims) {
                float accum = 0.0f;
                for (int y = 0; y < blockDim.y; y++) {
                    accum += shared[y * TILE_SIZE + tid_x * VEC + v];
                }
                output[outer_idx * inner_dims + inner_idx] = accum / reduction_size;
            }
        }
    }
}

__global__ void mean_reduce_inner1_kernel(const float* input, float* output, 
                                        int64_t outer_dims, int64_t reduction_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_dims) return;

    const float* base_ptr = input + idx * reduction_size;
    float sum = 0.0f;
    constexpr int VEC = 4;
    int r = 0;

    for (; r <= reduction_size - VEC; r += VEC) {
        float4 chunk = *reinterpret_cast<const float4*>(base_ptr + r);
        sum += chunk.x + chunk.y + chunk.z + chunk.w;
    }
    for (; r < reduction_size; r++) {
        sum += base_ptr[r];
    }
    output[idx] = sum / reduction_size;
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    auto shape = input.sizes();
    int ndim = shape.size();

    if (dim < 0) {
        dim = ndim + dim;
    }

    std::vector<int64_t> new_shape;
    for (int i=0; i<ndim; i++) {
        if (i != dim) {
            new_shape.push_back(shape[i]);
        }
    }

    int64_t outer_dims = 1;
    for (int i=0; i<dim; i++) {
        outer_dims *= shape[i];
    }
    int64_t reduction_size = shape[dim];
    int64_t inner_dims = 1;
    for (int i=dim+1; i<ndim; i++) {
        inner_dims *= shape[i];
    }

    int64_t total_output_elements = outer_dims * inner_dims;
    auto output = torch::zeros(new_shape, input.options());

    if (input.numel() == 0) {
        return output;
    }

    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Specialized path for inner_dims=1 with vectorization
    if (inner_dims == 1) {
        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (outer_dims + BLOCK_SIZE - 1) / BLOCK_SIZE;
        mean_reduce_inner1_kernel<<<grid_size, BLOCK_SIZE>>>(input_ptr, output_ptr, 
                                                             outer_dims, reduction_size);
    } 
    // Optimized tiled kernel for other cases
    else {
        constexpr int TILE_SIZE = 512;
        constexpr int VEC = 4;
        const int THREADS_X = TILE_SIZE / VEC;
        const int BLOCK_Y = 8;

        int grid_x = (inner_dims + TILE_SIZE - 1) / TILE_SIZE;
        dim3 grid(grid_x, outer_dims);
        dim3 block(THREADS_X, BLOCK_Y);

        size_t shared_mem_size = BLOCK_Y * TILE_SIZE * sizeof(float);
        mean_reduce_kernel<<<grid, block, shared_mem_size>>>(input_ptr, output_ptr, 
                                                            outer_dims, reduction_size, inner_dims);
    }
    
    return output;
}
// PART-END