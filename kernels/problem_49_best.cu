// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template <int DIM>
__global__ void max_reduce_first_pass_kernel(const float* __restrict__ input, 
                                           float* __restrict__ temp, 
                                           int64_t s0, int64_t s1, int64_t s2,
                                           int64_t st0, int64_t st1, int64_t st2,
                                           int64_t reduced_dim_size, int64_t M, int64_t grid_y) {
    constexpr int vector_size = 4;
    int total_threads = blockDim.x * blockDim.y;
    int thread_idx = blockIdx.x * total_threads + threadIdx.x * blockDim.y + threadIdx.y;
    if (thread_idx * vector_size >= M) return;

    int chunk_index = blockIdx.y;
    int64_t chunk_start = static_cast<int64_t>(chunk_index) * blockDim.x;
    int64_t chunk_end = min(reduced_dim_size, chunk_start + blockDim.x);

    int64_t nonreduced_idx_base = thread_idx * vector_size;
    int64_t base_offsets[vector_size];
    bool valid[vector_size];
    
    // DIM-specialized offset calculation
    for (int i = 0; i < vector_size; ++i) {
        int64_t nonreduced_idx = nonreduced_idx_base + i;
        valid[i] = (nonreduced_idx < M);
        if (valid[i]) {
            if constexpr (DIM == 0) {
                int idx1 = nonreduced_idx / s2;
                int idx2 = nonreduced_idx % s2;
                base_offsets[i] = idx1 * st1 + idx2 * st2;
            } else if constexpr (DIM == 1) {
                int idx0 = nonreduced_idx / s2;
                int idx2 = nonreduced_idx % s2;
                base_offsets[i] = idx0 * st0 + idx2 * st2;
            } else if constexpr (DIM == 2) {
                int idx0 = nonreduced_idx / s1;
                int idx1 = nonreduced_idx % s1;
                base_offsets[i] = idx0 * st0 + idx1 * st1;
            }
        }
    }
    
    float local_max[vector_size] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    int64_t stride_reduced = (DIM == 0) ? st0 : (DIM == 1) ? st1 : st2;
    int lane_id = threadIdx.x % warpSize;
    int num_segments = static_cast<int>((chunk_end - chunk_start + warpSize - 1) / warpSize);

    for (int seg = 0; seg < num_segments; ++seg) {
        int64_t r = chunk_start + static_cast<int64_t>(seg) * warpSize + lane_id;
        float vals[vector_size] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

        if (r < chunk_end) {
            for (int i = 0; i < vector_size; ++i) {
                if (valid[i]) vals[i] = __ldg(input + base_offsets[i] + r * stride_reduced);
            }
        }

        // Warp reduction
        unsigned int mask = 0xFFFFFFFF;
        for (int offset = 1; offset < warpSize; offset <<= 1) {
            for (int i = 0; i < vector_size; ++i) {
                float tmp = __shfl_xor_sync(mask, vals[i], offset);
                vals[i] = fmaxf(vals[i], tmp);
            }
        }

        // Accumulate results
        for (int i = 0; i < vector_size; ++i) {
            if (valid[i]) local_max[i] = fmaxf(local_max[i], vals[i]);
        }
    }

    int64_t temp_index_base = chunk_index * M + nonreduced_idx_base;
    if (nonreduced_idx_base + 3 < M) {
        float4 out_val = make_float4(
            local_max[0], local_max[1], local_max[2], local_max[3]
        );
        *reinterpret_cast<float4*>(&temp[temp_index_base]) = out_val;
    } else {
        for (int i = 0; i < vector_size; i++) {
            if (valid[i]) temp[temp_index_base + i] = local_max[i];
        }
    }
}

__global__ void max_reduce_second_pass_kernel(const float* __restrict__ temp, 
                                            float* __restrict__ output, 
                                            int64_t M, int64_t grid_y) {
    int total_threads = blockDim.x * blockDim.y;
    int global_idx = blockIdx.x * total_threads + threadIdx.x * blockDim.y + threadIdx.y;
    
    if (global_idx < M) {
        float my_max = -FLT_MAX;
        for (int j = 0; j < grid_y; j++) {
            int64_t temp_idx = static_cast<int64_t>(j) * M + global_idx;
            my_max = fmaxf(my_max, __ldg(temp + temp_idx));
        }
        output[global_idx] = my_max;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor max_reduce_cuda(torch::Tensor input, int dim) {
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D");
    TORCH_CHECK(dim >= 0 && dim <= 2, "dim must be 0, 1, or 2");
    
    auto sizes = input.sizes();
    int64_t s0 = sizes[0];
    int64_t s1 = sizes[1];
    int64_t s2 = sizes[2];
    auto strides = input.strides();
    
    int64_t reduced_dim_size = sizes[dim];
    int64_t M = input.numel() / reduced_dim_size;
    
    auto output = torch::empty({M}, input.options());
    
    constexpr int block_x = 128;
    constexpr int block_y = 8;
    constexpr int total_threads = block_x * block_y;
    constexpr int vector_size = 4;
    
    int64_t grid_y_val = (reduced_dim_size + block_x - 1) / block_x;
    int grid_y = grid_y_val > INT_MAX ? INT_MAX : static_cast<int>(grid_y_val);
    int64_t grid_x_val = (M + total_threads * vector_size - 1) / (total_threads * vector_size);
    int grid_x = grid_x_val > INT_MAX ? INT_MAX : static_cast<int>(grid_x_val);
    
    auto temp = torch::empty({grid_y, M}, input.options());
    
    dim3 grid_first(grid_x, grid_y);
    dim3 block_first(block_x, block_y);
    
    // Launch specialized kernel based on reduction dimension
    switch(dim) {
        case 0:
            max_reduce_first_pass_kernel<0><<<grid_first, block_first, 0>>>(
                input.data_ptr<float>(),
                temp.data_ptr<float>(),
                s0, s1, s2,
                strides[0], strides[1], strides[2],
                reduced_dim_size,
                M, grid_y
            );
            break;
        case 1:
            max_reduce_first_pass_kernel<1><<<grid_first, block_first, 0>>>(
                input.data_ptr<float>(),
                temp.data_ptr<float>(),
                s0, s1, s2,
                strides[0], strides[1], strides[2],
                reduced_dim_size,
                M, grid_y
            );
            break;
        case 2:
            max_reduce_first_pass_kernel<2><<<grid_first, block_first, 0>>>(
                input.data_ptr<float>(),
                temp.data_ptr<float>(),
                s0, s1, s2,
                strides[0], strides[1], strides[2],
                reduced_dim_size,
                M, grid_y
            );
            break;
    }
    
    int64_t grid_second_val = (M + total_threads - 1) / total_threads;
    int grid_second = grid_second_val > INT_MAX ? INT_MAX : static_cast<int>(grid_second_val);
    max_reduce_second_pass_kernel<<<grid_second, block_first, 0>>>(
        temp.data_ptr<float>(),
        output.data_ptr<float>(),
        M, grid_y
    );
    
    std::vector<int64_t> output_shape;
    for (int i = 0; i < 3; i++) {
        if (i != dim) output_shape.push_back(sizes[i]);
    }
    return output.view(output_shape);
}
// PART-END