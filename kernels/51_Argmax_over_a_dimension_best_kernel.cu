// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <numeric>
#include <functional>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
// Specialized kernel for reduction_dim=2 on [128,4096,4095] contiguous input
__global__ void argmax_specialized_dim2_kernel(const float* __restrict__ input, 
                                               int64_t* __restrict__ output,
                                               const int64_t stride0, 
                                               const int64_t stride1) {
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_id / 32;
    const int lane_id = global_id % 32;
    
    if (warp_id >= 128 * 4096) return;
    
    const int i = warp_id / 4096;
    const int j = warp_id % 4096;
    const float* row_ptr = input + i * stride0 + j * stride1;
    
    float thread_max = -FLT_MAX;
    int64_t thread_idx = -1;
    
    #pragma unroll
    for (int k = lane_id; k < 4095; k += 32) {
        float val = __ldg(row_ptr + k);
        if (val > thread_max) {
            thread_max = val;
            thread_idx = k;
        }
    }
    
    float warp_max = thread_max;
    int64_t warp_idx = thread_idx;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float temp_val = __shfl_down_sync(0xFFFFFFFF, warp_max, offset);
        int64_t temp_idx = __shfl_down_sync(0xFFFFFFFF, warp_idx, offset);
        if (temp_val > warp_max) {
            warp_max = temp_val;
            warp_idx = temp_idx;
        } else if (temp_val == warp_max && temp_idx < warp_idx) {
            warp_idx = temp_idx;
        }
    }
    
    if (lane_id == 0) {
        output[i * 4096 + j] = warp_idx;
    }
}

// General optimized kernel
__global__ void argmax_kernel(const float* __restrict__ input, 
                              int64_t* __restrict__ output, 
                              int reduction_dim, int reduction_dim_size,
                              int inner_dim_size, int segments_per_outer, int total_segments, 
                              int n_dims,
                              const int64_t* __restrict__ input_strides,
                              const int64_t* __restrict__ non_reduced_dims) {
    int segment_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment_id >= total_segments) return;
    
    int outer_idx = segment_id / segments_per_outer;
    int seg = segment_id % segments_per_outer;
    int k_base = seg << 2;

    int base_offset = 0;
    if (n_dims > 1) {
        base_offset = outer_idx * input_strides[non_reduced_dims[0]];
    }
    base_offset += k_base * input_strides[non_reduced_dims[n_dims - 2]];
    
    int reduction_stride = input_strides[reduction_dim];
    
    float4 max_values = make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    int64_t max_indices[4] = {0};

    for (int j = 0; j < reduction_dim_size; ++j) {
        const float* ptr = input + base_offset + j * reduction_stride;
        float4 vals;
        
        bool k0_valid = (k_base + 0) < inner_dim_size;
        bool k1_valid = (k_base + 1) < inner_dim_size;
        bool k2_valid = (k_base + 2) < inner_dim_size;
        bool k3_valid = (k_base + 3) < inner_dim_size;
        
        vals.x = k0_valid ? __ldg(ptr + 0) : -FLT_MAX;
        vals.y = k1_valid ? __ldg(ptr + 1) : -FLT_MAX;
        vals.z = k2_valid ? __ldg(ptr + 2) : -FLT_MAX;
        vals.w = k3_valid ? __ldg(ptr + 3) : -FLT_MAX;

        if (vals.x > max_values.x) {
            max_values.x = vals.x;
            max_indices[0] = j;
        }
        if (vals.y > max_values.y) {
            max_values.y = vals.y;
            max_indices[1] = j;
        }
        if (vals.z > max_values.z) {
            max_values.z = vals.z;
            max_indices[2] = j;
        }
        if (vals.w > max_values.w) {
            max_values.w = vals.w;
            max_indices[3] = j;
        }
    }

    for (int v = 0; v < 4; ++v) {
        int k = k_base + v;
        if (k < inner_dim_size) {
            output[outer_idx * inner_dim_size + k] = max_indices[v];
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor argmax_cuda(torch::Tensor input, int64_t reduction_dim) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() <= 8, "Input tensor must have <= 8 dimensions");
    
    if (input.dtype() != torch::kFloat32) {
        input = input.to(torch::kFloat32);
    }
    
    auto device = input.device();
    int n_dims = input.dim();
    int reduction_dim_size = input.size(reduction_dim);
    
    auto input_sizes = input.sizes();
    std::vector<int64_t> output_shape_vec;
    for (int i=0; i<n_dims; i++) {
        if (i != reduction_dim) {
            output_shape_vec.push_back(input_sizes[i]);
        }
    }
    auto output = torch::empty(output_shape_vec, torch::TensorOptions().dtype(torch::kLong).device(device));
    
    if (input.numel() == 0) {
        return output;
    }
    
    int64_t outer_size = 1;
    int inner_dim_size = 1;
    if (output_shape_vec.size() > 0) {
        inner_dim_size = output_shape_vec.back();
        if (output_shape_vec.size() > 1) {
            outer_size = std::accumulate(output_shape_vec.begin(), output_shape_vec.end()-1, 
                                        1, std::multiplies<int64_t>());
        }
    }
    
    int segments_per_outer = (inner_dim_size + 3) / 4;
    int total_segments = outer_size * segments_per_outer;

    std::vector<int64_t> non_reduced_dims_host;
    for (int i=0; i<n_dims; i++) {
        if (i != reduction_dim) {
            non_reduced_dims_host.push_back(i);
        }
    }
    non_reduced_dims_host.resize(8, 0);
    
    auto input_strides_host = input.strides();
    std::vector<int64_t> input_strides_vec(input_strides_host.begin(), input_strides_host.end());
    input_strides_vec.resize(8, 0);
    
    const int block_size = 256;
    int grid_size = (total_segments + block_size - 1) / block_size;
    
    auto input_strides_gpu = torch::tensor(input_strides_vec, torch::TensorOptions().dtype(torch::kLong).device(device));
    auto non_reduced_dims_gpu = torch::tensor(non_reduced_dims_host, torch::TensorOptions().dtype(torch::kLong).device(device));
    
    // Specialized handling for dim=2 reduction with specific input
    if (reduction_dim == 2 && n_dims == 3 && 
        input.size(0) == 128 && input.size(1) == 4096 && input.size(2) == 4095 && 
        input.is_contiguous()) {
        const int64_t total_rows = 128 * 4096;
        const int block_size_special = 256;
        const int total_threads = total_rows * 32;
        const int grid_size_special = (total_threads + block_size_special - 1) / block_size_special;
        
        int64_t stride0 = input.stride(0) / sizeof(float);
        int64_t stride1 = input.stride(1) / sizeof(float);
        
        argmax_specialized_dim2_kernel<<<grid_size_special, block_size_special, 0>>>(
            input.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            stride0,
            stride1
        );
    } else {
        argmax_kernel<<<grid_size, block_size, 0>>>(
            input.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            reduction_dim,
            reduction_dim_size,
            inner_dim_size,
            segments_per_outer,
            total_segments,
            n_dims,
            input_strides_gpu.data_ptr<int64_t>(),
            non_reduced_dims_gpu.data_ptr<int64_t>()
        );
    }
    
    return output;
}
// PART-END