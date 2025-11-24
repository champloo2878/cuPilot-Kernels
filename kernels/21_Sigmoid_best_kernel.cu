// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__device__ __forceinline__ int swizzle_block_idx(int block_idx, int grid_size) {
    const int group_size = 128;
    if (grid_size < group_size) {
        return block_idx;
    }

    int group = block_idx / group_size;
    int offset = block_idx % group_size;

    int new_offset = (offset % 8) * 16 + (offset / 8);
    int new_block_idx = group * group_size + new_offset;

    if (new_block_idx < grid_size) {
        return new_block_idx;
    }
    return block_idx;
}

__device__ __forceinline__ float fast_exp_neg(float x) {
    const float LOG2E = 1.4426950408889634f; // log2(e)
    float y = -x * LOG2E;
    float result;
    asm volatile ("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(y));
    return result;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void sigmoid_kernel_fp32(const float* input, float* output, int num_elements, int grid_size) {
    extern __shared__ char s_data_raw[];
    const int TILE_SIZE = 1024;
    float* s_data = reinterpret_cast<float*>(s_data_raw);

    int tile_idx = swizzle_block_idx(blockIdx.x, grid_size);
    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = min(tile_start + TILE_SIZE, num_elements);
    int tile_valid_count = tile_end - tile_start;

    int tid = threadIdx.x;
    int idx = tile_start + tid * 4;

    // Vectorized load
    if (tile_start + tid * 4 + 3 < tile_end) {
        float4 val = reinterpret_cast<const float4*>(&input[idx])[0];
        s_data[tid*4] = val.x;
        s_data[tid*4+1] = val.y;
        s_data[tid*4+2] = val.z;
        s_data[tid*4+3] = val.w;
    } else if (tile_valid_count > 0) {
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                s_data[tid*4+i] = input[tile_start + tid*4+i];
            }
        }
    }
    __syncthreads();

    // Compute sigmoid in shared memory using PTX exp approximation
    if (tid * 4 < tile_valid_count) {
        float results[4];
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                float val = s_data[tid*4+i];
                // Use specialized PTX exp approximation
                float exp_val = fast_exp_neg(val);
                results[i] = 1.0f / (1.0f + exp_val);
            }
        }

        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                s_data[tid*4+i] = results[i];
            }
        }
    }
    __syncthreads();

    // Vectorized store
    if (tile_start + tid * 4 + 3 < tile_end) {
        float4 out_val;
        out_val.x = s_data[tid*4];
        out_val.y = s_data[tid*4+1];
        out_val.z = s_data[tid*4+2];
        out_val.w = s_data[tid*4+3];
        reinterpret_cast<float4*>(&output[idx])[0] = out_val;
    } else if (tile_valid_count > 0) {
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                output[tile_start + tid*4+i] = s_data[tid*4+i];
            }
        }
    }
}

__global__ void sigmoid_kernel_fp16(const __half* input, __half* output, int num_elements, int grid_size) {
    extern __shared__ char s_data_raw[];
    const int TILE_SIZE = 1024;
    __half* s_data = reinterpret_cast<__half*>(s_data_raw);

    int tile_idx = swizzle_block_idx(blockIdx.x, grid_size);
    int tile_start = tile_idx * TILE_SIZE;
    int tile_end = min(tile_start + TILE_SIZE, num_elements);
    int tile_valid_count = tile_end - tile_start;

    int tid = threadIdx.x;
    int idx = tile_start + tid * 4;

    // Vectorized load
    if (tile_start + tid * 4 + 3 < tile_end) {
        uint4 in_val = reinterpret_cast<const uint4*>(&input[idx])[0];
        __half* in_val_half = reinterpret_cast<__half*>(&in_val);
        s_data[tid*4] = in_val_half[0];
        s_data[tid*4+1] = in_val_half[1];
        s_data[tid*4+2] = in_val_half[2];
        s_data[tid*4+3] = in_val_half[3];
    } else if (tile_valid_count > 0) {
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                s_data[tid*4+i] = input[tile_start + tid*4+i];
            }
        }
    }
    __syncthreads();

    // Compute sigmoid in FP32 using PTX exp approximation
    if (tid * 4 < tile_valid_count) {
        float results[4];
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                float val = __half2float(s_data[tid*4+i]);
                // Use specialized PTX exp approximation
                float exp_val = fast_exp_neg(val);
                results[i] = 1.0f / (1.0f + exp_val);
            }
        }

        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                s_data[tid*4+i] = __float2half(results[i]);
            }
        }
    }
    __syncthreads();

    // Vectorized store
    if (tile_start + tid * 4 + 3 < tile_end) {
        uint4 out_val;
        __half* out_val_half = reinterpret_cast<__half*>(&out_val);
        out_val_half[0] = s_data[tid*4];
        out_val_half[1] = s_data[tid*4+1];
        out_val_half[2] = s_data[tid*4+2];
        out_val_half[3] = s_data[tid*4+3];
        reinterpret_cast<uint4*>(&output[idx])[0] = out_val;
    } else if (tile_valid_count > 0) {
        for (int i = 0; i < 4; i++) {
            if (tid * 4 + i < tile_valid_count) {
                output[tile_start + tid*4+i] = s_data[tid*4+i];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor sigmoid_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    if (num_elements == 0) {
        return output;
    }

    const int block_size = 256;
    const int tile_size = 1024;
    int grid_size = (num_elements + tile_size - 1) / tile_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t shared_mem_size;
    if (input.dtype() == torch::kFloat32) {
        shared_mem_size = (tile_size + 1) * sizeof(float);
    } else if (input.dtype() == torch::kFloat16) {
        shared_mem_size = (tile_size + 1) * sizeof(__half);
    } else {
        AT_ERROR("Unsupported data type for sigmoid kernel");
    }

    if (input.dtype() == torch::kFloat32) {
        sigmoid_kernel_fp32<<<grid_size, block_size, shared_mem_size, stream>>>(
            input.data_ptr<float>(), 
            output.data_ptr<float>(),
            num_elements,
            grid_size
        );
    } else if (input.dtype() == torch::kFloat16) {
        sigmoid_kernel_fp16<<<grid_size, block_size, shared_mem_size, stream>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_elements,
            grid_size
        );
    } else {
        AT_ERROR("Unsupported data type for sigmoid kernel");
    }
    
    return output;
}
// PART-END