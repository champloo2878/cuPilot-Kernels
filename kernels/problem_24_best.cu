// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void log_softmax_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int num_rows, 
    int row_size
) {
    extern __shared__ float smem[];
    const int num_warps = 32;
    const int tile_size = 4096; // Optimized tile size for A100 shared memory
    const int num_tiles = (row_size + tile_size - 1) / tile_size;
    
    float* warp_max = smem;
    float* warp_sum = &smem[num_warps];
    float* block_max = &smem[2 * num_warps];
    float* block_sum = &block_max[1];
    float* buffer0 = &smem[2 * num_warps + 2];
    float* buffer1 = &buffer0[tile_size];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Double buffering variables
    float* current_buffer = buffer0;
    float* next_buffer = buffer1;
    bool buffer_flag = false;
    
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;
    
    // Pre-load first tile
    int tile = 0;
    int load_start = tile * tile_size;
    int load_size = min(tile_size, row_size - load_start);
    
    for (int base = tid * 4; base < load_size; base += blockDim.x * 4) {
        int global_idx = row * row_size + load_start + base;
        if (base + 3 < load_size) {
            float4 in = __ldg(reinterpret_cast<const float4*>(input + global_idx));
            current_buffer[base] = in.x;
            current_buffer[base + 1] = in.y;
            current_buffer[base + 2] = in.z;
            current_buffer[base + 3] = in.w;
        } else {
            // Handle boundary case
            for (int i = 0; i < 4; i++) {
                if (base + i < load_size) {
                    current_buffer[base + i] = input[global_idx + i];
                }
            }
        }
    }
    __syncthreads();
    
    // Process tiles with double buffering
    for (tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * tile_size;
        int tile_end = min((tile + 1) * tile_size, row_size);
        int current_tile_size = tile_end - tile_start;
        
        // Pre-load next tile if not the last tile
        if (tile < num_tiles - 1) {
            int next_tile_start = (tile + 1) * tile_size;
            int next_tile_size = min(tile_size, row_size - next_tile_start);
            
            for (int base = tid * 4; base < next_tile_size; base += blockDim.x * 4) {
                int global_idx = row * row_size + next_tile_start + base;
                if (base + 3 < next_tile_size) {
                    float4 in = __ldg(reinterpret_cast<const float4*>(input + global_idx));
                    next_buffer[base] = in.x;
                    next_buffer[base + 1] = in.y;
                    next_buffer[base + 2] = in.z;
                    next_buffer[base + 3] = in.w;
                } else {
                    for (int i = 0; i < 4; i++) {
                        if (base + i < next_tile_size) {
                            next_buffer[base + i] = input[global_idx + i];
                        }
                    }
                }
            }
        }
        
        // Process current tile - max computation
        for (int base = tid * 4; base < current_tile_size; base += blockDim.x * 16) {
            int max_base = min(base, current_tile_size - 4);
            if (max_base + 3 < current_tile_size) {
                thread_max = fmaxf(thread_max, current_buffer[max_base]);
                thread_max = fmaxf(thread_max, current_buffer[max_base + 1]);
                thread_max = fmaxf(thread_max, current_buffer[max_base + 2]);
                thread_max = fmaxf(thread_max, current_buffer[max_base + 3]);
            }
        }
        
        // Warp and block reduction for max
        float reduced_max = warpReduceMax(thread_max);
        if (lane_id == 0)
            warp_max[warp_id] = reduced_max;
        __syncthreads();
        
        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? warp_max[lane_id] : -INFINITY;
            val = warpReduceMax(val);
            if (lane_id == 0)
                *block_max = val;
        }
        __syncthreads();
        
        float row_max = *block_max;
        
        // Process current tile - sum computation
        for (int base = tid * 4; base < current_tile_size; base += blockDim.x * 16) {
            int sum_base = min(base, current_tile_size - 4);
            if (sum_base + 3 < current_tile_size) {
                thread_sum += expf(current_buffer[sum_base] - row_max) +
                             expf(current_buffer[sum_base + 1] - row_max) +
                             expf(current_buffer[sum_base + 2] - row_max) +
                             expf(current_buffer[sum_base + 3] - row_max);
            }
        }
        
        // Swap buffers for next iteration
        float* temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
        buffer_flag = !buffer_flag;
        
        __syncthreads();
    }
    
    // Final warp and block reduction for sum
    float reduced_sum = warpReduceSum(thread_sum);
    if (lane_id == 0)
        warp_sum[warp_id] = reduced_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sum[lane_id] : 0.0f;
        val = warpReduceSum(val);
        if (lane_id == 0)
            *block_sum = val;
    }
    __syncthreads();
    
    float log_sum = logf(*block_sum);
    float row_max = *block_max;
    
    // Final pass - compute output with double buffering
    current_buffer = buffer0;
    next_buffer = buffer1;
    buffer_flag = false;
    
    // Re-load first tile for output computation
    tile = 0;
    load_start = tile * tile_size;
    load_size = min(tile_size, row_size - load_start);
    
    for (int base = tid * 4; base < load_size; base += blockDim.x * 4) {
        int global_idx = row * row_size + load_start + base;
        if (base + 3 < load_size) {
            float4 in = __ldg(reinterpret_cast<const float4*>(input + global_idx));
            current_buffer[base] = in.x;
            current_buffer[base + 1] = in.y;
            current_buffer[base + 2] = in.z;
            current_buffer[base + 3] = in.w;
        } else {
            for (int i = 0; i < 4; i++) {
                if (base + i < load_size) {
                    current_buffer[base + i] = input[global_idx + i];
                }
            }
        }
    }
    __syncthreads();
    
    for (tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * tile_size;
        int tile_end = min((tile + 1) * tile_size, row_size);
        int current_tile_size = tile_end - tile_start;
        
        // Pre-load next tile if not the last tile
        if (tile < num_tiles - 1) {
            int next_tile_start = (tile + 1) * tile_size;
            int next_tile_size = min(tile_size, row_size - next_tile_start);
            
            for (int base = tid * 4; base < next_tile_size; base += blockDim.x * 4) {
                int global_idx = row * row_size + next_tile_start + base;
                if (base + 3 < next_tile_size) {
                    float4 in = __ldg(reinterpret_cast<const float4*>(input + global_idx));
                    next_buffer[base] = in.x;
                    next_buffer[base + 1] = in.y;
                    next_buffer[base + 2] = in.z;
                    next_buffer[base + 3] = in.w;
                } else {
                    for (int i = 0; i < 4; i++) {
                        if (base + i < next_tile_size) {
                            next_buffer[base + i] = input[global_idx + i];
                        }
                    }
                }
            }
        }
        
        // Process current tile - output computation
        for (int base = tid * 4; base < current_tile_size; base += blockDim.x * 4) {
            int out_base = min(base, current_tile_size - 4);
            int global_idx = row * row_size + tile_start + out_base;
            
            if (out_base + 3 < current_tile_size) {
                float4 out;
                out.x = current_buffer[out_base] - row_max - log_sum;
                out.y = current_buffer[out_base + 1] - row_max - log_sum;
                out.z = current_buffer[out_base + 2] - row_max - log_sum;
                out.w = current_buffer[out_base + 3] - row_max - log_sum;
                
                *reinterpret_cast<float4*>(output + global_idx) = out;
            } else {
                // Handle boundary case
                for (int i = 0; i < 4; i++) {
                    if (out_base + i < current_tile_size) {
                        output[global_idx + i] = current_buffer[out_base + i] - row_max - log_sum;
                    }
                }
            }
        }
        
        // Swap buffers for next iteration
        float* temp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = temp;
        buffer_flag = !buffer_flag;
        
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor log_softmax_cuda(torch::Tensor input, int dim) {
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(input.size(1) % 4 == 0, "Row size must be divisible by 4 for vectorization");
    
    auto output = torch::empty_like(input);
    int num_rows = input.size(0);
    int row_size = input.size(1);
    
    // Configure kernel launch parameters
    const int block_size = 1024;
    const int num_warps = block_size / 32;
    const int tile_size = 4096;
    
    // Calculate shared memory requirements for double buffering
    int shared_mem_size = (2 * num_warps + 2) * sizeof(float) +  // reduction arrays
                        2 * tile_size * sizeof(float);          // double buffers
    
    TORCH_CHECK(shared_mem_size <= 49152, "Shared memory requirement exceeds A100 limit");
    
    dim3 grid(num_rows);
    dim3 block(block_size);
    
    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    log_softmax_kernel<<<grid, block, shared_mem_size, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_rows,
        row_size
    );
    
    return output;
}
// PART-END