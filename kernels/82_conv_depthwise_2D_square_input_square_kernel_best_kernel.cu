// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_pipeline.h>
#include <cstdint>

// Fixed tile dimensions
constexpr int TILE_INPUT_DIM = 34;
constexpr int TILE_PADDED_STRIDE = 36;

// Constant memory for kernel weights
__constant__ float c_kernel[576];
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void depthwise_conv2d_kernel(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding,
    int output_height,
    int output_width
) {
    if (kernel_size != 3 || stride != 1 || padding != 0) return;
    
    // Calculate indices
    int channel = blockIdx.z % in_channels;
    int batch = blockIdx.z / in_channels;
    if (batch >= batch_size) return;

    extern __shared__ float shared_mem[];
    float (*tile0)[TILE_PADDED_STRIDE] = reinterpret_cast<float (*)[TILE_PADDED_STRIDE]>(shared_mem);
    float (*tile1)[TILE_PADDED_STRIDE] = reinterpret_cast<float (*)[TILE_PADDED_STRIDE]>(shared_mem + TILE_INPUT_DIM * TILE_PADDED_STRIDE);

    // Load kernel weights first to overlap with async copies
    float w[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        w[i] = c_kernel[channel * 9 + i];
    }

    // Starting positions for both tiles
    int start_x = blockIdx.x * (2 * blockDim.x * 2);
    int start_y = blockIdx.y * (2 * blockDim.y);
    
    // Thread indexing
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;
    int total_tasks = 2 * TILE_INPUT_DIM * 9;

    // Unified async loading for both tiles
    for (int task_id = tid; task_id < total_tasks; task_id += num_threads) {
        int tile_id = task_id / (TILE_INPUT_DIM * 9);
        int task_in_tile = task_id % (TILE_INPUT_DIM * 9);
        int i = task_in_tile / 9;
        int j = (task_in_tile % 9) * 4;
        int tile_offset = tile_id * 32;
        
        float (*tile_ptr)[TILE_PADDED_STRIDE] = (tile_id == 0) ? tile0 : tile1;
        int in_y = start_y + i;
        int in_x = start_x + tile_offset + j;
        
        if (in_y >= 0 && in_y < height && in_x >= 0 && in_x + 3 < width) {
            int input_idx = batch * (in_channels * height * width) 
                         + channel * (height * width) 
                         + in_y * width 
                         + in_x;
            uint64_t global_ptr = reinterpret_cast<uint64_t>(&input[input_idx]);
            uint32_t smem_addr = __cvta_generic_to_shared(&tile_ptr[i][j]);
            asm volatile (
                "cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(smem_addr), "l"(global_ptr)
            );
        }
        else {
            for (int k = 0; k < 4; k++) {
                int el_x = in_x + k;
                if (in_y >= 0 && in_y < height && el_x >= 0 && el_x < width) {
                    int input_idx = batch * (in_channels * height * width) 
                                 + channel * (height * width) 
                                 + in_y * width 
                                 + el_x;
                    tile_ptr[i][j+k] = input[input_idx];
                } else {
                    tile_ptr[i][j+k] = 0.0f;
                }
            }
        }
    }

    // Unified pipeline synchronization
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Process tile0
    int out_x0 = start_x + threadIdx.x * 2;
    int out_y0 = start_y + threadIdx.y * 2;
    int out_y1 = out_y0 + 1;

    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    #pragma unroll
    for (int ky = 0; ky < 3; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 3; kx++) {
            int shared_y = threadIdx.y * 2 + ky;
            int shared_x = threadIdx.x * 2 + kx;
            
            float v00 = tile0[shared_y][shared_x];
            float v01 = tile0[shared_y][shared_x+1];
            float v10 = tile0[shared_y+1][shared_x];
            float v11 = tile0[shared_y+1][shared_x+1];
            
            float weight = w[ky*3+kx];
            sum00 += v00 * weight;
            sum01 += v01 * weight;
            sum10 += v10 * weight;
            sum11 += v11 * weight;
        }
    }

    // Write tile0 results with simplified conditions
    if (out_y0 < output_height) {
        int base_idx = batch * (in_channels * output_height * output_width) 
                    + channel * (output_height * output_width) 
                    + out_y0 * output_width 
                    + out_x0;
        
        if (out_x0 < output_width - 1) {
            *reinterpret_cast<float2*>(&output[base_idx]) = make_float2(sum00, sum01);
        } else {
            if (out_x0 < output_width) output[base_idx] = sum00;
            if (out_x0 + 1 < output_width) output[base_idx + 1] = sum01;
        }
    }
    
    if (out_y1 < output_height) {
        int base_idx = batch * (in_channels * output_height * output_width) 
                    + channel * (output_height * output_width) 
                    + out_y1 * output_width 
                    + out_x0;
        
        if (out_x0 < output_width - 1) {
            *reinterpret_cast<float2*>(&output[base_idx]) = make_float2(sum10, sum11);
        } else {
            if (out_x0 < output_width) output[base_idx] = sum10;
            if (out_x0 + 1 < output_width) output[base_idx + 1] = sum11;
        }
    }

    // Process tile1
    int out_x1 = start_x + 32 + threadIdx.x * 2;
    sum00 = 0.0f; sum01 = 0.0f; sum10 = 0.0f; sum11 = 0.0f;

    #pragma unroll
    for (int ky = 0; ky < 3; ky++) {
        #pragma unroll
        for (int kx = 0; kx < 3; kx++) {
            int shared_y = threadIdx.y * 2 + ky;
            int shared_x = threadIdx.x * 2 + kx;
            
            float v00 = tile1[shared_y][shared_x];
            float v01 = tile1[shared_y][shared_x+1];
            float v10 = tile1[shared_y+1][shared_x];
            float v11 = tile1[shared_y+1][shared_x+1];
            
            float weight = w[ky*3+kx];
            sum00 += v00 * weight;
            sum01 += v01 * weight;
            sum10 += v10 * weight;
            sum11 += v11 * weight;
        }
    }

    // Write tile1 results with simplified conditions
    if (out_y0 < output_height) {
        int base_idx = batch * (in_channels * output_height * output_width) 
                    + channel * (output_height * output_width) 
                    + out_y0 * output_width 
                    + out_x1;
        
        if (out_x1 < output_width - 1) {
            *reinterpret_cast<float2*>(&output[base_idx]) = make_float2(sum00, sum01);
        } else {
            if (out_x1 < output_width) output[base_idx] = sum00;
            if (out_x1 + 1 < output_width) output[base_idx + 1] = sum01;
        }
    }
    
    if (out_y1 < output_height) {
        int base_idx = batch * (in_channels * output_height * output_width) 
                    + channel * (output_height * output_width) 
                    + out_y1 * output_width 
                    + out_x1;
        
        if (out_x1 < output_width - 1) {
            *reinterpret_cast<float2*>(&output[base_idx]) = make_float2(sum10, sum11);
        } else {
            if (out_x1 < output_width) output[base_idx] = sum10;
            if (out_x1 + 1 < output_width) output[base_idx + 1] = sum11;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    int kernel_size,
    int stride,
    int padding
) {
    TORCH_CHECK(kernel_size == 3 && stride == 1 && padding == 0,
                "Optimized kernel requires: kernel_size=3, stride=1, padding=0");
    TORCH_CHECK(input.size(1) == 64,
                "Optimized kernel requires in_channels=64");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int output_height = height - 2;
    int output_width = width - 2;

    auto output = torch::zeros({batch_size, in_channels, output_height, output_width}, input.options());
    if (output.numel() == 0) {
        return output;
    }

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(c_kernel, kernel.data_ptr<float>(), 64 * 9 * sizeof(float));

    // Block and grid dimensions
    dim3 block(16, 16);
    dim3 grid(
        (output_width + 4 * block.x - 1) / (4 * block.x),
        (output_height + 2 * block.y - 1) / (2 * block.y),
        batch_size * in_channels
    );

    // Simplified shared memory calculation
    size_t shared_mem_size = 2 * TILE_INPUT_DIM * TILE_PADDED_STRIDE * sizeof(float);

    depthwise_conv2d_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        output_height,
        output_width
    );

    return output;
}
// PART-END