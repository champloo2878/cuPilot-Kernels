// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // for FLT_MAX
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void max_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation
) {
    // Specialized branch for kernel_size=4, stride=1, padding=1, dilation=1
    if (kernel_size == 4 && stride == 1 && padding == 1 && dilation == 1) {
        // Dynamic shared memory for input tile
        extern __shared__ float smem_dynamic[];
        const int SHM_EXTRA = 3;  // kernel_size - 1
        const int SHM_HEIGHT = 32 + SHM_EXTRA;
        const int SHM_WIDTH = 32 + SHM_EXTRA;
        const int SHM_SIZE = SHM_HEIGHT * SHM_WIDTH;
        
        // Block and thread indexing
        const int n = blockIdx.z / channels;
        const int c = blockIdx.z % channels;
        const int tile_start_ph = blockIdx.y * 32;  // output tile height=32
        const int tile_start_pw = blockIdx.x * 32;  // output tile width=32
        const int ty = threadIdx.y;
        const int tx = threadIdx.x;
        
        // Global memory offsets
        const float* input_plane = input + n * channels * height * width + c * height * width;
        float* output_plane = output + n * channels * pooled_height * pooled_width + c * pooled_height * pooled_width;
        
        // Input tile start coordinates
        const int input_start_h = tile_start_ph - padding;
        const int input_start_w = tile_start_pw - padding;
        
        // Precompute boundary masks for the entire tile
        const int tile_end_h = input_start_h + SHM_HEIGHT;
        const int tile_end_w = input_start_w + SHM_WIDTH;
        
        // Cooperative shared memory loading with predicated boundary handling
        const int tid = ty * blockDim.x + tx;
        const int threads_per_block = blockDim.x * blockDim.y;
        for (int idx = tid; idx < SHM_SIZE; idx += threads_per_block) {
            const int h_in_tile = idx / SHM_WIDTH;
            const int w_in_tile = idx % SHM_WIDTH;
            const int h_global = input_start_h + h_in_tile;
            const int w_global = input_start_w + w_in_tile;
            
            // Predicated execution: load if within bounds, otherwise use -FLT_MAX
            float val = -FLT_MAX;
            if (h_global >= 0 && h_global < height && w_global >= 0 && w_global < width) {
                val = input_plane[h_global * width + w_global];
            }
            smem_dynamic[idx] = val;
        }
        __syncthreads();
        
        // Each thread computes 2x2 output elements
        const int ph_local = ty * 2;
        const int pw_local = tx * 2;
        
        // Load 5x5 neighborhood into registers for 2x2 outputs
        float reg_tile[25];
        const int base_idx = ph_local * SHM_WIDTH + pw_local;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                reg_tile[i*5 + j] = smem_dynamic[base_idx + i * SHM_WIDTH + j];
            }
        }
        
        // Compute max for all 4 output regions using predicated max operations
        // No boundary checks needed since shared memory contains padded values
        
        // Top-left output (rows 0-3, cols 0-3)
        float max00 = reg_tile[0];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                max00 = fmaxf(max00, reg_tile[i*5 + j]);
            }
        }
        
        // Top-right output (rows 0-3, cols 1-4)
        float max01 = reg_tile[1];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 1; j < 5; j++) {
                max01 = fmaxf(max01, reg_tile[i*5 + j]);
            }
        }
        
        // Bottom-left output (rows 1-4, cols 0-3)
        float max10 = reg_tile[5];
        #pragma unroll
        for (int i = 1; i < 5; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                max10 = fmaxf(max10, reg_tile[i*5 + j]);
            }
        }
        
        // Bottom-right output (rows 1-4, cols 1-4)
        float max11 = reg_tile[6];
        #pragma unroll
        for (int i = 1; i < 5; i++) {
            #pragma unroll
            for (int j = 1; j < 5; j++) {
                max11 = fmaxf(max11, reg_tile[i*5 + j]);
            }
        }
        
        // Precompute global output positions
        const int ph_global0 = tile_start_ph + ph_local;
        const int pw_global0 = tile_start_pw + pw_local;
        const int ph_global1 = ph_global0 + 1;
        const int pw_global1 = pw_global0 + 1;
        
        // Write results with boundary checks (only needed for output boundaries)
        // Use predicated stores to avoid branching
        if (ph_global0 < pooled_height) {
            if (pw_global0 < pooled_width) {
                output_plane[ph_global0 * pooled_width + pw_global0] = max00;
            }
            if (pw_global1 < pooled_width) {
                output_plane[ph_global0 * pooled_width + pw_global1] = max01;
            }
        }
        if (ph_global1 < pooled_height) {
            if (pw_global0 < pooled_width) {
                output_plane[ph_global1 * pooled_width + pw_global0] = max10;
            }
            if (pw_global1 < pooled_width) {
                output_plane[ph_global1 * pooled_width + pw_global1] = max11;
            }
        }
    }
    // Original optimized path for kernel_size=4 (non (1,1,1) parameters)
    else if (kernel_size == 4) {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= batch_size * channels * pooled_height * pooled_width) return;
        
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / (pooled_width * pooled_height)) % channels;
        const int n = index / (pooled_width * pooled_height * channels);
        
        const int h_start = ph * stride - padding;
        const int w_start = pw * stride - padding;
        const int base = n * channels * height * width + c * height * width;
        
        // Precompute boundary conditions for predicated execution
        const bool h_valid[4] = {
            (h_start + 0 * dilation) >= 0 && (h_start + 0 * dilation) < height,
            (h_start + 1 * dilation) >= 0 && (h_start + 1 * dilation) < height,
            (h_start + 2 * dilation) >= 0 && (h_start + 2 * dilation) < height,
            (h_start + 3 * dilation) >= 0 && (h_start + 3 * dilation) < height
        };
        
        const bool w_valid[4] = {
            (w_start + 0 * dilation) >= 0 && (w_start + 0 * dilation) < width,
            (w_start + 1 * dilation) >= 0 && (w_start + 1 * dilation) < width,
            (w_start + 2 * dilation) >= 0 && (w_start + 2 * dilation) < width,
            (w_start + 3 * dilation) >= 0 && (w_start + 3 * dilation) < width
        };
        
        float max_val = -FLT_MAX;
        
        #pragma unroll
        for (int kh = 0; kh < 4; kh++) {
            const int h = h_start + kh * dilation;
            #pragma unroll
            for (int kw = 0; kw < 4; kw++) {
                const int w = w_start + kw * dilation;
                // Predicated execution: condition is evaluated once per iteration
                if (h_valid[kh] && w_valid[kw]) {
                    const float val = input[base + h * width + w];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        output[index] = max_val;
    }
    // Fallback for other kernel sizes
    else {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= batch_size * channels * pooled_height * pooled_width) return;
        
        const int pw = index % pooled_width;
        const int ph = (index / pooled_width) % pooled_height;
        const int c = (index / (pooled_width * pooled_height)) % channels;
        const int n = index / (pooled_width * pooled_height * channels);
        
        const int h_start = ph * stride - padding;
        const int w_start = pw * stride - padding;
        const int h_end = min(h_start + (kernel_size - 1) * dilation + 1, height);
        const int w_end = min(w_start + (kernel_size - 1) * dilation + 1, width);
        const int h_start_clamped = max(h_start, 0);
        const int w_start_clamped = max(w_start, 0);
        const int base = n * channels * height * width + c * height * width;
        
        float max_val = -FLT_MAX;
        for (int h = h_start_clamped; h < h_end; h += dilation) {
            for (int w = w_start_clamped; w < w_end; w += dilation) {
                const float val = input[base + h * width + w];
                if (val > max_val) max_val = val;
            }
        }
        output[index] = max_val;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor max_pool2d_forward_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Get tensor dimensions
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    // Calculate output dimensions
    const int pooled_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int pooled_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int total_elements = batch_size * channels * pooled_height * pooled_width;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, channels, pooled_height, pooled_width}, input.options());
    
    // Specialized launch configuration for kernel_size=4, stride=1, padding=1, dilation=1
    if (kernel_size == 4 && stride == 1 && padding == 1 && dilation == 1 && total_elements > 0) {
        // Constants for shared memory
        const int SHM_EXTRA = 3;  // kernel_size - 1
        const int SHM_SIZE = (32 + SHM_EXTRA) * (32 + SHM_EXTRA) * sizeof(float);
        
        // Grid dimensions (ceil division) - optimized for 511x511 output
        const int grid_x = (pooled_width + 31) / 32;  // 16 tiles for 511 width
        const int grid_y = (pooled_height + 31) / 32; // 16 tiles for 511 height
        const int grid_z = batch_size * channels;
        
        // Block dimensions (16x16 threads per block)
        dim3 block_dim(16, 16);
        dim3 grid_dim(grid_x, grid_y, grid_z);
        
        // Launch kernel with dynamic shared memory
        max_pool2d_forward_kernel<<<grid_dim, block_dim, SHM_SIZE>>>(
            input.contiguous().data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, height, width,
            pooled_height, pooled_width,
            kernel_size, stride, padding, dilation
        );
    }
    // General case launch configuration
    else if (total_elements > 0) {
        const int block_size = 256;
        const int grid_size = (total_elements + block_size - 1) / block_size;
        
        max_pool2d_forward_kernel<<<grid_size, block_size>>>(
            input.contiguous().data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels, height, width,
            pooled_height, pooled_width,
            kernel_size, stride, padding, dilation
        );
    }
    
    return output;
}
// PART-END