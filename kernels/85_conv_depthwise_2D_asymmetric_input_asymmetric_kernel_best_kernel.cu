// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory padding to avoid bank conflicts
template<int BLOCK_SIZE_X, int KERNEL_H, int KERNEL_W>
__device__ __forceinline__ int get_padded_shared_index(int row, int col) {
    const int PADDED_WIDTH = (BLOCK_SIZE_X + KERNEL_W - 1 + 31) & ~31; // Pad to multiple of 32 to avoid bank conflicts
    return row * PADDED_WIDTH + col;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* __restrict__ output, 
    int batch_size, 
    int in_channels, 
    int height, 
    int width, 
    int kernel_size_h, 
    int kernel_size_w, 
    int stride_h, 
    int stride_w, 
    int padding_h, 
    int padding_w, 
    int dilation_h, 
    int dilation_w, 
    int H_out, 
    int W_out, 
    int total_elements
) {
    const int TILE_H = BLOCK_SIZE_Y;
    const int TILE_W = BLOCK_SIZE_X;
    const int PADDED_WIDTH = (TILE_W + kernel_size_w - 1 + 31) & ~31;
    
    extern __shared__ float shared_mem[];
    float* input_tile = shared_mem;
    float* kernel_tile = shared_mem + (TILE_H + kernel_size_h - 1) * PADDED_WIDTH;
    
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int tile_h_start = blockIdx.x * TILE_H;
    int tile_w_start = 0;
    
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    
    // Load kernel into shared memory (small, so load once per block)
    if (threadIdx.x < kernel_size_w && threadIdx.y < kernel_size_h) {
        int kernel_idx = (channel * kernel_size_h + threadIdx.y) * kernel_size_w + threadIdx.x;
        kernel_tile[threadIdx.y * kernel_size_w + threadIdx.x] = kernel[kernel_idx];
    }
    
    __syncthreads();
    
    for (int w_tile = 0; w_tile < (W_out + TILE_W - 1) / TILE_W; w_tile++) {
        tile_w_start = w_tile * TILE_W;
        
        // Load input tile into shared memory with padding to avoid bank conflicts
        for (int load_h = local_h; load_h < TILE_H + kernel_size_h - 1; load_h += BLOCK_SIZE_Y) {
            for (int load_w = local_w; load_w < TILE_W + kernel_size_w - 1; load_w += BLOCK_SIZE_X) {
                int global_h = tile_h_start + load_h - padding_h;
                int global_w = tile_w_start + load_w - padding_w;
                
                float value = 0.0f;
                if (global_h >= 0 && global_h < height && global_w >= 0 && global_w < width) {
                    int input_idx = ((batch * in_channels + channel) * height + global_h) * width + global_w;
                    value = input[input_idx];
                }
                
                int shared_idx = get_padded_shared_index<BLOCK_SIZE_X, 3, 7>(load_h, load_w);
                input_tile[shared_idx] = value;
            }
        }
        
        __syncthreads();
        
        // Compute convolution for this tile
        int output_h = tile_h_start + local_h;
        int output_w = tile_w_start + local_w;
        
        if (output_h < H_out && output_w < W_out) {
            float result = 0.0f;
            
            // Unroll the inner loops for fixed kernel size 3x7
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                #pragma unroll
                for (int kw = 0; kw < 7; kw++) {
                    int input_h = local_h + kh;
                    int input_w = local_w + kw;
                    
                    int shared_idx = get_padded_shared_index<BLOCK_SIZE_X, 3, 7>(input_h, input_w);
                    float input_val = input_tile[shared_idx];
                    float kernel_val = kernel_tile[kh * kernel_size_w + kw];
                    
                    result += input_val * kernel_val;
                }
            }
            
            int output_idx = ((batch * in_channels + channel) * H_out + output_h) * W_out + output_w;
            output[output_idx] = result;
        }
        
        __syncthreads();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input, 
    torch::Tensor kernel, 
    int kernel_size_h, 
    int kernel_size_w, 
    int stride_h, 
    int stride_w, 
    int padding_h, 
    int padding_w, 
    int dilation_h, 
    int dilation_w
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    int H_out = (height + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int W_out = (width + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;

    auto output = torch::zeros({batch_size, in_channels, H_out, W_out}, input.options());

    int total_elements = batch_size * in_channels * H_out * W_out;

    if (total_elements == 0) {
        return output;
    }

    // Optimized block dimensions for better coalescing on A100
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 8;
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((H_out + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, in_channels, batch_size);
    
    // Calculate shared memory size with padding to avoid bank conflicts
    const int PADDED_WIDTH = (BLOCK_SIZE_X + kernel_size_w - 1 + 31) & ~31;
    size_t shared_mem_size = (BLOCK_SIZE_Y + kernel_size_h - 1) * PADDED_WIDTH * sizeof(float) + 
                            kernel_size_h * kernel_size_w * sizeof(float);

    auto kernel_flat = kernel.view({-1});

    // Launch the templated kernel with optimized block size
    depthwise_conv2d_kernel<BLOCK_SIZE_X, BLOCK_SIZE_Y><<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(),
        kernel_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        H_out,
        W_out,
        total_elements
    );

    return output;
}
// PART-END