// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void reduce_mean_kernel(const float* input, float* output, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int total = N * H * W;
    
    // Reorganize thread indexing for better memory coalescing
    // Each thread handles consecutive elements in the spatial dimensions
    int tid = threadIdx.x;
    int threads_per_spatial = min(blockDim.x, H * W);
    int spatial_chunks = (H * W + threads_per_spatial - 1) / threads_per_spatial;
    
    float sum = 0.0f;
    
    // Process spatial dimensions with coalesced access pattern
    for (int spatial_idx = 0; spatial_idx < spatial_chunks; spatial_idx++) {
        int spatial_start = spatial_idx * threads_per_spatial;
        int spatial_pos = spatial_start + tid;
        
        if (spatial_pos < H * W) {
            int h = spatial_pos / W;
            int w = spatial_pos % W;
            
            // Process all batches for this spatial position
            for (int n = 0; n < N; n++) {
                int index = n * (C * H * W) + c * (H * W) + h * W + w;
                sum += input[index];
            }
        }
    }
    
    __shared__ float sdata[1024];
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction with better memory access pattern
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[c] = sdata[0] / total;
    }
}

__global__ void reduce_variance_kernel(const float* input, const float* mean, float* output, int N, int C, int H, int W) {
    int c = blockIdx.x;
    int total = N * H * W;
    float m = mean[c];
    
    // Reorganize thread indexing for better memory coalescing
    int tid = threadIdx.x;
    int threads_per_spatial = min(blockDim.x, H * W);
    int spatial_chunks = (H * W + threads_per_spatial - 1) / threads_per_spatial;
    
    float sum = 0.0f;
    
    // Process spatial dimensions with coalesced access pattern
    for (int spatial_idx = 0; spatial_idx < spatial_chunks; spatial_idx++) {
        int spatial_start = spatial_idx * threads_per_spatial;
        int spatial_pos = spatial_start + tid;
        
        if (spatial_pos < H * W) {
            int h = spatial_pos / W;
            int w = spatial_pos % W;
            
            // Process all batches for this spatial position
            for (int n = 0; n < N; n++) {
                int index = n * (C * H * W) + c * (H * W) + h * W + w;
                float diff = input[index] - m;
                sum += diff * diff;
            }
        }
    }
    
    __shared__ float sdata[1024];
    sdata[tid] = sum;
    __syncthreads();
    
    // Parallel reduction with better memory access pattern
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[c] = sdata[0] / total;
    }
}

__global__ void normalize_kernel(const float* input, const float* mean, const float* var, const float* gamma, const float* beta, float* output, int N, int C, int H, int W, float eps) {
    // Optimized indexing for better memory coalescing
    // Process elements in chunks where consecutive threads access consecutive memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    
    // Process multiple elements per thread to improve memory utilization
    const int elements_per_thread = 4;
    int total_threads = gridDim.x * blockDim.x;
    int total_work = (total_elements + elements_per_thread - 1) / elements_per_thread;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int global_idx = idx + i * total_threads;
        if (global_idx >= total_elements) break;
        
        int n = global_idx / (C * H * W);
        int rem = global_idx % (C * H * W);
        int c = rem / (H * W);
        rem = rem % (H * W);
        int h = rem / W;
        int w = rem % W;
        
        float m = mean[c];
        float v = var[c];
        float g = gamma[c];
        float b = beta[c];
        
        float inv_std = rsqrtf(v + eps);
        float x_hat = (input[global_idx] - m) * inv_std;
        output[global_idx] = g * x_hat + b;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor reduce_mean_cuda(torch::Tensor input) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::zeros({C}, input.options());
    
    // Optimized block size for A100 (1024 threads per block)
    dim3 grid(C);
    dim3 block(1024);  // Increased to utilize A100's 1024 threads per block
    reduce_mean_kernel<<<grid, block, block.x * sizeof(float)>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, H, W
    );
    return output;
}

torch::Tensor reduce_variance_cuda(torch::Tensor input, torch::Tensor mean) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::zeros({C}, input.options());
    
    // Optimized block size for A100
    dim3 grid(C);
    dim3 block(1024);  // Increased to utilize A100's 1024 threads per block
    reduce_variance_kernel<<<grid, block, block.x * sizeof(float)>>>(
        input.data_ptr<float>(), 
        mean.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, H, W
    );
    return output;
}

torch::Tensor normalize_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor var, torch::Tensor gamma, torch::Tensor beta, float eps) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::empty_like(input);
    
    int total = N * C * H * W;
    int block_size = 256;  // Optimal for memory coalescing
    int grid_size = (total + block_size * 4 - 1) / (block_size * 4);  // Adjusted for elements_per_thread
    
    normalize_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, eps
    );
    return output;
}
// PART-END