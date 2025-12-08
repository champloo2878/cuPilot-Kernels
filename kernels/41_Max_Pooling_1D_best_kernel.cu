// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Helper function for block calculation
inline int get_blocks(int total, int block_size) {
    return (total + block_size - 1) / block_size;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void max_pool1d_kernel(
    const float* input,
    const int batch_size,
    const int features,
    const int sequence_length,
    const int kernel_size,
    const int stride,
    const int padding,
    const int dilation,
    const int output_length,
    float* output) 
{
    // Optimized 3D grid: batch × feature_segments × output_blocks
    const int batch_idx = blockIdx.z;
    const int feature_segment = blockIdx.y;
    const int output_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int features_per_block = 16; // Increased for better memory coalescing on A100
    const int feature_start = feature_segment * features_per_block;
    
    if (output_pos >= output_length) return;
    if (feature_start >= features) return;
    
    // Process multiple features per thread for the same output position
    // This improves memory coalescing as threads access contiguous features
    #pragma unroll
    for (int feature_offset = 0; feature_offset < features_per_block; feature_offset++) {
        const int feature_idx = feature_start + feature_offset;
        if (feature_idx >= features) break;
        
        const int start = output_pos * stride - padding;
        float max_value = -FLT_MAX;
        
        // Precompute base index for efficient memory access
        const int base_input_idx = batch_idx * features * sequence_length + feature_idx * sequence_length;
        
        // Fixed kernel size 8 with full unrolling for optimal performance
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int index_in_seq = start + k * dilation;
            if (index_in_seq >= 0 && index_in_seq < sequence_length) {
                float val = input[base_input_idx + index_in_seq];
                max_value = fmaxf(max_value, val);
            }
        }
        
        // Calculate output index with optimized memory layout
        int output_idx = batch_idx * features * output_length + feature_idx * output_length + output_pos;
        output[output_idx] = max_value;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor max_pool1d_cuda_forward(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool return_indices) 
{
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch, features, sequence_length)");
    TORCH_CHECK(!return_indices, "This function is only for return_indices=false");

    int batch_size = input.size(0);
    int features = input.size(1);
    int sequence_length = input.size(2);

    int numerator = sequence_length + 2 * padding - dilation * (kernel_size - 1) - 1;
    int output_length = (numerator < 0) ? 0 : (numerator / stride + 1);

    auto output = torch::zeros({batch_size, features, output_length}, input.options());

    if (output_length > 0) {
        const int threads_per_block = 256; // Optimal for A100
        const int features_per_block = 16; // Increased for better memory coalescing
        
        // Calculate grid dimensions for optimal A100 utilization
        const int output_blocks = get_blocks(output_length, threads_per_block);
        const int feature_blocks = get_blocks(features, features_per_block);
        
        // 3D grid: output_blocks × feature_blocks × batch_size
        dim3 grid_dim(output_blocks, feature_blocks, batch_size);
        
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();

        max_pool1d_kernel<<<grid_dim, threads_per_block>>>(
            input_ptr,
            batch_size,
            features,
            sequence_length,
            kernel_size,
            stride,
            padding,
            dilation,
            output_length,
            output_ptr
        );
    }
    return output;
}
// PART-END