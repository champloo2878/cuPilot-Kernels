// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use CUDA's built-in float4 type for vectorized memory operations
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void reverse_cumsum_kernel_optimized(const float* __restrict__ input, float* __restrict__ output, int num_slices, int slice_size) {
    int slice_idx = blockIdx.x;
    if (slice_idx >= num_slices) return;

    const float* slice_in = input + slice_idx * slice_size;
    float* slice_out = output + slice_idx * slice_size;

    // Only thread 0 in each block processes the slice (simplified approach)
    if(threadIdx.x == 0){
        // Use vectorized loads/stores for the main loop
        const int vector_size = 4;
        const int remainder = slice_size % vector_size;
        
        // Handle remainder elements first (scalar) - start from the end
        float acc = 0.0f;
        for(int i = slice_size - 1; i >= slice_size - remainder; i--) {
            acc += slice_in[i];
            slice_out[i] = acc;
        }
        
        // Vectorized processing for the main part
        for(int i = slice_size - remainder - vector_size; i >= 0; i -= vector_size) {
            // Load 4 elements using float4
            float4 in_vec = *reinterpret_cast<const float4*>(slice_in + i);
            
            // Perform reverse cumulative sum
            float4 out_vec;
            out_vec.w = in_vec.w + acc;
            out_vec.z = in_vec.z + out_vec.w;
            out_vec.y = in_vec.y + out_vec.z;
            out_vec.x = in_vec.x + out_vec.y;
            
            // Store result
            *reinterpret_cast<float4*>(slice_out + i) = out_vec;
            
            // Update accumulator for next iteration
            acc = out_vec.x;
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int64_t num_slices = 1;
    for (int i = 0; i < dim; ++i) {
        num_slices *= sizes[i];
    }
    int64_t slice_size = sizes[dim];
    for (int i = dim + 1; i < input.dim(); ++i) {
        num_slices *= sizes[i];
    }

    auto input_2d = input.contiguous().view({num_slices, slice_size});
    auto output = torch::empty_like(input_2d);

    if (input.numel() == 0) {
        return output.view(sizes);
    }

    // Optimized configuration for A100 with 32768x32768 tensor
    // Use one block per slice with 256 threads for better occupancy
    dim3 grid(num_slices);
    dim3 block(256);  // Increased block size for better GPU utilization

    reverse_cumsum_kernel_optimized<<<grid, block>>>(
        input_2d.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)num_slices,
        (int)slice_size);

    cudaDeviceSynchronize();

    return output.view(sizes);
}
// PART-END