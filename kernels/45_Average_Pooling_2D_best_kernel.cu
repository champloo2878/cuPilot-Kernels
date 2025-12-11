// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// Mixed precision accumulation specialization for FP16
template<int KERNEL_H, int KERNEL_W, int STRIDE_H, int STRIDE_W, int PAD_H, int PAD_W>
struct PoolingSpecializationFP16 {
    static __device__ __forceinline__ void compute_pool(
        const half* __restrict__ input, half* __restrict__ output,
        const int batch_size, const int channels,
        const int input_height, const int input_width,
        const int output_height, const int output_width,
        const int output_idx) {
        
        // Calculate output indices
        const int n = output_idx / (channels * output_height * output_width);
        const int c = (output_idx / (output_height * output_width)) % channels;
        const int h = (output_idx / output_width) % output_height;
        const int w = output_idx % output_width;
        
        // Since padding=0 and stride=11, window is always fully within bounds
        const int h_start = h * STRIDE_H;
        const int w_start = w * STRIDE_W;
        
        // Get base pointer for this batch and channel
        const int64_t base_offset = ((int64_t)n * channels + c) * input_height * input_width;
        const half* base_ptr = input + base_offset;
        
        float sum = 0.0f;
        
        // Manually unrolled loops for 11x11 kernel with vectorized loads
        #pragma unroll
        for (int i = 0; i < 11; ++i) {
            const int row = h_start + i;
            const int64_t row_offset = (int64_t)row * input_width + w_start;
            
            // Process in chunks of 8 for vectorized loads
            #pragma unroll
            for (int j = 0; j < 11; j += 8) {
                if (j + 7 < 11) {
                    half2 val1 = *reinterpret_cast<const half2*>(&base_ptr[row_offset + j]);
                    half2 val2 = *reinterpret_cast<const half2*>(&base_ptr[row_offset + j + 2]);
                    half2 val3 = *reinterpret_cast<const half2*>(&base_ptr[row_offset + j + 4]);
                    half2 val4 = *reinterpret_cast<const half2*>(&base_ptr[row_offset + j + 6]);
                    
                    sum += __half2float(val1.x) + __half2float(val1.y) + 
                           __half2float(val2.x) + __half2float(val2.y) +
                           __half2float(val3.x) + __half2float(val3.y) + 
                           __half2float(val4.x) + __half2float(val4.y);
                } else {
                    // Handle remaining elements
                    for (int k = j; k < 11; ++k) {
                        sum += __half2float(base_ptr[row_offset + k]);
                    }
                }
            }
        }
        
        // Average and convert back to FP16
        constexpr int pool_size = 121;
        output[output_idx] = __float2half(sum / static_cast<float>(pool_size));
    }
};

// Mixed precision accumulation specialization for BF16
template<int KERNEL_H, int KERNEL_W, int STRIDE_H, int STRIDE_W, int PAD_H, int PAD_W>
struct PoolingSpecializationBF16 {
    static __device__ __forceinline__ void compute_pool(
        const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
        const int batch_size, const int channels,
        const int input_height, const int input_width,
        const int output_height, const int output_width,
        const int output_idx) {
        
        // Calculate output indices
        const int n = output_idx / (channels * output_height * output_width);
        const int c = (output_idx / (output_height * output_width)) % channels;
        const int h = (output_idx / output_width) % output_height;
        const int w = output_idx % output_width;
        
        // Since padding=0 and stride=11, window is always fully within bounds
        const int h_start = h * STRIDE_H;
        const int w_start = w * STRIDE_W;
        
        // Get base pointer for this batch and channel
        const int64_t base_offset = ((int64_t)n * channels + c) * input_height * input_width;
        const __nv_bfloat16* base_ptr = input + base_offset;
        
        float sum = 0.0f;
        
        // Manually unrolled loops for 11x11 kernel with vectorized loads
        #pragma unroll
        for (int i = 0; i < 11; ++i) {
            const int row = h_start + i;
            const int64_t row_offset = (int64_t)row * input_width + w_start;
            
            // Process in chunks of 8 for vectorized loads
            #pragma unroll
            for (int j = 0; j < 11; j += 8) {
                if (j + 7 < 11) {
                    __nv_bfloat162 val1 = *reinterpret_cast<const __nv_bfloat162*>(&base_ptr[row_offset + j]);
                    __nv_bfloat162 val2 = *reinterpret_cast<const __nv_bfloat162*>(&base_ptr[row_offset + j + 2]);
                    __nv_bfloat162 val3 = *reinterpret_cast<const __nv_bfloat162*>(&base_ptr[row_offset + j + 4]);
                    __nv_bfloat162 val4 = *reinterpret_cast<const __nv_bfloat162*>(&base_ptr[row_offset + j + 6]);
                    
                    sum += __bfloat162float(val1.x) + __bfloat162float(val1.y) + 
                           __bfloat162float(val2.x) + __bfloat162float(val2.y) +
                           __bfloat162float(val3.x) + __bfloat162float(val3.y) + 
                           __bfloat162float(val4.x) + __bfloat162float(val4.y);
                } else {
                    // Handle remaining elements
                    for (int k = j; k < 11; ++k) {
                        sum += __bfloat162float(base_ptr[row_offset + k]);
                    }
                }
            }
        }
        
        // Average and convert back to BF16
        constexpr int pool_size = 121;
        output[output_idx] = __float2bfloat16(sum / static_cast<float>(pool_size));
    }
};

// Mixed precision accumulation specialization for FP32 and FP64
template<int KERNEL_H, int KERNEL_W, int STRIDE_H, int STRIDE_W, int PAD_H, int PAD_W>
struct PoolingSpecializationMixed {
    template<typename T, typename AccumT>
    static __device__ __forceinline__ void compute_pool(
        const T* __restrict__ input, T* __restrict__ output,
        const int batch_size, const int channels,
        const int input_height, const int input_width,
        const int output_height, const int output_width,
        const int output_idx) {
        
        // Calculate output indices
        const int n = output_idx / (channels * output_height * output_width);
        const int c = (output_idx / (output_height * output_width)) % channels;
        const int h = (output_idx / output_width) % output_height;
        const int w = output_idx % output_width;
        
        // Since padding=0 and stride=11, window is always fully within bounds
        const int h_start = h * STRIDE_H;
        const int w_start = w * STRIDE_W;
        
        // Get base pointer for this batch and channel
        const int64_t base_offset = ((int64_t)n * channels + c) * input_height * input_width;
        const T* base_ptr = input + base_offset;
        
        AccumT sum = AccumT(0);
        
        // Manually unrolled loops for 11x11 kernel
        #pragma unroll
        for (int i = 0; i < 11; ++i) {
            const int row = h_start + i;
            const int64_t row_offset = (int64_t)row * input_width + w_start;
            
            #pragma unroll
            for (int j = 0; j < 11; ++j) {
                // Convert to accumulation type (FP32) for higher precision
                if constexpr (std::is_same<T, half>::value) {
                    sum += __half2float(base_ptr[row_offset + j]);
                } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                    sum += __bfloat162float(base_ptr[row_offset + j]);
                } else {
                    sum += static_cast<AccumT>(base_ptr[row_offset + j]);
                }
            }
        }
        
        // Average and convert back to output type
        constexpr int pool_size = 121;
        AccumT avg = sum / static_cast<AccumT>(pool_size);
        
        if constexpr (std::is_same<T, half>::value) {
            output[output_idx] = __float2half(avg);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            output[output_idx] = __float2bfloat16(avg);
        } else {
            output[output_idx] = static_cast<T>(avg);
        }
    }
};

// Generic pooling implementation with mixed precision
template<typename T, typename AccumT>
__device__ __forceinline__ void generic_pool_compute_mixed(
    const T* __restrict__ input, T* __restrict__ output,
    const int batch_size, const int channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int output_idx) {
    
    // Calculate output indices
    const int n = output_idx / (channels * output_height * output_width);
    const int c = (output_idx / (output_height * output_width)) % channels;
    const int h = (output_idx / output_width) % output_height;
    const int w = output_idx % output_width;
    
    // Calculate input window start positions
    const int h_start = h * stride_h - pad_h;
    const int w_start = w * stride_w - pad_w;
    const int h_end = min(h_start + kernel_h, input_height);
    const int w_end = min(w_start + kernel_w, input_width);
    const int h_start_clamped = max(h_start, 0);
    const int w_start_clamped = max(w_start, 0);
    
    // Calculate number of valid elements
    const int pool_size = (h_end - h_start_clamped) * (w_end - w_start_clamped);
    if (pool_size == 0) {
        if constexpr (std::is_same<T, half>::value) {
            output[output_idx] = __float2half(0.0f);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
            output[output_idx] = __float2bfloat16(0.0f);
        } else {
            output[output_idx] = T(0);
        }
        return;
    }
    
    // Get base pointer for this batch and channel
    const int64_t base_offset = ((int64_t)n * channels + c) * input_height * input_width;
    const T* base_ptr = input + base_offset;
    
    AccumT sum = AccumT(0);
    
    // Partially unrolled loops for generic case
    for (int i = h_start_clamped; i < h_end; ++i) {
        const int64_t row_offset = (int64_t)i * input_width;
        
        // Unroll by 4 for better ILP
        int j = w_start_clamped;
        for (; j + 3 < w_end; j += 4) {
            if constexpr (std::is_same<T, half>::value) {
                sum += __half2float(base_ptr[row_offset + j]);
                sum += __half2float(base_ptr[row_offset + j + 1]);
                sum += __half2float(base_ptr[row_offset + j + 2]);
                sum += __half2float(base_ptr[row_offset + j + 3]);
            } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                sum += __bfloat162float(base_ptr[row_offset + j]);
                sum += __bfloat162float(base_ptr[row_offset + j + 1]);
                sum += __bfloat162float(base_ptr[row_offset + j + 2]);
                sum += __bfloat162float(base_ptr[row_offset + j + 3]);
            } else {
                sum += static_cast<AccumT>(base_ptr[row_offset + j]);
                sum += static_cast<AccumT>(base_ptr[row_offset + j + 1]);
                sum += static_cast<AccumT>(base_ptr[row_offset + j + 2]);
                sum += static_cast<AccumT>(base_ptr[row_offset + j + 3]);
            }
        }
        
        // Handle remaining elements
        for (; j < w_end; ++j) {
            if constexpr (std::is_same<T, half>::value) {
                sum += __half2float(base_ptr[row_offset + j]);
            } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
                sum += __bfloat162float(base_ptr[row_offset + j]);
            } else {
                sum += static_cast<AccumT>(base_ptr[row_offset + j]);
            }
        }
    }
    
    // Average and convert back to output type
    AccumT avg = sum / static_cast<AccumT>(pool_size);
    
    if constexpr (std::is_same<T, half>::value) {
        output[output_idx] = __float2half(avg);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        output[output_idx] = __float2bfloat16(avg);
    } else {
        output[output_idx] = static_cast<T>(avg);
    }
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T, typename AccumT>
__global__ void avg_pool2d_kernel_mixed_precision(
    const T* __restrict__ input, T* __restrict__ output,
    const int batch_size, const int channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;
    
    // Specialized path for 11x11 kernel with stride 11 and no padding
    if (kernel_h == 11 && kernel_w == 11 && 
        stride_h == 11 && stride_w == 11 && 
        pad_h == 0 && pad_w == 0) {
        
        PoolingSpecializationMixed<11, 11, 11, 11, 0, 0>::compute_pool<T, AccumT>(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            output_idx
        );
    } else {
        // Generic path for other configurations
        generic_pool_compute_mixed<T, AccumT>(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            output_idx
        );
    }
}

// Specialized kernel for FP16 with vectorized loads
__global__ void avg_pool2d_kernel_fp16(
    const half* __restrict__ input, half* __restrict__ output,
    const int batch_size, const int channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;
    
    // Specialized path for 11x11 kernel with stride 11 and no padding
    if (kernel_h == 11 && kernel_w == 11 && 
        stride_h == 11 && stride_w == 11 && 
        pad_h == 0 && pad_w == 0) {
        
        PoolingSpecializationFP16<11, 11, 11, 11, 0, 0>::compute_pool(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            output_idx
        );
    } else {
        // Generic path for other configurations
        generic_pool_compute_mixed<half, float>(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            output_idx
        );
    }
}

// Specialized kernel for BF16 with vectorized loads
__global__ void avg_pool2d_kernel_bf16(
    const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
    const int batch_size, const int channels,
    const int input_height, const int input_width,
    const int output_height, const int output_width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w) {
    
    const int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_idx >= batch_size * channels * output_height * output_width) return;
    
    // Specialized path for 11x11 kernel with stride 11 and no padding
    if (kernel_h == 11 && kernel_w == 11 && 
        stride_h == 11 && stride_w == 11 && 
        pad_h == 0 && pad_w == 0) {
        
        PoolingSpecializationBF16<11, 11, 11, 11, 0, 0>::compute_pool(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            output_idx
        );
    } else {
        // Generic path for other configurations
        generic_pool_compute_mixed<__nv_bfloat16, float>(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            output_idx
        );
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor avg_pool2d_cuda(torch::Tensor input, int kernel_size, int stride, int padding) {
    // Get input dimensions
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    
    // Calculate output dimensions
    const int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    // Create output tensor with same type as input
    auto output = torch::empty({batch_size, channels, output_height, output_width}, 
                              input.options());
    
    // Set kernel parameters
    const int kernel_h = kernel_size;
    const int kernel_w = kernel_size;
    const int stride_h = stride;
    const int stride_w = stride;
    const int pad_h = padding;
    const int pad_w = padding;
    
    // Calculate total number of output elements
    const int total_output_elements = batch_size * channels * output_height * output_width;
    
    // Optimized launch configuration for A100
    // Use 1024 threads per block for maximum occupancy on A100's 108 SMs
    const int threads_per_block = 1024;
    const int blocks_per_grid = (total_output_elements + threads_per_block - 1) / threads_per_block;
    
    // Dispatch based on data type with specialized optimizations
    if (input.scalar_type() == torch::kHalf) {
        avg_pool2d_kernel_fp16<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<half*>(input.data_ptr<torch::Half>()),
            reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w
        );
    } else if (input.scalar_type() == torch::kBFloat16) {
        avg_pool2d_kernel_bf16<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w
        );
    } else if (input.scalar_type() == torch::kFloat) {
        avg_pool2d_kernel_mixed_precision<float, float><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w
        );
    } else if (input.scalar_type() == torch::kDouble) {
        avg_pool2d_kernel_mixed_precision<double, double><<<blocks_per_grid, threads_per_block>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w
        );
    }
    
    return output;
}
// PART-END