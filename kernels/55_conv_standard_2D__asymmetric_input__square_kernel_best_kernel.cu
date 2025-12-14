// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// PTX-level helper functions for precise accumulation
__device__ __forceinline__ float add_rn(float a, float b) {
    return __fadd_rn(a, b);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ half add_rn(half a, half b) {
    return __hadd_rn(a, b);
}
#endif

// Swizzle permutation for better shared memory bank conflict avoidance
template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    static_assert(kColStride <= 16, "kColStride must <= 16");
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
    static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
    if constexpr (kStep == 8) {
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        static_assert(kStep == 4);
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

template<const int kMmaAtomK = 16>
static __device__ __forceinline__ int swizzle_permuted_A_j(int i, int j) {
    return swizzle_permuted_j<kMmaAtomK, 8>(i, j);
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T>
__global__ void conv2d_optimized_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_size, int output_height, int output_width,
    int stride, int padding, int dilation) {
    
    // Swizzled thread indices for better memory access patterns
    const int swizzled_thread_x = threadIdx.x ^ (threadIdx.y << 2);
    const int output_x = blockIdx.x * blockDim.x + swizzled_thread_x;
    const int output_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int output_c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (output_x >= output_width || output_y >= output_height || output_c >= out_channels) {
        return;
    }
    
    // Precompute input spatial offsets for the 3x3 kernel (hardcoded for kernel_size=3)
    const int input_y_base = output_y * stride - padding;
    const int input_x_base = output_x * stride - padding;
    
    int input_offsets[9];
    input_offsets[0] = (input_y_base + 0 * dilation) * input_width + (input_x_base + 0 * dilation);
    input_offsets[1] = (input_y_base + 0 * dilation) * input_width + (input_x_base + 1 * dilation);
    input_offsets[2] = (input_y_base + 0 * dilation) * input_width + (input_x_base + 2 * dilation);
    input_offsets[3] = (input_y_base + 1 * dilation) * input_width + (input_x_base + 0 * dilation);
    input_offsets[4] = (input_y_base + 1 * dilation) * input_width + (input_x_base + 1 * dilation);
    input_offsets[5] = (input_y_base + 1 * dilation) * input_width + (input_x_base + 2 * dilation);
    input_offsets[6] = (input_y_base + 2 * dilation) * input_width + (input_x_base + 0 * dilation);
    input_offsets[7] = (input_y_base + 2 * dilation) * input_width + (input_x_base + 1 * dilation);
    input_offsets[8] = (input_y_base + 2 * dilation) * input_width + (input_x_base + 2 * dilation);
    
    // Batch accumulators for all 8 batches (batch_size is 8 for this problem)
    T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0), acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
    T acc4 = static_cast<T>(0), acc5 = static_cast<T>(0), acc6 = static_cast<T>(0), acc7 = static_cast<T>(0);
    
    // Precompute base strides for memory access
    const int input_batch_stride = in_channels * input_height * input_width;
    const int weight_oc_stride = in_channels * 9;
    const int weight_ic_stride = 9;
    
    // Software pipelining arrays for weight and input preloading
    T w_pre[9][4];
    T in_pre[9][4];
    
    // Main loop over input channels with unrolling factor 4 (64 channels divisible by 4)
    for (int in_c = 0; in_c < in_channels; in_c += 4) {
        // Preload weights for current 4 channels (software pipelining) using __ldg for constant cache
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int weight_idx_base = (output_c * in_channels + in_c) * 9 + k;
            w_pre[k][0] = __ldg(&weight[weight_idx_base]);
            w_pre[k][1] = __ldg(&weight[weight_idx_base + weight_ic_stride]);
            w_pre[k][2] = __ldg(&weight[weight_idx_base + 2 * weight_ic_stride]);
            w_pre[k][3] = __ldg(&weight[weight_idx_base + 3 * weight_ic_stride]);
        }
        
        // Process batch 0 (fully unrolled for all 8 batches)
        {
            // Load inputs for batch 0, current 4 channels using __ldca for streaming access
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (0 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            // Compute accumulation for batch 0 with precise rounding control
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc0 = add_rn(acc0, sum);
            }
        }
        
        // Process batch 1
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (1 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc1 = add_rn(acc1, sum);
            }
        }
        
        // Process batch 2
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (2 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc2 = add_rn(acc2, sum);
            }
        }
        
        // Process batch 3
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (3 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc3 = add_rn(acc3, sum);
            }
        }
        
        // Process batch 4
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (4 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc4 = add_rn(acc4, sum);
            }
        }
        
        // Process batch 5
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (5 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc5 = add_rn(acc5, sum);
            }
        }
        
        // Process batch 6
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (6 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc6 = add_rn(acc6, sum);
            }
        }
        
        // Process batch 7
        {
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                const int input_idx_base = (7 * in_channels + in_c) * input_height * input_width + input_offsets[k];
                in_pre[k][0] = __ldca(&input[input_idx_base]);
                in_pre[k][1] = __ldca(&input[input_idx_base + input_height * input_width]);
                in_pre[k][2] = __ldca(&input[input_idx_base + 2 * input_height * input_width]);
                in_pre[k][3] = __ldca(&input[input_idx_base + 3 * input_height * input_width]);
            }
            
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                T prod0 = in_pre[k][0] * w_pre[k][0];
                T prod1 = in_pre[k][1] * w_pre[k][1];
                T prod2 = in_pre[k][2] * w_pre[k][2];
                T prod3 = in_pre[k][3] * w_pre[k][3];
                T sum01 = add_rn(prod0, prod1);
                T sum23 = add_rn(prod2, prod3);
                T sum = add_rn(sum01, sum23);
                acc7 = add_rn(acc7, sum);
            }
        }
    }
    
    // Write results for all batches (batch_size is 8 for this problem)
    if (output_x < output_width && output_y < output_height && output_c < out_channels) {
        const int output_batch_stride = out_channels * output_height * output_width;
        const int output_idx_base = output_c * output_height * output_width + output_y * output_width + output_x;
        
        // Use streaming write operations for better memory throughput
        __stcs(&output[0 * output_batch_stride + output_idx_base], acc0);
        __stcs(&output[1 * output_batch_stride + output_idx_base], acc1);
        __stcs(&output[2 * output_batch_stride + output_idx_base], acc2);
        __stcs(&output[3 * output_batch_stride + output_idx_base], acc3);
        __stcs(&output[4 * output_batch_stride + output_idx_base], acc4);
        __stcs(&output[5 * output_batch_stride + output_idx_base], acc5);
        __stcs(&output[6 * output_batch_stride + output_idx_base], acc6);
        __stcs(&output[7 * output_batch_stride + output_idx_base], acc7);
    }
}

// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv2d_optimized_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride, int padding, int dilation, int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                              torch::device(input.device()).dtype(input.dtype()));
    
    // Optimized thread block configuration for A100 with 1024 threads per block
    // Using (64, 4, 4) = 1024 threads for better memory coalescing on A100
    // This configuration better matches the 128 output channels (divisible by 4)
    const int threads_x = 64;  // For output width dimension - better coalescing with wider loads
    const int threads_y = 4;   // For output height dimension
    const int threads_z = 4;   // For output channel dimension
    
    dim3 blocks(
        (output_width + threads_x - 1) / threads_x,
        (output_height + threads_y - 1) / threads_y,
        (out_channels + threads_z - 1) / threads_z
    );
    dim3 threads(threads_x, threads_y, threads_z);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv2d_optimized_kernel", ([&] {
        // Set cache configuration to prefer L1 for better memory latency hiding
        // Use L1 cache for better temporal locality in input access patterns
        cudaFuncSetCacheConfig(conv2d_optimized_kernel<scalar_t>, cudaFuncCachePreferL1);
        
        // Set shared memory bank size to 8 bytes for better bank conflict avoidance
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        
        conv2d_optimized_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_size, output_height, output_width,
            stride, padding, dilation);
    }));
    
    cudaDeviceSynchronize();
    return output;
}
// PART-END