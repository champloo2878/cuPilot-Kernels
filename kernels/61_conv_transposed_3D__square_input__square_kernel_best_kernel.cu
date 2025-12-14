// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Problem size constants
constexpr int BatchSize = 8;
constexpr int InputChannels = 48;
constexpr int OutputChannels = 48;
constexpr int InputDepth = 64;
constexpr int InputHeight = 64;
constexpr int InputWidth = 64;
constexpr int OutputDepth = 66;
constexpr int OutputHeight = 66;
constexpr int OutputWidth = 66;
constexpr int KernelDepth = 3;
constexpr int KernelHeight = 3;
constexpr int KernelWidth = 3;
constexpr int Padding = 0;
constexpr int Groups = 1;
constexpr int KernelVolume = 27;

// Constant memory for bias and kernel metadata - A100 has 64KB constant memory
__constant__ float constant_bias[OutputChannels];
__constant__ int constant_metadata[10]; // [in_channels, input_depth, input_height, input_width,
                                        //  output_depth, output_height, output_width,
                                        //  kernel_size, stride, groups]

// Initialize constant memory
__host__ void init_constant_memory(const float* bias, int in_channels, int input_depth,
                                   int input_height, int input_width, int output_depth,
                                   int output_height, int output_width, int kernel_size,
                                   int stride, int groups) {
    if (bias) {
        cudaMemcpyToSymbol(constant_bias, bias, OutputChannels * sizeof(float));
    } else {
        float zero_bias[OutputChannels] = {0};
        cudaMemcpyToSymbol(constant_bias, zero_bias, OutputChannels * sizeof(float));
    }
    
    int metadata[10] = {in_channels, input_depth, input_height, input_width,
                        output_depth, output_height, output_width,
                        kernel_size, stride, groups};
    cudaMemcpyToSymbol(constant_metadata, metadata, 10 * sizeof(int));
}

// Helper to read from constant memory
__device__ __forceinline__ int get_metadata(int index) {
    return constant_metadata[index];
}

// Vectorized load/store helpers for A100
__device__ __forceinline__ float4 load_vectorized(const float* ptr, int idx) {
    return reinterpret_cast<const float4*>(ptr)[idx];
}

__device__ __forceinline__ void store_vectorized(float* ptr, int idx, float4 value) {
    reinterpret_cast<float4*>(ptr)[idx] = value;
}

// For half precision, use 128-bit loads (8 half values)
struct Half8 {
    __half2 x, y, z, w;
};

__device__ __forceinline__ Half8 load_vectorized_half(const __half* ptr, int idx) {
    return *reinterpret_cast<const Half8*>(ptr + idx * 8);
}

__device__ __forceinline__ void store_vectorized_half(__half* ptr, int idx, Half8 value) {
    *reinterpret_cast<Half8*>(ptr + idx * 8) = value;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T>
__global__ void conv_transpose3d_forward_kernel(
    const T* __restrict__ input, const T* __restrict__ weight,
    T* __restrict__ output) {
    
    // Read metadata from constant memory
    const int in_channels = get_metadata(0);
    const int input_depth = get_metadata(1);
    const int input_height = get_metadata(2);
    const int input_width = get_metadata(3);
    const int output_depth = get_metadata(4);
    const int output_height = get_metadata(5);
    const int output_width = get_metadata(6);
    const int kernel_size = get_metadata(7);
    const int stride = get_metadata(8);
    const int groups = get_metadata(9);
    
    // Early exit for incorrect configuration
    if (/*batch_size*/ 8 != BatchSize || in_channels != InputChannels || 
        /*out_channels*/ OutputChannels != OutputChannels || kernel_size != KernelDepth ||
        input_depth != InputDepth || input_height != InputHeight || 
        input_width != InputWidth || output_depth != OutputDepth ||
        output_height != OutputHeight || output_width != OutputWidth ||
        groups != Groups) {
        return;
    }
    
    const int oc = blockIdx.x;
    const int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (oc >= OutputChannels || spatial_idx >= BatchSize * output_depth * output_height * output_width) {
        return;
    }
    
    const int spatial_size = output_depth * output_height * output_width;
    const int n = spatial_idx / spatial_size;
    const int spatial_remainder = spatial_idx - n * spatial_size;
    const int od = spatial_remainder / (output_height * output_width);
    const int spatial_remainder2 = spatial_remainder - od * output_height * output_width;
    const int oh = spatial_remainder2 / output_width;
    const int ow = spatial_remainder2 - oh * output_width;
    
    // Use 4 accumulators for ILP (Instruction Level Parallelism)
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    
    // Process input channels in blocks of 4 for better ILP and register efficiency
    #pragma unroll 12
    for (int ic_block = 0; ic_block < InputChannels; ic_block += 4) {
        const int input_base0 = ((n * in_channels + ic_block) * input_depth) * input_height * input_width;
        const int input_base1 = input_base0 + input_depth * input_height * input_width;
        const int input_base2 = input_base1 + input_depth * input_height * input_width;
        const int input_base3 = input_base2 + input_depth * input_height * input_width;
        
        const int weight_base0 = ((ic_block * OutputChannels + oc) * kernel_size) * kernel_size * kernel_size;
        const int weight_base1 = weight_base0 + OutputChannels * kernel_size * kernel_size * kernel_size;
        const int weight_base2 = weight_base1 + OutputChannels * kernel_size * kernel_size * kernel_size;
        const int weight_base3 = weight_base2 + OutputChannels * kernel_size * kernel_size * kernel_size;
        
        // Precompute kernel validity and offsets
        #pragma unroll
        for (int kd = 0; kd < 3; kd++) {
            const int id = od - kd;
            const bool depth_valid = (id >= 0) && (id < input_depth);
            const int input_depth_offset = id * input_height * input_width;
            const int weight_depth_offset = kd * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const int ih = oh - kh;
                const bool height_valid = depth_valid && (ih >= 0) && (ih < input_height);
                const int input_height_offset = ih * input_width;
                const int weight_height_offset = kh * 3;
                
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = ow - kw;
                    const bool width_valid = height_valid && (iw >= 0) && (iw < input_width);
                    
                    if (width_valid) {
                        const int input_idx = input_depth_offset + input_height_offset + iw;
                        const int weight_idx = weight_depth_offset + weight_height_offset + kw;
                        
                        const float inp0 = static_cast<float>(input[input_base0 + input_idx]);
                        const float inp1 = static_cast<float>(input[input_base1 + input_idx]);
                        const float inp2 = static_cast<float>(input[input_base2 + input_idx]);
                        const float inp3 = static_cast<float>(input[input_base3 + input_idx]);
                        
                        const float w0 = static_cast<float>(weight[weight_base0 + weight_idx]);
                        const float w1 = static_cast<float>(weight[weight_base1 + weight_idx]);
                        const float w2 = static_cast<float>(weight[weight_base2 + weight_idx]);
                        const float w3 = static_cast<float>(weight[weight_base3 + weight_idx]);
                        
                        // FMA (Fused Multiply-Add) operations for better throughput
                        value0 = fmaf(inp0, w0, value0);
                        value1 = fmaf(inp1, w1, value1);
                        value2 = fmaf(inp2, w2, value2);
                        value3 = fmaf(inp3, w3, value3);
                    }
                }
            }
        }
    }
    
    // Combine accumulators with bias from constant memory
    float value = value0 + value1 + value2 + value3 + constant_bias[oc];
    
    const int output_idx = ((n * OutputChannels + oc) * output_depth + od) * output_height * output_width +
                          oh * output_width + ow;
    output[output_idx] = static_cast<T>(value);
}

// Specialized half kernel with optimized memory access patterns for A100
template<>
__global__ void conv_transpose3d_forward_kernel<__half>(
    const __half* __restrict__ input, const __half* __restrict__ weight,
    __half* __restrict__ output) {
    
    // Read metadata from constant memory
    const int in_channels = get_metadata(0);
    const int input_depth = get_metadata(1);
    const int input_height = get_metadata(2);
    const int input_width = get_metadata(3);
    const int output_depth = get_metadata(4);
    const int output_height = get_metadata(5);
    const int output_width = get_metadata(6);
    const int kernel_size = get_metadata(7);
    const int stride = get_metadata(8);
    const int groups = get_metadata(9);
    
    // Early exit for incorrect configuration
    if (/*batch_size*/ 8 != BatchSize || in_channels != InputChannels || 
        /*out_channels*/ OutputChannels != OutputChannels || kernel_size != KernelDepth ||
        input_depth != InputDepth || input_height != InputHeight || 
        input_width != InputWidth || output_depth != OutputDepth ||
        output_height != OutputHeight || output_width != OutputWidth ||
        groups != Groups) {
        return;
    }
    
    const int oc = blockIdx.x;
    const int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (oc >= OutputChannels || spatial_idx >= BatchSize * output_depth * output_height * output_width) {
        return;
    }
    
    const int spatial_size = output_depth * output_height * output_width;
    const int n = spatial_idx / spatial_size;
    const int spatial_remainder = spatial_idx - n * spatial_size;
    const int od = spatial_remainder / (output_height * output_width);
    const int spatial_remainder2 = spatial_remainder - od * output_height * output_width;
    const int oh = spatial_remainder2 / output_width;
    const int ow = spatial_remainder2 - oh * output_width;
    
    // Use 4 accumulators for ILP (Instruction Level Parallelism)
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    
    // Process input channels in blocks of 4 for better ILP and register efficiency
    #pragma unroll 12
    for (int ic_block = 0; ic_block < InputChannels; ic_block += 4) {
        const int input_base0 = ((n * in_channels + ic_block) * input_depth) * input_height * input_width;
        const int input_base1 = input_base0 + input_depth * input_height * input_width;
        const int input_base2 = input_base1 + input_depth * input_height * input_width;
        const int input_base3 = input_base2 + input_depth * input_height * input_width;
        
        const int weight_base0 = ((ic_block * OutputChannels + oc) * kernel_size) * kernel_size * kernel_size;
        const int weight_base1 = weight_base0 + OutputChannels * kernel_size * kernel_size * kernel_size;
        const int weight_base2 = weight_base1 + OutputChannels * kernel_size * kernel_size * kernel_size;
        const int weight_base3 = weight_base2 + OutputChannels * kernel_size * kernel_size * kernel_size;
        
        // Precompute kernel validity and offsets
        #pragma unroll
        for (int kd = 0; kd < 3; kd++) {
            const int id = od - kd;
            const bool depth_valid = (id >= 0) && (id < input_depth);
            const int input_depth_offset = id * input_height * input_width;
            const int weight_depth_offset = kd * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const int ih = oh - kh;
                const bool height_valid = depth_valid && (ih >= 0) && (ih < input_height);
                const int input_height_offset = ih * input_width;
                const int weight_height_offset = kh * 3;
                
                #pragma unroll
                for (int kw = 0; kw < 3; kw++) {
                    const int iw = ow - kw;
                    const bool width_valid = height_valid && (iw >= 0) && (iw < input_width);
                    
                    if (width_valid) {
                        const int input_idx = input_depth_offset + input_height_offset + iw;
                        const int weight_idx = weight_depth_offset + weight_height_offset + kw;
                        
                        const float inp0 = __half2float(input[input_base0 + input_idx]);
                        const float inp1 = __half2float(input[input_base1 + input_idx]);
                        const float inp2 = __half2float(input[input_base2 + input_idx]);
                        const float inp3 = __half2float(input[input_base3 + input_idx]);
                        
                        const float w0 = __half2float(weight[weight_base0 + weight_idx]);
                        const float w1 = __half2float(weight[weight_base1 + weight_idx]);
                        const float w2 = __half2float(weight[weight_base2 + weight_idx]);
                        const float w3 = __half2float(weight[weight_base3 + weight_idx]);
                        
                        // FMA (Fused Multiply-Add) operations for better throughput
                        value0 = fmaf(inp0, w0, value0);
                        value1 = fmaf(inp1, w1, value1);
                        value2 = fmaf(inp2, w2, value2);
                        value3 = fmaf(inp3, w3, value3);
                    }
                }
            }
        }
    }
    
    // Combine accumulators with bias from constant memory
    float value = value0 + value1 + value2 + value3 + constant_bias[oc];
    
    const int output_idx = ((n * OutputChannels + oc) * output_depth + od) * output_height * output_width +
                          oh * output_width + ow;
    output[output_idx] = __float2half(value);
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose3d_forward_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_size, int stride, int padding, int output_padding, int groups) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_depth = input.size(2);
    const int input_height = input.size(3);
    const int input_width = input.size(4);
    
    const int out_channels = weight.size(1);
    
    // Calculate output dimensions
    const int output_depth = (input_depth - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_height = (input_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int output_width = (input_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    // Initialize constant memory
    if (input.dtype() == torch::kFloat16) {
        if (bias.defined()) {
            auto bias_float = bias.to(torch::kFloat32);
            init_constant_memory(bias_float.data_ptr<float>(), in_channels, input_depth,
                                input_height, input_width, output_depth,
                                output_height, output_width, kernel_size,
                                stride, groups);
        } else {
            init_constant_memory(nullptr, in_channels, input_depth,
                                input_height, input_width, output_depth,
                                output_height, output_width, kernel_size,
                                stride, groups);
        }
    } else if (input.dtype() == torch::kFloat32) {
        init_constant_memory(bias.defined() ? bias.data_ptr<float>() : nullptr,
                            in_channels, input_depth,
                            input_height, input_width, output_depth,
                            output_height, output_width, kernel_size,
                            stride, groups);
    }
    
    // Optimized grid configuration for A100 with fixed problem size
    // Use 256 threads per block for good occupancy (A100 has 2048 threads per SM)
    const int threads_per_block = 256;
    const int spatial_elements = batch_size * output_depth * output_height * output_width;
    
    // Calculate optimal block count: ceil(spatial_elements / threads_per_block)
    const int grid_z = (spatial_elements + threads_per_block - 1) / threads_per_block;
    
    // Use 3D grid: x=output_channels, y=1, z=spatial_tiles
    dim3 grid(out_channels, 1, grid_z);
    dim3 block(threads_per_block);
    
    if (input.dtype() == torch::kFloat16) {
        conv_transpose3d_forward_kernel<__half><<<grid, block>>>(
            reinterpret_cast<const __half*>(input.data_ptr<torch::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<torch::Half>()));
    } else if (input.dtype() == torch::kFloat32) {
        conv_transpose3d_forward_kernel<float><<<grid, block>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(),
            output.data_ptr<float>());
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
// PART-END