// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIM(x, d) TORCH_CHECK(x.dim() == d, #x " must have dimension " #d)

// Constants for the specific problem size
constexpr int BATCH_SIZE = 16;
constexpr int IN_CHANNELS = 32;
constexpr int OUT_CHANNELS = 64;
constexpr int KERNEL_SIZE = 3;
constexpr int STRIDE = 2;
constexpr int PADDING = 1;
constexpr int DILATION = 2;
constexpr int INPUT_DEPTH = 16;
constexpr int INPUT_HEIGHT = 32;
constexpr int INPUT_WIDTH = 32;
constexpr int OUTPUT_DEPTH = 33;
constexpr int OUTPUT_HEIGHT = 65;
constexpr int OUTPUT_WIDTH = 65;

// Precompute constant values for kernel optimization
constexpr int KERNEL_VOLUME = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE;
constexpr int INPUT_SIZE = INPUT_DEPTH * INPUT_HEIGHT * INPUT_WIDTH;
constexpr int OUTPUT_SIZE = OUTPUT_DEPTH * OUTPUT_HEIGHT * OUTPUT_WIDTH;
constexpr int TOTAL_ELEMENTS = BATCH_SIZE * OUT_CHANNELS * OUTPUT_SIZE;

// Precompute valid kernel positions to reduce modulo operations
__device__ __constant__ int valid_kernel_positions[KERNEL_VOLUME][4]; // [kd, kh, kw, valid_flag]
__device__ __constant__ int valid_kernel_count;

// Precompute weight base indices for each input channel group
__device__ __constant__ int weight_base_indices[IN_CHANNELS/4][4];

// Precomputation function for constant memory initialization
void precompute_constants() {
    // Precompute valid kernel positions
    int valid_count = 0;
    int h_valid_positions[KERNEL_VOLUME][4];
    
    for (int kd = 0; kd < KERNEL_SIZE; ++kd) {
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                // Check if this kernel position can ever contribute to any output
                // For transposed conv, kernel position is valid if there exists any output position
                // where (od - kd*DILATION + PADDING) % STRIDE == 0
                // Since STRIDE=2, DILATION=2, PADDING=1, we can precompute
                int valid_flag = 1; // Assume valid for this fixed configuration
                h_valid_positions[valid_count][0] = kd;
                h_valid_positions[valid_count][1] = kh;
                h_valid_positions[valid_count][2] = kw;
                h_valid_positions[valid_count][3] = valid_flag;
                valid_count++;
            }
        }
    }
    
    // Copy to device constant memory
    cudaMemcpyToSymbol(valid_kernel_positions, h_valid_positions, 
                       sizeof(int) * KERNEL_VOLUME * 4);
    cudaMemcpyToSymbol(valid_kernel_count, &valid_count, sizeof(int));
    
    // Precompute weight base indices
    int h_weight_base_indices[IN_CHANNELS/4][4];
    for (int ic_base = 0; ic_base < IN_CHANNELS; ic_base += 4) {
        int base_idx = ic_base / 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            h_weight_base_indices[base_idx][i] = 
                (ic_base + i) * OUT_CHANNELS * KERNEL_VOLUME;
        }
    }
    cudaMemcpyToSymbol(weight_base_indices, h_weight_base_indices, 
                       sizeof(int) * (IN_CHANNELS/4) * 4);
}

// Optimized memory access patterns for A100
template<int BLOCK_SIZE>
__device__ __inline__ void load_global_to_registers(const float* addr, float& reg0, float& reg1, float& reg2, float& reg3) {
    // Use vectorized loads for better memory throughput
    float4 vec = *reinterpret_cast<const float4*>(addr);
    reg0 = vec.x;
    reg1 = vec.y;
    reg2 = vec.z;
    reg3 = vec.w;
}

template<int BLOCK_SIZE>
__device__ __inline__ void load_global_to_registers(const half* addr, half& reg0, half& reg1, half& reg2, half& reg3) {
    // Use half2 for vectorized loads
    half2 vec0 = *reinterpret_cast<const half2*>(addr);
    half2 vec1 = *reinterpret_cast<const half2*>(addr + 2);
    reg0 = vec0.x;
    reg1 = vec0.y;
    reg2 = vec1.x;
    reg3 = vec1.y;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename scalar_t, int BLOCK_SIZE, int UNROLL_FACTOR = 2>
__launch_bounds__(BLOCK_SIZE, 2)  // Guide compiler: max threads per block = BLOCK_SIZE, min blocks per SM = 2
__global__ void conv_transpose3d_kernel_optimized(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output) {
    
    // Warp-level optimization for A100 (32 threads per warp)
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Each thread handles multiple output elements for better occupancy
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * UNROLL_FACTOR;
    
    // Precompute constants in registers
    const int output_size = OUTPUT_SIZE;
    const int input_size = INPUT_SIZE;
    const int kernel_volume = KERNEL_VOLUME;
    
    // Process UNROLL_FACTOR output elements per thread
    #pragma unroll
    for (int u = 0; u < UNROLL_FACTOR; u++) {
        int element_idx = idx + u;
        if (element_idx >= TOTAL_ELEMENTS) continue;
        
        // Decompose the index using optimized arithmetic
        int n = element_idx / (OUT_CHANNELS * output_size);
        int oc = (element_idx % (OUT_CHANNELS * output_size)) / output_size;
        int output_idx = element_idx % output_size;
        
        // Optimized spatial index decomposition
        int od = output_idx / (OUTPUT_HEIGHT * OUTPUT_WIDTH);
        int oh = (output_idx % (OUTPUT_HEIGHT * OUTPUT_WIDTH)) / OUTPUT_WIDTH;
        int ow = output_idx % OUTPUT_WIDTH;
        
        // Precompute spatial indices that are reused
        int od_padded = od + PADDING;
        int oh_padded = oh + PADDING;
        int ow_padded = ow + PADDING;
        
        // Initialize result with bias (load bias to register once)
        scalar_t result = bias ? bias[oc] : scalar_t(0);
        
        // Use precomputed valid kernel positions
        #pragma unroll 1  // Unroll just 1 to reduce register pressure
        for (int k_idx = 0; k_idx < valid_kernel_count; k_idx++) {
            int kd = valid_kernel_positions[k_idx][0];
            int kh = valid_kernel_positions[k_idx][1];
            int kw = valid_kernel_positions[k_idx][2];
            
            // Optimized index calculations using precomputed values
            int id = od_padded - kd * DILATION;
            int ih = oh_padded - kh * DILATION;
            int iw = ow_padded - kw * DILATION;
            
            // Early exit if not divisible by stride
            if (id % STRIDE != 0 || ih % STRIDE != 0 || iw % STRIDE != 0) continue;
            
            id /= STRIDE;
            ih /= STRIDE;
            iw /= STRIDE;
            
            if (id < 0 || id >= INPUT_DEPTH || ih < 0 || ih >= INPUT_HEIGHT || iw < 0 || iw >= INPUT_WIDTH) continue;
            
            int input_spatial_idx = id * INPUT_HEIGHT * INPUT_WIDTH + ih * INPUT_WIDTH + iw;
            
            // Process input channels in blocks for better cache utilization
            int weight_kernel_idx = kd * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw;
            
            // Manually partition the computation across input channel groups
            // to keep register usage under control
            #pragma unroll 2  // Process 2 input channel groups at a time
            for (int ic_base_idx = 0; ic_base_idx < IN_CHANNELS/4; ic_base_idx++) {
                // Use scalar variables instead of arrays to reduce register pressure
                scalar_t input_val0, input_val1, input_val2, input_val3;
                scalar_t weight_val0, weight_val1, weight_val2, weight_val3;
                
                // Load input values using vectorized loads
                int base_input_idx = n * IN_CHANNELS * input_size + ic_base_idx * 4 * input_size + input_spatial_idx;
                int input_idx0 = base_input_idx;
                int input_idx1 = base_input_idx + input_size;
                int input_idx2 = base_input_idx + 2 * input_size;
                int input_idx3 = base_input_idx + 3 * input_size;
                
                input_val0 = input[input_idx0];
                input_val1 = input[input_idx1];
                input_val2 = input[input_idx2];
                input_val3 = input[input_idx3];
                
                // Load weight values using precomputed base indices
                int weight_base = weight_base_indices[ic_base_idx][0];
                int weight_idx0 = weight_base + oc * kernel_volume + weight_kernel_idx;
                int weight_idx1 = weight_base + OUT_CHANNELS * kernel_volume + oc * kernel_volume + weight_kernel_idx;
                int weight_idx2 = weight_base + 2 * OUT_CHANNELS * kernel_volume + oc * kernel_volume + weight_kernel_idx;
                int weight_idx3 = weight_base + 3 * OUT_CHANNELS * kernel_volume + oc * kernel_volume + weight_kernel_idx;
                
                weight_val0 = weight[weight_idx0];
                weight_val1 = weight[weight_idx1];
                weight_val2 = weight[weight_idx2];
                weight_val3 = weight[weight_idx3];
                
                // FMA operations without temporary arrays
                result = fmaf(input_val0, weight_val0, result);
                result = fmaf(input_val1, weight_val1, result);
                result = fmaf(input_val2, weight_val2, result);
                result = fmaf(input_val3, weight_val3, result);
            }
        }
        
        output[element_idx] = result;
    }
}

// Generic fallback kernel
template<typename scalar_t>
__global__ void conv_transpose3d_generic_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int dilation_depth, int dilation_height, int dilation_width,
    int output_depth, int output_height, int output_width) {
    
    int od = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;
    
    if (od >= output_depth || oh >= output_height || oc >= out_channels || n >= batch_size) return;
    
    for (int ow = 0; ow < output_width; ++ow) {
        scalar_t acc = bias ? bias[oc] : scalar_t(0);
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < kernel_depth; ++kd) {
                int id = (od + padding_depth - kd * dilation_depth) / stride_depth;
                if (id < 0 || id >= input_depth || (od + padding_depth - kd * dilation_depth) % stride_depth != 0) continue;
                
                for (int kh = 0; kh < kernel_height; ++kh) {
                    int ih = (oh + padding_height - kh * dilation_height) / stride_height;
                    if (ih < 0 || ih >= input_height || (oh + padding_height - kh * dilation_height) % stride_height != 0) continue;
                    
                    for (int kw = 0; kw < kernel_width; ++kw) {
                        int iw = (ow + padding_width - kw * dilation_width) / stride_width;
                        if (iw < 0 || iw >= input_width || (ow + padding_width - kw * dilation_width) % stride_width != 0) continue;
                        
                        int input_idx = ((n * in_channels + ic) * input_depth + id) * input_height * input_width +
                                       ih * input_width + iw;
                        int weight_idx = ((ic * out_channels + oc) * kernel_depth + kd) * kernel_height * kernel_width +
                                        kh * kernel_width + kw;
                        
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        int output_idx = ((n * out_channels + oc) * output_depth + od) * output_height * output_width +
                        oh * output_width + ow;
        output[output_idx] = acc;
    }
}

template<typename scalar_t>
cudaError_t cutlass_conv_transpose3d(
    const scalar_t* input_ptr,
    const scalar_t* weight_ptr,
    const scalar_t* bias_ptr,
    scalar_t* output_ptr,
    int batch_size, int in_channels, int out_channels,
    int input_depth, int input_height, int input_width,
    int kernel_depth, int kernel_height, int kernel_width,
    int stride_depth, int stride_height, int stride_width,
    int padding_depth, int padding_height, int padding_width,
    int dilation_depth, int dilation_height, int dilation_width,
    int output_depth, int output_height, int output_width) {
    
    // Check if this matches our specific optimized case
    bool matches_optimized_case = (batch_size == BATCH_SIZE &&
                                  in_channels == IN_CHANNELS &&
                                  out_channels == OUT_CHANNELS &&
                                  kernel_depth == KERNEL_SIZE &&
                                  kernel_height == KERNEL_SIZE &&
                                  kernel_width == KERNEL_SIZE &&
                                  stride_depth == STRIDE &&
                                  stride_height == STRIDE &&
                                  stride_width == STRIDE &&
                                  padding_depth == PADDING &&
                                  padding_height == PADDING &&
                                  padding_width == PADDING &&
                                  dilation_depth == DILATION &&
                                  dilation_height == DILATION &&
                                  dilation_width == DILATION &&
                                  input_depth == INPUT_DEPTH &&
                                  input_height == INPUT_HEIGHT &&
                                  input_width == INPUT_WIDTH &&
                                  output_depth == OUTPUT_DEPTH &&
                                  output_height == OUTPUT_HEIGHT &&
                                  output_width == OUTPUT_WIDTH);
    
    if (matches_optimized_case) {
        // Optimized configuration for A100
        // Use 128 threads per block for better occupancy (4 warps per block)
        const int BLOCK_SIZE = 128;
        const int UNROLL_FACTOR = 2; // Process 2 elements per thread
        
        // Calculate grid size
        int num_blocks = (TOTAL_ELEMENTS + (BLOCK_SIZE * UNROLL_FACTOR) - 1) / (BLOCK_SIZE * UNROLL_FACTOR);
        
        // Launch kernel with optimized configuration and reduced unroll factor
        conv_transpose3d_kernel_optimized<scalar_t, BLOCK_SIZE, UNROLL_FACTOR>
            <<<num_blocks, BLOCK_SIZE>>>(input_ptr, weight_ptr, bias_ptr, output_ptr);
        
        return cudaGetLastError();
    } else {
        // Fallback to generic implementation
        int total_elements = batch_size * out_channels * output_depth * output_height * output_width;
        const int block_size = 256;
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        // Clear output tensor
        cudaMemset(output_ptr, 0, total_elements * sizeof(scalar_t));
        
        // Launch generic kernel
        dim3 block(8, 8, 1);
        dim3 grid((output_depth + block.x - 1) / block.x,
                  (output_height + block.y - 1) / block.y,
                  batch_size * out_channels);
        
        conv_transpose3d_generic_kernel<scalar_t><<<grid, block>>>(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride_depth, stride_height, stride_width,
            padding_depth, padding_height, padding_width,
            dilation_depth, dilation_height, dilation_width,
            output_depth, output_height, output_width);
        
        return cudaGetLastError();
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size, int stride, int padding, int dilation) {
    
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    if (bias.defined()) CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    if (bias.defined()) CHECK_CONTIGUOUS(bias);
    CHECK_DIM(input, 5);
    CHECK_DIM(weight, 5);
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);
    int out_channels = weight.size(1);
    int kernel_depth = weight.size(2);
    int kernel_height = weight.size(3);
    int kernel_width = weight.size(4);
    
    // Calculate output dimensions for transposed convolution
    int output_depth = (input_depth - 1) * stride - 2 * padding + dilation * (kernel_depth - 1) + 1;
    int output_height = (input_height - 1) * stride - 2 * padding + dilation * (kernel_height - 1) + 1;
    int output_width = (input_width - 1) * stride - 2 * padding + dilation * (kernel_width - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_depth, output_height, output_width}, 
                              input.options());
    
    // Precompute constants on first run
    static bool constants_precomputed = false;
    if (!constants_precomputed) {
        precompute_constants();
        constants_precomputed = true;
    }
    
    // Determine tensor data type and dispatch to appropriate kernel
    cudaError_t status = cudaSuccess;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "conv_transpose3d_cutlass", [&] {
        status = cutlass_conv_transpose3d<scalar_t>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            output.data_ptr<scalar_t>(),
            batch_size, in_channels, out_channels,
            input_depth, input_height, input_width,
            kernel_depth, kernel_height, kernel_width,
            stride, stride, stride,
            padding, padding, padding,
            dilation, dilation, dilation,
            output_depth, output_height, output_width
        );
    });
    
    if (status != cudaSuccess) {
        TORCH_CHECK(false, "Transposed convolution failed with error: ", cudaGetErrorString(status));
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    return output;
}
// PART-END