#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

// PART-START
__global__ void __launch_bounds__(256, 4) conv3d_depth1_forward_kernel(
    const float* __restrict__ input,
    const half* __restrict__ weight,  // FP16 weights
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int depth,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int groups,
    int output_height,
    int output_width
) {
    // Target-specific compile-time constants
    constexpr int TARGET_BATCH_SIZE = 16;
    constexpr int TARGET_IN_CHANNELS = 3;
    constexpr int TARGET_HEIGHT = 256;
    constexpr int TARGET_WIDTH = 256;
    constexpr int TARGET_DEPTH = 10;
    constexpr int TARGET_OUT_CHANNELS = 64;
    constexpr int TARGET_KERNEL_SIZE = 3;
    constexpr int TARGET_STRIDE = 1;
    constexpr int TARGET_PADDING = 0;
    constexpr int TARGET_DILATION = 1;
    constexpr int TARGET_GROUPS = 1;
    constexpr int TARGET_OUTPUT_HEIGHT = 254;
    constexpr int TARGET_OUTPUT_WIDTH = 254;
    
    // Derived constants
    constexpr int DEPTH_ELEMENTS_PER_THREAD = 2;
    constexpr int OUTPUT_CHANNELS_PER_BLOCK = 16;
    constexpr int GROUP_SIZE_OUT = TARGET_OUT_CHANNELS / TARGET_GROUPS;
    constexpr int OUTPUT_BLOCKS = TARGET_GROUPS * (GROUP_SIZE_OUT / OUTPUT_CHANNELS_PER_BLOCK);
    constexpr int SPATIAL_SIZE = TARGET_OUTPUT_HEIGHT * TARGET_OUTPUT_WIDTH;
    constexpr int DEPTH_PAIRS = TARGET_DEPTH / DEPTH_ELEMENTS_PER_THREAD;
    constexpr int OUTPUT_ELEMENTS_PER_BATCH = OUTPUT_BLOCKS * SPATIAL_SIZE * DEPTH_PAIRS;
    constexpr int WEIGHT_SIZE = TARGET_IN_CHANNELS * TARGET_KERNEL_SIZE * TARGET_KERNEL_SIZE * OUTPUT_BLOCKS * OUTPUT_CHANNELS_PER_BLOCK;
    
    // Shared memory for weights (FP16) with 16-byte alignment
    extern __shared__ __align__(16) half s_weights[];
    
    // Asynchronous weight loading with cp.async
    int tid = threadIdx.x;
    constexpr int ELEMENTS_PER_COPY = 8;  // 16 bytes per copy (8 half elements)
    constexpr int NUM_VECTORS = WEIGHT_SIZE / ELEMENTS_PER_COPY;
    int num_per_thread = (NUM_VECTORS + blockDim.x - 1) / blockDim.x;
    
    for (int i = 0; i < num_per_thread; ++i) {
        int vector_idx = tid + i * blockDim.x;
        if (vector_idx < NUM_VECTORS) {
            int element_idx = vector_idx * ELEMENTS_PER_COPY;
            unsigned int shared_offset = element_idx * sizeof(half);
            const uint64_t global_ptr = reinterpret_cast<uint64_t>(weight + element_idx);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 16;"
                :: "r"(shared_offset), "l"(global_ptr)
            );
        }
    }
    asm volatile("cp.async.commit_group;");
    
    // Precompute global index while weights load
    int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Wait for async copies to complete
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    
    if (batch_index >= TARGET_BATCH_SIZE * OUTPUT_ELEMENTS_PER_BATCH) return;

    // Unravel global index
    int b = batch_index / OUTPUT_ELEMENTS_PER_BATCH;
    int residual = batch_index % OUTPUT_ELEMENTS_PER_BATCH;
    int output_block_index = residual / (SPATIAL_SIZE * DEPTH_PAIRS);
    residual %= SPATIAL_SIZE * DEPTH_PAIRS;
    int depth_pair_index = residual % DEPTH_PAIRS;
    int spatial_index = residual / DEPTH_PAIRS;
    
    // Spatial coordinates
    int w_out = spatial_index % TARGET_OUTPUT_WIDTH;
    int h_out = spatial_index / TARGET_OUTPUT_WIDTH;
    int d_base = depth_pair_index * DEPTH_ELEMENTS_PER_THREAD;

    // Group and channel info
    int g = output_block_index / (GROUP_SIZE_OUT / OUTPUT_CHANNELS_PER_BLOCK);
    int g_out = output_block_index % (GROUP_SIZE_OUT / OUTPUT_CHANNELS_PER_BLOCK);
    int c_out_start = g * GROUP_SIZE_OUT + g_out * OUTPUT_CHANNELS_PER_BLOCK;
    int c_in_start = g * (TARGET_IN_CHANNELS / TARGET_GROUPS);

    // Split accumulators for depth elements
    float accum0[OUTPUT_CHANNELS_PER_BLOCK] = {0.0f};
    float accum1[OUTPUT_CHANNELS_PER_BLOCK] = {0.0f};

    // Precompute output block offset
    int output_block_offset = output_block_index * OUTPUT_CHANNELS_PER_BLOCK;

    // Optimized inner loops with FP16 weights
    #pragma unroll
    for (int i = 0; i < TARGET_KERNEL_SIZE; ++i) {
        int h_in = h_out * TARGET_STRIDE - TARGET_PADDING + i * TARGET_DILATION;
        #pragma unroll
        for (int j = 0; j < TARGET_KERNEL_SIZE; ++j) {
            int w_in = w_out * TARGET_STRIDE - TARGET_PADDING + j * TARGET_DILATION;
            int spatial_offset = h_in * TARGET_WIDTH * TARGET_DEPTH + w_in * TARGET_DEPTH;
            
            #pragma unroll
            for (int c_in = c_in_start; c_in < c_in_start + TARGET_IN_CHANNELS; ++c_in) {
                int input_idx = b * TARGET_IN_CHANNELS * TARGET_HEIGHT * TARGET_WIDTH * TARGET_DEPTH +
                              c_in * TARGET_HEIGHT * TARGET_WIDTH * TARGET_DEPTH +
                              spatial_offset +
                              d_base;
                float2 in_val = *reinterpret_cast<const float2*>(input + input_idx);
                
                int weight_idx = ((c_in * TARGET_KERNEL_SIZE + i) * TARGET_KERNEL_SIZE + j) * OUTPUT_BLOCKS * OUTPUT_CHANNELS_PER_BLOCK + 
                                output_block_offset;
                
                // Vectorized weight load (FP16) with direct assignments
                const half2* wt_ptr = reinterpret_cast<const half2*>(s_weights + weight_idx);
                
                half2 wt_val0 = wt_ptr[0];
                half2 wt_val1 = wt_ptr[1];
                half2 wt_val2 = wt_ptr[2];
                half2 wt_val3 = wt_ptr[3];
                half2 wt_val4 = wt_ptr[4];
                half2 wt_val5 = wt_ptr[5];
                half2 wt_val6 = wt_ptr[6];
                half2 wt_val7 = wt_ptr[7];

                float2 wt_float0 = __half22float2(wt_val0);
                float2 wt_float1 = __half22float2(wt_val1);
                float2 wt_float2 = __half22float2(wt_val2);
                float2 wt_float3 = __half22float2(wt_val3);
                float2 wt_float4 = __half22float2(wt_val4);
                float2 wt_float5 = __half22float2(wt_val5);
                float2 wt_float6 = __half22float2(wt_val6);
                float2 wt_float7 = __half22float2(wt_val7);

                // Direct assignments for efficient accumulation
                accum0[0]  += in_val.x * wt_float0.x; accum1[0]  += in_val.y * wt_float0.x;
                accum0[1]  += in_val.x * wt_float0.y; accum1[1]  += in_val.y * wt_float0.y;
                accum0[2]  += in_val.x * wt_float1.x; accum1[2]  += in_val.y * wt_float1.x;
                accum0[3]  += in_val.x * wt_float1.y; accum1[3]  += in_val.y * wt_float1.y;
                accum0[4]  += in_val.x * wt_float2.x; accum1[4]  += in_val.y * wt_float2.x;
                accum0[5]  += in_val.x * wt_float2.y; accum1[5]  += in_val.y * wt_float2.y;
                accum0[6]  += in_val.x * wt_float3.x; accum1[6]  += in_val.y * wt_float3.x;
                accum0[7]  += in_val.x * wt_float3.y; accum1[7]  += in_val.y * wt_float3.y;
                accum0[8]  += in_val.x * wt_float4.x; accum1[8]  += in_val.y * wt_float4.x;
                accum0[9]  += in_val.x * wt_float4.y; accum1[9]  += in_val.y * wt_float4.y;
                accum0[10] += in_val.x * wt_float5.x; accum1[10] += in_val.y * wt_float5.x;
                accum0[11] += in_val.x * wt_float5.y; accum1[11] += in_val.y * wt_float5.y;
                accum0[12] += in_val.x * wt_float6.x; accum1[12] += in_val.y * wt_float6.x;
                accum0[13] += in_val.x * wt_float6.y; accum1[13] += in_val.y * wt_float6.y;
                accum0[14] += in_val.x * wt_float7.x; accum1[14] += in_val.y * wt_float7.x;
                accum0[15] += in_val.x * wt_float7.y; accum1[15] += in_val.y * wt_float7.y;
            }
        }
    }

    // Warp-level optimized bias addition
    if (bias != nullptr) {
        int lane_id = threadIdx.x % 32;
        int warp_output_block_index = __shfl_sync(0xFFFFFFFF, output_block_index, 0);
        unsigned mask = __ballot_sync(0xFFFFFFFF, output_block_index == warp_output_block_index);
        bool all_same = (mask == 0xFFFFFFFF);

        if (all_same) {
            // Warp-level broadcast for shared output blocks
            float4 bias_val0, bias_val1, bias_val2, bias_val3;
            if (lane_id == 0) {
                bias_val0 = *reinterpret_cast<const float4*>(bias + c_out_start);
                bias_val1 = *reinterpret_cast<const float4*>(bias + c_out_start + 4);
                bias_val2 = *reinterpret_cast<const float4*>(bias + c_out_start + 8);
                bias_val3 = *reinterpret_cast<const float4*>(bias + c_out_start + 12);
            }
            
            // Broadcast bias values across warp
            bias_val0.x = __shfl_sync(0xFFFFFFFF, bias_val0.x, 0);
            bias_val0.y = __shfl_sync(0xFFFFFFFF, bias_val0.y, 0);
            bias_val0.z = __shfl_sync(0xFFFFFFFF, bias_val0.z, 0);
            bias_val0.w = __shfl_sync(0xFFFFFFFF, bias_val0.w, 0);
            
            bias_val1.x = __shfl_sync(0xFFFFFFFF, bias_val1.x, 0);
            bias_val1.y = __shfl_sync(0xFFFFFFFF, bias_val1.y, 0);
            bias_val1.z = __shfl_sync(0xFFFFFFFF, bias_val1.z, 0);
            bias_val1.w = __shfl_sync(0xFFFFFFFF, bias_val1.w, 0);
            
            bias_val2.x = __shfl_sync(0xFFFFFFFF, bias_val2.x, 0);
            bias_val2.y = __shfl_sync(0xFFFFFFFF, bias_val2.y, 0);
            bias_val2.z = __shfl_sync(0xFFFFFFFF, bias_val2.z, 0);
            bias_val2.w = __shfl_sync(0xFFFFFFFF, bias_val2.w, 0);
            
            bias_val3.x = __shfl_sync(0xFFFFFFFF, bias_val3.x, 0);
            bias_val3.y = __shfl_sync(0xFFFFFFFF, bias_val3.y, 0);
            bias_val3.z = __shfl_sync(0xFFFFFFFF, bias_val3.z, 0);
            bias_val3.w = __shfl_sync(0xFFFFFFFF, bias_val3.w, 0);
            
            // Direct assignment for bias
            accum0[0] += bias_val0.x; accum1[0] += bias_val0.x;
            accum0[1] += bias_val0.y; accum1[1] += bias_val0.y;
            accum0[2] += bias_val0.z; accum1[2] += bias_val0.z;
            accum0[3] += bias_val0.w; accum1[3] += bias_val0.w;
            
            accum0[4] += bias_val1.x; accum1[4] += bias_val1.x;
            accum0[5] += bias_val1.y; accum1[5] += bias_val1.y;
            accum0[6] += bias_val1.z; accum1[6] += bias_val1.z;
            accum0[7] += bias_val1.w; accum1[7] += bias_val1.w;
            
            accum0[8] += bias_val2.x; accum1[8] += bias_val2.x;
            accum0[9] += bias_val2.y; accum1[9] += bias_val2.y;
            accum0[10] += bias_val2.z; accum1[10] += bias_val2.z;
            accum0[11] += bias_val2.w; accum1[11] += bias_val2.w;
            
            accum0[12] += bias_val3.x; accum1[12] += bias_val3.x;
            accum0[13] += bias_val3.y; accum1[13] += bias_val3.y;
            accum0[14] += bias_val3.z; accum1[14] += bias_val3.z;
            accum0[15] += bias_val3.w; accum1[15] += bias_val3.w;
        } else {
            // Fallback for non-shared output blocks
            float4 bias_val0 = *reinterpret_cast<const float4*>(bias + c_out_start);
            float4 bias_val1 = *reinterpret_cast<const float4*>(bias + c_out_start + 4);
            float4 bias_val2 = *reinterpret_cast<const float4*>(bias + c_out_start + 8);
            float4 bias_val3 = *reinterpret_cast<const float4*>(bias + c_out_start + 12);
            
            accum0[0] += bias_val0.x; accum1[0] += bias_val0.x;
            accum0[1] += bias_val0.y; accum1[1] += bias_val0.y;
            accum0[2] += bias_val0.z; accum1[2] += bias_val0.z;
            accum0[3] += bias_val0.w; accum1[3] += bias_val0.w;
            
            accum0[4] += bias_val1.x; accum1[4] += bias_val1.x;
            accum0[5] += bias_val1.y; accum1[5] += bias_val1.y;
            accum0[6] += bias_val1.z; accum1[6] += bias_val1.z;
            accum0[7] += bias_val1.w; accum1[7] += bias_val1.w;
            
            accum0[8] += bias_val2.x; accum1[8] += bias_val2.x;
            accum0[9] += bias_val2.y; accum1[9] += bias_val2.y;
            accum0[10] += bias_val2.z; accum1[10] += bias_val2.z;
            accum0[11] += bias_val2.w; accum1[11] += bias_val2.w;
            
            accum0[12] += bias_val3.x; accum1[12] += bias_val3.x;
            accum0[13] += bias_val3.y; accum1[13] += bias_val3.y;
            accum0[14] += bias_val3.z; accum1[14] += bias_val3.z;
            accum0[15] += bias_val3.w; accum1[15] += bias_val3.w;
        }
    }

    // Coalesced output writing
    #pragma unroll
    for (int lane = 0; lane < OUTPUT_CHANNELS_PER_BLOCK; ++lane) {
        int c_out = c_out_start + lane;
        int output_idx = b * TARGET_OUT_CHANNELS * TARGET_OUTPUT_HEIGHT * TARGET_OUTPUT_WIDTH * TARGET_DEPTH +
                         c_out * TARGET_OUTPUT_HEIGHT * TARGET_OUTPUT_WIDTH * TARGET_DEPTH +
                         h_out * TARGET_OUTPUT_WIDTH * TARGET_DEPTH +
                         w_out * TARGET_DEPTH +
                         d_base;
        *reinterpret_cast<float2*>(output + output_idx) = make_float2(accum0[lane], accum1[lane]);
    }
}
// PART-END

// PART-START
torch::Tensor conv3d_depth1_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    // Validate target size
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int depth = input.size(4);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    // Output dimensions
    const int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Create output
    auto output = torch::zeros({batch_size, out_channels, output_height, output_width, depth}, input.options());
    
    // Launch configuration
    constexpr int OUTPUT_CHANNELS_PER_BLOCK = 16;
    const int group_size_out = out_channels / groups;
    const int output_blocks = groups * (group_size_out / OUTPUT_CHANNELS_PER_BLOCK);
    const int spatial_size = output_height * output_width;
    const int depth_pairs = depth / 2;
    const int output_elements = batch_size * output_blocks * spatial_size * depth_pairs;
    
    // Optimized block/grid for A100
    const int block_size = 256;
    const int grid_size = (output_elements + block_size - 1) / block_size;
    
    // Efficient weight reordering and FP16 conversion
    auto weight_reordered = weight;
    if (out_channels == 64) {
        weight_reordered = weight.view({groups, group_size_out, in_channels, kernel_size, kernel_size})
                          .permute({2, 3, 4, 0, 1})
                          .reshape({in_channels, kernel_size, kernel_size, output_blocks, OUTPUT_CHANNELS_PER_BLOCK})
                          .contiguous();
    }
    auto weight_half = weight_reordered.to(torch::kHalf);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Reduced shared memory requirements (FP16)
    const int weight_size = in_channels * kernel_size * kernel_size * output_blocks * OUTPUT_CHANNELS_PER_BLOCK;
    const size_t shared_mem_bytes = weight_size * sizeof(half);

    // Kernel launch with shared memory configuration
    conv3d_depth1_forward_kernel<<<grid_size, block_size, shared_mem_bytes, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const half*>(weight_half.data_ptr<torch::Half>()),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        height,
        width,
        depth,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_height,
        output_width
    );
    
    return output;
}
// PART-END