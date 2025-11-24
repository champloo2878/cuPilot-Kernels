#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

template<int GROUP_SIZE>
__device__ __forceinline__ void load_weights(float* shmem, float* weights, int k_offset) {
    if constexpr (GROUP_SIZE == 16) {
        *reinterpret_cast<float4*>(weights) = *reinterpret_cast<float4*>(shmem + k_offset);
        *reinterpret_cast<float4*>(weights + 4) = *reinterpret_cast<float4*>(shmem + k_offset + 4);
        *reinterpret_cast<float4*>(weights + 8) = *reinterpret_cast<float4*>(shmem + k_offset + 8);
        *reinterpret_cast<float4*>(weights + 12) = *reinterpret_cast<float4*>(shmem + k_offset + 12);
    } else if constexpr (GROUP_SIZE == 32) {
        *reinterpret_cast<float4*>(weights) = *reinterpret_cast<float4*>(shmem + k_offset);
        *reinterpret_cast<float4*>(weights + 4) = *reinterpret_cast<float4*>(shmem + k_offset + 4);
        *reinterpret_cast<float4*>(weights + 8) = *reinterpret_cast<float4*>(shmem + k_offset + 8);
        *reinterpret_cast<float4*>(weights + 12) = *reinterpret_cast<float4*>(shmem + k_offset + 12);
        *reinterpret_cast<float4*>(weights + 16) = *reinterpret_cast<float4*>(shmem + k_offset + 16);
        *reinterpret_cast<float4*>(weights + 20) = *reinterpret_cast<float4*>(shmem + k_offset + 20);
        *reinterpret_cast<float4*>(weights + 24) = *reinterpret_cast<float4*>(shmem + k_offset + 24);
        *reinterpret_cast<float4*>(weights + 28) = *reinterpret_cast<float4*>(shmem + k_offset + 28);
    } else {
        for (int c = 0; c < GROUP_SIZE; c++) {
            weights[c] = shmem[k_offset + c];
        }
    }
}

//PART-START conv_transpose1d_kernel
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int kernel_size,
    int input_length,
    int output_length,
    int stride,
    int padding,
    int dilation
) {
    extern __shared__ float dynamic_shared[];
    const int TILE_SIZE = 128;
    const int GROUP_SIZE = 32;
    const int IN_CHANNELS = 32;
    const int OUT_CHANNELS = 64;
    const int KERNEL_SIZE = 3;
    const int HALO = 1;
    const int TILES_PER_BLOCK = 8;
    
    float* s_input_buffers = dynamic_shared;
    float* s_weights = dynamic_shared + 2 * IN_CHANNELS * (TILE_SIZE + 2);
    
    int b = blockIdx.x;
    int c_out_group = blockIdx.y;
    int block_tile_start = blockIdx.z * TILES_PER_BLOCK;
    int thread_id = threadIdx.x;
    int size_per_channel = TILE_SIZE + 2;
    int total_size = IN_CHANNELS * size_per_channel;
    
    for (int idx = thread_id; idx < IN_CHANNELS * KERNEL_SIZE * GROUP_SIZE; idx += blockDim.x) {
        int k = idx / (IN_CHANNELS * GROUP_SIZE);
        int rem = idx % (IN_CHANNELS * GROUP_SIZE);
        int c_in = rem / GROUP_SIZE;
        int c_offset = rem % GROUP_SIZE;
        int c_out = c_out_group * GROUP_SIZE + c_offset;
        if (c_out < OUT_CHANNELS) {
            s_weights[k * (IN_CHANNELS * GROUP_SIZE) + c_in * GROUP_SIZE + c_offset] = 
                weight[c_in * OUT_CHANNELS * KERNEL_SIZE + c_out * KERNEL_SIZE + k];
        } else {
            s_weights[k * (IN_CHANNELS * GROUP_SIZE) + c_in * GROUP_SIZE + c_offset] = 0.0f;
        }
    }
    __syncthreads();
    
    int first_global_tile = block_tile_start;
    int first_tile_start = first_global_tile * TILE_SIZE;
    for (int idx = thread_id; idx < total_size; idx += blockDim.x) {
        int c_in = idx / size_per_channel;
        int spatial_offset = idx % size_per_channel;
        int l_in = first_tile_start - HALO + spatial_offset;
        float* dst = &s_input_buffers[c_in * size_per_channel + spatial_offset];
        if (l_in >= 0 && l_in < input_length) {
            const float* src = &input[b * IN_CHANNELS * input_length + c_in * input_length + l_in];
            __pipeline_memcpy_async(dst, src, sizeof(float));
        } else {
            *dst = 0.0f;
        }
    }
    __pipeline_commit();
    
    for (int i = 0; i < TILES_PER_BLOCK; i++) {
        int global_tile_index = block_tile_start + i;
        int tile_start = global_tile_index * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, output_length / 2 + (output_length % 2));
        int l_out_index = tile_start + thread_id;
        
        __pipeline_wait_prior(0);
        __syncthreads();
        
        float* curr_buffer = s_input_buffers + (i % 2) * total_size;
        
        if (i < TILES_PER_BLOCK - 1) {
            int next_global_tile = global_tile_index + 1;
            int next_tile_start = next_global_tile * TILE_SIZE;
            float* next_buffer = s_input_buffers + ((i+1) % 2) * total_size;
            
            for (int idx = thread_id; idx < total_size; idx += blockDim.x) {
                int c_in = idx / size_per_channel;
                int spatial_offset = idx % size_per_channel;
                int l_in = next_tile_start - HALO + spatial_offset;
                float* dst = &next_buffer[c_in * size_per_channel + spatial_offset];
                if (l_in >= 0 && l_in < input_length) {
                    const float* src = &input[b * IN_CHANNELS * input_length + c_in * input_length + l_in];
                    __pipeline_memcpy_async(dst, src, sizeof(float));
                } else {
                    *dst = 0.0f;
                }
            }
            __pipeline_commit();
        }
        
        if (l_out_index < tile_end) {
            float acc[GROUP_SIZE] = {0.0f};
            int spatial_base = thread_id;
            
            #pragma unroll 8
            for (int c_in = 0; c_in < IN_CHANNELS; c_in++) {
                float in0 = curr_buffer[c_in * size_per_channel + spatial_base + 2];
                float in1 = curr_buffer[c_in * size_per_channel + spatial_base + 1];
                float in2 = curr_buffer[c_in * size_per_channel + spatial_base];
                
                float w0[GROUP_SIZE], w1[GROUP_SIZE], w2[GROUP_SIZE];
                float* w_base0 = s_weights + (0 * IN_CHANNELS + c_in) * GROUP_SIZE;
                float* w_base1 = s_weights + (1 * IN_CHANNELS + c_in) * GROUP_SIZE;
                float* w_base2 = s_weights + (2 * IN_CHANNELS + c_in) * GROUP_SIZE;
                load_weights<GROUP_SIZE>(w_base0, w0, 0);
                load_weights<GROUP_SIZE>(w_base1, w1, 0);
                load_weights<GROUP_SIZE>(w_base2, w2, 0);
                
                #pragma unroll
                for (int c = 0; c < GROUP_SIZE; c++) {
                    acc[c] = fmaf(in0, w0[c], acc[c]);
                    acc[c] = fmaf(in1, w1[c], acc[c]);
                    acc[c] = fmaf(in2, w2[c], acc[c]);
                }
            }
            
            int l_out = 2 * l_out_index + 1;
            int c_out_base = c_out_group * GROUP_SIZE;
            if (c_out_base < OUT_CHANNELS && l_out < output_length) {
                for (int c = 0; c < GROUP_SIZE; c++) {
                    if (c_out_base + c < OUT_CHANNELS) {
                        float val = acc[c];
                        if (bias) val += bias[c_out_base + c];
                        output[b * OUT_CHANNELS * output_length + (c_out_base + c) * output_length + l_out] = val;
                    }
                }
            }
        }
    }
}
//PART-END conv_transpose1d_kernel

//PART-START conv_transpose1d_cuda
torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);
    
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());
    
    auto input_ptr = input.contiguous().data_ptr<float>();
    auto weight_ptr = weight.contiguous().data_ptr<float>();
    auto bias_ptr = bias.defined() ? bias.contiguous().data_ptr<float>() : nullptr;
    auto output_ptr = output.data_ptr<float>();
    
    const int TILE_SIZE = 128;
    const int GROUP_SIZE = 32;
    const int TILES_PER_BLOCK = 8;
    int spatial_points = output_length / 2 + (output_length % 2);
    int n_tiles = (spatial_points + TILE_SIZE - 1) / TILE_SIZE;
    int block_tiles = (n_tiles + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;
    
    dim3 grid(batch_size, (out_channels + GROUP_SIZE - 1) / GROUP_SIZE, block_tiles);
    size_t shared_size = (2 * in_channels * (TILE_SIZE + 2) + in_channels * kernel_size * GROUP_SIZE) * sizeof(float);
    
    conv_transpose1d_kernel<<<grid, 256, shared_size>>>(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        input_length,
        output_length,
        stride,
        padding,
        dilation
    );
    
    return output;
}
//PART-END conv_transpose1d_cuda