#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_D 4
#define TILE_H 8
#define TILE_W 8
#define KERNEL_SIZE 3
#define INPUT_TILE_D (TILE_D + KERNEL_SIZE - 1)
#define INPUT_TILE_H (TILE_H + KERNEL_SIZE - 1)
#define INPUT_TILE_W_PADDED 12
#define BLOCK_OUT_CHANNELS 32
#define IN_CHANNELS 3
#define KERNEL_VOLUME 27

//PART-START part2
__global__ void conv3d_forward_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int depth, int height, int width,
    int kernel_size, int stride, int padding,
    int out_depth, int out_height, int out_width,
    int block_out_channel_blocks, int num_tiles
) {
    extern __shared__ float smem[];
    
    float* input_sm = smem;
    float* weight_sm = &smem[INPUT_TILE_D * in_channels * INPUT_TILE_H * INPUT_TILE_W_PADDED];
    float* bias_sm = &weight_sm[in_channels * KERNEL_VOLUME * BLOCK_OUT_CHANNELS];
    
    int total_blocks_per_batch = block_out_channel_blocks * num_tiles;
    int batch_idx = blockIdx.x / total_blocks_per_batch;
    int rem = blockIdx.x % total_blocks_per_batch;
    int out_channel_block = rem / num_tiles;
    int tile_index = rem % num_tiles;
    
    int num_tiles_w = (out_width + TILE_W - 1) / TILE_W;
    int num_tiles_h = (out_height + TILE_H - 1) / TILE_H;
    int tile_w = tile_index % num_tiles_w;
    int tile_h = (tile_index / num_tiles_w) % num_tiles_h;
    int tile_d = tile_index / (num_tiles_w * num_tiles_h);
    
    int tile_start_d = tile_d * TILE_D;
    int tile_start_h = tile_h * TILE_H;
    int tile_start_w = tile_w * TILE_W;
    
    int input_start_d = tile_start_d * stride - padding;
    int input_start_h = tile_start_h * stride - padding;
    int input_start_w = tile_start_w * stride - padding;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * TILE_H * TILE_W + ty * TILE_W + tx;
    int block_threads = blockDim.x * blockDim.y * blockDim.z;
    
    // Vectorized bias loading
    float4* bias_sm4 = reinterpret_cast<float4*>(bias_sm);
    int num_bias_vectors = BLOCK_OUT_CHANNELS / 4;
    if (tid < num_bias_vectors) {
        int global_channel_base = out_channel_block * BLOCK_OUT_CHANNELS + tid * 4;
        float4 bias4;
        bias4.x = (bias != nullptr && global_channel_base + 0 < out_channels) ? 
                  __ldg(bias + global_channel_base + 0) : 0.0f;
        bias4.y = (bias != nullptr && global_channel_base + 1 < out_channels) ? 
                  __ldg(bias + global_channel_base + 1) : 0.0f;
        bias4.z = (bias != nullptr && global_channel_base + 2 < out_channels) ? 
                  __ldg(bias + global_channel_base + 2) : 0.0f;
        bias4.w = (bias != nullptr && global_channel_base + 3 < out_channels) ? 
                  __ldg(bias + global_channel_base + 3) : 0.0f;
        bias_sm4[tid] = bias4;
    }
    
    // Vectorized input loading with depth-first memory coalescing
    int vectors_per_tile = INPUT_TILE_D * in_channels * INPUT_TILE_H * (INPUT_TILE_W_PADDED / 4);
    for (int idx_vector = tid; idx_vector < vectors_per_tile; idx_vector += block_threads) {
        int w0_index = idx_vector % (INPUT_TILE_W_PADDED / 4);
        int h_index = (idx_vector / (INPUT_TILE_W_PADDED / 4)) % INPUT_TILE_H;
        int d_index = (idx_vector / (INPUT_TILE_W_PADDED / 4 * INPUT_TILE_H)) % INPUT_TILE_D;
        int c_in_index = (idx_vector / (INPUT_TILE_W_PADDED / 4 * INPUT_TILE_H * INPUT_TILE_D)) % in_channels;
        
        int w0 = w0_index * 4;
        int d_abs = input_start_d + d_index;
        int h_abs = input_start_h + h_index;
        int w_abs_base = input_start_w + w0;
        
        float4 val4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (d_abs >= 0 && d_abs < depth && h_abs >= 0 && h_abs < height) {
            int global_base = ((batch_idx * in_channels + c_in_index) * depth + d_abs) * height * width + h_abs * width;
            
            if (w_abs_base >= 0 && w_abs_base + 3 < width) {
                val4 = *reinterpret_cast<const float4*>(input + global_base + w_abs_base);
            } else {
                if (w_abs_base + 0 >= 0 && w_abs_base + 0 < width) 
                    val4.x = __ldg(input + global_base + w_abs_base + 0);
                if (w_abs_base + 1 >= 0 && w_abs_base + 1 < width) 
                    val4.y = __ldg(input + global_base + w_abs_base + 1);
                if (w_abs_base + 2 >= 0 && w_abs_base + 2 < width) 
                    val4.z = __ldg(input + global_base + w_abs_base + 2);
                if (w_abs_base + 3 >= 0 && w_abs_base + 3 < width) 
                    val4.w = __ldg(input + global_base + w_abs_base + 3);
            }
        }
        
        int sm_index = d_index * in_channels * INPUT_TILE_H * INPUT_TILE_W_PADDED + 
                       c_in_index * INPUT_TILE_H * INPUT_TILE_W_PADDED + 
                       h_index * INPUT_TILE_W_PADDED + w0;
        *reinterpret_cast<float4*>(input_sm + sm_index) = val4;
    }
    
    // Vectorized weight loading
    float4* weight_sm4 = reinterpret_cast<float4*>(weight_sm);
    int num_weight_vectors = in_channels * KERNEL_VOLUME * (BLOCK_OUT_CHANNELS / 4);
    for (int idx_vector = tid; idx_vector < num_weight_vectors; idx_vector += block_threads) {
        int c_in = idx_vector / (KERNEL_VOLUME * (BLOCK_OUT_CHANNELS / 4));
        int rem = idx_vector % (KERNEL_VOLUME * (BLOCK_OUT_CHANNELS / 4));
        int k_index = rem / (BLOCK_OUT_CHANNELS / 4);
        int vector_index = rem % (BLOCK_OUT_CHANNELS / 4);
        
        int c_out_local_base = vector_index * 4;
        float4 w4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = 0; i < 4; i++) {
            int c_out_local = c_out_local_base + i;
            int global_out_channel = out_channel_block * BLOCK_OUT_CHANNELS + c_out_local;
            if (global_out_channel < out_channels && c_in < in_channels) {
                int weight_idx = (global_out_channel * in_channels + c_in) * KERNEL_VOLUME + k_index;
                (&w4.x)[i] = __ldg(weight + weight_idx);
            }
        }
        
        int sm_idx = c_in * KERNEL_VOLUME * (BLOCK_OUT_CHANNELS / 4) + k_index * (BLOCK_OUT_CHANNELS / 4) + vector_index;
        weight_sm4[sm_idx] = w4;
    }
    
    __syncthreads();
    
    float4 accum_vec[BLOCK_OUT_CHANNELS / 4];
    #pragma unroll
    for (int i = 0; i < BLOCK_OUT_CHANNELS / 4; i++) {
        accum_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    
    #pragma unroll
    for (int c_in = 0; c_in < in_channels; c_in++) {
        #pragma unroll
        for (int kd = 0; kd < KERNEL_SIZE; kd++) {
            int d_in = tz + kd;
            #pragma unroll
            for (int kh = 0; kh < KERNEL_SIZE; kh++) {
                int h_in = ty + kh;
                #pragma unroll
                for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                    int w_in = tx + kw;
                    
                    int sm_index = d_in * in_channels * INPUT_TILE_H * INPUT_TILE_W_PADDED + 
                                  c_in * INPUT_TILE_H * INPUT_TILE_W_PADDED + 
                                  h_in * INPUT_TILE_W_PADDED + 
                                  w_in;
                    float input_val = input_sm[sm_index];
                    
                    int k_index = kd * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw;
                    const float4* weight_ptr4 = weight_sm4 + (c_in * KERNEL_VOLUME + k_index) * (BLOCK_OUT_CHANNELS / 4);
                    
                    #pragma unroll
                    for (int c4 = 0; c4 < BLOCK_OUT_CHANNELS / 4; c4++) {
                        float4 w4 = weight_ptr4[c4];
                        accum_vec[c4].x += input_val * w4.x;
                        accum_vec[c4].y += input_val * w4.y;
                        accum_vec[c4].z += input_val * w4.z;
                        accum_vec[c4].w += input_val * w4.w;
                    }
                }
            }
        }
    }
    
    if (tz < TILE_D && ty < TILE_H && tx < TILE_W) {
        int out_d = tile_start_d + tz;
        int out_h = tile_start_h + ty;
        int out_w = tile_start_w + tx;
        
        if (out_d < out_depth && out_h < out_height && out_w < out_width) {
            int volume = out_depth * out_height * out_width;
            int spatial_index = out_d * out_height * out_width + out_h * out_width + out_w;
            int base_output = batch_idx * out_channels * volume;
            int base_channel = out_channel_block * BLOCK_OUT_CHANNELS;
            
            const float4* bias_vec = reinterpret_cast<const float4*>(bias_sm);
            #pragma unroll
            for (int c4 = 0; c4 < BLOCK_OUT_CHANNELS / 4; c4++) {
                float4 out_val = accum_vec[c4];
                float4 bias4 = bias_vec[c4];
                out_val.x += bias4.x;
                out_val.y += bias4.y;
                out_val.z += bias4.z;
                out_val.w += bias4.w;
                
                for (int i = 0; i < 4; i++) {
                    int channel = base_channel + c4 * 4 + i;
                    if (channel < out_channels) {
                        output[base_output + channel * volume + spatial_index] = (&out_val.x)[i];
                    }
                }
            }
        }
    }
}
//PART-END

//PART-START part3
torch::Tensor conv3d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int stride, int padding
) {
    input = input.contiguous();
    weight = weight.contiguous();
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    int block_out_channel_blocks = (out_channels + BLOCK_OUT_CHANNELS - 1) / BLOCK_OUT_CHANNELS;
    int num_tiles_w = (out_width + TILE_W - 1) / TILE_W;
    int num_tiles_h = (out_height + TILE_H - 1) / TILE_H;
    int num_tiles_d = (out_depth + TILE_D - 1) / TILE_D;
    int num_tiles = num_tiles_d * num_tiles_h * num_tiles_w;
    int total_blocks = batch_size * block_out_channel_blocks * num_tiles;
    
    dim3 grid(total_blocks);
    dim3 block_size(TILE_W, TILE_H, TILE_D);
    
    int smem_size = (INPUT_TILE_D * in_channels * INPUT_TILE_H * INPUT_TILE_W_PADDED +
                     in_channels * KERNEL_VOLUME * BLOCK_OUT_CHANNELS +
                     BLOCK_OUT_CHANNELS) * sizeof(float);
    
    float* bias_ptr = nullptr;
    if (bias.defined()) {
        bias = bias.contiguous();
        bias_ptr = bias.data_ptr<float>();
    }
    
    conv3d_forward_kernel<<<grid, block_size, smem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        out_depth, out_height, out_width,
        block_out_channel_blocks, num_tiles
    );
    
    return output;
}
//PART-END