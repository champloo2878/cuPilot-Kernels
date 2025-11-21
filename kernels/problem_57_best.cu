#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#define TILE_DIM 16
#define BLOCK_SIZE 16
#define CHUNK_SIZE 16
#define KERNEL_SIZE 3
// PART-END

// PART-START
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int padding,
    const int out_height,
    const int out_width,
    const int num_channel_groups
) {
    const int weight_total = 3 * 3 * CHUNK_SIZE;
    extern __shared__ float sh[];
    float* sh_weight = sh;
    float* sh_input = sh + 16 * weight_total;

    // Remapped grid indexing for improved L2 locality
    const int c_out_octet = blockIdx.x % num_channel_groups;
    const int tile_x = blockIdx.x / num_channel_groups;
    const int tile_y = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int x_out = tile_x * TILE_DIM + threadIdx.x;
    const int y_out = tile_y * TILE_DIM + threadIdx.y;
    const int c_out0 = c_out_octet * 16;
    const int num_active = min(16, out_channels - c_out0);

    const int x0_out = tile_x * TILE_DIM;
    const int y0_out = tile_y * TILE_DIM;
    const int x0_in = x0_out + padding - 2;
    const int y0_in = y0_out + padding - 2;
    
    bool is_interior_tile = (x0_in >= 0 && y0_in >= 0 && 
                            (x0_in + 18) <= in_width && 
                            (y0_in + 18) <= in_height);

    float accum[16] = {0.0f};

    #pragma unroll 4
    for (int c_in_base = 0; c_in_base < in_channels; c_in_base += CHUNK_SIZE) {
        // Vectorized weight loading
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; 
             idx < 9 * CHUNK_SIZE;
             idx += blockDim.x * blockDim.y) {
            const int c_rel = idx % CHUNK_SIZE;
            const int k_idx = idx / CHUNK_SIZE;
            const int ky = k_idx / 3;
            const int kx = k_idx % 3;
            const int dst_base = ky * 3 * CHUNK_SIZE + kx * CHUNK_SIZE + c_rel;
            
            #pragma unroll 4
            for (int i = 0; i < 16; i += 4) {
                float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (i < num_active) {
                    const int src_idx0 = (c_in_base + c_rel) * (out_channels * 9)
                                     + (c_out0 + i) * 9 + ky * 3 + kx;
                    vals.x = weight[src_idx0];
                }
                if (i+1 < num_active) {
                    const int src_idx1 = (c_in_base + c_rel) * (out_channels * 9)
                                     + (c_out0 + i+1) * 9 + ky * 3 + kx;
                    vals.y = weight[src_idx1];
                }
                if (i+2 < num_active) {
                    const int src_idx2 = (c_in_base + c_rel) * (out_channels * 9)
                                     + (c_out0 + i+2) * 9 + ky * 3 + kx;
                    vals.z = weight[src_idx2];
                }
                if (i+3 < num_active) {
                    const int src_idx3 = (c_in_base + c_rel) * (out_channels * 9)
                                     + (c_out0 + i+3) * 9 + ky * 3 + kx;
                    vals.w = weight[src_idx3];
                }
                
                sh_weight[(i+0)*weight_total + dst_base] = vals.x;
                sh_weight[(i+1)*weight_total + dst_base] = vals.y;
                sh_weight[(i+2)*weight_total + dst_base] = vals.z;
                sh_weight[(i+3)*weight_total + dst_base] = vals.w;
            }
        }

        // Vectorized input loading
        for (int idx = threadIdx.y * blockDim.x + threadIdx.x; 
             idx < 18*18; 
             idx += blockDim.x * blockDim.y) {
            const int rel_x = idx % 18;
            const int rel_y = idx / 18;
            const int x_in = x0_in + rel_x;
            const int y_in = y0_in + rel_y;
            
            const int base_idx = batch_idx * (in_channels * in_height * in_width)
                               + c_in_base * (in_height * in_width)
                               + y_in * in_width + x_in;
            
            for (int c = 0; c < CHUNK_SIZE; c += 4) {
                float4 vals = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (is_interior_tile) {
                    vals.x = input[base_idx + (c+0) * (in_height*in_width)];
                    vals.y = input[base_idx + (c+1) * (in_height*in_width)];
                    vals.z = input[base_idx + (c+2) * (in_height*in_width)];
                    vals.w = input[base_idx + (c+3) * (in_height*in_width)];
                } else {
                    if (x_in >= 0 && x_in < in_width && y_in >= 0 && y_in < in_height) {
                        vals.x = input[base_idx + (c+0) * (in_height*in_width)];
                        vals.y = input[base_idx + (c+1) * (in_height*in_width)];
                        vals.z = input[base_idx + (c+2) * (in_height*in_width)];
                        vals.w = input[base_idx + (c+3) * (in_height*in_width)];
                    }
                }
                
                sh_input[(c+0)*324 + rel_y*18 + rel_x] = vals.x;
                sh_input[(c+1)*324 + rel_y*18 + rel_x] = vals.y;
                sh_input[(c+2)*324 + rel_y*18 + rel_x] = vals.z;
                sh_input[(c+3)*324 + rel_y*18 + rel_x] = vals.w;
            }
        }

        __syncthreads();

        // Vectorized accumulation
        if (x_out < out_width && y_out < out_height) {
            const int x_rel = threadIdx.x + 2;
            const int y_rel = threadIdx.y + 2;
            
            #pragma unroll
            for (int ky = 0; ky < 3; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < 3; ++kx) {
                    const int x_src = x_rel - kx;
                    const int y_src = y_rel - ky;
                    const int input_base = y_src * 18 + x_src;
                    const int weight_base = ky * 3 * CHUNK_SIZE + kx * CHUNK_SIZE;
                    
                    for (int c = 0; c < CHUNK_SIZE; ++c) {
                        const float in_val = sh_input[c*324 + input_base];
                        #pragma unroll
                        for (int i = 0; i < 16; i += 4) {
                            accum[i+0] += in_val * sh_weight[(i+0)*weight_total + weight_base + c];
                            accum[i+1] += in_val * sh_weight[(i+1)*weight_total + weight_base + c];
                            accum[i+2] += in_val * sh_weight[(i+2)*weight_total + weight_base + c];
                            accum[i+3] += in_val * sh_weight[(i+3)*weight_total + weight_base + c];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Vectorized output writing
    if (x_out < out_width && y_out < out_height) {
        #pragma unroll 4
        for (int i = 0; i < 16; i += 4) {
            if (i < num_active) {
                const int out_idx0 = batch_idx * (out_channels * out_height * out_width)
                                 + (c_out0 + i) * (out_height * out_width)
                                 + y_out * out_width + x_out;
                output[out_idx0] = accum[i];
            }
            if (i+1 < num_active) {
                const int out_idx1 = batch_idx * (out_channels * out_height * out_width)
                                 + (c_out0 + i+1) * (out_height * out_width)
                                 + y_out * out_width + x_out;
                output[out_idx1] = accum[i+1];
            }
            if (i+2 < num_active) {
                const int out_idx2 = batch_idx * (out_channels * out_height * out_width)
                                 + (c_out0 + i+2) * (out_height * out_width)
                                 + y_out * out_width + x_out;
                output[out_idx2] = accum[i+2];
            }
            if (i+3 < num_active) {
                const int out_idx3 = batch_idx * (out_channels * out_height * out_width)
                                 + (c_out0 + i+3) * (out_height * out_width)
                                 + y_out * out_width + x_out;
                output[out_idx3] = accum[i+3];
            }
        }
    }
}
// PART-END

// PART-START
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");
    TORCH_CHECK(groups == 1, "Only groups=1 supported");
    TORCH_CHECK(weight.size(2) == 3, "This kernel only supports 3x3 kernels");
    TORCH_CHECK(weight.size(3) == 3, "This kernel only supports 3x3 kernels");
    TORCH_CHECK(input.size(1) % CHUNK_SIZE == 0, 
                "in_channels must be divisible by CHUNK_SIZE (16)");

    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = 3;

    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());

    // Precomputed channel groups and spatial dimensions
    const int num_channel_groups = (out_channels + 15) / 16;
    const int grid_x_spatial = (out_width + TILE_DIM - 1) / TILE_DIM;
    const int grid_y_spatial = (out_height + TILE_DIM - 1) / TILE_DIM;

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(grid_x_spatial * num_channel_groups, 
              grid_y_spatial, 
              batch_size);

    const int weight_total = 3 * 3 * CHUNK_SIZE;
    const int input_total = CHUNK_SIZE * 18 * 18;
    size_t shared_mem_size = (16 * weight_total + input_total) * sizeof(float);
    conv_transpose2d_kernel<<<grid, block, shared_mem_size>>>(
        input.contiguous().data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        padding,
        out_height,
        out_width,
        num_channel_groups
    );

    return output;
}
// PART-END