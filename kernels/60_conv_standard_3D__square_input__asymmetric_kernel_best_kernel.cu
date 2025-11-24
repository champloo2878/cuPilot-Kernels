// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int kernel_input_offset[105];   // Maximum kernel volume = 3*5*7
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
#define TILE_W 8
#define TILE_H 8
#define TILE_D 16  // Increased from 8 to process more depth per block
#define TILE_OC 8
#define KERNEL_VOLUME 105

__global__ void conv3d_forward_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int width, int height, int depth,
    int kernel_w, int kernel_h, int kernel_d,
    int stride, int padding, int dilation,
    int width_out, int height_out, int depth_out,
    bool has_bias
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int w_tile = blockIdx.x;
    int h_tile = blockIdx.y;
    int combined = blockIdx.z;
    
    int num_tiles_depth = (depth_out + TILE_D - 1) / TILE_D;
    int num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;
    
    int batch_index = combined / (num_tiles_depth * num_oc_tiles);
    int rem = combined % (num_tiles_depth * num_oc_tiles);
    int d_tile = rem / num_oc_tiles;
    int oc_tile = rem % num_oc_tiles;
    
    int w_start = w_tile * TILE_W;
    int w_end = min(w_start + TILE_W, width_out);
    int h_start = h_tile * TILE_H;
    int h_end = min(h_start + TILE_H, height_out);
    int d_start = d_tile * TILE_D;
    int d_end = min(d_start + TILE_D, depth_out);
    
    int input_w_start = w_start * stride - padding;
    int input_h_start = h_start * stride - padding;
    int input_d_start = d_start * stride - padding;
    
    int input_tile_w = (TILE_W - 1) * stride + 1 + (kernel_w - 1) * dilation;
    int input_tile_h = (TILE_H - 1) * stride + 1 + (kernel_h - 1) * dilation;
    int input_tile_d = (TILE_D - 1) * stride + 1 + (kernel_d - 1) * dilation;
    int input_tile_stride = input_tile_h * input_tile_d;
    
    extern __shared__ float shm[];
    float* input_tile_shm = shm;
    float* weight_tile_shm = &input_tile_shm[in_channels * input_tile_w * input_tile_h * input_tile_d];
    
    int input_tile_size = in_channels * input_tile_w * input_tile_h * input_tile_d;
    int weight_tile_size = in_channels * KERNEL_VOLUME * TILE_OC;
    
    int tid = tz * (blockDim.x * blockDim.y) + ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    // Calculate kernel_volume for flexible loop control
    int kernel_volume = kernel_w * kernel_h * kernel_d;
    
    // Cooperative input loading
    for (int idx = tid; idx < input_tile_size; idx += num_threads) {
        int ic = idx / (input_tile_w * input_tile_h * input_tile_d);
        int rem = idx % (input_tile_w * input_tile_h * input_tile_d);
        int iw = rem / (input_tile_h * input_tile_d);
        rem %= (input_tile_h * input_tile_d);
        int ih = rem / input_tile_d;
        int id = rem % input_tile_d;
        
        int w_global = input_w_start + iw;
        int h_global = input_h_start + ih;
        int d_global = input_d_start + id;
        
        if (w_global >= 0 && w_global < width && 
            h_global >= 0 && h_global < height && 
            d_global >= 0 && d_global < depth) {
            input_tile_shm[idx] = input[batch_index * (in_channels * width * height * depth)
                + ic * (width * height * depth)
                + w_global * (height * depth)
                + h_global * depth
                + d_global];
        } else {
            input_tile_shm[idx] = 0.0f;
        }
    }
    
    // Warp-level weight loading
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = num_threads / 32;

    if (warp_id < in_channels) {
        int ic = warp_id;
        for (int k = lane_id; k < kernel_volume; k += 32) {
            int base_global = ic * kernel_volume + k;
            int base_shm = (ic * KERNEL_VOLUME + k) * TILE_OC;
            for (int oc_local = 0; oc_local < TILE_OC; oc_local++) {
                int oc_global = oc_tile * TILE_OC + oc_local;
                if (oc_global < out_channels) {
                    weight_tile_shm[base_shm + oc_local] = 
                        weight[oc_global * (in_channels * kernel_volume) + base_global];
                } else {
                    weight_tile_shm[base_shm + oc_local] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    int w0_global = w_start + 2 * tx;
    int w1_global = w0_global + 1;
    int h0_global = h_start + 2 * ty;
    int h1_global = h0_global + 1;
    int d_global = d_start + tz;
    
    bool valid_d = (d_global < depth_out);
    bool valid_w0 = (w0_global < width_out);
    bool valid_w1 = (w1_global < width_out);
    bool valid_h0 = (h0_global < height_out);
    bool valid_h1 = (h1_global < height_out);
    
    // Vector accumulators [position][channel-half]
    float4 acc00_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc00_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc10_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc10_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc01_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc01_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc11_0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 acc11_1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    int base_input00 = (w0_global - input_w_start) * input_tile_stride +
                       (h0_global - input_h_start) * input_tile_d +
                       (d_global - input_d_start);
    int base_input10 = (w1_global - input_w_start) * input_tile_stride +
                       (h0_global - input_h_start) * input_tile_d +
                       (d_global - input_d_start);
    int base_input01 = (w0_global - input_w_start) * input_tile_stride +
                       (h1_global - input_h_start) * input_tile_d +
                       (d_global - input_d_start);
    int base_input11 = (w1_global - input_w_start) * input_tile_stride +
                       (h1_global - input_h_start) * input_tile_d +
                       (d_global - input_d_start);

    if (valid_d && (valid_w0 || valid_w1 || valid_h0 || valid_h1)) {
        #pragma unroll 3  // Unroll in_channels loop for better ILP
        for (int ic = 0; ic < in_channels; ic++) {
            int input_base = ic * input_tile_w * input_tile_stride;
            #pragma unroll 7  // Unroll kernel_volume loop
            for (int k = 0; k < kernel_volume; k++) {
                float in00 = input_tile_shm[input_base + base_input00 + kernel_input_offset[k]];
                float in10 = input_tile_shm[input_base + base_input10 + kernel_input_offset[k]];
                float in01 = input_tile_shm[input_base + base_input01 + kernel_input_offset[k]];
                float in11 = input_tile_shm[input_base + base_input11 + kernel_input_offset[k]];
                
                int weight_idx = (ic * KERNEL_VOLUME + k) * TILE_OC;
                float4 w0 = *reinterpret_cast<float4*>(&weight_tile_shm[weight_idx]);
                float4 w1 = *reinterpret_cast<float4*>(&weight_tile_shm[weight_idx + 4]);
                
                // Temporary variables to break dependencies
                float4 t00_0 = acc00_0, t00_1 = acc00_1;
                float4 t10_0 = acc10_0, t10_1 = acc10_1;
                float4 t01_0 = acc01_0, t01_1 = acc01_1;
                float4 t11_0 = acc11_0, t11_1 = acc11_1;
                
                // Position 00
                t00_0.x = __fmaf_rn(in00, w0.x, t00_0.x);
                t00_0.y = __fmaf_rn(in00, w0.y, t00_0.y);
                t00_0.z = __fmaf_rn(in00, w0.z, t00_0.z);
                t00_0.w = __fmaf_rn(in00, w0.w, t00_0.w);
                t00_1.x = __fmaf_rn(in00, w1.x, t00_1.x);
                t00_1.y = __fmaf_rn(in00, w1.y, t00_1.y);
                t00_1.z = __fmaf_rn(in00, w1.z, t00_1.z);
                t00_1.w = __fmaf_rn(in00, w1.w, t00_1.w);
                
                // Position 10
                t10_0.x = __fmaf_rn(in10, w0.x, t10_0.x);
                t10_0.y = __fmaf_rn(in10, w0.y, t10_0.y);
                t10_0.z = __fmaf_rn(in10, w0.z, t10_0.z);
                t10_0.w = __fmaf_rn(in10, w0.w, t10_0.w);
                t10_1.x = __fmaf_rn(in10, w1.x, t10_1.x);
                t10_1.y = __fmaf_rn(in10, w1.y, t10_1.y);
                t10_1.z = __fmaf_rn(in10, w1.z, t10_1.z);
                t10_1.w = __fmaf_rn(in10, w1.w, t10_1.w);
                
                // Position 01
                t01_0.x = __fmaf_rn(in01, w0.x, t01_0.x);
                t01_0.y = __fmaf_rn(in01, w0.y, t01_0.y);
                t01_0.z = __fmaf_rn(in01, w0.z, t01_0.z);
                t01_0.w = __fmaf_rn(in01, w0.w, t01_0.w);
                t01_1.x = __fmaf_rn(in01, w1.x, t01_1.x);
                t01_1.y = __fmaf_rn(in01, w1.y, t01_1.y);
                t01_1.z = __fmaf_rn(in01, w1.z, t01_1.z);
                t01_1.w = __fmaf_rn(in01, w1.w, t01_1.w);
                
                // Position 11
                t11_0.x = __fmaf_rn(in11, w0.x, t11_0.x);
                t11_0.y = __fmaf_rn(in11, w0.y, t11_0.y);
                t11_0.z = __fmaf_rn(in11, w0.z, t11_0.z);
                t11_0.w = __fmaf_rn(in11, w0.w, t11_0.w);
                t11_1.x = __fmaf_rn(in11, w1.x, t11_1.x);
                t11_1.y = __fmaf_rn(in11, w1.y, t11_1.y);
                t11_1.z = __fmaf_rn(in11, w1.z, t11_1.z);
                t11_1.w = __fmaf_rn(in11, w1.w, t11_1.w);
                
                acc00_0 = t00_0; acc00_1 = t00_1;
                acc10_0 = t10_0; acc10_1 = t10_1;
                acc01_0 = t01_0; acc01_1 = t01_1;
                acc11_0 = t11_0; acc11_1 = t11_1;
            }
        }
    }
    
    int channel_stride = width_out * height_out * depth_out;
    for (int c = 0; c < TILE_OC; c++) {
        int oc_global = oc_tile * TILE_OC + c;
        if (oc_global < out_channels) {
            float bias_val = has_bias ? bias[oc_global] : 0.0f;
            float val;
            
            if (valid_d && valid_w0 && valid_h0) {
                if (c < 4) val = (&acc00_0.x)[c];
                else val = (&acc00_1.x)[c-4];
                int output_index = batch_index * (out_channels * channel_stride)
                                 + oc_global * channel_stride
                                 + w0_global * (height_out * depth_out)
                                 + h0_global * depth_out
                                 + d_global;
                output[output_index] = val + bias_val;
            }
            if (valid_d && valid_w1 && valid_h0) {
                if (c < 4) val = (&acc10_0.x)[c];
                else val = (&acc10_1.x)[c-4];
                int output_index = batch_index * (out_channels * channel_stride)
                                 + oc_global * channel_stride
                                 + w1_global * (height_out * depth_out)
                                 + h0_global * depth_out
                                 + d_global;
                output[output_index] = val + bias_val;
            }
            if (valid_d && valid_w0 && valid_h1) {
                if (c < 4) val = (&acc01_0.x)[c];
                else val = (&acc01_1.x)[c-4];
                int output_index = batch_index * (out_channels * channel_stride)
                                 + oc_global * channel_stride
                                 + w0_global * (height_out * depth_out)
                                 + h1_global * depth_out
                                 + d_global;
                output[output_index] = val + bias_val;
            }
            if (valid_d && valid_w1 && valid_h1) {
                if (c < 4) val = (&acc11_0.x)[c];
                else val = (&acc11_1.x)[c-4];
                int output_index = batch_index * (out_channels * channel_stride)
                                 + oc_global * channel_stride
                                 + w1_global * (height_out * depth_out)
                                 + h1_global * depth_out
                                 + d_global;
                output[output_index] = val + bias_val;
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv3d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int width = input.size(2);
    int height = input.size(3);
    int depth = input.size(4);
    
    int out_channels = weight.size(0);
    int kernel_w = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_d = weight.size(4);
    int kernel_vol = kernel_w * kernel_h * kernel_d;
    
    int width_out = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;
    int height_out = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int depth_out = (depth + 2 * padding - dilation * (kernel_d - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, width_out, height_out, depth_out}, input.options());
    
    int num_tiles_width = (width_out + TILE_W - 1) / TILE_W;
    int num_tiles_height = (height_out + TILE_H - 1) / TILE_H;
    int num_tiles_depth = (depth_out + TILE_D - 1) / TILE_D;
    int num_oc_tiles = (out_channels + TILE_OC - 1) / TILE_OC;
    
    int grid_z = batch_size * num_tiles_depth * num_oc_tiles;
    dim3 grid(num_tiles_width, num_tiles_height, grid_z);
    dim3 block(4, 4, 16);  // Increased z-dimension to match TILE_D=16
    
    int input_tile_w = (TILE_W - 1) * stride + 1 + (kernel_w - 1) * dilation;
    int input_tile_h = (TILE_H - 1) * stride + 1 + (kernel_h - 1) * dilation;
    int input_tile_d = (TILE_D - 1) * stride + 1 + (kernel_d - 1) * dilation;
    
    size_t input_shm_size = in_channels * input_tile_w * input_tile_h * input_tile_d * sizeof(float);
    size_t weight_shm_size = in_channels * KERNEL_VOLUME * TILE_OC * sizeof(float);
    size_t shm_size = input_shm_size + weight_shm_size;
    
    auto input_ptr = input.contiguous().data_ptr<float>();
    auto weight_ptr = weight.contiguous().data_ptr<float>();
    auto bias_ptr = has_bias ? bias.contiguous().data_ptr<float>() : nullptr;
    auto output_ptr = output.data_ptr<float>();
    
    static int last_kernel_w = -1, last_kernel_h = -1, last_kernel_d = -1;
    static int last_stride = -1, last_dilation = -1;
    static int h_kernel_input_offset[105];
    
    if (kernel_w != last_kernel_w || kernel_h != last_kernel_h || kernel_d != last_kernel_d ||
        stride != last_stride || dilation != last_dilation) {
        for (int kw = 0; kw < kernel_w; kw++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kd = 0; kd < kernel_d; kd++) {
                    int k = kw * (kernel_h * kernel_d) + kh * kernel_d + kd;
                    h_kernel_input_offset[k] = kw * dilation * (input_tile_h * input_tile_d) + 
                                              kh * dilation * input_tile_d + 
                                              kd * dilation;
                }
            }
        }
        cudaMemcpyToSymbol(kernel_input_offset, h_kernel_input_offset, sizeof(int)*kernel_vol);
        last_kernel_w = kernel_w;
        last_kernel_h = kernel_h;
        last_kernel_d = kernel_d;
        last_stride = stride;
        last_dilation = dilation;
    }
    
    conv3d_forward_kernel<<<grid, block, shm_size>>>(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, in_channels, out_channels,
        width, height, depth,
        kernel_w, kernel_h, kernel_d,
        stride, padding, dilation,
        width_out, height_out, depth_out,
        has_bias
    );
    
    return output;
}
// PART-END