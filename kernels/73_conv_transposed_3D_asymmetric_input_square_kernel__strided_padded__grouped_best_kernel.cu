//PART-START part1
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
//PART-END part1

//PART-START part2
__global__ void conv_transpose3d_kernel(
    const float* input, 
    const float* weight, 
    const float* bias, 
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int groups,
    int D_in, 
    int H_in, 
    int W_in, 
    int D_out, 
    int H_out, 
    int W_out,
    int H_out_tile,
    int W_out_tile
) {
    __shared__ struct {
        bool has_bias;
        float* bias_ptr;
    } s_bias_info;

    if (threadIdx.x == 0) {
        s_bias_info.has_bias = (bias != nullptr);
        s_bias_info.bias_ptr = const_cast<float*>(bias);
    }
    __syncthreads();

    const int kernel_size = 3;
    const int stride = 2;
    const int padding = 1;
    const int kernel_size3 = 27;
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int channel_stride = D_in * H_in * W_in;
    
    int total_tiles = batch_size * ((out_channels + 3) / 4) * D_out * H_out_tile * W_out_tile;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tiles) return;

    int n = tid / (((out_channels + 3) / 4) * D_out * H_out_tile * W_out_tile);
    int rem1 = tid % (((out_channels + 3) / 4) * D_out * H_out_tile * W_out_tile);
    int c_out_quad = rem1 / (D_out * H_out_tile * W_out_tile);
    int rem2 = rem1 % (D_out * H_out_tile * W_out_tile);
    int d = rem2 / (H_out_tile * W_out_tile);
    int rem3 = rem2 % (H_out_tile * W_out_tile);
    int h_base = rem3 / W_out_tile;
    int w_base = rem3 % W_out_tile;

    int c_out0 = c_out_quad * 4;
    int c_out1 = c_out0 + 1;
    int c_out2 = c_out0 + 2;
    int c_out3 = c_out0 + 3;
    if (c_out0 >= out_channels) return;
    
    int group_id0 = c_out0 / out_channels_per_group;
    int c_out_in_group0 = c_out0 % out_channels_per_group;
    int group_id1 = (c_out1 < out_channels) ? c_out1 / out_channels_per_group : -1;
    int c_out_in_group1 = (c_out1 < out_channels) ? c_out1 % out_channels_per_group : 0;
    int group_id2 = (c_out2 < out_channels) ? c_out2 / out_channels_per_group : -1;
    int c_out_in_group2 = (c_out2 < out_channels) ? c_out2 % out_channels_per_group : 0;
    int group_id3 = (c_out3 < out_channels) ? c_out3 / out_channels_per_group : -1;
    int c_out_in_group3 = (c_out3 < out_channels) ? c_out3 % out_channels_per_group : 0;
    
    bool valid_channel1 = (c_out1 < out_channels) && (group_id0 == group_id1);
    bool valid_channel2 = (c_out2 < out_channels) && (group_id0 == group_id2);
    bool valid_channel3 = (c_out3 < out_channels) && (group_id0 == group_id3);

    const int weight_group_stride = out_channels_per_group * kernel_size3 * in_channels_per_group;
    const int weight_channel_stride = kernel_size3 * in_channels_per_group;
    const int weight_group_base0 = group_id0 * weight_group_stride
                                + c_out_in_group0 * weight_channel_stride;
    const int weight_group_base1 = valid_channel1 ? 
                                 (group_id0 * weight_group_stride
                                 + c_out_in_group1 * weight_channel_stride) : 0;
    const int weight_group_base2 = valid_channel2 ? 
                                 (group_id0 * weight_group_stride
                                 + c_out_in_group2 * weight_channel_stride) : 0;
    const int weight_group_base3 = valid_channel3 ? 
                                 (group_id0 * weight_group_stride
                                 + c_out_in_group3 * weight_channel_stride) : 0;
    const int group_base0 = group_id0 * in_channels_per_group * channel_stride;

    float out_val0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_val1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_val2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out_val3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int h_arr[4] = {h_base*2, h_base*2, h_base*2+1, h_base*2+1};
    int w_arr[4] = {w_base*2, w_base*2+1, w_base*2, w_base*2+1};
    bool valid_out[4] = {
        (h_arr[0] < H_out && w_arr[0] < W_out),
        (h_arr[1] < H_out && w_arr[1] < W_out),
        (h_arr[2] < H_out && w_arr[2] < W_out),
        (h_arr[3] < H_out && w_arr[3] < W_out)
    };

    float bias_val0 = 0.0f;
    float bias_val1 = 0.0f;
    float bias_val2 = 0.0f;
    float bias_val3 = 0.0f;
    if (s_bias_info.has_bias) {
        int warp_lane_id = threadIdx.x % 32;
        float b0_tmp = 0.0f, b1_tmp = 0.0f, b2_tmp = 0.0f, b3_tmp = 0.0f;
        if (warp_lane_id == 0) {
            b0_tmp = s_bias_info.bias_ptr[c_out0];
            if (valid_channel1) b1_tmp = s_bias_info.bias_ptr[c_out1];
            if (valid_channel2) b2_tmp = s_bias_info.bias_ptr[c_out2];
            if (valid_channel3) b3_tmp = s_bias_info.bias_ptr[c_out3];
        }
        bias_val0 = __shfl_sync(0xFFFFFFFF, b0_tmp, 0);
        bias_val1 = __shfl_sync(0xFFFFFFFF, b1_tmp, 0);
        bias_val2 = __shfl_sync(0xFFFFFFFF, b2_tmp, 0);
        bias_val3 = __shfl_sync(0xFFFFFFFF, b3_tmp, 0);
    }

    #pragma unroll
    for (int kd = 0; kd < 3; kd++) {
        int d_in_val = d + padding - kd;
        if (d_in_val & 1) continue;
        int d_in = d_in_val >> 1;
        if (d_in < 0 || d_in >= D_in) continue;
        
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                const int kidx = kd * 9 + kh * 3 + kw;
                const int base_weight_idx0 = weight_group_base0 + kidx * in_channels_per_group;
                const int base_weight_idx1 = valid_channel1 ? 
                                          (weight_group_base1 + kidx * in_channels_per_group) : 0;
                const int base_weight_idx2 = valid_channel2 ? 
                                          (weight_group_base2 + kidx * in_channels_per_group) : 0;
                const int base_weight_idx3 = valid_channel3 ? 
                                          (weight_group_base3 + kidx * in_channels_per_group) : 0;

                for (int i = 0; i < 4; i++) {
                    if (!valid_out[i]) continue;
                    
                    const int h = h_arr[i];
                    const int w = w_arr[i];
                    const int h_in_val = h + padding - kh;
                    const int w_in_val = w + padding - kw;
                    
                    if (h_in_val & 1 || w_in_val & 1) continue;
                    const int h_in = h_in_val >> 1;
                    const int w_in = w_in_val >> 1;
                    if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;
                    
                    int base_input_idx = n * (in_channels * channel_stride)
                                      + d_in * (H_in * W_in)
                                      + h_in * W_in
                                      + w_in
                                      + group_base0;
                    
                    for (int cg = 0; cg < in_channels_per_group; cg += 4) {
                        float4 weight4_0 = reinterpret_cast<const float4*>(weight + base_weight_idx0 + cg)[0];
                        float4 weight4_1;
                        float4 weight4_2;
                        float4 weight4_3;
                        if (valid_channel1) {
                            weight4_1 = reinterpret_cast<const float4*>(weight + base_weight_idx1 + cg)[0];
                        }
                        if (valid_channel2) {
                            weight4_2 = reinterpret_cast<const float4*>(weight + base_weight_idx2 + cg)[0];
                        }
                        if (valid_channel3) {
                            weight4_3 = reinterpret_cast<const float4*>(weight + base_weight_idx3 + cg)[0];
                        }
                        
                        int c_in_start = base_input_idx + cg * channel_stride;
                        #pragma unroll
                        for (int off = 0; off < 4; off++) {
                            int c_in_idx = c_in_start + off * channel_stride;
                            float input_val = input[c_in_idx];
                            out_val0[i] += input_val * ((float*)&weight4_0)[off];
                            if (valid_channel1) {
                                out_val1[i] += input_val * ((float*)&weight4_1)[off];
                            }
                            if (valid_channel2) {
                                out_val2[i] += input_val * ((float*)&weight4_2)[off];
                            }
                            if (valid_channel3) {
                                out_val3[i] += input_val * ((float*)&weight4_3)[off];
                            }
                        }
                    }
                }
            }
        }
    }

    int output_base0 = n * (out_channels * D_out * H_out * W_out)
                   + c_out0 * (D_out * H_out * W_out)
                   + d * (H_out * W_out);
    int output_base1 = n * (out_channels * D_out * H_out * W_out)
                   + c_out1 * (D_out * H_out * W_out)
                   + d * (H_out * W_out);
    int output_base2 = n * (out_channels * D_out * H_out * W_out)
                   + c_out2 * (D_out * H_out * W_out)
                   + d * (H_out * W_out);
    int output_base3 = n * (out_channels * D_out * H_out * W_out)
                   + c_out3 * (D_out * H_out * W_out)
                   + d * (H_out * W_out);
    
    for (int i = 0; i < 4; i++) {
        if (valid_out[i]) {
            output[output_base0 + h_arr[i] * W_out + w_arr[i]] = out_val0[i] + bias_val0;
            if (valid_channel1) {
                output[output_base1 + h_arr[i] * W_out + w_arr[i]] = out_val1[i] + bias_val1;
            }
            if (valid_channel2) {
                output[output_base2 + h_arr[i] * W_out + w_arr[i]] = out_val2[i] + bias_val2;
            }
            if (valid_channel3) {
                output[output_base3 + h_arr[i] * W_out + w_arr[i]] = out_val3[i] + bias_val3;
            }
        }
    }
}
//PART-END part2

//PART-START part3
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int groups
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int out_channels = weight.size(1) * groups;
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int D_out = (D_in - 1) * stride - 2 * padding + kernel_size;
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size;
    int H_out_tile = CEIL_DIV(H_out, 2);
    int W_out_tile = CEIL_DIV(W_out, 2);

    auto weight_restructured = weight.view({groups, in_channels_per_group, out_channels_per_group, 
                                          kernel_size, kernel_size, kernel_size})
                         .permute({0, 2, 3, 4, 5, 1})
                         .contiguous();
    auto weight_ptr_restructured = weight_restructured.data_ptr<float>();

    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());
    int total_tiles = batch_size * ((out_channels + 3) / 4) * D_out * H_out_tile * W_out_tile;

    const int threads = 256;
    int blocks = CEIL_DIV(total_tiles, threads);

    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight_ptr_restructured,
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        groups,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        H_out_tile,
        W_out_tile
    );

    return output;
}
//PART-END part3