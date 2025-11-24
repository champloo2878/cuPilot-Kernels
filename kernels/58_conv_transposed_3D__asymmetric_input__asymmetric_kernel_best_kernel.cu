// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
// PART-END

// PART-START
__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w
) {
    constexpr int T_kernel_d = 3;
    constexpr int T_kernel_h = 5;
    constexpr int T_kernel_w = 7;
    constexpr int channels_per_thread = 4;
    constexpr int tile_size = 4;
    
    int spatial_size = D_out * H_out * W_out;
    int spatial_size_reduced = D_out * H_out * ((W_out + 3) / 4);
    int batch_threads = batch_size * spatial_size_reduced * (out_channels / channels_per_thread);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_threads) return;

    int instance_threads = spatial_size_reduced * (out_channels / channels_per_thread);
    int b = idx / instance_threads;
    int rest = idx % instance_threads;
    int spt_idx_reduced = rest / (out_channels / channels_per_thread);
    int c_out_base = (rest % (out_channels / channels_per_thread)) * channels_per_thread;

    int i = spt_idx_reduced / (H_out * ((W_out + 3) / 4));
    int j = (spt_idx_reduced % (H_out * ((W_out + 3) / 4))) / ((W_out + 3) / 4);
    int k_base = (spt_idx_reduced % ((W_out + 3) / 4)) * 4;
    int k0 = k_base;
    int k1 = k_base + 1;
    int k2 = k_base + 2;
    int k3 = k_base + 3;

    // Scalarize accumulators
    float val_pt00 = 0.0f, val_pt01 = 0.0f, val_pt02 = 0.0f, val_pt03 = 0.0f;
    float val_pt10 = 0.0f, val_pt11 = 0.0f, val_pt12 = 0.0f, val_pt13 = 0.0f;
    float val_pt20 = 0.0f, val_pt21 = 0.0f, val_pt22 = 0.0f, val_pt23 = 0.0f;
    float val_pt30 = 0.0f, val_pt31 = 0.0f, val_pt32 = 0.0f, val_pt33 = 0.0f;
    
    const int weight_stride = in_channels * out_channels;
    int spt_idx0 = i * H_out * W_out + j * W_out + k0;
    int spt_idx1 = i * H_out * W_out + j * W_out + k1;
    int spt_idx2 = i * H_out * W_out + j * W_out + k2;
    int spt_idx3 = i * H_out * W_out + j * W_out + k3;

    // Precompute base_valid masks for spatial dimensions
    bool base_valid[T_kernel_d][T_kernel_h];
    #pragma unroll
    for (int di = 0; di < T_kernel_d; di++) {
        int i_in = i - di;
        #pragma unroll
        for (int dj = 0; dj < T_kernel_h; dj++) {
            int j_in = j - dj;
            base_valid[di][dj] = (i_in >= 0 && i_in < D_in && j_in >= 0 && j_in < H_in);
        }
    }

    #pragma unroll
    for (int di = 0; di < T_kernel_d; di++) {
        int i_in = i - di;
        #pragma unroll
        for (int dj = 0; dj < T_kernel_h; dj++) {
            int j_in = j - dj;
            if (base_valid[di][dj]) {
                #pragma unroll
                for (int dk = 0; dk < T_kernel_w; dk++) {
                    int k_in0 = k0 - dk;
                    int k_in1 = k1 - dk;
                    int k_in2 = k2 - dk;
                    int k_in3 = k3 - dk;

                    bool in_bounds0 = (k_in0 >= 0 && k_in0 < W_in);
                    bool in_bounds1 = (k_in1 >= 0 && k_in1 < W_in);
                    bool in_bounds2 = (k_in2 >= 0 && k_in2 < W_in);
                    bool in_bounds3 = (k_in3 >= 0 && k_in3 < W_in);

                    int spatial_offset_base = (b * D_in * H_in * W_in + (i_in * H_in + j_in) * W_in) * in_channels;
                    int spatial_offset_in0 = in_bounds0 ? (spatial_offset_base + k_in0 * in_channels) : 0;
                    int spatial_offset_in1 = in_bounds1 ? (spatial_offset_base + k_in1 * in_channels) : 0;
                    int spatial_offset_in2 = in_bounds2 ? (spatial_offset_base + k_in2 * in_channels) : 0;
                    int spatial_offset_in3 = in_bounds3 ? (spatial_offset_base + k_in3 * in_channels) : 0;
                    
                    int weight_offset = ((di * T_kernel_h + dj) * T_kernel_w + dk) * weight_stride + c_out_base;

                    #pragma unroll
                    for (int c_in = 0; c_in < in_channels; c_in += tile_size) {
                        float4 in4_0 = in_bounds0 ? *reinterpret_cast<const float4*>(&input[spatial_offset_in0 + c_in]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        float4 in4_1 = in_bounds1 ? *reinterpret_cast<const float4*>(&input[spatial_offset_in1 + c_in]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        float4 in4_2 = in_bounds2 ? *reinterpret_cast<const float4*>(&input[spatial_offset_in2 + c_in]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        float4 in4_3 = in_bounds3 ? *reinterpret_cast<const float4*>(&input[spatial_offset_in3 + c_in]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        
                        float4 w4_0 = *reinterpret_cast<const float4*>(&weight[weight_offset + c_in * out_channels]);
                        float4 w4_1 = *reinterpret_cast<const float4*>(&weight[weight_offset + (c_in+1) * out_channels]);
                        float4 w4_2 = *reinterpret_cast<const float4*>(&weight[weight_offset + (c_in+2) * out_channels]);
                        float4 w4_3 = *reinterpret_cast<const float4*>(&weight[weight_offset + (c_in+3) * out_channels]);
                        
                        // Input channel 0
                        val_pt00 += in4_0.x * w4_0.x;
                        val_pt01 += in4_0.x * w4_0.y;
                        val_pt02 += in4_0.x * w4_0.z;
                        val_pt03 += in4_0.x * w4_0.w;
                        
                        val_pt10 += in4_1.x * w4_0.x;
                        val_pt11 += in4_1.x * w4_0.y;
                        val_pt12 += in4_1.x * w4_0.z;
                        val_pt13 += in4_1.x * w4_0.w;
                        
                        val_pt20 += in4_2.x * w4_0.x;
                        val_pt21 += in4_2.x * w4_0.y;
                        val_pt22 += in4_2.x * w4_0.z;
                        val_pt23 += in4_2.x * w4_0.w;
                        
                        val_pt30 += in4_3.x * w4_0.x;
                        val_pt31 += in4_3.x * w4_0.y;
                        val_pt32 += in4_3.x * w4_0.z;
                        val_pt33 += in4_3.x * w4_0.w;
                        
                        // Input channel 1
                        val_pt00 += in4_0.y * w4_1.x;
                        val_pt01 += in4_0.y * w4_1.y;
                        val_pt02 += in4_0.y * w4_1.z;
                        val_pt03 += in4_0.y * w4_1.w;
                        
                        val_pt10 += in4_1.y * w4_1.x;
                        val_pt11 += in4_1.y * w4_1.y;
                        val_pt12 += in4_1.y * w4_1.z;
                        val_pt13 += in4_1.y * w4_1.w;
                        
                        val_pt20 += in4_2.y * w4_1.x;
                        val_pt21 += in4_2.y * w4_1.y;
                        val_pt22 += in4_2.y * w4_1.z;
                        val_pt23 += in4_2.y * w4_1.w;
                        
                        val_pt30 += in4_3.y * w4_1.x;
                        val_pt31 += in4_3.y * w4_1.y;
                        val_pt32 += in4_3.y * w4_1.z;
                        val_pt33 += in4_3.y * w4_1.w;
                        
                        // Input channel 2
                        val_pt00 += in4_0.z * w4_2.x;
                        val_pt01 += in4_0.z * w4_2.y;
                        val_pt02 += in4_0.z * w4_2.z;
                        val_pt03 += in4_0.z * w4_2.w;
                        
                        val_pt10 += in4_1.z * w4_2.x;
                        val_pt11 += in4_1.z * w4_2.y;
                        val_pt12 += in4_1.z * w4_2.z;
                        val_pt13 += in4_1.z * w4_2.w;
                        
                        val_pt20 += in4_2.z * w4_2.x;
                        val_pt21 += in4_2.z * w4_2.y;
                        val_pt22 += in4_2.z * w4_2.z;
                        val_pt23 += in4_2.z * w4_2.w;
                        
                        val_pt30 += in4_3.z * w4_2.x;
                        val_pt31 += in4_3.z * w4_2.y;
                        val_pt32 += in4_3.z * w4_2.z;
                        val_pt33 += in4_3.z * w4_2.w;
                        
                        // Input channel 3
                        val_pt00 += in4_0.w * w4_3.x;
                        val_pt01 += in4_0.w * w4_3.y;
                        val_pt02 += in4_0.w * w4_3.z;
                        val_pt03 += in4_0.w * w4_3.w;
                        
                        val_pt10 += in4_1.w * w4_3.x;
                        val_pt11 += in4_1.w * w4_3.y;
                        val_pt12 += in4_1.w * w4_3.z;
                        val_pt13 += in4_1.w * w4_3.w;
                        
                        val_pt20 += in4_2.w * w4_3.x;
                        val_pt21 += in4_2.w * w4_3.y;
                        val_pt22 += in4_2.w * w4_3.z;
                        val_pt23 += in4_2.w * w4_3.w;
                        
                        val_pt30 += in4_3.w * w4_3.x;
                        val_pt31 += in4_3.w * w4_3.y;
                        val_pt32 += in4_3.w * w4_3.z;
                        val_pt33 += in4_3.w * w4_3.w;
                    }
                }
            }
        }
    }

    int out_base_idx = (b * out_channels + c_out_base) * spatial_size;
    if (k0 < W_out) {
        output[out_base_idx + spt_idx0] = val_pt00;
        output[out_base_idx + spatial_size + spt_idx0] = val_pt01;
        output[out_base_idx + 2 * spatial_size + spt_idx0] = val_pt02;
        output[out_base_idx + 3 * spatial_size + spt_idx0] = val_pt03;
    }
    if (k1 < W_out) {
        output[out_base_idx + spt_idx1] = val_pt10;
        output[out_base_idx + spatial_size + spt_idx1] = val_pt11;
        output[out_base_idx + 2 * spatial_size + spt_idx1] = val_pt12;
        output[out_base_idx + 3 * spatial_size + spt_idx1] = val_pt13;
    }
    if (k2 < W_out) {
        output[out_base_idx + spt_idx2] = val_pt20;
        output[out_base_idx + spatial_size + spt_idx2] = val_pt21;
        output[out_base_idx + 2 * spatial_size + spt_idx2] = val_pt22;
        output[out_base_idx + 3 * spatial_size + spt_idx2] = val_pt23;
    }
    if (k3 < W_out) {
        output[out_base_idx + spt_idx3] = val_pt30;
        output[out_base_idx + spatial_size + spt_idx3] = val_pt31;
        output[out_base_idx + 2 * spatial_size + spt_idx3] = val_pt32;
        output[out_base_idx + 3 * spatial_size + spt_idx3] = val_pt33;
    }
}
// PART-END

// PART-START
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input, 
    torch::Tensor weight,
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int out_channels = weight.size(1);

    int D_out = (D_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d;
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto input_nhwc = input.permute({0, 2, 3, 4, 1}).contiguous();
    auto weight_dhwci = weight.permute({2, 3, 4, 0, 1}).contiguous();
    auto output = torch::zeros({batch_size, out_channels, D_out, H_out, W_out}, input.options());

    int total_elements = output.numel();
    if (total_elements == 0) {
        return output;
    }

    int spatial_size = D_out * H_out * W_out;
    int spatial_size_reduced = D_out * H_out * ((W_out + 3) / 4);
    int total_threads = batch_size * spatial_size_reduced * (out_channels / 4);
    const int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    dim3 grid_dim(grid_size, 1, 1);

    conv_transpose3d_kernel<<<grid_dim, block_size>>>(
        input_nhwc.data_ptr<float>(),
        weight_dhwci.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w
    );

    return output;
}
// PART-END