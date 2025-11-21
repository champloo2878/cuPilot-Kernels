// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv_transpose3d_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int depth,
    int height,
    int width,
    int kernel_depth,
    int kernel_height,
    int kernel_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int padding_d,
    int padding_h,
    int padding_w,
    int groups,
    int depth_out,
    int height_out,
    int width_out,
    bool bias_enabled
) {
    // Precompute output strides
    const int output_stride_c = depth_out * height_out * width_out;
    const int output_stride_d = height_out * width_out;
    const int output_stride_h = width_out;
    
    // Calculate plane size
    const int plane_size = batch_size * (out_channels / 8) * height_out * width_out;
    const int idx_in_plane = blockIdx.x * blockDim.x + threadIdx.x;
    const int d_out = blockIdx.z;
    
    // Bounds check
    if (idx_in_plane >= plane_size) return;
    
    // Unflatten indices
    const int w_out = idx_in_plane % width_out;
    const int h_out = (idx_in_plane / width_out) % height_out;
    const int c_out_base = (idx_in_plane / (width_out * height_out)) % (out_channels / 8);
    const int n = idx_in_plane / (width_out * height_out * (out_channels / 8));
    
    // Group processing
    const int out_channels_per_group = out_channels / groups;
    const int in_channels_per_group = in_channels / groups;
    const int c_out0 = c_out_base * 8;
    const int group_idx = c_out0 / out_channels_per_group;
    const int c_out_local0 = c_out0 % out_channels_per_group;
    const int c_in_start = group_idx * in_channels_per_group;
    const int c_in_end = c_in_start + in_channels_per_group;

    // Precompute strides
    const int input_stride = depth * height * width;
    const int spatial_stride = height * width;
    const int weight_stride = out_channels_per_group;
    const int weight_stride_in = kernel_depth * kernel_height * kernel_width * weight_stride;

    // Precompute input ranges
    const int d_in_min = max(0, (d_out + padding_d - kernel_depth + stride_d) / stride_d);
    const int d_in_max = min(depth-1, (d_out + padding_d) / stride_d);
    const int h_in_min = max(0, (h_out + padding_h - kernel_height + stride_h) / stride_h);
    const int h_in_max = min(height-1, (h_out + padding_h) / stride_h);
    const int w_in_min = max(0, (w_out + padding_w - kernel_width + stride_w) / stride_w);
    const int w_in_max = min(width-1, (w_out + padding_w) / stride_w);

    // FP32 accumulators
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Optimized loops with fixed unrolling
    #pragma unroll 1
    for (int d_in = d_in_min; d_in <= d_in_max; d_in++) {
        const int kd = d_out - d_in * stride_d + padding_d;
        if (kd < 0 || kd >= kernel_depth) continue;
        
        #pragma unroll 1
        for (int h_in = h_in_min; h_in <= h_in_max; h_in++) {
            const int kh = h_out - h_in * stride_h + padding_h;
            if (kh < 0 || kh >= kernel_height) continue;
            
            #pragma unroll
            for (int w_in = w_in_min; w_in <= w_in_max; w_in++) {
                const int kw = w_out - w_in * stride_w + padding_w;
                if (kw < 0 || kw >= kernel_width) continue;

                // Spatial offset
                const int spatial_offset = d_in * spatial_stride + h_in * width + w_in;
                const int input_base = n * in_channels * input_stride + spatial_offset;
                
                // Weight index
                const int weight_base = c_in_start * weight_stride_in +
                                      kd * kernel_height * kernel_width * weight_stride +
                                      kh * kernel_width * weight_stride +
                                      kw * weight_stride +
                                      c_out_local0;
                
                // Channel processing
                for (int c_in = c_in_start; c_in < c_in_end; c_in++) {
                    // Load and convert input
                    const float input_val = __bfloat162float(
                        __ldg(&input[input_base + c_in * input_stride])
                    );
                    
                    // Vectorized weight loading
                    const __nv_bfloat162* w_ptr = reinterpret_cast<const __nv_bfloat162*>(
                        &weight[weight_base + (c_in - c_in_start) * weight_stride_in]
                    );
                    const __nv_bfloat162 w0 = __ldg(w_ptr);
                    const __nv_bfloat162 w1 = __ldg(w_ptr+1);
                    const __nv_bfloat162 w2 = __ldg(w_ptr+2);
                    const __nv_bfloat162 w3 = __ldg(w_ptr+3);
                    
                    // FMA with conversion
                    accum[0] += input_val * __bfloat162float(w0.x);
                    accum[1] += input_val * __bfloat162float(w0.y);
                    accum[2] += input_val * __bfloat162float(w1.x);
                    accum[3] += input_val * __bfloat162float(w1.y);
                    accum[4] += input_val * __bfloat162float(w2.x);
                    accum[5] += input_val * __bfloat162float(w2.y);
                    accum[6] += input_val * __bfloat162float(w3.x);
                    accum[7] += input_val * __bfloat162float(w3.y);
                }
            }
        }
    }

    // Add bias if enabled
    if (bias_enabled) {
        const float4 bias_vec1 = __ldg(reinterpret_cast<const float4*>(&bias[c_out0]));
        const float4 bias_vec2 = __ldg(reinterpret_cast<const float4*>(&bias[c_out0 + 4]));
        accum[0] += bias_vec1.x;
        accum[1] += bias_vec1.y;
        accum[2] += bias_vec1.z;
        accum[3] += bias_vec1.w;
        accum[4] += bias_vec2.x;
        accum[5] += bias_vec2.y;
        accum[6] += bias_vec2.z;
        accum[7] += bias_vec2.w;
    }
    
    // Output indexing
    const int output_base_idx = n * out_channels * output_stride_c +
                               d_out * output_stride_d +
                               h_out * output_stride_h +
                               w_out;
    
    // Store results
    output[output_base_idx + c_out0 * output_stride_c] = accum[0];
    output[output_base_idx + (c_out0 + 1) * output_stride_c] = accum[1];
    output[output_base_idx + (c_out0 + 2) * output_stride_c] = accum[2];
    output[output_base_idx + (c_out0 + 3) * output_stride_c] = accum[3];
    output[output_base_idx + (c_out0 + 4) * output_stride_c] = accum[4];
    output[output_base_idx + (c_out0 + 5) * output_stride_c] = accum[5];
    output[output_base_idx + (c_out0 + 6) * output_stride_c] = accum[6];
    output[output_base_idx + (c_out0 + 7) * output_stride_c] = accum[7];
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int output_padding_d, int output_padding_h, int output_padding_w,
    int groups
) {
    // Input dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    // Weight dimensions
    const int kernel_depth = weight.size(2);
    const int kernel_height = weight.size(3);
    const int kernel_width = weight.size(4);
    const int out_channels = weight.size(1) * groups;
    
    // Output dimensions
    const int depth_out = (depth - 1) * stride_d - 2 * padding_d + kernel_depth + output_padding_d;
    const int height_out = (height - 1) * stride_h - 2 * padding_h + kernel_height + output_padding_h;
    const int width_out = (width - 1) * stride_w - 2 * padding_w + kernel_width + output_padding_w;
    
    // Validate kernel size
    TORCH_CHECK(kernel_depth == 3 && kernel_height == 5 && kernel_width == 5,
                "conv_transpose3d_cuda: only kernel_size=(3,5,5) is supported");
    
    // Reorder weight
    auto weight_reordered = weight.permute({0, 2, 3, 4, 1}).contiguous();
    
    // Convert to BF16
    auto input_bf16 = input.to(torch::kBFloat16);
    auto weight_bf16 = weight_reordered.to(torch::kBFloat16);
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, depth_out, height_out, width_out}, 
                             input.options().dtype(torch::kFloat));
    
    // Validate vectorization
    if (out_channels % 8 != 0) {
        AT_ERROR("conv_transpose3d_cuda: out_channels must be divisible by 8");
    }
    
    // Grid configuration
    const int block_size = 256;
    const int plane_size = batch_size * (out_channels / 8) * height_out * width_out;
    const int grid_size_x = (plane_size + block_size - 1) / block_size;
    dim3 grid(grid_size_x, 1, depth_out);
    
    // Bias handling
    const bool bias_enabled = bias.has_value();
    const float* bias_ptr = bias_enabled ? bias->data_ptr<float>() : nullptr;
    
    // Launch kernel
    conv_transpose3d_kernel<<<grid, block_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_bf16.data_ptr<torch::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16.data_ptr<torch::BFloat16>()),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_depth,
        kernel_height,
        kernel_width,
        stride_d,
        stride_h,
        stride_w,
        padding_d,
        padding_h,
        padding_w,
        groups,
        depth_out,
        height_out,
        width_out,
        bias_enabled
    );
    
    return output;
}
// PART-END