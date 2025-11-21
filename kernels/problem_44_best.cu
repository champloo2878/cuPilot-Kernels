// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding
) {
    const int batch_channel_index = blockIdx.y;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx_base = tid * 4;
    if (out_idx_base >= output_length) return;
    
    const int input_offset = batch_channel_index * input_length;
    const int output_offset = batch_channel_index * output_length;
    
    uintptr_t output_ptr = reinterpret_cast<uintptr_t>(output + output_offset + out_idx_base);
    bool base_aligned = (output_ptr % 16) == 0;
    float out_vals[4] = {0.0f};

    if (kernel_size == 8 && stride == 1 && padding == 4) {
        const int base_input_idx = out_idx_base - padding;
        if (base_input_idx >= 0 && base_input_idx + 12 <= input_length) {
            const uint4* vec_ptr = reinterpret_cast<const uint4*>(input + input_offset + base_input_idx);
            uint4 vec0 = vec_ptr[0];
            uint4 vec1 = vec_ptr[1];
            uint4 vec2 = vec_ptr[2];
            
            float a0 = __uint_as_float(vec0.x);
            float a1 = __uint_as_float(vec0.y);
            float a2 = __uint_as_float(vec0.z);
            float a3 = __uint_as_float(vec0.w);
            float a4 = __uint_as_float(vec1.x);
            float a5 = __uint_as_float(vec1.y);
            float a6 = __uint_as_float(vec1.z);
            float a7 = __uint_as_float(vec1.w);
            float a8 = __uint_as_float(vec2.x);
            float a9 = __uint_as_float(vec2.y);
            float a10 = __uint_as_float(vec2.z);
            
            // Compute initial window sum
            float sum0 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            // Update sums using sliding window
            float sum1 = sum0 - a0 + a8;
            float sum2 = sum1 - a1 + a9;
            float sum3 = sum2 - a2 + a10;
            
            const float inv_k = 0.125f;
            out_vals[0] = sum0 * inv_k;
            out_vals[1] = sum1 * inv_k;
            out_vals[2] = sum2 * inv_k;
            out_vals[3] = sum3 * inv_k;
        } else {
            float arr[12];
            for (int i = 0; i < 12; i++) {
                int idx = base_input_idx + i;
                arr[i] = (idx >= 0 && idx < input_length) ? input[input_offset + idx] : 0.0f;
            }
            
            float sum0 = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
            float sum1 = sum0 - arr[0] + arr[8];
            float sum2 = sum1 - arr[1] + arr[9];
            float sum3 = sum2 - arr[2] + arr[10];
            
            const float inv_k = 0.125f;
            out_vals[0] = sum0 * inv_k;
            out_vals[1] = sum1 * inv_k;
            out_vals[2] = sum2 * inv_k;
            out_vals[3] = sum3 * inv_k;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int out_idx = out_idx_base + k;
            if (out_idx < output_length) {
                const int input_start = out_idx * stride - padding;
                const int input_end = input_start + kernel_size;
                const int valid_start = max(0, input_start);
                const int valid_end = min(input_length, input_end);
                const int valid_count = valid_end - valid_start;
                
                float sum = 0.0f;
                for (int i = valid_start; i < valid_end; i++) {
                    sum += input[input_offset + i];
                }
                out_vals[k] = valid_count > 0 ? sum / valid_count : 0.0f;
            }
        }
    }

    if (base_aligned && (out_idx_base + 3 < output_length)) {
        uint4 store_val;
        store_val.x = __float_as_uint(out_vals[0]);
        store_val.y = __float_as_uint(out_vals[1]);
        store_val.z = __float_as_uint(out_vals[2]);
        store_val.w = __float_as_uint(out_vals[3]);
        *reinterpret_cast<uint4*>(output + output_offset + out_idx_base) = store_val;
    } else {
        for (int k = 0; k < 4; k++) {
            if (out_idx_base + k < output_length) {
                output[output_offset + out_idx_base + k] = out_vals[k];
            }
        }
    }
}

__global__ void avg_pool1d_kernel_half(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int input_length,
    const int output_length,
    const int kernel_size,
    const int stride,
    const int padding
) {
    const int batch_channel_index = blockIdx.y;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx_base = tid * 4;
    if (out_idx_base >= output_length) return;
    
    const int input_offset = batch_channel_index * input_length;
    const int output_offset = batch_channel_index * output_length;
    
    uintptr_t output_ptr = reinterpret_cast<uintptr_t>(output + output_offset + out_idx_base);
    bool base_aligned = (output_ptr % 8) == 0;
    __half out_vals_half[4] = {__float2half(0.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f)};

    if (kernel_size == 8 && stride == 1 && padding == 4) {
        const int base_input_idx = out_idx_base - padding;
        if (base_input_idx >= 0 && base_input_idx + 12 <= input_length) {
            const half2* in_ptr = reinterpret_cast<const half2*>(input + input_offset + base_input_idx);
            half2 v[6];
            v[0] = in_ptr[0];
            v[1] = in_ptr[1];
            v[2] = in_ptr[2];
            v[3] = in_ptr[3];
            v[4] = in_ptr[4];
            v[5] = in_ptr[5];

            half2 sum_vec0 = __hadd2(v[0], v[1]);
            sum_vec0 = __hadd2(sum_vec0, v[2]);
            sum_vec0 = __hadd2(sum_vec0, v[3]);
            half total0 = __hadd(__low2half(sum_vec0), __high2half(sum_vec0));
            
            half total1 = __hadd(__hsub(total0, __low2half(v[0])), __low2half(v[4]));
            half total2 = __hadd(__hsub(total1, __high2half(v[0])), __high2half(v[4]));
            half total3 = __hadd(__hsub(total2, __low2half(v[1])), __low2half(v[5]));
            
            const half inv_k = __float2half_rn(0.125f);
            out_vals_half[0] = __hmul(total0, inv_k);
            out_vals_half[1] = __hmul(total1, inv_k);
            out_vals_half[2] = __hmul(total2, inv_k);
            out_vals_half[3] = __hmul(total3, inv_k);
        } else {
            __half arr[12];
            for (int i = 0; i < 12; i++) {
                int idx = base_input_idx + i;
                arr[i] = (idx >= 0 && idx < input_length) ? input[input_offset + idx] : __float2half(0.0f);
            }
            __half total0 = arr[0];
            total0 = __hadd(total0, arr[1]);
            total0 = __hadd(total0, arr[2]);
            total0 = __hadd(total0, arr[3]);
            total0 = __hadd(total0, arr[4]);
            total0 = __hadd(total0, arr[5]);
            total0 = __hadd(total0, arr[6]);
            total0 = __hadd(total0, arr[7]);
            
            __half total1 = __hadd(__hsub(total0, arr[0]), arr[8]);
            __half total2 = __hadd(__hsub(total1, arr[1]), arr[9]);
            __half total3 = __hadd(__hsub(total2, arr[2]), arr[10]);
            
            const __half inv_k = __float2half_rn(0.125f);
            out_vals_half[0] = __hmul(total0, inv_k);
            out_vals_half[1] = __hmul(total1, inv_k);
            out_vals_half[2] = __hmul(total2, inv_k);
            out_vals_half[3] = __hmul(total3, inv_k);
        }
    } else {
        float out_vals[4] = {0.0f};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int out_idx = out_idx_base + k;
            if (out_idx < output_length) {
                const int input_start = out_idx * stride - padding;
                const int input_end = input_start + kernel_size;
                const int valid_start = max(0, input_start);
                const int valid_end = min(input_length, input_end);
                const int valid_count = valid_end - valid_start;
                
                float sum = 0.0f;
                for (int i = valid_start; i < valid_end; i++) {
                    sum += __half2float(input[input_offset + i]);
                }
                out_vals[k] = valid_count > 0 ? sum / valid_count : 0.0f;
            }
        }
        out_vals_half[0] = __float2half(out_vals[0]);
        out_vals_half[1] = __float2half(out_vals[1]);
        out_vals_half[2] = __float2half(out_vals[2]);
        out_vals_half[3] = __float2half(out_vals[3]);
    }

    if (base_aligned && (out_idx_base + 3 < output_length)) {
        uint2 store_val;
        reinterpret_cast<__half*>(&store_val)[0] = out_vals_half[0];
        reinterpret_cast<__half*>(&store_val)[1] = out_vals_half[1];
        reinterpret_cast<__half*>(&store_val)[2] = out_vals_half[2];
        reinterpret_cast<__half*>(&store_val)[3] = out_vals_half[3];
        *reinterpret_cast<uint2*>(output + output_offset + out_idx_base) = store_val;
    } else {
        for (int k = 0; k < 4; k++) {
            if (out_idx_base + k < output_length) {
                output[output_offset + out_idx_base + k] = out_vals_half[k];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor avg_pool1d_cuda(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding
) {
    const auto batch_size = input.size(0);
    const auto in_channels = input.size(1);
    const auto input_length = input.size(2);
    
    const int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    auto output = torch::empty({batch_size, in_channels, output_length}, input.options());
    
    if (output_length <= 0) return output;
    
    const int threads = 256;
    const int total_threads_needed = (output_length + 3) / 4;
    const int blocks_x = (total_threads_needed + threads - 1) / threads;
    
    const int total_pairs = batch_size * in_channels;
    
    const dim3 blocks(blocks_x, total_pairs);
    
    if (input.dtype() == torch::kHalf) {
        avg_pool1d_kernel_half<<<blocks, threads>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            batch_size,
            in_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding
        );
    } else {
        avg_pool1d_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding
        );
    }
    
    return output;
}
// PART-END