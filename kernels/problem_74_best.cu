#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int const_batch;
__constant__ int const_L_in;
__constant__ int const_L_out;
__constant__ int const_in_channels;
__constant__ int const_out_channels;

__global__ void conv_transpose1d_kernel_optimized(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output
) {
    constexpr int VEC = 8;
    constexpr int POS_PER_THREAD = 4;
    constexpr int KERNEL_SIZE = 5;
    constexpr int DILATION = 3;
    constexpr int MAX_DILATION_OFFSET = (KERNEL_SIZE - 1) * DILATION;
    constexpr int WINDOW_SIZE = MAX_DILATION_OFFSET + POS_PER_THREAD;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int out_ceil = (const_out_channels + VEC - 1) / VEC;
    int total_pos_blocks = (const_L_out + POS_PER_THREAD - 1) / POS_PER_THREAD;
    int total_elements = const_batch * out_ceil * total_pos_blocks;
    if (tid >= total_elements) return;

    int b = tid / (out_ceil * total_pos_blocks);
    int rem = tid % (out_ceil * total_pos_blocks);
    int c_block = rem / total_pos_blocks;
    int pos_block = rem % total_pos_blocks;
    int base_pos = pos_block * POS_PER_THREAD;
    int c_start = c_block * VEC;

    float accum[POS_PER_THREAD][VEC] = {{0.0f}};
    
    bool is_full_internal = (base_pos >= MAX_DILATION_OFFSET) && 
                            (base_pos + POS_PER_THREAD - 1 < const_L_in);
    
    if (is_full_internal) {
        for (int c_in = 0; c_in < const_in_channels; c_in++) {
            const float* input_ptr = input + b * (const_in_channels * const_L_in) + c_in * const_L_in;
            int base_n_min = base_pos - MAX_DILATION_OFFSET;
            float window[WINDOW_SIZE];
            
            float4 v0 = *reinterpret_cast<const float4*>(input_ptr + base_n_min);
            float4 v1 = *reinterpret_cast<const float4*>(input_ptr + base_n_min + 4);
            float4 v2 = *reinterpret_cast<const float4*>(input_ptr + base_n_min + 8);
            float4 v3 = *reinterpret_cast<const float4*>(input_ptr + base_n_min + 12);
            window[0] = v0.x; window[1] = v0.y; window[2] = v0.z; window[3] = v0.w;
            window[4] = v1.x; window[5] = v1.y; window[6] = v1.z; window[7] = v1.w;
            window[8] = v2.x; window[9] = v2.y; window[10] = v2.z; window[11] = v2.w;
            window[12] = v3.x; window[13] = v3.y; window[14] = v3.z; window[15] = v3.w;

            #pragma unroll
            for (int j = 0; j < KERNEL_SIZE; j++) {
                int offset = MAX_DILATION_OFFSET - j * DILATION;
                float in_val[POS_PER_THREAD] = {
                    window[offset],
                    window[offset + 1],
                    window[offset + 2],
                    window[offset + 3]
                };

                float4 weight_vec0 = *reinterpret_cast<const float4*>(weight + c_in * (KERNEL_SIZE * const_out_channels) + j * const_out_channels + c_start);
                float4 weight_vec1 = *reinterpret_cast<const float4*>(weight + c_in * (KERNEL_SIZE * const_out_channels) + j * const_out_channels + c_start + 4);
                float w[VEC] = {weight_vec0.x, weight_vec0.y, weight_vec0.z, weight_vec0.w,
                                weight_vec1.x, weight_vec1.y, weight_vec1.z, weight_vec1.w};

                #pragma unroll
                for (int p = 0; p < POS_PER_THREAD; p++) {
                    #pragma unroll
                    for (int k = 0; k < VEC; k++) {
                        accum[p][k] += in_val[p] * w[k];
                    }
                }
            }
        }
        
        for (int k = 0; k < VEC; k++) {
            int output_idx = b * (const_out_channels * const_L_out) + (c_start + k) * const_L_out + base_pos;
            float4 val = make_float4(accum[0][k], accum[1][k], accum[2][k], accum[3][k]);
            *reinterpret_cast<float4*>(output + output_idx) = val;
        }
    } else {
        for (int c_in = 0; c_in < const_in_channels; c_in++) {
            const float* input_ptr = input + b * (const_in_channels * const_L_in) + c_in * const_L_in;
            #pragma unroll
            for (int j = 0; j < KERNEL_SIZE; j++) {
                int base_n = base_pos - j * DILATION;
                float4 weight_vec0 = *reinterpret_cast<const float4*>(weight + c_in * (KERNEL_SIZE * const_out_channels) + j * const_out_channels + c_start);
                float4 weight_vec1 = *reinterpret_cast<const float4*>(weight + c_in * (KERNEL_SIZE * const_out_channels) + j * const_out_channels + c_start + 4);
                float w[VEC] = {weight_vec0.x, weight_vec0.y, weight_vec0.z, weight_vec0.w,
                                weight_vec1.x, weight_vec1.y, weight_vec1.z, weight_vec1.w};

                if (base_n >=0 && base_n + POS_PER_THREAD - 1 < const_L_in) {
                    float in_val[POS_PER_THREAD];
                    if (base_n % 4 == 0) {
                        float4 in_val4 = *reinterpret_cast<const float4*>(input_ptr + base_n);
                        in_val[0] = in_val4.x;
                        in_val[1] = in_val4.y;
                        in_val[2] = in_val4.z;
                        in_val[3] = in_val4.w;
                    }
                    else if (base_n % 2 == 0) {
                        float2 in_val2a = *reinterpret_cast<const float2*>(input_ptr + base_n);
                        float2 in_val2b = *reinterpret_cast<const float2*>(input_ptr + base_n + 2);
                        in_val[0] = in_val2a.x;
                        in_val[1] = in_val2a.y;
                        in_val[2] = in_val2b.x;
                        in_val[3] = in_val2b.y;
                    }
                    else {
                        in_val[0] = input_ptr[base_n];
                        in_val[1] = input_ptr[base_n+1];
                        in_val[2] = input_ptr[base_n+2];
                        in_val[3] = input_ptr[base_n+3];
                    }
                    
                    #pragma unroll
                    for (int p = 0; p < POS_PER_THREAD; p++) {
                        #pragma unroll
                        for (int k = 0; k < VEC; k++) {
                            accum[p][k] += in_val[p] * w[k];
                        }
                    }
                }
                else {
                    #pragma unroll
                    for (int p = 0; p < POS_PER_THREAD; p++) {
                        int n = base_n + p;
                        if (n >= 0 && n < const_L_in) {
                            float in_val = input_ptr[n];
                            #pragma unroll
                            for (int k = 0; k < VEC; k++) {
                                accum[p][k] += in_val * w[k];
                            }
                        }
                    }
                }
            }
        }
        
        for (int k = 0; k < VEC; k++) {
            if (c_start + k < const_out_channels) {
                #pragma unroll
                for (int p = 0; p < POS_PER_THREAD; p++) {
                    int pos = base_pos + p;
                    if (pos < const_L_out) {
                        int output_idx = b * (const_out_channels * const_L_out) + (c_start + k) * const_L_out + pos;
                        output[output_idx] = accum[p][k];
                    }
                }
            }
        }
    }
}

__global__ void conv_transpose1d_kernel_general(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch,
    int L_in,
    int L_out,
    int in_channels,
    int out_channels,
    int kernel_size,
    int dilation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * L_out;
    if (idx >= total_elements) return;

    int b = idx / (out_channels * L_out);
    int rem = idx % (out_channels * L_out);
    int c_out = rem / L_out;
    int pos_out = rem % L_out;

    float result = 0.0f;

    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int k = 0; k < kernel_size; k++) {
            int pos_in = pos_out - k * dilation;
            if (pos_in >= 0 && pos_in < L_in) {
                int input_idx = b * (in_channels * L_in) + c_in * L_in + pos_in;
                int weight_idx = c_in * (out_channels * kernel_size) + c_out * kernel_size + k;
                result += input[input_idx] * weight[weight_idx];
            }
        }
    }

    output[b * (out_channels * L_out) + c_out * L_out + pos_out] = result;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int batch,
    int L_in,
    int L_out,
    int in_channels,
    int out_channels,
    int kernel_size,
    int dilation
) {
    auto output = torch::zeros({batch, out_channels, L_out}, input.options());

    if (kernel_size == 5 && dilation == 3 && out_channels % 8 == 0) {
        auto weight_reshaped = weight.view({in_channels, out_channels, kernel_size});
        auto weight_transposed = weight_reshaped.permute({0, 2, 1}).contiguous();

        cudaMemcpyToSymbol(const_batch, &batch, sizeof(int));
        cudaMemcpyToSymbol(const_L_in, &L_in, sizeof(int));
        cudaMemcpyToSymbol(const_L_out, &L_out, sizeof(int));
        cudaMemcpyToSymbol(const_in_channels, &in_channels, sizeof(int));
        cudaMemcpyToSymbol(const_out_channels, &out_channels, sizeof(int));
        
        const int VEC = 8;
        const int POS_PER_THREAD = 4;
        int out_ceil = (out_channels + VEC - 1) / VEC;
        int total_pos_blocks = (L_out + POS_PER_THREAD - 1) / POS_PER_THREAD;
        int total_elements = batch * out_ceil * total_pos_blocks;
        int block_size = 128;
        int grid_size = (total_elements + block_size - 1) / block_size;

        conv_transpose1d_kernel_optimized<<<grid_size, block_size>>>(
            input.contiguous().data_ptr<float>(),
            weight_transposed.data_ptr<float>(),
            output.data_ptr<float>()
        );
    } else {
        int total_elements = batch * out_channels * L_out;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;

        conv_transpose1d_kernel_general<<<grid_size, block_size>>>(
            input.contiguous().data_ptr<float>(),
            weight.contiguous().data_ptr<float>(),
            output.data_ptr<float>(),
            batch,
            L_in,
            L_out,
            in_channels,
            out_channels,
            kernel_size,
            dilation
        );
    }

    return output;
}