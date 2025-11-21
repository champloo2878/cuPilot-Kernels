// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

__device__ __forceinline__ float tanh_approx(float x) {
    const float limit = 4.0f;
    if (x > limit) return 1.0f;
    if (x < -limit) return -1.0f;
    
    const float x2 = x * x;
    float term = __fmaf_rn(17.0f/315.0f, x2, 2.0f/15.0f);
    term = __fmaf_rn(-term, x2, 1.0f/3.0f);
    term = -term;
    return __fmaf_rn(x, term * x2, x);
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void __launch_bounds__(512, 4)
gelu_kernel(const float* __restrict__ input, float* __restrict__ output, const int num_elements) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    float sqrt_2_over_pi_val;
    float fused_k_val;
    
    if (lane_id == 0) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float k = 0.044715f;
        sqrt_2_over_pi_val = sqrt_2_over_pi;
        fused_k_val = sqrt_2_over_pi * k;
    }
    
    sqrt_2_over_pi_val = __shfl_sync(0xFFFFFFFF, sqrt_2_over_pi_val, 0);
    fused_k_val = __shfl_sync(0xFFFFFFFF, fused_k_val, 0);
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid * 4;

    if (idx + 3 < num_elements) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        
        #pragma unroll
        for(int i = 0; i < 4; i++) {
            float x = (&in.x)[i];
            const float x_sq = x * x;
            const float temp = __fmaf_rn(fused_k_val, x_sq, sqrt_2_over_pi_val);
            const float inner = x * temp;
            const float tanh_val = tanh_approx(inner);
            const float common = 0.5f * x;
            (&out.x)[i] = __fmaf_rn(common, tanh_val, common);
        }
        
        *reinterpret_cast<float4*>(output + idx) = out;
    } else {
        for(int i = 0; i < 4; i++) {
            const int element_idx = idx + i;
            if(element_idx < num_elements) {
                const float x = input[element_idx];
                const float x_sq = x * x;
                const float temp = __fmaf_rn(fused_k_val, x_sq, sqrt_2_over_pi_val);
                const float inner = x * temp;
                const float tanh_val = tanh_approx(inner);
                const float common = 0.5f * x;
                output[element_idx] = __fmaf_rn(common, tanh_val, common);
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor gelu_forward_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    constexpr int block_size = 512;
    const int grid_size = (num_elements + 4 * block_size - 1) / (4 * block_size);
    
    gelu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements
    );
    
    return output;
}
// PART-END