// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__device__ __inline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) 
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void mse_atomic_kernel(const float* predictions, const float* targets, float* total_sum, int total_elements) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    uint index = blockIdx.x * blockDim.x * 8 + tid * 8;
    float sum = 0.0f;
    
    if (index + 7 < total_elements) {
        float4 pred1 = *reinterpret_cast<const float4*>(predictions + index);
        float4 pred2 = *reinterpret_cast<const float4*>(predictions + index + 4);
        float4 target1 = *reinterpret_cast<const float4*>(targets + index);
        float4 target2 = *reinterpret_cast<const float4*>(targets + index + 4);

        float diff1[4] = {pred1.x - target1.x, pred1.y - target1.y, pred1.z - target1.z, pred1.w - target1.w};
        float diff2[4] = {pred2.x - target2.x, pred2.y - target2.y, pred2.z - target2.z, pred2.w - target2.w};

        for (int i = 0; i < 4; i++) sum += diff1[i] * diff1[i];
        for (int i = 0; i < 4; i++) sum += diff2[i] * diff2[i];
    } else {
        for (int i = 0; i < 8; i++) {
            if (index + i < total_elements) {
                float diff = predictions[index+i] - targets[index+i];
                sum += diff * diff;
            }
        }
    }

    float warp_sum = warpReduceSum(sum);
    if (lane_id == 0) sdata[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < blockDim.x / 32) ? sdata[lane_id] : 0.0f;
        float block_sum = warpReduceSum(v);
        if (lane_id == 0) atomicAdd(total_sum, block_sum);
    }
}

__global__ void mse_divide_kernel(const float* total_sum, int total_elements, float* result) {
    *result = *total_sum / total_elements;
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor mse_cuda_forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input sizes must match");
    
    predictions = predictions.contiguous();
    targets = targets.contiguous();
    
    int64_t total_elements = predictions.numel();
    if (total_elements == 0) {
        return torch::zeros({}, predictions.options());
    }
    
    const int block_size = 512;
    const int elements_per_thread = 8;
    const int elements_per_block = block_size * elements_per_thread;
    int grid_size = (total_elements + elements_per_block - 1) / elements_per_block;
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    
    auto total_sum = torch::zeros({}, predictions.options());
    
    mse_atomic_kernel<<<grid_size, block_size, shared_mem_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        total_sum.data_ptr<float>(),
        total_elements
    );
    
    auto result = torch::empty({}, predictions.options());
    mse_divide_kernel<<<1, 1>>>(total_sum.data_ptr<float>(), total_elements, result.data_ptr<float>());
    
    return result;
}
// PART-END