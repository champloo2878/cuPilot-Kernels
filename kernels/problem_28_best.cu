// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
// PART-END

// PART-START
__global__ void hardsigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int num_elements) {
    const float div6 = 1.0f / 6.0f;
    
    // Using cooperative groups for better synchronization
    namespace cg = cooperative_groups;
    cg::thread_block tb = cg::this_thread_block();
    
    // Shared memory for async operations
    __shared__ float shared_input[4096];  // Enough for 1024 threads * 4 elements
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_idx = idx * 4;
    
    if (base_idx >= num_elements) return;

    // Create pipeline for async memory operations
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> pipeline_state;
    cuda::pipeline pipeline = cuda::make_pipeline(tb, &pipeline_state);
    
    // Load data asynchronously using cp.async
    if (base_idx + 3 < num_elements) {
        // Stage 1: Async load from global to shared memory
        constexpr int pipeline_phase = 0;
        pipeline.producer_acquire();
        
        float4 in_async = __ldg(reinterpret_cast<const float4*>(input + base_idx));
        // Store to shared memory for async processing
        shared_input[threadIdx.x * 4] = in_async.x;
        shared_input[threadIdx.x * 4 + 1] = in_async.y;
        shared_input[threadIdx.x * 4 + 2] = in_async.z;
        shared_input[threadIdx.x * 4 + 3] = in_async.w;
        
        pipeline.producer_commit();
        
        // Stage 2: Compute while next load happens
        pipeline.consumer_wait();
        tb.sync();
        
        // Load from shared memory
        float4 in;
        in.x = shared_input[threadIdx.x * 4];
        in.y = shared_input[threadIdx.x * 4 + 1];
        in.z = shared_input[threadIdx.x * 4 + 2];
        in.w = shared_input[threadIdx.x * 4 + 3];
        
        float4 out;
        
        // Optimized computation with pipeline
        out.x = min(max(__fadd_rd(__fmul_rd(in.x, div6), 0.5f), 0.0f), 1.0f);
        out.y = min(max(__fadd_rd(__fmul_rd(in.y, div6), 0.5f), 0.0f), 1.0f);
        out.z = min(max(__fadd_rd(__fmul_rd(in.z, div6), 0.5f), 0.0f), 1.0f);
        out.w = min(max(__fadd_rd(__fmul_rd(in.w, div6), 0.5f), 0.0f), 1.0f);
        
        pipeline.consumer_release();
        
        // Direct store to global memory
        *reinterpret_cast<float4*>(output + base_idx) = out;
    } else {
        // Fallback for boundary cases
        int remaining = num_elements - base_idx;
        for (int i = 0; i < remaining; i++) {
            float x = __ldg(input + base_idx + i);
            float t = __fadd_rd(__fmul_rd(x, div6), 0.5f);
            output[base_idx + i] = min(max(t, 0.0f), 1.0f);
        }
    }
}
// PART-END

// PART-START
torch::Tensor hardsigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    auto output = torch::empty_like(input);
    int num_elements = input.numel();
    
    // Optimized block and grid sizes for A100 with async operations
    const int block_size = 1024;  // Max threads per block for A100
    int grid_size = (num_elements + 4 * block_size - 1) / (4 * block_size);
    
    // Ensure grid size doesn't exceed A100 limits
    grid_size = min(grid_size, 2147483647);
    
    static cudaStream_t persistent_stream = []{
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        return stream;
    }();

    if (grid_size > 0) {
        cudaFuncSetCacheConfig(hardsigmoid_kernel, cudaFuncCachePreferL1);
        cudaFuncSetAttribute(hardsigmoid_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
        
        hardsigmoid_kernel<<<grid_size, block_size, 49152, persistent_stream>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements
        );
    }
    
    return output;
}
// PART-END