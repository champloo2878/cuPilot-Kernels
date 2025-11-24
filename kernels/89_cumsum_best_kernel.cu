// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void inclusive_scan_dim1_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int n) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int row = blockIdx.x * 32 + warp_id;
    
    if (row >= batch_size) return;
    
    // PRE-COMPUTE ROW OFFSET ONCE
    int row_offset = row * n;
    int num_segments = (n + 127) / 128;
    float base = 0.0f;
    
    #pragma unroll 8
    for (int seg = 0; seg < num_segments; seg++) {
        int segment_start = seg * 128;
        int thread_start = segment_start + lane * 4;
        // USE PRE-COMPUTED OFFSET
        int global_idx_base = row_offset + thread_start;
        
        // Vectorized load with boundary checks using texture cache
        float4 in_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (thread_start < n) {
            if (thread_start + 3 < n) {
                in_val = __ldg(reinterpret_cast<const float4*>(input + global_idx_base));
            } else {
                // Handle partial vector at segment end
                for (int i = 0; i < 4 && (thread_start + i) < n; i++) {
                    ((float*)&in_val)[i] = __ldg(input + global_idx_base + i);
                }
            }
        }

        // Parallel prefetch next segment via L2 using multiple threads
        if (seg + 1 < num_segments && lane < 4) {
            int next_seg_start = (seg + 1) * 128;
            // USE PRE-COMPUTED OFFSET
            int next_global_idx_base = row_offset + next_seg_start;
            // Prefetch cache lines in parallel (each thread handles one)
            const void* addr = (const void*)(input + next_global_idx_base + lane * 32);
            asm volatile ("prefetch.global.L2 [%0];" : : "l"(addr));
        }

        // Thread-local inclusive scan
        float4 out_val;
        out_val.x = in_val.x;
        out_val.y = in_val.x + in_val.y;
        out_val.z = out_val.y + in_val.z;
        out_val.w = out_val.z + in_val.w;
        float thread_total = out_val.w;

        // Optimized warp-level scan using butterfly reduction
        float val = thread_total;
        float shuffle_val;
        shuffle_val = __shfl_up_sync(0xFFFFFFFF, val, 1);
        if (lane >= 1) val += shuffle_val;
        shuffle_val = __shfl_up_sync(0xFFFFFFFF, val, 2);
        if (lane >= 2) val += shuffle_val;
        shuffle_val = __shfl_up_sync(0xFFFFFFFF, val, 4);
        if (lane >= 4) val += shuffle_val;
        shuffle_val = __shfl_up_sync(0xFFFFFFFF, val, 8);
        if (lane >= 8) val += shuffle_val;
        shuffle_val = __shfl_up_sync(0xFFFFFFFF, val, 16);
        if (lane >= 16) val += shuffle_val;
        
        float warp_exclusive = val - thread_total;
        float warp_total = __shfl_sync(0xFFFFFFFF, val, 31);

        // Add base and warp prefix to local results
        float offset_val = base + warp_exclusive;
        out_val.x += offset_val;
        out_val.y += offset_val;
        out_val.z += offset_val;
        out_val.w += offset_val;

        // Vectorized store with boundary checks
        if (thread_start < n) {
            if (thread_start + 3 < n) {
                *reinterpret_cast<float4*>(output + global_idx_base) = out_val;
            } else {
                // Handle partial vector at segment end
                for (int i = 0; i < 4 && (thread_start + i) < n; i++) {
                    output[global_idx_base + i] = ((float*)&out_val)[i];
                }
            }
        }

        // Update base for next segment
        base += warp_total;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor inclusive_scan_dim1_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int n = input.size(1);
    int block_size = 1024;
    
    auto output = torch::empty_like(input);
    dim3 grid((batch_size + 31) / 32);
    dim3 block(block_size);
    
    inclusive_scan_dim1_kernel<<<grid, block, 0>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        n
    );
    
    return output;
}
// PART-END