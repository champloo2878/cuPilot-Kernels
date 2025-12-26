// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

/**
 * Helper function for inclusive warp scan using shuffle instructions.
 * This performs an inclusive product scan across a 32-thread warp.
 */
__device__ __forceinline__ float scan_warp_inclusive(float val) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float t = __shfl_up_sync(mask, val, offset);
        if ((threadIdx.x & 31) >= offset) val *= t;
    }
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void cumulative_product_segment_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    float* segment_products, 
    int outer_size, 
    int scan_size, 
    int num_segments
) {
    /**
     * Optimized Single-Pass Per-Row Cumulative Product Kernel
     * Optimization Strategies:
     * 1. Vectorized Memory Access: Uses float4 for 128-bit memory transactions.
     * 2. Boundary Handling: Explicit checks to handle arbitrary row sizes safely.
     * 3. Hierarchical Scan: Register -> Warp -> Block level scan.
     * 4. Persistent Accumulator: Maintains running product in register to minimize shared memory use.
     * 5. Reduced Shared Memory: Uses only 32 floats for warp aggregates.
     */

    int row = blockIdx.x;
    if (row >= outer_size) return;

    // Use long long for offset calculation to prevent overflow with large tensors
    const long long row_offset = (long long)row * scan_size;
    
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    // Shared memory for warp-level prefix products (32 warps * 4 bytes = 128 bytes)
    __shared__ float s_warp_prods[32];

    // Carry the cumulative product multiplier across chunks of the row
    float running_prod = 1.0f;

    // Process the row in chunks of BLOCK_SIZE * 4 elements
    // For 32768 elements and 1024 threads, this loop runs 8 times
    for (int chunk_start = 0; chunk_start < scan_size; chunk_start += BLOCK_SIZE * 4) {
        int idx = chunk_start + tid * 4;
        float4 v = {1.0f, 1.0f, 1.0f, 1.0f};

        // 1. Vectorized Load with Boundary Check
        if (idx + 3 < scan_size) {
            // Safe to perform aligned float4 load
            v = reinterpret_cast<const float4*>(input + row_offset + idx)[0];
        } else {
            // Fallback to scalar loads for boundary elements
            if (idx < scan_size) v.x = input[row_offset + idx];
            if (idx + 1 < scan_size) v.y = input[row_offset + idx + 1];
            if (idx + 2 < scan_size) v.z = input[row_offset + idx + 2];
            if (idx + 3 < scan_size) v.w = input[row_offset + idx + 3];
        }

        // 2. Thread-local inclusive product scan of the 4 elements in registers
        v.y *= v.x;
        v.z *= v.y;
        v.w *= v.z;
        float thread_total = v.w; 

        // 3. Warp-level inclusive scan of thread totals
        float warp_incl = scan_warp_inclusive(thread_total);
        
        // 4. Store warp aggregates to shared memory for block-level scan
        if (lane == 31) {
            s_warp_prods[wid] = warp_incl;
        }
        __syncthreads();

        // 5. Scan warp aggregates using the first warp (wid == 0)
        if (wid == 0) {
            float wp = (lane < 32) ? s_warp_prods[lane] : 1.0f;
            wp = scan_warp_inclusive(wp);
            if (lane < 32) {
                s_warp_prods[lane] = wp;
            }
        }
        __syncthreads();

        // 6. Apply hierarchical scaling factor
        // block_prefix: product of all preceding warps in this chunk
        float block_prefix = (wid > 0) ? s_warp_prods[wid - 1] : 1.0f;
        // thread_prefix: product of all preceding threads in this warp
        float thread_prefix = __shfl_up_sync(0xffffffff, warp_incl, 1);
        if (lane == 0) thread_prefix = 1.0f;
        
        // Final scale factor combines running product from previous chunks, 
        // preceding warps, and preceding threads.
        float total_scale = running_prod * block_prefix * thread_prefix;

        v.x *= total_scale;
        v.y *= total_scale;
        v.z *= total_scale;
        v.w *= total_scale;

        // 7. Vectorized Write Back with Boundary Check
        if (idx + 3 < scan_size) {
            reinterpret_cast<float4*>(output + row_offset + idx)[0] = v;
        } else {
            if (idx < scan_size) output[row_offset + idx] = v.x;
            if (idx + 1 < scan_size) output[row_offset + idx + 1] = v.y;
            if (idx + 2 < scan_size) output[row_offset + idx + 2] = v.z;
            if (idx + 3 < scan_size) output[row_offset + idx + 3] = v.w;
        }

        // 8. Update running_prod for the next chunk. 
        // s_warp_prods[31] contains the total product of the current chunk.
        running_prod *= s_warp_prods[31];
        __syncthreads();
    }
}

__global__ void compute_scaling_factors_kernel(float* segment_products, float* scaling_factors, int outer_size, int num_segments) {}
__global__ void apply_scaling_kernel(float* output, const float* scaling_factors, int outer_size, int scan_size, int num_segments) {}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor custom_cumprod_2d(torch::Tensor input, int dim) {
    auto original_shape = input.sizes();
    int ndim = original_shape.size();
    
    // Create permutation to move target dim to last position
    std::vector<int64_t> perm(ndim);
    for (int i = 0; i < ndim; ++i) perm[i] = i;
    if (dim != ndim - 1) {
        perm[dim] = ndim - 1;
        perm[ndim - 1] = dim;
    }
    
    // Permute and reshape to 2D tensor
    auto input_perm = input.permute(perm).contiguous();
    int64_t outer_size = input_perm.numel() / input_perm.size(-1);
    int64_t scan_size = input_perm.size(-1);
    auto input_2d = input_perm.view({outer_size, scan_size});
    
    // Allocate output buffer
    auto output_2d = torch::empty_like(input_2d);

    // Launch configuration: One block per row for single-pass processing
    dim3 grid(outer_size);
    dim3 block(BLOCK_SIZE);
    
    cumulative_product_segment_kernel<<<grid, block>>>(
        input_2d.data_ptr<float>(),
        output_2d.data_ptr<float>(),
        nullptr,
        (int)outer_size,
        (int)scan_size,
        0
    );
    
    // Reshape back to original dimensions
    auto output_perm = output_2d.view(input_perm.sizes());
    auto output = output_perm.permute(perm).contiguous();
    
    return output;
}
// PART-END