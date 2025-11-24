// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Memory coalescing optimization helper
template<int kBlockSize, int kElementsPerThread>
__device__ __forceinline__ int get_coalesced_col_idx(int block_col, int thread_id) {
    const int warp_id = thread_id / 32;
    const int lane_id = thread_id % 32;
    const int elements_per_warp = 32 * kElementsPerThread;
    const int warp_col_offset = (warp_id * elements_per_warp) % kBlockSize;
    return block_col + warp_col_offset + lane_id * kElementsPerThread;
}

// Warp-level reduction helper for better Tensor Core utilization
template<int WarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction helper
template<int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warp_reduce_sum<32>(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    if (threadIdx.x < BlockSize / 32) {
        val = shared[threadIdx.x];
    } else {
        val = 0;
    }
    
    if (wid == 0) {
        val = warp_reduce_sum<BlockSize / 32>(val);
    }
    return val;
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void compute_norm_kernel(const float* input, float* norms, int N, int D) {
    const int R = 4;   // Reduced rows per block for better occupancy
    const int T = 256; // Increased threads per row for better parallelism
    
    const int row_base = blockIdx.x * R;
    const int local_row = threadIdx.x / T;
    const int tid_in_row = threadIdx.x % T;
    const int global_row = row_base + local_row;
    
    if (global_row >= N) return;
    
    extern __shared__ float sdata[];
    float* row_sdata = &sdata[local_row * (T + 8)]; // Add padding to avoid bank conflicts
    
    const int D_vec = D & ~7; // Process 8 elements at a time for better memory throughput
    float sq_sum = 0.0f;
    
    // Vectorized reduction with improved memory coalescing
    int base_idx = 8 * tid_in_row;
    const float* row_ptr = input + global_row * D;
    
    while (base_idx < D_vec) {
        float4 val4_1 = reinterpret_cast<const float4*>(row_ptr + base_idx)[0];
        float4 val4_2 = reinterpret_cast<const float4*>(row_ptr + base_idx + 4)[0];
        
        sq_sum += val4_1.x * val4_1.x + val4_1.y * val4_1.y + 
                  val4_1.z * val4_1.z + val4_1.w * val4_1.w +
                  val4_2.x * val4_2.x + val4_2.y * val4_2.y + 
                  val4_2.z * val4_2.z + val4_2.w * val4_2.w;
        base_idx += T * 8;
    }
    
    // Handle remainder with 4-element vectorization
    int rem_start = D_vec + tid_in_row * 4;
    for (int idx = rem_start; idx < D; idx += T * 4) {
        if (idx + 3 < D) {
            float4 val4 = reinterpret_cast<const float4*>(row_ptr + idx)[0];
            sq_sum += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
        } else {
            for (int i = 0; i < 4 && idx + i < D; i++) {
                float val = row_ptr[idx + i];
                sq_sum += val * val;
            }
        }
    }
    
    // Use warp-level reduction first for better efficiency
    float warp_sum = warp_reduce_sum<32>(sq_sum);
    
    if (tid_in_row % 32 == 0) {
        row_sdata[tid_in_row / 32] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid_in_row < 8) { // 256 threads / 32 = 8 warps per row
        float val = (tid_in_row < (T / 32)) ? row_sdata[tid_in_row] : 0.0f;
        val = warp_reduce_sum<8>(val);
        
        if (tid_in_row == 0) {
            norms[global_row] = sqrtf(val + 1e-8f);
        }
    }
}

__global__ void normalize_kernel(const float* input, const float* norms, float* output, int N, int D) {
    const int BLOCK_SIZE = 256;  // Reduced block size for better occupancy
    const int ELEMENTS_PER_THREAD = 8; // Increased elements per thread for better ILP
    
    const int row = blockIdx.x;
    const int block_col = blockIdx.y * (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    const int thread_id = threadIdx.x;
    
    if (row >= N) return;
    
    float norm_val = norms[row];
    const float inv_norm = 1.0f / (norm_val + 1e-8f);
    
    // Process 8 elements per thread with coalesced access pattern
    int col_start = get_coalesced_col_idx<BLOCK_SIZE, ELEMENTS_PER_THREAD>(block_col, thread_id);
    
    // Use vectorized loads/stores for better memory throughput
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
        int col = col_start + i;
        if (col + 3 < D) {
            float4 val4 = reinterpret_cast<const float4*>(input + row * D + col)[0];
            float4 out4;
            out4.x = val4.x * inv_norm;
            out4.y = val4.y * inv_norm;
            out4.z = val4.z * inv_norm;
            out4.w = val4.w * inv_norm;
            reinterpret_cast<float4*>(output + row * D + col)[0] = out4;
        } else if (col < D) {
            for (int j = 0; j < 4 && col + j < D; j++) {
                float val = input[row * D + col + j];
                output[row * D + col + j] = val * inv_norm;
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor l2_norm_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on GPU");
    
    auto orig_shape = input.sizes();
    auto input_flat = input.view({-1, input.size(-2)});
    const int N = input_flat.size(0);
    const int D = input_flat.size(1);
    
    auto output_flat = torch::empty_like(input_flat);
    auto norms = torch::empty(N, input_flat.options());
    
    // Optimized compute kernel launch for A100
    const int R = 4;  // Reduced rows per block for better occupancy
    const int T = 256; // Increased threads for better parallelism
    const int reduce_threads = R * T;
    const int reduce_blocks = (N + R - 1) / R;
    const size_t shared_mem = R * (T + 8) * sizeof(float); // Padded shared memory
    
    compute_norm_kernel<<<reduce_blocks, reduce_threads, shared_mem>>>(
        input_flat.data_ptr<float>(),
        norms.data_ptr<float>(),
        N,
        D
    );
    
    // Optimized normalize kernel launch for A100
    const int BLOCK_SIZE = 256;
    const int ELEMENTS_PER_THREAD = 8;
    const int ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int blocks_per_row = (D + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    dim3 normalize_blocks(N, blocks_per_row);
    const int normalize_threads = BLOCK_SIZE;
    
    normalize_kernel<<<normalize_blocks, normalize_threads>>>(
        input_flat.data_ptr<float>(),
        norms.data_ptr<float>(),
        output_flat.data_ptr<float>(),
        N,
        D
    );
    
    auto output = output_flat.view(orig_shape);
    return output;
}
// PART-END