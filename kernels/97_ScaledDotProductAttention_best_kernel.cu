// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <c10/cuda/CUDAStream.h>

const int MAX_SEQ_LENGTH = 512;
const int EMBED_DIM = 1024;
const int WARPS_PER_BLOCK = 4;
const int THREADS_PER_BLOCK = 128;
const int WARP_SIZE = 32;
const int ELEMENTS_PER_THREAD = 32;
const int FLOAT4_PER_THREAD = 4;

// Swizzle function for attention score array to reduce bank conflicts
template<const int kColStride = 16, const int kStep = 8>
static __device__ __forceinline__ int swizzle_permuted_j(int i, int j) {
    static_assert(kColStride <= 16, "kColStride must <= 16");
    static_assert(kStep == 4 || kStep == 8, "kStep must be 8 or 4.");
    static_assert(kColStride % kStep == 0, "kColStride must be multiple of kStep.");
    if constexpr (kStep == 8) {
        return (((j >> 3) ^ (i >> 2)) % (kColStride >> 3)) << 3;
    } else {
        static_assert(kStep == 4);
        return (((j >> 2) ^ (i >> 2)) % (kColStride >> 2)) << 2;
    }
}

// Swizzle addressing for attention score array with XOR-based permutation
static __device__ __forceinline__ int swizzle_attention_index(int warp_id, int seq_idx, int seq_len) {
    // XOR-based swizzle with 128-byte (32 float) base stride for 32 banks
    const int linear_idx = warp_id * seq_len + seq_idx;
    const int swizzled_idx = (linear_idx ^ ((linear_idx >> 5) & 0x1F)) & ((WARPS_PER_BLOCK * seq_len) - 1);
    return swizzled_idx;
}

// Optimized softmax implementation using warp-level primitives with swizzled memory access
template<int SEQ_LEN>
__device__ __forceinline__ void warp_softmax_parallel(float* attention_scores, int warp_id, int lane_id) {
    const int scores_per_thread = SEQ_LEN / WARP_SIZE;
    float thread_max = -1e30f;
    
    // Each thread processes 16 scores (512/32) with swizzled indexing
    #pragma unroll
    for (int i = 0; i < scores_per_thread; i++) {
        int idx = lane_id * scores_per_thread + i;
        int swizzled_idx = swizzle_attention_index(warp_id, idx, SEQ_LEN);
        thread_max = fmaxf(thread_max, attention_scores[swizzled_idx]);
    }
    
    // Warp-wide max reduction
    float max_val = thread_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        max_val = fmaxf(max_val, other);
    }
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    
    // Compute exponentials and sum with swizzled memory access
    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < scores_per_thread; i++) {
        int idx = lane_id * scores_per_thread + i;
        int swizzled_idx = swizzle_attention_index(warp_id, idx, SEQ_LEN);
        float exp_val = expf(attention_scores[swizzled_idx] - max_val);
        attention_scores[swizzled_idx] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-wide sum reduction
    float sum = thread_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum += other;
    }
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    
    // Normalize with reciprocal approximation and swizzled write-back
    float recip_sum = __frcp_rn(sum);
    #pragma unroll
    for (int i = 0; i < scores_per_thread; i++) {
        int idx = lane_id * scores_per_thread + i;
        int swizzled_idx = swizzle_attention_index(warp_id, idx, SEQ_LEN);
        attention_scores[swizzled_idx] *= recip_sum;
    }
}

// Function to compute QK^T with swizzled score storage
template<int SEQ_LEN, int EMBED_DIM>
__device__ __forceinline__ void compute_qk_matmul(
    const float* q_reg, 
    const half* K_head,
    float* attention_scores,
    int warp_id,
    int lane_id,
    float scale_factor) {
    
    const int q_start_col = lane_id * ELEMENTS_PER_THREAD;
    
    #pragma unroll 16
    for (int k = 0; k < SEQ_LEN; k++) {
        float dot_product = 0.0f;
        
        // Vectorized load of K values
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            int col = q_start_col + i * 8;
            float4 k_val = *reinterpret_cast<const float4*>(&K_head[k * EMBED_DIM + col]);
            half* k_half = reinterpret_cast<half*>(&k_val);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                dot_product += q_reg[i * 8 + j] * __half2float(k_half[j]);
            }
        }
        
        // Warp reduction for dot product
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_product += __shfl_down_sync(0xFFFFFFFF, dot_product, offset);
        }
        
        // Store scaled attention score with swizzled indexing
        if (lane_id == 0) {
            int swizzled_idx = swizzle_attention_index(warp_id, k, SEQ_LEN);
            attention_scores[swizzled_idx] = dot_product * scale_factor;
        }
    }
}
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void fused_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int embed_dim,
    const float scale_factor) {
    
    // Shared memory for attention scores with swizzle-optimized layout
    extern __shared__ float attention_scores[];
    
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int token_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    // Early exit if token index out of bounds
    if (token_idx >= seq_len) return;
    
    const long head_offset = (long)blockIdx.z * num_heads * seq_len * embed_dim + 
                             (long)blockIdx.y * seq_len * embed_dim;
    
    const half* Q_head = Q + head_offset;
    const half* K_head = K + head_offset;
    const half* V_head = V + head_offset;
    half* out_head = output + head_offset;
    
    // Load query vector into registers using vectorized loads
    float q_reg[ELEMENTS_PER_THREAD];
    const int q_row = token_idx;
    const int q_start_col = lane_id * ELEMENTS_PER_THREAD;
    
    #pragma unroll
    for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
        int col = q_start_col + i * 8;
        float4 q_val = *reinterpret_cast<const float4*>(&Q_head[q_row * embed_dim + col]);
        half* q_half = reinterpret_cast<half*>(&q_val);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            q_reg[i * 8 + j] = __half2float(q_half[j]);
        }
    }
    
    // Compute Q*K^T with swizzled score storage
    compute_qk_matmul<MAX_SEQ_LENGTH, EMBED_DIM>(
        q_reg, K_head, attention_scores, warp_id, lane_id, scale_factor);
    
    __syncthreads();
    
    // Parallel warp-level softmax with swizzled memory access
    warp_softmax_parallel<MAX_SEQ_LENGTH>(attention_scores, warp_id, lane_id);
    
    __syncthreads();
    
    // Register tiling for output accumulation
    float out_acc0[ELEMENTS_PER_THREAD];
    float out_acc1[ELEMENTS_PER_THREAD];
    float out_acc2[ELEMENTS_PER_THREAD];
    float out_acc3[ELEMENTS_PER_THREAD];
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        out_acc0[i] = 0.0f;
        out_acc1[i] = 0.0f;
        out_acc2[i] = 0.0f;
        out_acc3[i] = 0.0f;
    }
    
    // Weighted accumulation with swizzled attention weight access
    #pragma unroll 4
    for (int k = 0; k < seq_len; k++) {
        int swizzled_idx = swizzle_attention_index(warp_id, k, seq_len);
        float attn_weight = attention_scores[swizzled_idx];
        
        // Vectorized load and accumulation of V values
        #pragma unroll
        for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
            int col = q_start_col + i * 8;
            float4 v_val = *reinterpret_cast<const float4*>(&V_head[k * embed_dim + col]);
            half* v_half = reinterpret_cast<half*>(&v_val);
            
            // Distribute accumulation across accumulator registers
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float v_float = __half2float(v_half[j]);
                int element_idx = i * 8 + j;
                if (element_idx % 4 == 0) out_acc0[element_idx] += attn_weight * v_float;
                else if (element_idx % 4 == 1) out_acc1[element_idx] += attn_weight * v_float;
                else if (element_idx % 4 == 2) out_acc2[element_idx] += attn_weight * v_float;
                else out_acc3[element_idx] += attn_weight * v_float;
            }
        }
    }
    
    // Combine accumulator sets into final output
    float out_final[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        out_final[i] = out_acc0[i] + out_acc1[i] + out_acc2[i] + out_acc3[i];
    }
    
    // Write output to global memory with vectorized stores
    #pragma unroll
    for (int i = 0; i < FLOAT4_PER_THREAD; i++) {
        int col = q_start_col + i * 8;
        float4 out_val;
        half* out_half = reinterpret_cast<half*>(&out_val);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            out_half[j] = __float2half_rn(out_final[i * 8 + j]);
        }
        *reinterpret_cast<float4*>(&out_head[token_idx * embed_dim + col]) = out_val;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor fused_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K, 
    torch::Tensor V) {
    
    auto sizes = Q.sizes();
    int batch_size = sizes[0];
    int num_heads = sizes[1];
    int seq_len = sizes[2];
    int embed_dim = sizes[3];
    
    TORCH_CHECK(seq_len <= MAX_SEQ_LENGTH, "Sequence length exceeds MAX_SEQ_LENGTH.");
    TORCH_CHECK(embed_dim == EMBED_DIM, "Embedding dimension must be 1024.");
    TORCH_CHECK(embed_dim % WARP_SIZE == 0, "Embedding dimension must be divisible by warp size.");
    
    auto output = torch::empty_like(Q);
    
    float scale_factor = 1.0f / sqrtf(static_cast<float>(embed_dim));
    
    // Calculate shared memory size for attention scores
    size_t shared_mem_size = WARPS_PER_BLOCK * seq_len * sizeof(float);
    
    // Optimized grid configuration for A100
    dim3 blocks((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 
                num_heads, 
                batch_size);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    fused_attention_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size, stream>>>(
        reinterpret_cast<half*>(Q.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(K.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(V.data_ptr<torch::Half>()),
        reinterpret_cast<half*>(output.data_ptr<torch::Half>()),
        batch_size,
        num_heads,
        seq_len,
        embed_dim,
        scale_factor);
    
    return output;
}
// PART-END