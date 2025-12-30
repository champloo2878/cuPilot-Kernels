// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Helper struct for fast integer division
struct FastDiv {
    int d;
    double magic;

    __host__ __device__ FastDiv() : d(1), magic(1.0) {}
    
    __host__ void init(int divisor) {
        d = divisor;
        if (d == 0) d = 1;
        magic = 1.0 / (double)d;
    }

    __device__ __forceinline__ int div(int n) const {
        return (int)((double)n * magic);
    }

    __device__ __forceinline__ void divmod(int n, int &q, int &r) const {
        q = div(n);
        r = n - q * d;
    }
};

// Async Copy Macros
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(dst), "l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile("cp.async.commit_group;")

#define CP_ASYNC_WAIT_GROUP(N) \
    asm volatile("cp.async.wait_group %0;" :: "n"(N))
// PART-END

// PART-START
#define BM 128
#define BN 128
#define BK 16
#define PAD_A 4
#define PAD_B 8

// Helper for ldmatrix on Matrix A (16x8 TF32)
// This loads a 16x8 TF32 tile from shared memory into registers.
// It maps the 16x8 TF32 layout to the 16x16 B16 layout expected by ldmatrix.x4.
__device__ __forceinline__ void load_a_ldmatrix(unsigned* reg, float* smem_ptr, int stride_in_floats) {
    int tid = threadIdx.x & 31;
    int row = tid % 16;
    int col_off = (tid / 16) * 4; 
    
    float* ptr = smem_ptr + row * stride_in_floats + col_off;
    uint32_t smem_addr = __cvta_generic_to_shared(ptr);
    
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" 
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3]) 
        : "r"(smem_addr));
}

template<int STRIDE, int DILATION, int KERNEL_SIZE, int IN_CHANNELS>
__global__ __launch_bounds__(256) void conv1d_forward_kernel_opt(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    const int out_channels,
    const int input_length,
    const int output_length,
    const int total_elements,
    const FastDiv div_length
) {
    const int total_n = total_elements;
    
    // Grid indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    const int n_start = bx * BN;
    const int m_start = by * BM;

    // Early exit
    if (m_start >= out_channels || n_start >= total_n) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    constexpr int K_total = IN_CHANNELS * KERNEL_SIZE;

    // Shared Memory Setup
    extern __shared__ float smem[];
    
    const int sA_stride = BK + PAD_A; // 20
    const int sA_size = BM * sA_stride; // 2560
    
    const int sB_stride = BN + PAD_B; // 136
    const int sB_size = BK * sB_stride; // 2176
    
    const int sC_stride = BN + PAD_B; // 136
    
    float* smem_A = smem; 
    float* smem_B = smem + 2 * sA_size;
    float* smem_C = smem; 

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][4];

    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int warp_row = warp_id / 2; // 0..3
    const int warp_col = warp_id % 2; // 0..1

    // Precompute N-related indices
    const int tid_n = tid % 128;
    const int global_n = n_start + tid_n;
    
    int b_idx = 0, l_out = 0;
    const float* input_base_n = nullptr;
    bool valid_n = (global_n < total_n);
    
    if (valid_n) {
        div_length.divmod(global_n, b_idx, l_out);
        input_base_n = input + (size_t)b_idx * IN_CHANNELS * input_length;
    }
    
    const int l_out_stride_val = l_out * STRIDE;

    // Prologue: Load Tile 0
    {
        const int write_stage = 0;
        float* smem_A_tile = smem_A + write_stage * sA_size;
        float* smem_B_tile = smem_B + write_stage * sB_size;

        // Load A: [BM, BK]
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = tid + i * 256; 
            int row = idx / 4; 
            int col = (idx % 4) * 4; 

            int global_r = m_start + row;
            float* dst_ptr = smem_A_tile + row * sA_stride + col;
            uint32_t dst_addr = __cvta_generic_to_shared(dst_ptr);

            if (global_r < out_channels) {
                const void* src = (weight + (size_t)global_r * K_total + col);
                CP_ASYNC_CG(dst_addr, src, 16);
            } else {
                *reinterpret_cast<float4*>(dst_ptr) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Load B: [BK, BN]
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = (tid / 128) + i * 2; 
            float* dst_ptr = smem_B_tile + r * sB_stride + tid_n;
            uint32_t dst_addr = __cvta_generic_to_shared(dst_ptr);

            if (valid_n) {
                int c_in = r / KERNEL_SIZE;
                int k_kern = r % KERNEL_SIZE;
                size_t offset = (size_t)c_in * input_length + (l_out_stride_val + k_kern * DILATION);
                const float* src_ptr = input_base_n + offset;
                CP_ASYNC_CG(dst_addr, src_ptr, 4);
            } else {
                *dst_ptr = 0.0f;
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    // Main Loop with Split Pipeline and ldmatrix
    for (int k_step = 0; k_step < K_total; k_step += BK) {
        int next_k = k_step + BK;
        int read_stage = (k_step / BK) % 2;
        int write_stage = 1 - read_stage;
        
        float* smem_A_curr = smem_A + read_stage * sA_size;
        float* smem_B_curr = smem_B + read_stage * sB_size;

        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();

        // Split Compute 1: k=0..8
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            load_a_ldmatrix(reinterpret_cast<unsigned*>(a_frag[i].x), 
                          smem_A_curr + (warp_row * 32 + i * 16) * sA_stride + 0, 
                          sA_stride);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(b_frag[j], smem_B_curr + 0 * sB_stride + (warp_col * 64 + j * 16), sB_stride);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }

        // Issue Load for Next Tile
        if (next_k < K_total) {
            float* smem_A_next = smem_A + write_stage * sA_size;
            float* smem_B_next = smem_B + write_stage * sB_size;

            // Load A
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int idx = tid + i * 256;
                int row = idx / 4;
                int col = (idx % 4) * 4;
                int global_r = m_start + row;
                int global_k = next_k + col;

                float* dst_ptr = smem_A_next + row * sA_stride + col;
                uint32_t dst_addr = __cvta_generic_to_shared(dst_ptr);

                if (global_r < out_channels) {
                    const void* src = (weight + (size_t)global_r * K_total + global_k);
                    CP_ASYNC_CG(dst_addr, src, 16);
                } else {
                    *reinterpret_cast<float4*>(dst_ptr) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }

            // Load B
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int r = (tid / 128) + i * 2;
                int global_k = next_k + r;
                float* dst_ptr = smem_B_next + r * sB_stride + tid_n;
                uint32_t dst_addr = __cvta_generic_to_shared(dst_ptr);

                if (valid_n) {
                    int c_in = global_k / KERNEL_SIZE;
                    int k_kern = global_k % KERNEL_SIZE;
                    size_t offset = (size_t)c_in * input_length + (l_out_stride_val + k_kern * DILATION);
                    const float* src_ptr = input_base_n + offset;
                    CP_ASYNC_CG(dst_addr, src_ptr, 4);
                } else {
                    *dst_ptr = 0.0f;
                }
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        // Split Compute 2: k=8..16
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            load_a_ldmatrix(reinterpret_cast<unsigned*>(a_frag[i].x), 
                          smem_A_curr + (warp_row * 32 + i * 16) * sA_stride + 8, 
                          sA_stride);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::load_matrix_sync(b_frag[j], smem_B_curr + 8 * sB_stride + (warp_col * 64 + j * 16), sB_stride);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    // Store to Shared C
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::store_matrix_sync(smem_C + (warp_row * 32 + i * 16) * sC_stride + (warp_col * 64 + j * 16), acc[i][j], sC_stride, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // Optimized Epilogue with Vectorized Stores
    const int c = (tid * 4) % BN;
    const int r_start = (tid * 4) / BN;
    const int global_n_out = n_start + c;
    
    int b = 0, l_out_idx = 0;
    bool valid_n_base = (global_n_out < total_n);
    bool safe_n = false;
    size_t base_offset = 0;
    bool is_aligned = false;

    if (valid_n_base) {
        div_length.divmod(global_n_out, b, l_out_idx);
        if (l_out_idx + 3 < output_length && global_n_out + 3 < total_n) {
            safe_n = true;
        }
        base_offset = (size_t)b * out_channels * output_length + l_out_idx;
        size_t first_offset = base_offset + (size_t)(m_start + r_start) * output_length;
        is_aligned = (first_offset % 4 == 0);
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int r = r_start + i * 8;
        if (r < BM) {
            int global_r = m_start + r;
            if (global_r < out_channels) {
                float4 val = *reinterpret_cast<float4*>(&smem_C[r * sC_stride + c]);
                if (bias != nullptr) {
                   float b_val = bias[global_r];
                   val.x += b_val; val.y += b_val; val.z += b_val; val.w += b_val;
                }
                if (valid_n_base) {
                    float* out_ptr = output + base_offset + (size_t)global_r * output_length;
                    if (safe_n) {
                        if (is_aligned) {
                            *reinterpret_cast<float4*>(out_ptr) = val;
                        } else {
                            out_ptr[0] = val.x; out_ptr[1] = val.y; out_ptr[2] = val.z; out_ptr[3] = val.w;
                        }
                    } else {
                        float* v = (float*)&val;
                        #pragma unroll
                        for (int k=0; k<4; ++k) {
                            int curr_n = global_n_out + k;
                            if (curr_n < total_n) {
                                int b_k, l_k;
                                div_length.divmod(curr_n, b_k, l_k);
                                size_t out_idx = (size_t)b_k * out_channels * output_length + (size_t)global_r * output_length + l_k;
                                output[out_idx] = v[k];
                            }
                        }
                    }
                }
            }
        }
    }
}
// PART-END

// PART-START
torch::Tensor conv1d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int dilation,
    int output_length
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int input_length = input.size(2);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    auto output = torch::zeros({batch_size, out_channels, output_length}, input.options());
    
    const int total_n = batch_size * output_length;
    
    int grid_x = (total_n + 127) / 128;
    int grid_y = (out_channels + 127) / 128;
    
    dim3 blocks(grid_x, grid_y, 1);
    dim3 threads(256, 1, 1);
    
    size_t shmem_size = 71680; // 70KB
    
    cudaFuncSetAttribute(conv1d_forward_kernel_opt<3, 4, 3, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

    FastDiv div_length;
    div_length.init(output_length);

    conv1d_forward_kernel_opt<3, 4, 3, 64><<<blocks, threads, shmem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        out_channels,
        input_length,
        output_length,
        total_n, 
        div_length
    );
    
    return output;
}
// PART-END