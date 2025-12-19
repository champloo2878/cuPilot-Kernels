// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#define TILE_COUT 128

// Tuning parameters for A100 TF32 Tensor Cores
// M = Batch * LengthOut, N = OutChannels, K = InChannels * KernelSize
// Optimization: Increased BN to 128 to maximize Input (A) reuse. 
// Input A is loaded once for 128 output channels instead of twice (for BN=64).
#define BM 128   // Number of output time steps per block
#define BN 128   // Number of output channels per block
#define BK 16    // Accumulation chunk size (K dimension)

// Shared Memory Padding to avoid bank conflicts
// PITCH 17 (17 floats = 68 bytes) ensures that column accesses with stride 17
// cycle through all 32 banks (GCD(17, 32) = 1).
#define PITCH_A 17 
#define PITCH_B 17

// Output Shared Memory Pitch
// We store [BN][BM] block (Transposed: N is major, M is minor).
// Optimization: Use PITCH 132 (128 + 4) instead of 136 (128 + 8).
// 136 (Offset 8) causes bank conflicts every 4 rows (32/8=4).
// 132 (Offset 4) causes bank conflicts every 8 rows (32/4=8), reducing conflicts in 16-wide tiles.
// 132 is also a multiple of 4, ensuring 16-byte alignment for float4 writes.
#define PITCH_OUT 132
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
using namespace nvcuda;

__global__ void conv1d_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* output,
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int length_in, 
    int length_out,
    int kernel_size, 
    int stride, 
    int padding, 
    int dilation, 
    bool use_bias
) {
    // Shared Memory Allocation
    // Optimization: Store 3 tiles of Input (for k_tap=0,1,2) to enable loop unrolling and pipelining.
    // s_a layout: 3 blocks of [BM][PITCH_A]
    // s_b layout: [3][BN][PITCH_B] (Stored linearly, BN=128)
    extern __shared__ float smem[];
    
    // Pointers for Input tiles (A)
    float* s_a_k0 = smem;
    float* s_a_k1 = smem + BM * PITCH_A;
    float* s_a_k2 = smem + 2 * BM * PITCH_A;
    
    // Pointers for Weight tiles (B) - starts after the 3 input tiles
    // Offset = 3 * 128 * 17 = 6528 floats
    float* s_b_base = smem + 3 * BM * PITCH_A;
    float* s_b_k0 = s_b_base;
    float* s_b_k1 = s_b_base + BN * PITCH_B;
    float* s_b_k2 = s_b_base + 2 * BN * PITCH_B;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Global coordinates for the top-left of the tile
    int m_global_start = bx * BM;
    int n_global_start = by * BN;

    // Optimization: Precompute Batch and Output Time indices once per thread
    int m_local = tid; 
    int m_global = m_global_start + m_local;
    
    int b_idx = 0;
    int t_out = 0;
    bool m_in_bounds = (m_global < batch_size * length_out);
    if (m_in_bounds) {
        b_idx = m_global / length_out;
        t_out = m_global % length_out;
    }

    // Accumulators: 4 warps (128 threads)
    // Optimization: Increased N tile size requires more accumulators.
    // BN=128, M=128. 4 Warps.
    // Each warp computes 32 (M) x 128 (N).
    // WMMA tile is 16x16. M dimension: 32/16 = 2 tiles. N dimension: 128/16 = 8 tiles.
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][8];

    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    
    // Precompute input addresses and validity for the 3 taps
    // Optimization: Interleaved load logic
    int t_in_base = t_out * stride - padding;
    int t_in_0 = t_in_base;
    int t_in_1 = t_in_base + dilation;
    int t_in_2 = t_in_base + 2 * dilation;
    
    bool valid_0 = m_in_bounds && (t_in_0 >= 0 && t_in_0 < length_in);
    bool valid_1 = m_in_bounds && (t_in_1 >= 0 && t_in_1 < length_in);
    bool valid_2 = m_in_bounds && (t_in_2 >= 0 && t_in_2 < length_in);

    // Outer Loop: Iterate over Input Channels (chunks of BK)
    for (int ic_base = 0; ic_base < in_channels; ic_base += BK) {
        
        // ----------------------------------------------------------------
        // 1. Load Weight Tiles (B) for all k_taps into Shared Memory
        // ----------------------------------------------------------------
        // Total floats to load: BN * 3 * BK. With BN=128, BK=16 -> 6144 floats.
        // float4 loads: 1536. Threads: 128. Iterations: 12.
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            int load_idx = tid + i * 128; 
            int row = load_idx / 12;      // n index
            int col_vec = load_idx % 12;  // float4 index
            int col = col_vec * 4;        // float index
            
            int n_glob = n_global_start + row;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            if (n_glob < out_channels) {
                // Address: n_glob * (in_channels * 3) + ic_base * 3 + col
                size_t offset = (size_t)n_glob * (in_channels * 3) + ic_base * 3 + col;
                val = *reinterpret_cast<const float4*>(&weight[offset]);
            }
            
            // Distribute to s_b_k0, s_b_k1, s_b_k2
            #pragma unroll
            for(int el=0; el<4; ++el) {
                 float v_elem = (el==0) ? val.x : ((el==1) ? val.y : ((el==2) ? val.z : val.w));
                 int c = col + el;
                 int k = c % 3;
                 int ic = c / 3;
                 float* dst = (k == 0) ? s_b_k0 : ((k == 1) ? s_b_k1 : s_b_k2);
                 dst[row * PITCH_B + ic] = v_elem;
            }
        }

        // ----------------------------------------------------------------
        // 2. Load Input Tiles (A) for all 3 k_taps interleaved
        // ----------------------------------------------------------------
        // Base address for this batch and channel block
        size_t base_offset = (size_t)b_idx * in_channels * length_in + ic_base * length_in;
        
        const float* src_0 = input + base_offset + t_in_0;
        const float* src_1 = input + base_offset + t_in_1;
        const float* src_2 = input + base_offset + t_in_2;

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            float v2 = 0.0f;
            
            if (valid_0) v0 = *src_0;
            if (valid_1) v1 = *src_1;
            if (valid_2) v2 = *src_2;
            
            s_a_k0[m_local * PITCH_A + k] = v0;
            s_a_k1[m_local * PITCH_A + k] = v1;
            s_a_k2[m_local * PITCH_A + k] = v2;
            
            // Stride in memory is length_in (moving to next channel)
            src_0 += length_in;
            src_1 += length_in;
            src_2 += length_in;
        }

        __syncthreads();

        // ----------------------------------------------------------------
        // 3. Compute Matrix Multiply (WMMA) - Unrolled k_tap loop
        // ----------------------------------------------------------------
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_frag;
        
        int m_warp_offset = warp_id * 32;

        // Unroll k_tap = 0
        {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                int m_tile_idx = m_warp_offset + ii * 16;
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    int k_offset = kk * 8;
                    wmma::load_matrix_sync(a_frag, &s_a_k0[m_tile_idx * PITCH_A + k_offset], PITCH_A);
                    #pragma unroll
                    for (int jj = 0; jj < 8; ++jj) { // Expanded to 8 for BN=128
                        int n_tile_idx = jj * 16;
                        wmma::load_matrix_sync(b_frag, &s_b_k0[n_tile_idx * PITCH_B + k_offset], PITCH_B);
                        wmma::mma_sync(acc[ii][jj], a_frag, b_frag, acc[ii][jj]);
                    }
                }
            }
        }

        // Unroll k_tap = 1
        {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                int m_tile_idx = m_warp_offset + ii * 16;
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    int k_offset = kk * 8;
                    wmma::load_matrix_sync(a_frag, &s_a_k1[m_tile_idx * PITCH_A + k_offset], PITCH_A);
                    #pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        int n_tile_idx = jj * 16;
                        wmma::load_matrix_sync(b_frag, &s_b_k1[n_tile_idx * PITCH_B + k_offset], PITCH_B);
                        wmma::mma_sync(acc[ii][jj], a_frag, b_frag, acc[ii][jj]);
                    }
                }
            }
        }

        // Unroll k_tap = 2
        {
            #pragma unroll
            for (int ii = 0; ii < 2; ++ii) {
                int m_tile_idx = m_warp_offset + ii * 16;
                #pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    int k_offset = kk * 8;
                    wmma::load_matrix_sync(a_frag, &s_a_k2[m_tile_idx * PITCH_A + k_offset], PITCH_A);
                    #pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        int n_tile_idx = jj * 16;
                        wmma::load_matrix_sync(b_frag, &s_b_k2[n_tile_idx * PITCH_B + k_offset], PITCH_B);
                        wmma::mma_sync(acc[ii][jj], a_frag, b_frag, acc[ii][jj]);
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // 4. Store Output
    // ----------------------------------------------------------------
    // Reuse smem for output. Layout: [BN][PITCH_OUT] (N-major, M-minor)
    float* s_out = smem; 
    
    int m_warp_offset_out = warp_id * 32;
    
    // Store Accumulators to Shared Memory with Transposed Layout
    #pragma unroll
    for (int ii = 0; ii < 2; ++ii) {
        int m_tile_base = m_warp_offset_out + ii * 16;
        #pragma unroll
        for (int jj = 0; jj < 8; ++jj) { // Expanded to 8
            int n_tile_base = jj * 16;
            
            float* dst_ptr = &s_out[n_tile_base * PITCH_OUT + m_tile_base];
            wmma::store_matrix_sync(dst_ptr, acc[ii][jj], PITCH_OUT, wmma::mem_col_major);
        }
    }
    
    __syncthreads();

    // Vectorized Write to Global Memory
    int b_start = m_global_start / length_out;
    int l_start = m_global_start % length_out;

    // Loop over columns assigned to this warp
    // With BN=128, loop condition handles the increased range
    for (int n_c = warp_id; n_c < BN; n_c += 4) {
        int n_abs = n_global_start + n_c;
        if (n_abs >= out_channels) continue;
        
        float bias_val = (use_bias) ? bias[n_abs] : 0.0f;
        
        // Each thread handles 4 contiguous M elements: lane_id*4 ... lane_id*4+3
        int m_local_lane = lane_id * 4;
        
        // Read float4 from SMEM
        float4 v = *(float4*)&s_out[n_c * PITCH_OUT + m_local_lane];
        
        v.x += bias_val;
        v.y += bias_val;
        v.z += bias_val;
        v.w += bias_val;
        
        int m_abs = m_global_start + m_local_lane;
        
        int l_curr = l_start + m_local_lane;
        int b_curr = b_start;
        // Handle wrap-around if block crosses batch boundary
        if (l_curr >= length_out) {
            b_curr += l_curr / length_out;
            l_curr %= length_out;
        }
        
        bool safe = (m_abs + 3 < batch_size * length_out);
        bool contiguous = (l_curr + 3 < length_out); 
        
        long long base_addr = (long long)b_curr * out_channels * length_out + (long long)n_abs * length_out + l_curr;
        
        if (safe && contiguous) {
            // Optimization: check 16-byte alignment of global address
            if ((size_t)(&output[base_addr]) % 16 == 0) {
                *(float4*)&output[base_addr] = v;
            } else {
                output[base_addr]     = v.x;
                output[base_addr + 1] = v.y;
                output[base_addr + 2] = v.z;
                output[base_addr + 3] = v.w;
            }
        } else {
             if (m_abs < batch_size * length_out) {
                 output[base_addr] = v.x;
             }
             
             int m1 = m_abs + 1;
             if (m1 < batch_size * length_out) {
                 int b1 = m1 / length_out;
                 int l1 = m1 % length_out;
                 output[(long long)b1 * out_channels * length_out + (long long)n_abs * length_out + l1] = v.y;
             }
             
             int m2 = m_abs + 2;
             if (m2 < batch_size * length_out) {
                 int b2 = m2 / length_out;
                 int l2 = m2 % length_out;
                 output[(long long)b2 * out_channels * length_out + (long long)n_abs * length_out + l2] = v.z;
             }
             
             int m3 = m_abs + 3;
             if (m3 < batch_size * length_out) {
                 int b3 = m3 / length_out;
                 int l3 = m3 % length_out;
                 output[(long long)b3 * out_channels * length_out + (long long)n_abs * length_out + l3] = v.w;
             }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool use_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int length_in = input.size(2);
    
    int out_channels = weight.size(0);
    int length_out = (length_in + 2 * padding - dilation * (kernel_size - 1) -1) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, length_out}, input.options());

    long long total_m = (long long)batch_size * length_out;
    
    // Grid Dimensions
    // BM = 128, BN = 128 (Increased from 64 to 128)
    int grid_x = (total_m + 128 - 1) / 128;
    int grid_y = (out_channels + 128 - 1) / 128;
    
    dim3 blocks(grid_x, grid_y, 1);
    dim3 threads(128); // 4 Warps
    
    // Shared Memory Size Calculation
    // Optimized for kernel_size=3 loop unrolling and BN=128
    // Input: 3 tiles * 128 (BM) * 17 (PITCH_A) * 4 bytes = 26,112 bytes
    // Weight: 3 * 128 (BN) * 17 (PITCH_B) * 4 bytes = 26,112 bytes
    // Total Input + Weight = 52,224 bytes
    // Output: 128 (BN) * 132 (PITCH_OUT) * 4 bytes = 67,584 bytes
    // Max required = 67,584 bytes (~66 KB)
    int smem_size = 67584; 

    // Enable dynamic shared memory > 48KB for A100
    cudaFuncSetAttribute(conv1d_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    conv1d_kernel<<<blocks, threads, smem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        use_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        length_in,
        length_out,
        kernel_size,
        stride,
        padding,
        dilation,
        use_bias
    );

    cudaDeviceSynchronize();

    return output;
}
// PART-END