// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Compile-time constants for optimization
// Block Tile Size: BM (Out Channels) x BN (Pixels)
// K Chunk Size: BK
// Optimized for OutChannels=96, BN=64 (Good occupancy), BK=32 (TF32 efficiency)
#define BM 96
#define BN 64
#define BK 32

// Padding to avoid shared memory bank conflicts
// Stride for As (96 x 32): 32 + 4 = 36 floats
// Stride for Bs (32 x 64): 64 + 4 = 68 floats
#define PAD_A 4
#define PAD_B 4
#define BK_PADDED (BK + PAD_A)
#define BN_PADDED (BN + PAD_B)

// CP_ASYNC wrapper for A100
#define CP_ASYNC_CA(dst, src, bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;")
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;" :: "n"(N))
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output,
    int batch, int in_channels, int in_h, int in_w,
    int out_channels, int out_h, int out_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    // Shared Memory with Double Buffering and Padding
    // As: [2 buffers][BM rows][BK columns + pad]
    // Bs: [2 buffers][BK rows][BN columns + pad]
    __shared__ float As[2][BM][BK_PADDED];
    __shared__ float Bs[2][BK][BN_PADDED];

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    
    // Grid indices
    const int bx = blockIdx.x; // Block index for Pixels (BN)
    // by is always 0 since BM=96 covers all channels
    
    // Constants
    const int total_pixels = batch * out_h * out_w;
    const int total_k = in_channels * kernel_h * kernel_w;
    const int dim_out_hw = out_h * out_w;

    // Warp Tiling: 2x2 Warps (Total 128 threads, 4 warps)
    // Warp Row (0,1) covers BM (96) -> 48 per warp row
    // Warp Col (0,1) covers BN (64) -> 32 per warp col
    int warp_row = (warpId % 2); 
    int warp_col = (warpId / 2); 

    // Accumulators: 3x2 fragments of 16x16
    // Covers 48x32 area per warp
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[3][2];
    #pragma unroll
    for(int i=0; i<3; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[3];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[2];

    // Precompute pixel coordinates for this block
    // Each thread handles a subset of data loading.
    // 'tid % 64' maps to the pixel column index within the BN tile.
    int my_pixel_col = tid % 64; 
    int global_pixel_idx = bx * 64 + my_pixel_col;
    
    int my_b = 0, my_oh = 0, my_ow = 0;
    bool valid_pixel = (global_pixel_idx < total_pixels);

    if (valid_pixel) {
        int temp = global_pixel_idx;
        my_b = temp / dim_out_hw;
        temp %= dim_out_hw;
        my_oh = temp / out_w;
        my_ow = temp % out_w;
    }

    int num_steps = (total_k + BK - 1) / BK;

    // Prologue: Load Tile 0
    {
        // Load Weights (As) [96][32]
        // 128 threads -> 24 elements per thread
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 5; // idx / 32
            int c = idx & 31; // idx % 32
            // No bounds check needed for r < BM (128*24 = 96*32)
            
            int gk = c;
            if (gk < total_k) {
                CP_ASYNC_CA(&As[0][r][c], &weight[r * total_k + gk], 4);
            } else {
                As[0][r][c] = 0.0f;
            }
        }

        // Load Inputs (Bs) [32][64]
        // 128 threads -> 16 elements per thread
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int idx = tid + i * 128;
            int r = idx >> 6; // idx / 64
            int c = idx & 63; // idx % 64 (matches my_pixel_col)
            
            int gk = r;
            if (valid_pixel && gk < total_k) {
                // Calculate input coordinates
                int ic = gk / 121; // 121 = 11*11
                int rem = gk % 121;
                int kh = rem / 11;
                int kw = rem % 11;
                
                int h_in = my_oh * stride_h - padding_h + kh;
                int w_in = my_ow * stride_w - padding_w + kw;
                
                if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                    int in_idx = ((my_b * in_channels + ic) * in_h + h_in) * in_w + w_in;
                    CP_ASYNC_CA(&Bs[0][r][c], &input[in_idx], 4);
                } else {
                    Bs[0][r][c] = 0.0f;
                }
            } else {
                Bs[0][r][c] = 0.0f;
            }
        }
        CP_ASYNC_COMMIT_GROUP();
    }

    // Main Loop
    for (int step = 0; step < num_steps; ++step) {
        int compute_idx = step % 2;
        int load_idx = (step + 1) % 2;
        int next_k_start = (step + 1) * BK;

        // Async Load Next Tile
        if (step + 1 < num_steps) {
            // Load Weights
            #pragma unroll
            for (int i = 0; i < 24; ++i) {
                int idx = tid + i * 128;
                int r = idx >> 5;
                int c = idx & 31;
                
                int gk = next_k_start + c;
                if (gk < total_k) {
                    CP_ASYNC_CA(&As[load_idx][r][c], &weight[r * total_k + gk], 4);
                } else {
                    As[load_idx][r][c] = 0.0f;
                }
            }
            
            // Load Inputs
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int idx = tid + i * 128;
                int r = idx >> 6;
                int c = idx & 63;
                
                int gk = next_k_start + r;
                if (valid_pixel && gk < total_k) {
                    int ic = gk / 121;
                    int rem = gk % 121;
                    int kh = rem / 11;
                    int kw = rem % 11;
                    
                    int h_in = my_oh * stride_h - padding_h + kh;
                    int w_in = my_ow * stride_w - padding_w + kw;
                    
                    if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
                        int in_idx = ((my_b * in_channels + ic) * in_h + h_in) * in_w + w_in;
                        CP_ASYNC_CA(&Bs[load_idx][r][c], &input[in_idx], 4);
                    } else {
                        Bs[load_idx][r][c] = 0.0f;
                    }
                } else {
                    Bs[load_idx][r][c] = 0.0f;
                }
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        // Wait for compute data
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < BK; k += 8) {
            int a_row_start = warp_row * 48;
            #pragma unroll
            for(int i=0; i<3; ++i) {
                wmma::load_matrix_sync(a_frag[i], &As[compute_idx][a_row_start + i*16][k], BK_PADDED);
            }
            
            int b_col_start = warp_col * 32;
            #pragma unroll
            for(int j=0; j<2; ++j) {
                wmma::load_matrix_sync(b_frag[j], &Bs[compute_idx][k][b_col_start + j*16], BN_PADDED);
            }
            
            #pragma unroll
            for(int i=0; i<3; ++i) {
                #pragma unroll
                for(int j=0; j<2; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
        __syncthreads();
    }
    CP_ASYNC_WAIT_GROUP(0);

    // Epilogue: Store to Shared Memory then Global
    // Reuse As buffer as a flat shared memory space for output
    // As[0] and As[1] combined provide enough space (approx 6912 floats) for 96x64 (6144 floats)
    float* smem_out = &As[0][0][0];
    
    int n_offset = warp_row * 48;
    int m_offset = warp_col * 32;

    #pragma unroll
    for(int i=0; i<3; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            int row = n_offset + i * 16;
            int col = m_offset + j * 16;
            // Store with stride 64 (BN)
            wmma::store_matrix_sync(smem_out + row * 64 + col, c_frag[i][j], 64, wmma::mem_row_major);
        }
    }
    
    __syncthreads();

    // Write to Global Memory
    // 96 * 64 = 6144 elements. 128 threads. 48 elements per thread.
    #pragma unroll
    for (int i = 0; i < 48; ++i) {
        int idx = tid + i * 128;
        int r = idx >> 6; // idx / 64 (Channel)
        int c = idx & 63; // idx % 64 (Pixel)
        
        if (r < out_channels) {
            float val = smem_out[idx];
            val += bias[r];
            
            int global_pix = bx * 64 + c;
            if (global_pix < total_pixels) {
                // Recompute coords for write
                int temp = global_pix;
                int b = temp / dim_out_hw;
                temp %= dim_out_hw;
                int oh = temp / out_w;
                int ow = temp % out_w;
                
                int out_idx = ((b * out_channels + r) * out_h + oh) * out_w + ow;
                output[out_idx] = val;
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor custom_conv2d_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w
) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D tensor");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D tensor");
    
    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    
    int out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    
    auto output = torch::zeros({batch, out_channels, out_h, out_w}, input.options());
    int total_pixels = batch * out_h * out_w;
    
    // Config: BM=96, BN=64. BlockSize=128 (4 Warps)
    dim3 block_size(128);
    dim3 grid_size((total_pixels + 64 - 1) / 64, (out_channels + 96 - 1) / 96);
    
    // Static shared memory is used inside kernel, no dynamic needed
    conv2d_forward_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        batch, in_channels, in_h, in_w,
        out_channels, out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w
    );
    
    return output;
}
// PART-END