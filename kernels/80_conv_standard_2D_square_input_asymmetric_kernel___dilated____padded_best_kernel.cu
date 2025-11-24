// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <tuple>
using namespace nvcuda;
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv2d_kernel(const half* __restrict__ input, const half* __restrict__ weight, const half* __restrict__ bias, float* output) {
    // Hardcoded parameters for fixed problem size
    constexpr int BATCH_SIZE = 8;
    constexpr int IN_CHANNELS = 32;
    constexpr int OUT_CHANNELS = 64;
    constexpr int IN_HEIGHT = 512;
    constexpr int IN_WIDTH = 512;
    constexpr int OUT_HEIGHT = 508;
    constexpr int OUT_WIDTH = 496;
    constexpr int KERNEL_H = 5;
    constexpr int KERNEL_W = 9;
    constexpr int STRIDE = 1;
    constexpr int PADDING_H = 2;
    constexpr int PADDING_W = 4;
    constexpr int DILATION_H = 2;
    constexpr int DILATION_W = 3;
    
    // Precomputed constants
    constexpr int K = IN_CHANNELS * KERNEL_H * KERNEL_W;  // 32*5*9 = 1440
    constexpr int OUTPUT_STRIDE = OUT_HEIGHT * OUT_WIDTH;
    constexpr int CHANNEL_STRIDE = OUT_CHANNELS * OUTPUT_STRIDE;
    
    // Tile dimensions optimized for Tensor Cores
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int TILE_OUT_C = 64;
    constexpr int K_TILE = 16;
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr int WARP_SIZE = 32;
    constexpr int n_chunks = K / K_TILE;  // 1440/16 = 90
    
    // Shared memory allocation
    extern __shared__ __align__(16) char smem[];
    half (*A_shared)[64][16] = reinterpret_cast<half(*)[64][16]>(smem);
    half (*B_shared)[64][16] = reinterpret_cast<half(*)[64][16]>(smem + 2 * 64 * 16 * sizeof(half));
    float (*C_shared)[64] = reinterpret_cast<float(*)[64]>(smem + 4 * 64 * 16 * sizeof(half));
    float* bias_shared = reinterpret_cast<float*>(smem + 4 * 64 * 16 * sizeof(half) + 64 * 64 * sizeof(float));
    
    // Thread indexing
    int batch_idx = blockIdx.z;
    int start_h_out = blockIdx.y * TILE_H;
    int start_w_out = blockIdx.x * TILE_W;
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_m = warp_id / 4;
    int warp_n = warp_id % 4;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Base coordinates (with boundary adjustments)
    int base_h = start_h_out - PADDING_H;
    int base_w = start_w_out - PADDING_W;
    
    // OPTIMIZED: Parallel bias loading using all available threads
    if (threadIdx.x < OUT_CHANNELS) {
        float bias_val = (bias != nullptr) ? __half2float(__ldg(bias + threadIdx.x)) : 0.0f;
        bias_shared[threadIdx.x] = bias_val;
    }
    __syncthreads();
    
    // WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_c;
    nvcuda::wmma::fill_fragment(frag_c, 0.0f);
    
    // Preload first chunk
    int k_start0 = 0;
    for (int idx = threadIdx.x; idx < 64*16; idx += blockDim.x) {
        int spatial_idx = idx / 16;
        int k_local = idx % 16;
        int k_global = k_start0 + k_local;
        
        int ic = k_global / (KERNEL_H * KERNEL_W);
        int khw = k_global % (KERNEL_H * KERNEL_W);
        int kh = khw / KERNEL_W;
        int kw = khw % KERNEL_W;
        
        int h_out = spatial_idx / TILE_W;
        int w_out = spatial_idx % TILE_W;
        int h_in = base_h + h_out * STRIDE + kh * DILATION_H;
        int w_in = base_w + w_out * STRIDE + kw * DILATION_W;
        
        half val = __float2half(0.0f);
        if (h_in >= 0 && h_in < IN_HEIGHT && w_in >= 0 && w_in < IN_WIDTH) {
            int input_idx = ((batch_idx * IN_CHANNELS + ic) * IN_HEIGHT + h_in) * IN_WIDTH + w_in;
            val = __ldg(input + input_idx);
        }
        A_shared[0][spatial_idx][k_local] = val;
    }
    
    constexpr int VEC_ELEMS = 8;
    constexpr int WEIGHT_VECS_PER_CHUNK = (K_TILE * TILE_OUT_C) / VEC_ELEMS;
    for (int vec_idx = threadIdx.x; vec_idx < WEIGHT_VECS_PER_CHUNK; vec_idx += blockDim.x) {
        int oc = vec_idx / (K_TILE / VEC_ELEMS);
        int seg = vec_idx % (K_TILE / VEC_ELEMS);
        int k_local = seg * VEC_ELEMS;
        int weight_idx = oc * K + k_local;
        const uint4* weight_vec_ptr = reinterpret_cast<const uint4*>(weight + weight_idx);
        uint4 vec = __ldg(weight_vec_ptr);
        half* vals = reinterpret_cast<half*>(&vec);
        
        #pragma unroll
        for (int i = 0; i < VEC_ELEMS; ++i) {
            B_shared[0][oc][k_local + i] = vals[i];
        }
    }
    __syncthreads();
    
    // Main computation loop with overlapped loading
    if (n_chunks > 1) {
        for (int chunk = 0; chunk < n_chunks-1; chunk++) {
            int buf_current = chunk % 2;
            int buf_next = (chunk + 1) % 2;
            
            // WMMA computation for current chunk
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
            
            nvcuda::wmma::load_matrix_sync(frag_a, &A_shared[buf_current][warp_m * 16][0], 16);
            nvcuda::wmma::load_matrix_sync(frag_b, &B_shared[buf_current][warp_n * 16][0], 16);
            nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
            
            // Preload next chunk while computing current
            int k_start_next = (chunk+1) * K_TILE;
            for (int idx = threadIdx.x; idx < 64*16; idx += blockDim.x) {
                int spatial_idx = idx / 16;
                int k_local = idx % 16;
                int k_global = k_start_next + k_local;
                
                int ic = k_global / (KERNEL_H * KERNEL_W);
                int khw = k_global % (KERNEL_H * KERNEL_W);
                int kh = khw / KERNEL_W;
                int kw = khw % KERNEL_W;
                
                int h_out = spatial_idx / TILE_W;
                int w_out = spatial_idx % TILE_W;
                int h_in = base_h + h_out * STRIDE + kh * DILATION_H;
                int w_in = base_w + w_out * STRIDE + kw * DILATION_W;
                
                half val = __float2half(0.0f);
                if (h_in >= 0 && h_in < IN_HEIGHT && w_in >= 0 && w_in < IN_WIDTH) {
                    int input_idx = ((batch_idx * IN_CHANNELS + ic) * IN_HEIGHT + h_in) * IN_WIDTH + w_in;
                    val = __ldg(input + input_idx);
                }
                A_shared[buf_next][spatial_idx][k_local] = val;
            }
            
            for (int vec_idx = threadIdx.x; vec_idx < WEIGHT_VECS_PER_CHUNK; vec_idx += blockDim.x) {
                int oc = vec_idx / (K_TILE / VEC_ELEMS);
                int seg = vec_idx % (K_TILE / VEC_ELEMS);
                int k_local = seg * VEC_ELEMS;
                int weight_idx = oc * K + k_start_next + k_local;
                const uint4* weight_vec_ptr = reinterpret_cast<const uint4*>(weight + weight_idx);
                uint4 vec = __ldg(weight_vec_ptr);
                half* vals = reinterpret_cast<half*>(&vec);
                
                #pragma unroll
                for (int i = 0; i < VEC_ELEMS; ++i) {
                    B_shared[buf_next][oc][k_local + i] = vals[i];
                }
            }
            __syncthreads();
        }
    }
    
    // Process last chunk
    if (n_chunks > 0) {
        int buf_last = (n_chunks-1) % 2;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> frag_a;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> frag_b;
        
        nvcuda::wmma::load_matrix_sync(frag_a, &A_shared[buf_last][warp_m * 16][0], 16);
        nvcuda::wmma::load_matrix_sync(frag_b, &B_shared[buf_last][warp_n * 16][0], 16);
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    // Store accumulated results
    nvcuda::wmma::store_matrix_sync(&C_shared[warp_n * 16][warp_m * 16], frag_c, 64, nvcuda::wmma::mem_col_major);
    __syncthreads();
    
    // OPTIMIZED: Warp-strided output writing for coalesced memory access
    int warp_chan_base = warp_id * 4;
    for (int local_chan = 0; local_chan < 4; local_chan++) {
        int oc = warp_chan_base + local_chan;
        if (oc >= OUT_CHANNELS) continue;
        
        for (int spatial_idx = lane_id; spatial_idx < 64; spatial_idx += WARP_SIZE) {
            int tile_h = spatial_idx / TILE_W;
            int tile_w = spatial_idx % TILE_W;
            int h_out = start_h_out + tile_h;
            int w_out = start_w_out + tile_w;
            
            if (h_out < OUT_HEIGHT && w_out < OUT_WIDTH) {
                float value = C_shared[oc][spatial_idx];
                if (bias) value += bias_shared[oc];
                int output_idx = batch_idx * CHANNEL_STRIDE + oc * OUTPUT_STRIDE + h_out * OUT_WIDTH + w_out;
                output[output_idx] = value;
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor custom_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                 int stride, std::tuple<int, int> padding, std::tuple<int, int> dilation) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be on CUDA");
    
    // Validate fixed parameters
    TORCH_CHECK(input.size(0) == 8, "Batch size must be 8");
    TORCH_CHECK(input.size(1) == 32, "Input channels must be 32");
    TORCH_CHECK(input.size(2) == 512 && input.size(3) == 512, "Input dimensions must be 512x512");
    TORCH_CHECK(weight.size(0) == 64, "Output channels must be 64");
    TORCH_CHECK(weight.size(2) == 5 && weight.size(3) == 9, "Kernel size must be 5x9");
    TORCH_CHECK(stride == 1, "Stride must be 1");
    TORCH_CHECK(std::get<0>(padding) == 2 && std::get<1>(padding) == 4, "Padding must be (2,4)");
    TORCH_CHECK(std::get<0>(dilation) == 2 && std::get<1>(dilation) == 3, "Dilation must be (2,3)");
    
    // Fixed output dimensions
    constexpr int height_out = 508;
    constexpr int width_out = 496;
    
    auto output = torch::zeros({8, 64, height_out, width_out}, input.options());
    if (output.numel() == 0) {
        return output;
    }
    
    // Prepare tensors
    auto input_fp16 = input.to(torch::kFloat16);
    auto weight_reshaped = weight.view({64, -1}).to(torch::kFloat16);
    const half* bias_ptr = nullptr;
    if (bias.defined()) {
        auto bias_fp16 = bias.to(torch::kFloat16);
        bias_ptr = reinterpret_cast<const half*>(bias_fp16.data_ptr<torch::Half>());
    }
    
    // Optimized kernel parameters
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int BLOCK_SIZE = 512;
    constexpr int grid_x = (width_out + TILE_W - 1) / TILE_W;   // 496/8 = 62
    constexpr int grid_y = (height_out + TILE_H - 1) / TILE_H;  // 508/8 = 64
    dim3 grid(grid_x, grid_y, 8);
    
    // Shared memory calculation
    constexpr size_t smem_size = 4 * 64 * 16 * sizeof(half) +  // A/B shared (double buffered)
                                 64 * 64 * sizeof(float) +     // C shared
                                 64 * sizeof(float);           // Bias shared
    
    // Launch optimized kernel
    conv2d_kernel<<<grid, BLOCK_SIZE, smem_size>>>(
        reinterpret_cast<const half*>(input_fp16.data_ptr<torch::Half>()),
        reinterpret_cast<const half*>(weight_reshaped.data_ptr<torch::Half>()),
        bias_ptr,
        output.data_ptr<float>()
    );
    
    return output;
}
// PART-END