// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_H 16
#define TILE_W 64
#define BLOCK_SIZE 256
#define C_OUT_PER_GROUP 16
#define K_H 3
#define K_W 5
#define SMEM_IN_H 16
#define SMEM_IN_W 32

// Helper for async copy
__device__ __forceinline__ void cp_async4(void* smem_ptr, const void* glob_ptr) {
    unsigned int smem = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem), "l"(glob_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;");
    else if (n == 1) asm volatile("cp.async.wait_group 1;");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}
// PART-END

// PART-START
__global__ void conv_transpose2d_kernel(
    torch::PackedTensorAccessor32<float, 4> input,
    torch::PackedTensorAccessor32<float, 4> weight,
    const float* __restrict__ bias,
    torch::PackedTensorAccessor32<float, 4> output,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int in_channels,
    int in_h, int in_w,
    int out_channels,
    int out_h, int out_w,
    int kernel_h, int kernel_w
) {
    // Grid: x=tile_x, y=tile_y, z=batch*groups
    int b = blockIdx.z / groups;
    int g = blockIdx.z % groups;
    int tile_y = blockIdx.y;
    int tile_x = blockIdx.x;

    int boy = tile_y * TILE_H; // Base Output Y
    int box = tile_x * TILE_W; // Base Output X

    int tid = threadIdx.x;

    // Registers for accumulation
    // Each thread computes 2 pixels.
    // Pixel 1: (oy1, ox)
    // Pixel 2: (oy2, ox)
    // Due to the specific stride/dilation/padding configuration, only odd rows receive convolution updates.
    float acc1[C_OUT_PER_GROUP];
    float acc2[C_OUT_PER_GROUP];
    
    #pragma unroll
    for (int c = 0; c < C_OUT_PER_GROUP; ++c) {
        acc1[c] = 0.0f;
        acc2[c] = 0.0f;
    }

    // Shared Memory Double Buffers
    // Input: [2][SMEM_IN_H][SMEM_IN_W]
    // Weight: [2][C_OUT_PER_GROUP][K_H][K_W]
    __shared__ float smem_input[2][SMEM_IN_H][SMEM_IN_W];
    __shared__ float smem_weight[2][C_OUT_PER_GROUP][K_H][K_W];

    // Compute input loading base
    // h_in = (h_out + 1 - 2*kh) / 2
    int smem_base_h = (boy + 1) / 2 - 2; 
    int smem_base_w = (box + 2) / 3 - 2;

    int in_c_per_group = in_channels / groups; // 8

    // Thread Mapping
    // 256 threads cover 16x64 tile.
    // Valid convolution rows are odd rows relative to boy: 1, 3, 5, ... 15 (8 rows)
    // 8 rows * 64 cols = 512 pixels.
    // Each thread takes 2 pixels.
    // Mapping: 
    //   lane_y1 = tid / 64 (0..3) -> corresponds to valid rows 0..3 -> (2*lane_y1 + 1)
    //   lane_y2 = lane_y1 + 4 (4..7) -> corresponds to valid rows 4..7 -> (2*lane_y2 + 1)
    
    int lane_x = tid % 64;
    int lane_y1 = tid / 64;
    int lane_y2 = lane_y1 + 4;

    // Relative output Y within tile (odd rows)
    int py1 = lane_y1 * 2 + 1;
    int py2 = lane_y2 * 2 + 1;

    // Global Output Coordinates
    int oy1 = boy + py1;
    int oy2 = boy + py2;
    int ox = box + lane_x;

    // Precompute invariant parts of row calculation
    int base_r1 = (oy1 + 1) / 2 - smem_base_h;
    int base_r2 = (oy2 + 1) / 2 - smem_base_h;

    // Precompute start_kw and step_kw for this column
    int mod = (ox + 2) % 3;
    int start_kw, step_kw;
    if (mod == 0) { start_kw = 0; step_kw = 3; }
    else if (mod == 1) { start_kw = 1; step_kw = 3; }
    else { start_kw = 2; step_kw = 10; }

    // Prologue: Load Stage 0
    {
        int cin = g * in_c_per_group; // step 0
        
        // Load Input: 256 threads * 2 = 512 elements = 16 * 32
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = tid + i * 256;
            int r = load_idx / SMEM_IN_W;
            int c = load_idx % SMEM_IN_W;
            int global_h = smem_base_h + r;
            int global_w = smem_base_w + c;
            
            float* dst = &smem_input[0][r][c];
            const float* src = (global_h >= 0 && global_h < in_h && global_w >= 0 && global_w < in_w) ? 
                               &input[b][cin][global_h][global_w] : nullptr;
            if (src) cp_async4(dst, src);
            else *dst = 0.0f;
        }

        // Load Weight: 240 elements < 256 threads
        if (tid < 240) {
            int rem = tid;
            int kw_idx = rem % 5; rem /= 5;
            int kh_idx = rem % 3; rem /= 3;
            int cout_idx = rem; 
            
            float* dst = &smem_weight[0][cout_idx][kh_idx][kw_idx];
            const float* src = &weight[cin][cout_idx][kh_idx][kw_idx];
            cp_async4(dst, src);
        }
        cp_async_commit();
    }

    // Main Loop
    #pragma unroll 1
    for (int step = 0; step < in_c_per_group; ++step) {
        int next_step = step + 1;
        int buf_idx = step % 2;
        int next_buf_idx = (step + 1) % 2;

        if (next_step < in_c_per_group) {
            int cin_next = g * in_c_per_group + next_step;
            
            // Load Next Input
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                int load_idx = tid + i * 256;
                int r = load_idx / SMEM_IN_W;
                int c = load_idx % SMEM_IN_W;
                int global_h = smem_base_h + r;
                int global_w = smem_base_w + c;
                
                float* dst = &smem_input[next_buf_idx][r][c];
                const float* src = (global_h >= 0 && global_h < in_h && global_w >= 0 && global_w < in_w) ? 
                                   &input[b][cin_next][global_h][global_w] : nullptr;
                if (src) cp_async4(dst, src);
                else *dst = 0.0f;
            }

            // Load Next Weight
            if (tid < 240) {
                int rem = tid;
                int kw_idx = rem % 5; rem /= 5;
                int kh_idx = rem % 3; rem /= 3;
                int cout_idx = rem;
                
                float* dst = &smem_weight[next_buf_idx][cout_idx][kh_idx][kw_idx];
                const float* src = &weight[cin_next][cout_idx][kh_idx][kw_idx];
                cp_async4(dst, src);
            }
            cp_async_commit();
        }

        cp_async_wait_group(next_step < in_c_per_group ? 1 : 0);
        __syncthreads();

        // Compute
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            int r1 = base_r1 - kh;
            int r2 = base_r2 - kh;
            
            bool valid1 = (r1 >= 0 && r1 < SMEM_IN_H);
            bool valid2 = (r2 >= 0 && r2 < SMEM_IN_H);

            if (valid1 || valid2) {
                for (int kw = start_kw; kw < 5; kw += step_kw) {
                    int w_in = (ox + 2 - kw) / 3;
                    int c_idx = w_in - smem_base_w;

                    if (c_idx >= 0 && c_idx < SMEM_IN_W) {
                        float i1 = valid1 ? smem_input[buf_idx][r1][c_idx] : 0.0f;
                        float i2 = valid2 ? smem_input[buf_idx][r2][c_idx] : 0.0f;

                        #pragma unroll
                        for (int co = 0; co < C_OUT_PER_GROUP; ++co) {
                            float w = smem_weight[buf_idx][co][kh][kw];
                            acc1[co] += i1 * w;
                            acc2[co] += i2 * w;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    cp_async_wait_all();

    // Output Writing
    int out_group_offset = g * C_OUT_PER_GROUP;

    // Write Pixel 1 (Computed Odd Row)
    if (oy1 < out_h && ox < out_w) {
        #pragma unroll
        for (int c = 0; c < C_OUT_PER_GROUP; ++c) {
            float val = acc1[c];
            if (bias != nullptr) val += bias[out_group_offset + c];
            output[b][out_group_offset + c][oy1][ox] = val;
        }
    }
    // Write Bias to intermediate even row (oy1-1)
    // For this specific kernel config (S=2, P=1, D=2, K=3), even rows receive no convolution contribution.
    int oy1_even = oy1 - 1;
    if (oy1_even >= 0 && oy1_even < out_h && ox < out_w) {
         #pragma unroll
         for (int c = 0; c < C_OUT_PER_GROUP; ++c) {
             float val = (bias != nullptr) ? bias[out_group_offset + c] : 0.0f;
             output[b][out_group_offset + c][oy1_even][ox] = val;
         }
    }

    // Write Pixel 2 (Computed Odd Row)
    if (oy2 < out_h && ox < out_w) {
        #pragma unroll
        for (int c = 0; c < C_OUT_PER_GROUP; ++c) {
            float val = acc2[c];
            if (bias != nullptr) val += bias[out_group_offset + c];
            output[b][out_group_offset + c][oy2][ox] = val;
        }
    }
    // Write Bias to intermediate even row (oy2-1)
    int oy2_even = oy2 - 1;
    if (oy2_even >= 0 && oy2_even < out_h && ox < out_w) {
         #pragma unroll
         for (int c = 0; c < C_OUT_PER_GROUP; ++c) {
             float val = (bias != nullptr) ? bias[out_group_offset + c] : 0.0f;
             output[b][out_group_offset + c][oy2_even][ox] = val;
         }
    }
}
// PART-END

// PART-START
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int out_h, int out_w
) {
    auto output = torch::zeros({input.size(0), weight.size(1) * groups, out_h, out_w}, input.options());
    const float* bias_data = bias.defined() ? bias.data_ptr<float>() : nullptr;

    auto input_a = input.packed_accessor32<float, 4>();
    auto weight_a = weight.packed_accessor32<float, 4>();
    auto output_a = output.packed_accessor32<float, 4>();

    // Grid Dimensions
    // Output Tile 16x64
    int grid_y = (out_h + TILE_H - 1) / TILE_H;
    int grid_x = (out_w + TILE_W - 1) / TILE_W;
    int grid_z = input.size(0) * groups;

    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(BLOCK_SIZE);

    conv_transpose2d_kernel<<<grid, block>>>(
        input_a, weight_a, bias_data, output_a,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        input.size(1), input.size(2), input.size(3),
        output.size(1), out_h, out_w,
        weight.size(2), weight.size(3)
    );

    return output;
}
// PART-END