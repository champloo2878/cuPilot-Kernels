// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
__global__ void conv_transpose3d_inner_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch, int in_channels_padded, int out_channels_padded,
    int depth, int height, int width,
    int kernel_size, int stride, int padding, int dilation,
    int D_out, int H_out, int W_out,
    int grid_d, int grid_h, int grid_w,
    int in_channels_orig, int out_channels_orig
) {
    constexpr int TILE_IC = 8;
    constexpr int TILE_OC = 24;
    constexpr int TILE_D = 4;
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VEC = TILE_OC / VEC_SIZE;
    constexpr int NUM_KERNELS = 27;
    constexpr int INPUT_SHMEM_STRIDE = TILE_IC + 1; // Add padding to avoid bank conflicts
    constexpr int INPUT_SHMEM_SIZE = INPUT_SHMEM_STRIDE * (TILE_D+2) * (TILE_H+2) * (TILE_W+2);
    constexpr int WEIGHT_SHMEM_SIZE = NUM_KERNELS * TILE_IC * TILE_OC;
    
    extern __shared__ float shmem[];
    float* input_shared = shmem;
    float* weight_shared = shmem + INPUT_SHMEM_SIZE;

    int n = blockIdx.x;
    int oc_block = blockIdx.y;
    int inner_spatial_block = blockIdx.z;
    
    int inner_grid_h = (grid_h - 2);
    int inner_grid_w = (grid_w - 2);
    int inner_grid_d = (grid_d - 2);
    
    int d_block = inner_spatial_block / (inner_grid_h * inner_grid_w) + 1;
    int hw_block = inner_spatial_block % (inner_grid_h * inner_grid_w);
    int h_block = hw_block / inner_grid_w + 1;
    int w_block = hw_block % inner_grid_w + 1;
    
    int tid_w = threadIdx.x;
    int tid_h = threadIdx.y;
    int tid_d = threadIdx.z;
    
    int d_out = d_block * TILE_D + tid_d;
    int h_out = h_block * TILE_H + tid_h;
    int w_out = w_block * TILE_W + tid_w;
    
    float4 accum4[NUM_VEC];
    #pragma unroll
    for (int i = 0; i < NUM_VEC; i++) {
        accum4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    bool valid = (d_out < D_out) && (h_out < H_out) && (w_out < W_out);

    int d_in_start = d_block * TILE_D - 2;
    int d_in_size = TILE_D + 2;
    int h_in_start = h_block * TILE_H - 2;
    int h_in_size = TILE_H + 2;
    int w_in_start = w_block * TILE_W - 2;
    int w_in_size = TILE_W + 2;

    int d_base = d_out - d_in_start;
    int h_base = h_out - h_in_start;
    int w_base = w_out - w_in_start;

    int num_ic_blocks = in_channels_padded / TILE_IC;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    
    for (int ic_block = 0; ic_block < in_channels_padded; ic_block += TILE_IC) {
        int tid = tid_d * blockDim.x * blockDim.y + tid_h * blockDim.x + tid_w;
        
        for (int idx = tid; idx < d_in_size * h_in_size * w_in_size * TILE_IC; idx += total_threads) {
            int ic_local = idx % TILE_IC;
            int spatial_idx = idx / TILE_IC;
            int w_in = spatial_idx % w_in_size;
            int h_in = (spatial_idx / w_in_size) % h_in_size;
            int d_in = spatial_idx / (w_in_size * h_in_size);
            
            int d_val = d_in_start + d_in;
            int h_val = h_in_start + h_in;
            int w_val = w_in_start + w_in;
            
            float val = 0.0f;
            if (ic_block + ic_local < in_channels_orig &&
                d_val >= 0 && d_val < depth &&
                h_val >= 0 && h_val < height &&
                w_val >= 0 && w_val < width) {
                int input_idx = n * (in_channels_orig * depth * height * width)
                              + (ic_block + ic_local) * (depth * height * width)
                              + d_val * (height * width)
                              + h_val * width
                              + w_val;
                val = input[input_idx];
            }
            // Apply padding to avoid bank conflicts
            input_shared[spatial_idx * INPUT_SHMEM_STRIDE + ic_local] = val;
        }
        
        // Vectorized weight loading
        int num_float4_per_weight_block = (NUM_KERNELS * TILE_IC * TILE_OC) / 4;
        int ic_block_block = ic_block / TILE_IC;
        int base_offset = (oc_block * num_ic_blocks + ic_block_block) * NUM_KERNELS * TILE_IC * TILE_OC;
        for (int idx4 = tid; idx4 < num_float4_per_weight_block; idx4 += total_threads) {
            int global_idx = base_offset + idx4 * 4;
            float4 val4 = reinterpret_cast<const float4*>(weight)[global_idx / 4];
            reinterpret_cast<float4*>(weight_shared)[idx4] = val4;
        }
        
        __syncthreads();
        
        if (valid) {
            #pragma unroll
            for (int kd = 0; kd < 3; kd++) {
                #pragma unroll
                for (int kh = 0; kh < 3; kh++) {
                    #pragma unroll
                    for (int kw = 0; kw < 3; kw++) {
                        int d_rel = d_base - kd;
                        int h_rel = h_base - kh;
                        int w_rel = w_base - kw;
                        int spatial_idx = d_rel * (h_in_size * w_in_size) + h_rel * w_in_size + w_rel;
                        
                        #pragma unroll
                        for (int ic = 0; ic < TILE_IC; ic++) {
                            // Use padded indexing for conflict-free access
                            float in_val = input_shared[spatial_idx * INPUT_SHMEM_STRIDE + ic];
                            int weight_base = (kd*9 + kh*3 + kw) * (TILE_IC * TILE_OC) + ic * TILE_OC;
                            
                            float4* weight_ptr = reinterpret_cast<float4*>(weight_shared + weight_base);
                            #pragma unroll
                            for (int vec_idx = 0; vec_idx < NUM_VEC; vec_idx++) {
                                float4 w4 = weight_ptr[vec_idx];
                                accum4[vec_idx].x += in_val * w4.x;
                                accum4[vec_idx].y += in_val * w4.y;
                                accum4[vec_idx].z += in_val * w4.z;
                                accum4[vec_idx].w += in_val * w4.w;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (valid) {
        float* accum = reinterpret_cast<float*>(accum4);
        for (int oc_local = 0; oc_local < TILE_OC; oc_local++) {
            if (oc_block * TILE_OC + oc_local < out_channels_orig) {
                int output_idx = n * (out_channels_orig * D_out * H_out * W_out)
                              + (oc_block * TILE_OC + oc_local) * (D_out * H_out * W_out)
                              + d_out * (H_out * W_out)
                              + h_out * W_out
                              + w_out;
                output[output_idx] = accum[oc_local];
            }
        }
    }
}

__global__ void conv_transpose3d_boundary_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int* __restrict__ boundary_indices,
    float* __restrict__ output,
    int batch, int in_channels_padded, int out_channels_padded,
    int depth, int height, int width,
    int kernel_size, int stride, int padding, int dilation,
    int D_out, int H_out, int W_out,
    int grid_d, int grid_h, int grid_w,
    int boundary_spatial_size,
    int in_channels_orig, int out_channels_orig
) {
    constexpr int TILE_IC = 8;
    constexpr int TILE_OC = 24;
    constexpr int TILE_D = 4;
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;
    constexpr int VEC_SIZE = 4;
    constexpr int NUM_VEC = TILE_OC / VEC_SIZE;
    constexpr int NUM_KERNELS = 27;
    constexpr int INPUT_SHMEM_STRIDE = TILE_IC + 1; // Add padding to avoid bank conflicts
    constexpr int INPUT_SHMEM_SIZE = INPUT_SHMEM_STRIDE * (TILE_D+2) * (TILE_H+2) * (TILE_W+2);
    constexpr int WEIGHT_SHMEM_SIZE = NUM_KERNELS * TILE_IC * TILE_OC;
    
    extern __shared__ float shmem[];
    float* input_shared = shmem;
    float* weight_shared = shmem + INPUT_SHMEM_SIZE;

    int n = blockIdx.x;
    int oc_block = blockIdx.y;
    int boundary_idx = blockIdx.z;
    
    int spatial_block = boundary_indices[boundary_idx];
    int d_block = spatial_block / (grid_h * grid_w);
    int hw_block = spatial_block % (grid_h * grid_w);
    int h_block = hw_block / grid_w;
    int w_block = hw_block % grid_w;
    
    int tid_w = threadIdx.x;
    int tid_h = threadIdx.y;
    int tid_d = threadIdx.z;
    
    int d_out = d_block * TILE_D + tid_d;
    int h_out = h_block * TILE_H + tid_h;
    int w_out = w_block * TILE_W + tid_w;
    
    float4 accum4[NUM_VEC];
    #pragma unroll
    for (int i = 0; i < NUM_VEC; i++) {
        accum4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    bool valid = (d_out < D_out) && (h_out < H_out) && (w_out < W_out);

    int d_in_start = max(0, d_block * TILE_D - 2);
    int d_in_end = min(depth, d_block * TILE_D + TILE_D);
    int d_in_size = d_in_end - d_in_start;
    int h_in_start = max(0, h_block * TILE_H - 2);
    int h_in_end = min(height, h_block * TILE_H + TILE_H);
    int h_in_size = h_in_end - h_in_start;
    int w_in_start = max(0, w_block * TILE_W - 2);
    int w_in_end = min(width, w_block * TILE_W + TILE_W);
    int w_in_size = w_in_end - w_in_start;

    int d_base = d_out - d_in_start;
    int h_base = h_out - h_in_start;
    int w_base = w_out - w_in_start;

    // Precompute kernel validity mask
    unsigned int mask = 0;
    #pragma unroll
    for (int kd = 0; kd < 3; kd++) {
        int d_rel = d_base - kd;
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int h_rel = h_base - kh;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int w_rel = w_base - kw;
                if (d_rel >= 0 && d_rel < d_in_size && 
                    h_rel >= 0 && h_rel < h_in_size && 
                    w_rel >= 0 && w_rel < w_in_size) {
                    mask |= (1U << (kd*9 + kh*3 + kw));
                }
            }
        }
    }

    int num_ic_blocks = in_channels_padded / TILE_IC;
    int total_threads = blockDim.x * blockDim.y * blockDim.z;
    
    for (int ic_block = 0; ic_block < in_channels_padded; ic_block += TILE_IC) {
        int tid = tid_d * blockDim.x * blockDim.y + tid_h * blockDim.x + tid_w;
        
        for (int idx = tid; idx < d_in_size * h_in_size * w_in_size * TILE_IC; idx += total_threads) {
            int ic_local = idx % TILE_IC;
            int spatial_idx = idx / TILE_IC;
            int w_in = spatial_idx % w_in_size;
            int h_in = (spatial_idx / w_in_size) % h_in_size;
            int d_in = spatial_idx / (w_in_size * h_in_size);
            
            int d_val = d_in_start + d_in;
            int h_val = h_in_start + h_in;
            int w_val = w_in_start + w_in;
            
            float val = 0.0f;
            if (ic_block + ic_local < in_channels_orig &&
                d_val < depth && h_val < height && w_val < width) {
                int input_idx = n * (in_channels_orig * depth * height * width)
                              + (ic_block + ic_local) * (depth * height * width)
                              + d_val * (height * width)
                              + h_val * width
                              + w_val;
                val = input[input_idx];
            }
            // Apply padding to avoid bank conflicts
            input_shared[spatial_idx * INPUT_SHMEM_STRIDE + ic_local] = val;
        }
        
        // Vectorized weight loading
        int num_float4_per_weight_block = (NUM_KERNELS * TILE_IC * TILE_OC) / 4;
        int ic_block_block = ic_block / TILE_IC;
        int base_offset = (oc_block * num_ic_blocks + ic_block_block) * NUM_KERNELS * TILE_IC * TILE_OC;
        for (int idx4 = tid; idx4 < num_float4_per_weight_block; idx4 += total_threads) {
            int global_idx = base_offset + idx4 * 4;
            float4 val4 = reinterpret_cast<const float4*>(weight)[global_idx / 4];
            reinterpret_cast<float4*>(weight_shared)[idx4] = val4;
        }
        
        __syncthreads();
        
        if (valid) {
            #pragma unroll
            for (int k = 0; k < NUM_KERNELS; k++) {
                if (mask & (1U << k)) {
                    // Extract kernel indices
                    int kd = k / 9;
                    int kh = (k % 9) / 3;
                    int kw = k % 3;
                    
                    int d_rel = d_base - kd;
                    int h_rel = h_base - kh;
                    int w_rel = w_base - kw;
                    
                    int spatial_idx = d_rel * (h_in_size * w_in_size) + h_rel * w_in_size + w_rel;
                    
                    #pragma unroll
                    for (int ic = 0; ic < TILE_IC; ic++) {
                        // Use padded indexing for conflict-free access
                        float in_val = input_shared[spatial_idx * INPUT_SHMEM_STRIDE + ic];
                        int weight_base = k * (TILE_IC * TILE_OC) + ic * TILE_OC;
                        
                        float4* weight_ptr = reinterpret_cast<float4*>(weight_shared + weight_base);
                        #pragma unroll
                        for (int vec_idx = 0; vec_idx < NUM_VEC; vec_idx++) {
                            float4 w4 = weight_ptr[vec_idx];
                            accum4[vec_idx].x += in_val * w4.x;
                            accum4[vec_idx].y += in_val * w4.y;
                            accum4[vec_idx].z += in_val * w4.z;
                            accum4[vec_idx].w += in_val * w4.w;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (valid) {
        float* accum = reinterpret_cast<float*>(accum4);
        for (int oc_local = 0; oc_local < TILE_OC; oc_local++) {
            if (oc_block * TILE_OC + oc_local < out_channels_orig) {
                int output_idx = n * (out_channels_orig * D_out * H_out * W_out)
                              + (oc_block * TILE_OC + oc_local) * (D_out * H_out * W_out)
                              + d_out * (H_out * W_out)
                              + h_out * W_out
                              + w_out;
                output[output_idx] = accum[oc_local];
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor conv_transpose3d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_padding
) {
    if (kernel_size != 3) {
        AT_ERROR("This optimized kernel only supports kernel_size=3");
    }
    if (stride != 1) {
        AT_ERROR("This optimized kernel only supports stride=1");
    }
    if (padding != 0) {
        AT_ERROR("This optimized kernel only supports padding=0");
    }
    if (dilation != 1) {
        AT_ERROR("This optimized kernel only supports dilation=1");
    }
    if (output_padding != 0) {
        AT_ERROR("This optimized kernel only supports output_padding=0");
    }
    
    int batch = input.size(0);
    int in_channels_orig = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    int out_channels_orig = weight.size(1);

    int D_out = (depth - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int H_out = (height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    int W_out = (width - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    auto output = torch::empty({batch, out_channels_orig, D_out, H_out, W_out}, input.options());
    
    constexpr int TILE_IC = 8;
    constexpr int TILE_OC = 24;
    constexpr int TILE_D = 4;
    constexpr int TILE_H = 8;
    constexpr int TILE_W = 8;

    int num_ic_blocks = (in_channels_orig + TILE_IC - 1) / TILE_IC;
    int num_oc_blocks = (out_channels_orig + TILE_OC - 1) / TILE_OC;
    int in_channels_padded = num_ic_blocks * TILE_IC;
    int out_channels_padded = num_oc_blocks * TILE_OC;

    auto weight_padded = torch::zeros({in_channels_padded, out_channels_padded, 3, 3, 3}, weight.options());
    weight_padded.slice(0, 0, in_channels_orig).slice(1, 0, out_channels_orig).copy_(weight);
    auto weight_reordered = weight_padded.permute({1,0,2,3,4})
                               .reshape({num_oc_blocks, TILE_OC, num_ic_blocks, TILE_IC, 3, 3, 3})
                               .permute({0, 2, 4, 5, 6, 3, 1})
                               .contiguous();

    int grid_d = (D_out + TILE_D - 1) / TILE_D;
    int grid_h = (H_out + TILE_H - 1) / TILE_H;
    int grid_w = (W_out + TILE_W - 1) / TILE_W;
    int grid_spatial = grid_d * grid_h * grid_w;
    
    int inner_grid_d = grid_d - 2;
    int inner_grid_h = grid_h - 2;
    int inner_grid_w = grid_w - 2;
    int inner_spatial = inner_grid_d * inner_grid_h * inner_grid_w;
    int boundary_spatial = grid_spatial - inner_spatial;
    
    auto boundary_indices = torch::empty({boundary_spatial}, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto boundary_indices_ptr = boundary_indices.data_ptr<int>();
    int idx = 0;
    for (int d = 0; d < grid_d; d++) {
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                if (d < 1 || d >= grid_d-1 || h < 1 || h >= grid_h-1 || w < 1 || w >= grid_w-1) {
                    boundary_indices_ptr[idx++] = d * (grid_h * grid_w) + h * grid_w + w;
                }
            }
        }
    }
    boundary_indices = boundary_indices.to(input.device());
    
    constexpr int TILE_IC_padded = TILE_IC + 1;
    constexpr int INPUT_SHMEM_SIZE = TILE_IC_padded * (TILE_D+2) * (TILE_H+2) * (TILE_W+2);
    constexpr int WEIGHT_SHMEM_SIZE = 27 * TILE_IC * TILE_OC;
    size_t shared_mem_size = (INPUT_SHMEM_SIZE + WEIGHT_SHMEM_SIZE) * sizeof(float);
    
    dim3 block(TILE_W, TILE_H, TILE_D);
    
    if (inner_spatial > 0) {
        dim3 inner_grid(batch, num_oc_blocks, inner_spatial);
        conv_transpose3d_inner_kernel<<<inner_grid, block, shared_mem_size>>>(
            input.data_ptr<float>(),
            weight_reordered.data_ptr<float>(),
            output.data_ptr<float>(),
            batch, in_channels_padded, out_channels_padded,
            depth, height, width,
            kernel_size, stride, padding, dilation,
            D_out, H_out, W_out,
            grid_d, grid_h, grid_w,
            in_channels_orig, out_channels_orig
        );
    }
    
    if (boundary_spatial > 0) {
        dim3 boundary_grid(batch, num_oc_blocks, boundary_spatial);
        conv_transpose3d_boundary_kernel<<<boundary_grid, block, shared_mem_size>>>(
            input.data_ptr<float>(),
            weight_reordered.data_ptr<float>(),
            boundary_indices.data_ptr<int>(),
            output.data_ptr<float>(),
            batch, in_channels_padded, out_channels_padded,
            depth, height, width,
            kernel_size, stride, padding, dilation,
            D_out, H_out, W_out,
            grid_d, grid_h, grid_w,
            boundary_spatial,
            in_channels_orig, out_channels_orig
        );
    }

    return output;
}
// PART-END