// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template <int REDUCTION_SIZE, int DIM, bool UNIT_STRIDE>
__global__ void min_reduction_3d_template_kernel(const float* in_ptr, float* out_ptr, 
    int n0, int n1, int n2, 
    int s0, int s1, int s2, 
    int d) {

    extern __shared__ __align__(16) char shared_mem[];
    float* smem_data = reinterpret_cast<float*>(shared_mem);
    
    namespace cg = cooperative_groups;
    using MemCpyPipeline = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    __shared__ MemCpyPipeline shared_pipeline;
    
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * blockDim.x + lane_id;
    const int num_warps = blockDim.y;
    constexpr int OUTPUT_TILE_SIZE = 32;
    
    constexpr int TILE_SIZE = 256;
    constexpr int NUM_TILES = (REDUCTION_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    constexpr int ELEMS_PER_THREAD = UNIT_STRIDE ? 8 : 1;
    
    int outer0, outer1, s_outer0, s_outer1, s_inner;
    if constexpr (DIM == 0) {
        outer0 = n1;
        outer1 = n2;
        s_outer0 = s1;
        s_outer1 = s2;
        s_inner = s0;
    } else if constexpr (DIM == 1) {
        outer0 = n0;
        outer1 = n2;
        s_outer0 = s0;
        s_outer1 = s2;
        s_inner = s1;
    } else if constexpr (DIM == 2) {
        outer0 = n0;
        outer1 = n1;
        s_outer0 = s0;
        s_outer1 = s1;
        s_inner = s2;
    }

    int i_outer0 = blockIdx.y * OUTPUT_TILE_SIZE + warp_id;
    int i_outer1_start = blockIdx.x * OUTPUT_TILE_SIZE;
    int i_outer1 = i_outer1_start + lane_id;

    if (i_outer0 >= outer0 || i_outer1_start >= outer1) return;

    float* warp_smem = smem_data + warp_id * (2 * TILE_SIZE);
    float* buf0 = warp_smem;
    float* buf1 = warp_smem + TILE_SIZE;
    
    float partial_min = FLT_MAX;
    cg::thread_block block = cg::this_thread_block();
    auto pipeline = cuda::make_pipeline(block, &shared_pipeline);

    #pragma unroll
    for (int tile_idx = 0; tile_idx < NUM_TILES; tile_idx++) {
        int stage = tile_idx & 1;
        float* shared_buf = (stage == 0) ? buf0 : buf1;
        const int tile_start = tile_idx * TILE_SIZE;
        
        pipeline.producer_acquire();
        if constexpr (UNIT_STRIDE) {
            const int offset = tile_start + lane_id * ELEMS_PER_THREAD;
            if (offset < REDUCTION_SIZE) {
                const size_t base_offset = static_cast<size_t>(i_outer0) * s_outer0 + 
                                          static_cast<size_t>(i_outer1) * s_outer1;
                if (offset + ELEMS_PER_THREAD - 1 < REDUCTION_SIZE) {
                    cuda::memcpy_async(block, shared_buf + lane_id * ELEMS_PER_THREAD, 
                                     in_ptr + base_offset + offset, 
                                     ELEMS_PER_THREAD * sizeof(float), pipeline);
                } else {
                    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                        if (offset + i < REDUCTION_SIZE) {
                            shared_buf[lane_id * ELEMS_PER_THREAD + i] = 
                                in_ptr[base_offset + offset + i];
                        }
                    }
                }
            }
        } else {
            const int offset = tile_start + lane_id;
            if (offset < REDUCTION_SIZE && i_outer1 < outer1) {
                size_t base_offset = static_cast<size_t>(i_outer0) * s_outer0 + 
                                     static_cast<size_t>(i_outer1) * s_outer1;
                size_t global_offset = base_offset + static_cast<size_t>(offset) * s_inner;
                cuda::memcpy_async(block, shared_buf + lane_id, 
                                 in_ptr + global_offset, 
                                 sizeof(float), pipeline);
            }
        }
        pipeline.producer_commit();
        
        if (tile_idx > 0) {
            const int prev_stage = 1 - stage;
            float* prev_buf = (prev_stage == 0) ? buf0 : buf1;
            pipeline.consumer_wait();
            
            if constexpr (UNIT_STRIDE) {
                #pragma unroll
                for (int j = 0; j < ELEMS_PER_THREAD; j++) {
                    const int idx = lane_id * ELEMS_PER_THREAD + j;
                    if (idx < TILE_SIZE) {
                        const float val = prev_buf[idx];
                        partial_min = fminf(partial_min, val);
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < TILE_SIZE; j += 32) {
                    const int idx = j + lane_id;
                    if (idx < TILE_SIZE) {
                        const float val = prev_buf[idx];
                        partial_min = fminf(partial_min, val);
                    }
                }
            }
            pipeline.consumer_release();
        }
    }
    
    pipeline.consumer_wait();
    const int last_stage = (NUM_TILES - 1) & 1;
    float* last_buf = (last_stage == 0) ? buf0 : buf1;
    const int last_tile_start = (NUM_TILES - 1) * TILE_SIZE;
    const int num_valid = REDUCTION_SIZE - last_tile_start;
    
    if constexpr (UNIT_STRIDE) {
        #pragma unroll
        for (int j = 0; j < ELEMS_PER_THREAD; j++) {
            const int idx = lane_id * ELEMS_PER_THREAD + j;
            if (idx < num_valid) {
                const float val = last_buf[idx];
                partial_min = fminf(partial_min, val);
            }
        }
    } else {
        for (int j = 0; j < TILE_SIZE; j += 32) {
            const int idx = j + lane_id;
            if (idx < num_valid) {
                const float val = last_buf[idx];
                partial_min = fminf(partial_min, val);
            }
        }
    }
    pipeline.consumer_release();

    if (i_outer1 < outer1) {
        out_ptr[i_outer0 * outer1 + i_outer1] = partial_min;
    }
}

__global__ void min_reduction_3d_kernel(const float* in_ptr, float* out_ptr, 
    int n0, int n1, int n2, 
    int s0, int s1, int s2, 
    int d, int reduction_size) {

    extern __shared__ __align__(16) char shared_mem[];
    float* smem_data = reinterpret_cast<float*>(shared_mem);
    
    namespace cg = cooperative_groups;
    using MemCpyPipeline = cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;
    __shared__ MemCpyPipeline shared_pipeline;
    
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * blockDim.x + lane_id;
    const int num_warps = blockDim.y;
    const int output_tile_size = 32;
    
    const int TILE_SIZE = 256;
    const int num_tiles = (reduction_size + TILE_SIZE - 1) / TILE_SIZE;
    
    int outer0, outer1, s_outer0, s_outer1, s_inner;
    if (d == 0) {
        outer0 = n1;
        outer1 = n2;
        s_outer0 = s1;
        s_outer1 = s2;
        s_inner = s0;
    } else if (d == 1) {
        outer0 = n0;
        outer1 = n2;
        s_outer0 = s0;
        s_outer1 = s2;
        s_inner = s1;
    } else {
        outer0 = n0;
        outer1 = n1;
        s_outer0 = s0;
        s_outer1 = s1;
        s_inner = s2;
    }

    int i_outer0 = blockIdx.y * output_tile_size + warp_id;
    int i_outer1_start = blockIdx.x * output_tile_size;
    int i_outer1 = i_outer1_start + lane_id;

    if (i_outer0 >= outer0 || i_outer1_start >= outer1) return;

    float* warp_smem = smem_data + warp_id * (2 * TILE_SIZE);
    float* buf0 = warp_smem;
    float* buf1 = warp_smem + TILE_SIZE;
    
    float partial_min = FLT_MAX;
    cg::thread_block block = cg::this_thread_block();
    auto pipeline = cuda::make_pipeline(block, &shared_pipeline);

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int stage = tile_idx & 1;
        float* shared_buf = (stage == 0) ? buf0 : buf1;
        int tile_start = tile_idx * TILE_SIZE;
        
        pipeline.producer_acquire();
        int offset = tile_start + lane_id;
        if (offset < reduction_size && i_outer1 < outer1) {
            size_t global_offset = static_cast<size_t>(i_outer0) * s_outer0 + 
                                  static_cast<size_t>(offset) * s_inner + 
                                  static_cast<size_t>(i_outer1) * s_outer1;
            cuda::memcpy_async(block, shared_buf + lane_id, 
                             in_ptr + global_offset, 
                             sizeof(float), pipeline);
        }
        pipeline.producer_commit();
        
        if (tile_idx > 0) {
            int prev_stage = 1 - stage;
            float* prev_buf = (prev_stage == 0) ? buf0 : buf1;
            pipeline.consumer_wait();
            for (int j = 0; j < TILE_SIZE; j += 32) {
                int idx = j + lane_id;
                if (idx < TILE_SIZE) {
                    float val = prev_buf[idx];
                    partial_min = fminf(partial_min, val);
                }
            }
            pipeline.consumer_release();
        }
    }
    
    pipeline.consumer_wait();
    int last_stage = (num_tiles - 1) & 1;
    float* last_buf = (last_stage == 0) ? buf0 : buf1;
    const int last_tile_start = (num_tiles - 1) * TILE_SIZE;
    const int num_valid = reduction_size - last_tile_start;
    
    for (int j = 0; j < TILE_SIZE; j += 32) {
        int idx = j + lane_id;
        if (idx < num_valid) {
            float val = last_buf[idx];
            partial_min = fminf(partial_min, val);
        }
    }
    pipeline.consumer_release();

    if (i_outer1 < outer1) {
        out_ptr[i_outer0 * outer1 + i_outer1] = partial_min;
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor min_reduction_3d_cuda(torch::Tensor in, int dim) {
    TORCH_CHECK(in.dim() == 3, "min_reduction_3d_cuda: input must be 3D");
    TORCH_CHECK(dim >=0 && dim <=2, "min_reduction_3d_cuda: dim must be 0,1, or 2");
    TORCH_CHECK(in.dtype() == torch::kFloat32, "min_reduction_3d_cuda: input must be float32");
    TORCH_CHECK(in.is_cuda(), "min_reduction_3d_cuda: input must be on GPU");

    auto sizes = in.sizes();
    int n0 = sizes[0];
    int n1 = sizes[1];
    int n2 = sizes[2];
    auto strides = in.strides();
    int s0 = strides[0];
    int s1 = strides[1];
    int s2 = strides[2];
    int reduction_size = sizes[dim];

    std::vector<int64_t> output_shape;
    for (int i=0; i<3; i++) {
        if (i != dim) {
            output_shape.push_back(sizes[i]);
        }
    }
    auto out = torch::empty(output_shape, in.options());

    int num_output_elements = out.numel();
    if (num_output_elements == 0) {
        return out;
    }

    constexpr int OUTPUT_TILE_SIZE = 32;
    dim3 block, grid;

    if (dim == 0) {
        block = dim3(32, 16);
        grid = dim3((n2 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (n1 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);
    } else if (dim == 1) {
        block = dim3(32, 8);
        grid = dim3((n2 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (n0 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);
    } else {
        block = dim3(32, 16);
        grid = dim3((n1 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE, (n0 + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);
    }

    constexpr int TILE_SIZE = 256;
    size_t shared_mem_per_warp = 2 * TILE_SIZE * sizeof(float);
    size_t shared_mem_size = block.y * shared_mem_per_warp + sizeof(cuda::pipeline_shared_state<cuda::thread_scope_block, 2>);
    
    if (reduction_size == 4096) {
        int s_inner = -1;
        if (dim == 0) {
            s_inner = s0;
        } else if (dim == 1) {
            s_inner = s1;
        } else {
            s_inner = s2;
        }
        
        if (s_inner == 1) {
            if (dim == 0) {
                min_reduction_3d_template_kernel<4096, 0, true><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            } else if (dim == 1) {
                min_reduction_3d_template_kernel<4096, 1, true><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            } else if (dim == 2) {
                min_reduction_3d_template_kernel<4096, 2, true><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            }
        } else {
            if (dim == 0) {
                min_reduction_3d_template_kernel<4096, 0, false><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            } else if (dim == 1) {
                min_reduction_3d_template_kernel<4096, 1, false><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            } else if (dim == 2) {
                min_reduction_3d_template_kernel<4096, 2, false><<<grid, block, shared_mem_size>>>(
                    in.data_ptr<float>(), 
                    out.data_ptr<float>(),
                    n0, n1, n2,
                    s0, s1, s2,
                    dim
                );
            }
        }
    } else {
        min_reduction_3d_kernel<<<grid, block, shared_mem_size>>>(
            in.data_ptr<float>(), 
            out.data_ptr<float>(),
            n0, n1, n2,
            s0, s1, s2,
            dim, reduction_size
        );
    }

    return out;
}
// PART-END