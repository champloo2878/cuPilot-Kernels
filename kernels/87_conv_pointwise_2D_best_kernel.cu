// Part 1: (top-level header files and preprocessing functions)
// PART-START
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/util/Optional.h>
#include <type_traits>

// Helper for vectorized loads/stores
template<typename T, int VecSize>
struct VectorType;

template<>
struct VectorType<float, 8> {
    using Type = float4[2];
};

template<>
struct VectorType<at::Half, 16> {
    using Type = uint4[2];
};

template<>
struct VectorType<at::BFloat16, 16> {
    using Type = uint4[2];
};

template<>
struct VectorType<double, 4> {
    using Type = double2[2];
};

template<typename T, int VecSize>
using VecType = typename VectorType<T, VecSize>::Type;

template<typename T>
__device__ __inline__ T zero() {
    return static_cast<T>(0.0f);
}

template<>
__device__ __inline__ at::Half zero<at::Half>() {
    return __float2half(0.0f);
}

template<>
__device__ __inline__ at::BFloat16 zero<at::BFloat16>() {
    return __float2bfloat16(0.0f);
}

template<>
__device__ __inline__ double zero<double>() {
    return 0.0;
}

// Base template declarations
template<typename T, int VecSize>
__device__ __inline__ void load_vector(VecType<T, VecSize>& vec, const T* ptr);

template<typename T, int VecSize>
__device__ __inline__ void store_vector(T* ptr, const VecType<T, VecSize>& vec);

template<typename T, int VecSize>
__device__ __inline__ void unpack_vector(T regs[VecSize], const VecType<T, VecSize>& vec);

// Vectorized load for FP32 (8 elements = 32 bytes × 4 = 128 bytes)
template<>
__device__ __inline__ void load_vector<float, 8>(VecType<float, 8>& vec, const float* ptr) {
    vec[0] = *reinterpret_cast<const float4*>(ptr);
    vec[1] = *reinterpret_cast<const float4*>(ptr + 4);
}

// Vectorized load for FP16/BF16 (16 elements = 32 bytes × 4 = 128 bytes)
template<>
__device__ __inline__ void load_vector<at::Half, 16>(VecType<at::Half, 16>& vec, const at::Half* ptr) {
    vec[0] = *reinterpret_cast<const uint4*>(ptr);
    vec[1] = *reinterpret_cast<const uint4*>(ptr + 8);
}

template<>
__device__ __inline__ void load_vector<at::BFloat16, 16>(VecType<at::BFloat16, 16>& vec, const at::BFloat16* ptr) {
    vec[0] = *reinterpret_cast<const uint4*>(ptr);
    vec[1] = *reinterpret_cast<const uint4*>(ptr + 8);
}

// Vectorized load for double (4 elements = 32 bytes)
template<>
__device__ __inline__ void load_vector<double, 4>(VecType<double, 4>& vec, const double* ptr) {
    vec[0] = *reinterpret_cast<const double2*>(ptr);
    vec[1] = *reinterpret_cast<const double2*>(ptr + 2);
}

// Vectorized store for FP32
template<>
__device__ __inline__ void store_vector<float, 8>(float* ptr, const VecType<float, 8>& vec) {
    *reinterpret_cast<float4*>(ptr) = vec[0];
    *reinterpret_cast<float4*>(ptr + 4) = vec[1];
}

// Vectorized store for FP16/BF16
template<>
__device__ __inline__ void store_vector<at::Half, 16>(at::Half* ptr, const VecType<at::Half, 16>& vec) {
    *reinterpret_cast<uint4*>(ptr) = vec[0];
    *reinterpret_cast<uint4*>(ptr + 8) = vec[1];
}

template<>
__device__ __inline__ void store_vector<at::BFloat16, 16>(at::BFloat16* ptr, const VecType<at::BFloat16, 16>& vec) {
    *reinterpret_cast<uint4*>(ptr) = vec[0];
    *reinterpret_cast<uint4*>(ptr + 8) = vec[1];
}

// Vectorized store for double
template<>
__device__ __inline__ void store_vector<double, 4>(double* ptr, const VecType<double, 4>& vec) {
    *reinterpret_cast<double2*>(ptr) = vec[0];
    *reinterpret_cast<double2*>(ptr + 2) = vec[1];
}

// Unpack vectorized FP32 data to registers
template<>
__device__ __inline__ void unpack_vector<float, 8>(float regs[8], const VecType<float, 8>& vec) {
    #pragma unroll
    for (int i = 0; i < 4; i++) regs[i] = reinterpret_cast<const float*>(&vec[0])[i];
    #pragma unroll
    for (int i = 0; i < 4; i++) regs[i+4] = reinterpret_cast<const float*>(&vec[1])[i];
}

// Unpack vectorized FP16/BF16 data to registers
template<>
__device__ __inline__ void unpack_vector<at::Half, 16>(at::Half regs[16], const VecType<at::Half, 16>& vec) {
    #pragma unroll
    for (int i = 0; i < 8; i++) regs[i] = reinterpret_cast<const at::Half*>(&vec[0])[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) regs[i+8] = reinterpret_cast<const at::Half*>(&vec[1])[i];
}

template<>
__device__ __inline__ void unpack_vector<at::BFloat16, 16>(at::BFloat16 regs[16], const VecType<at::BFloat16, 16>& vec) {
    #pragma unroll
    for (int i = 0; i < 8; i++) regs[i] = reinterpret_cast<const at::BFloat16*>(&vec[0])[i];
    #pragma unroll
    for (int i = 0; i < 8; i++) regs[i+8] = reinterpret_cast<const at::BFloat16*>(&vec[1])[i];
}

// Unpack vectorized double data to registers
template<>
__device__ __inline__ void unpack_vector<double, 4>(double regs[4], const VecType<double, 4>& vec) {
    #pragma unroll
    for (int i = 0; i < 2; i++) regs[i] = reinterpret_cast<const double*>(&vec[0])[i];
    #pragma unroll
    for (int i = 0; i < 2; i++) regs[i+2] = reinterpret_cast<const double*>(&vec[1])[i];
}

// Swizzle function for improved memory access patterns
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
// PART-END

// Part 2: (the main body of the custom kernel function)
// PART-START
template<typename T>
__global__ void pointwise_conv_kernel(
    const T* __restrict__ input, 
    const T* __restrict__ weight, 
    const T* __restrict__ bias,
    T* __restrict__ output, 
    int batch_size, int in_channels, int out_channels, 
    int height, int width) {
    
    // Optimized tile dimensions for A100 with specific problem size
    constexpr int TILE_OUT_C = 128;    // Match out_channels=128 for complete coverage
    constexpr int TILE_SPATIAL = 64;   // Increased spatial tile for better L1 cache utilization
    constexpr int TILE_IN_C = 32;      // Increased input channel tile to reduce loops
    
    // Determine vectorization size based on data type
    constexpr int VEC_SIZE = sizeof(T) == 4 ? 8 : (sizeof(T) == 2 ? 16 : 4);
    constexpr int THREADS_PER_BLOCK = 256;
    
    // Enhanced thread organization for memory coalescing
    const int thread_id = threadIdx.x;
    
    // Reorganized mapping: consecutive threads handle consecutive spatial positions
    const int thread_spatial = thread_id % 16;      // 0-15: handles 4 spatial positions each
    const int thread_out_c = thread_id / 16;        // 0-15: handles 8 output channels each
    
    // Block indices
    const int out_c_tile_idx = blockIdx.y;
    const int spatial_tile_idx = blockIdx.x;
    
    // Global starting indices
    const int out_c_start = out_c_tile_idx * TILE_OUT_C;
    const int spatial_start = spatial_tile_idx * TILE_SPATIAL;
    
    // Flatten spatial dimension
    const int num_spatial = batch_size * height * width;
    
    // Thread-specific output channels (8 per thread for TILE_OUT_C=128)
    const int out_c_base = out_c_start + thread_out_c * 8;
    
    // Thread-specific spatial positions (4 per thread for TILE_SPATIAL=64)
    const int spatial_base = spatial_start + thread_spatial * 4;
    
    // Check if this thread has valid work
    if (out_c_base >= out_channels || spatial_base >= num_spatial) {
        return;
    }
    
    // Register accumulators: 8 output channels × 4 spatial positions
    T acc[8][4];
    #pragma unroll
    for (int oc = 0; oc < 8; oc++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            acc[oc][s] = zero<T>();
        }
    }
    
    // Shared memory buffers with swizzled layout for better bank conflict avoidance
    __shared__ T input_smem[TILE_IN_C][TILE_SPATIAL + 1];  // +1 for bank conflict padding
    __shared__ T weight_smem[TILE_OUT_C][TILE_IN_C + 1];   // +1 for bank conflict padding
    
    // Loop over input channels in tiles
    for (int in_c_offset = 0; in_c_offset < in_channels; in_c_offset += TILE_IN_C) {
        const int in_c_end = min(in_c_offset + TILE_IN_C, in_channels);
        const int valid_in_c = in_c_end - in_c_offset;
        
        // Collaborative loading of input tile with optimized memory coalescing
        #pragma unroll 2
        for (int load_idx = thread_id; load_idx < TILE_IN_C * TILE_SPATIAL / VEC_SIZE; load_idx += THREADS_PER_BLOCK) {
            // Calculate tile indices with coalesced access pattern
            int s_tile_base = (load_idx % (TILE_SPATIAL / VEC_SIZE)) * VEC_SIZE;
            int c_in_tile = load_idx / (TILE_SPATIAL / VEC_SIZE);
            
            int global_spatial_base = spatial_start + s_tile_base;
            
            if (c_in_tile < valid_in_c) {
                int c_in = in_c_offset + c_in_tile;
                T loaded_vals[VEC_SIZE];
                
                // Load vectorized data with optimized pattern
                if (s_tile_base + VEC_SIZE <= TILE_SPATIAL && global_spatial_base + VEC_SIZE <= num_spatial) {
                    // Fast path: full vector load
                    int n = global_spatial_base / (height * width);
                    int hw_base = global_spatial_base % (height * width);
                    int h_base = hw_base / width;
                    int w_base = hw_base % width;
                    
                    // Calculate base pointer for vectorized load
                    int input_idx = ((n * in_channels + c_in) * height + h_base) * width + w_base;
                    VecType<T, VEC_SIZE> vec;
                    load_vector<T, VEC_SIZE>(vec, &input[input_idx]);
                    unpack_vector<T, VEC_SIZE>(loaded_vals, vec);
                } else {
                    // Slow path: handle boundary conditions
                    #pragma unroll
                    for (int v = 0; v < VEC_SIZE; v++) {
                        int s_tile = s_tile_base + v;
                        int global_spatial = global_spatial_base + v;
                        
                        T val = zero<T>();
                        if (s_tile < TILE_SPATIAL && global_spatial < num_spatial) {
                            int n = global_spatial / (height * width);
                            int hw = global_spatial % (height * width);
                            int h = hw / width;
                            int w = hw % width;
                            int input_idx = ((n * in_channels + c_in) * height + h) * width + w;
                            val = input[input_idx];
                        }
                        loaded_vals[v] = val;
                    }
                }
                
                // Store to shared memory with swizzled layout
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; v++) {
                    int s_tile = s_tile_base + v;
                    input_smem[c_in_tile][s_tile] = loaded_vals[v];
                }
            }
        }
        
        // Collaborative loading of weight tile with optimized memory coalescing
        #pragma unroll 4
        for (int load_idx = thread_id; load_idx < TILE_OUT_C * TILE_IN_C / VEC_SIZE; load_idx += THREADS_PER_BLOCK) {
            // Calculate tile indices with coalesced access pattern
            int c_in_tile_base = (load_idx % (TILE_IN_C / VEC_SIZE)) * VEC_SIZE;
            int out_c_tile = load_idx / (TILE_IN_C / VEC_SIZE);
            
            if (out_c_tile < TILE_OUT_C) {
                int out_c = out_c_start + out_c_tile;
                T loaded_vals[VEC_SIZE];
                
                // Load vectorized data with optimized pattern
                if (c_in_tile_base + VEC_SIZE <= TILE_IN_C && c_in_tile_base + VEC_SIZE <= valid_in_c) {
                    // Fast path: full vector load
                    int weight_idx = out_c * in_channels + in_c_offset + c_in_tile_base;
                    VecType<T, VEC_SIZE> vec;
                    load_vector<T, VEC_SIZE>(vec, &weight[weight_idx]);
                    unpack_vector<T, VEC_SIZE>(loaded_vals, vec);
                } else {
                    // Slow path: handle boundary conditions
                    #pragma unroll
                    for (int v = 0; v < VEC_SIZE; v++) {
                        int c_in_tile = c_in_tile_base + v;
                        
                        T val = zero<T>();
                        if (c_in_tile < valid_in_c && out_c < out_channels) {
                            int c_in = in_c_offset + c_in_tile;
                            int weight_idx = out_c * in_channels + c_in;
                            val = weight[weight_idx];
                        }
                        loaded_vals[v] = val;
                    }
                }
                
                // Store to shared memory with swizzled layout
                #pragma unroll
                for (int v = 0; v < VEC_SIZE; v++) {
                    int c_in_tile = c_in_tile_base + v;
                    weight_smem[out_c_tile][c_in_tile] = loaded_vals[v];
                }
            }
        }
        
        __syncthreads();
        
        // Main computation loop with proper bounds checking
        for (int k = 0; k < valid_in_c; k++) {
            // Load input values for this thread
            T input_val[4];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                int s_tile = thread_spatial * 4 + s;
                input_val[s] = input_smem[k][s_tile];
            }
            
            // Load weight values for this thread
            T weight_val[8];
            #pragma unroll
            for (int oc = 0; oc < 8; oc++) {
                int out_c_tile_idx = thread_out_c * 8 + oc;
                weight_val[oc] = weight_smem[out_c_tile_idx][k];
            }
            
            // Accumulate results
            #pragma unroll
            for (int oc = 0; oc < 8; oc++) {
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    if constexpr (std::is_same<T, float>::value) {
                        acc[oc][s] = __fmaf_rn(weight_val[oc], input_val[s], acc[oc][s]);
                    } else {
                        acc[oc][s] = weight_val[oc] * input_val[s] + acc[oc][s];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Add bias if present with intrinsic
    if (bias != nullptr) {
        #pragma unroll
        for (int oc = 0; oc < 8; oc++) {
            int out_c = out_c_base + oc;
            if (out_c < out_channels) {
                T bias_val = bias[out_c];
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    if constexpr (std::is_same<T, float>::value) {
                        acc[oc][s] = __fadd_rn(acc[oc][s], bias_val);
                    } else {
                        acc[oc][s] = acc[oc][s] + bias_val;
                    }
                }
            }
        }
    }
    
    // Write results to global memory with optimized vectorization
    #pragma unroll
    for (int oc = 0; oc < 8; oc++) {
        int out_c = out_c_base + oc;
        
        if (out_c < out_channels) {
            #pragma unroll
            for (int s_base = 0; s_base < 4; s_base += VEC_SIZE/2) {
                int global_spatial_base = spatial_base + s_base;
                
                if (global_spatial_base + VEC_SIZE/2 <= num_spatial) {
                    // Fast path: vectorized store
                    int n = global_spatial_base / (height * width);
                    int hw_base = global_spatial_base % (height * width);
                    int h_base = hw_base / width;
                    int w_base = hw_base % width;
                    
                    int output_idx = ((n * out_channels + out_c) * height + h_base) * width + w_base;
                    
                    if constexpr (std::is_same<T, float>::value) {
                        float4 vec;
                        #pragma unroll
                        for (int i = 0; i < 4; i++) reinterpret_cast<float*>(&vec)[i] = acc[oc][s_base + i];
                        *reinterpret_cast<float4*>(&output[output_idx]) = vec;
                    } else if constexpr (std::is_same<T, at::Half>::value || std::is_same<T, at::BFloat16>::value) {
                        uint4 vec;
                        #pragma unroll
                        for (int i = 0; i < 8; i++) reinterpret_cast<T*>(&vec)[i] = acc[oc][s_base + i];
                        *reinterpret_cast<uint4*>(&output[output_idx]) = vec;
                    } else if constexpr (std::is_same<T, double>::value) {
                        double2 vec;
                        #pragma unroll
                        for (int i = 0; i < 2; i++) reinterpret_cast<double*>(&vec)[i] = acc[oc][s_base + i];
                        *reinterpret_cast<double2*>(&output[output_idx]) = vec;
                    }
                } else {
                    // Slow path: scalar store for boundary
                    #pragma unroll
                    for (int v = 0; v < VEC_SIZE/2; v++) {
                        int global_spatial = global_spatial_base + v;
                        if (global_spatial < num_spatial) {
                            int n = global_spatial / (height * width);
                            int hw = global_spatial % (height * width);
                            int h = hw / width;
                            int w = hw % width;
                            int output_idx = ((n * out_channels + out_c) * height + h) * width + w;
                            output[output_idx] = acc[oc][s_base + v];
                        }
                    }
                }
            }
        }
    }
}
// PART-END

// Part 3: (the top-level interface for encapsulating into a Torch operator)
// PART-START
torch::Tensor pointwise_conv_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    const auto& bias = bias_opt.has_value() ? *bias_opt : torch::Tensor();
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);
    auto out_channels = weight.size(0);
    
    auto output = torch::empty({batch_size, out_channels, height, width}, 
                              input.options());
    
    // Optimized tile configuration for specific problem size
    constexpr int TILE_OUT_C = 128;   // Match out_channels=128 for minimal grid
    constexpr int TILE_SPATIAL = 64;  // Increased spatial tile for better cache utilization
    
    const int num_spatial = batch_size * height * width;
    const int num_spatial_tiles = (num_spatial + TILE_SPATIAL - 1) / TILE_SPATIAL;
    const int num_out_c_tiles = (out_channels + TILE_OUT_C - 1) / TILE_OUT_C;
    
    // Optimized block and grid configuration for A100 with memory coalescing
    dim3 block(256);  // 1D block for optimal warp utilization
    dim3 grid(num_spatial_tiles, num_out_c_tiles);
    
    // Validate grid dimensions
    if (grid.x == 0 || grid.y == 0) {
        return output;
    }
    
    // Launch kernel with optimal configuration
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), 
        "pointwise_conv_cuda", [&] {
            pointwise_conv_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
                output.data_ptr<scalar_t>(),
                batch_size, in_channels, out_channels, height, width
            );
        });
    
    return output;
}
// PART-END