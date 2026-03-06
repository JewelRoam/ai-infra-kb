/**
 * CUDA 归约操作示例 - 多种优化技术
 * 
 * 演示内容：
 * 1. 朴素归约
 * 2. 共享内存归约
 * 3. Warp Shuffle优化
 * 4. 多版本性能对比
 * 
 * 编译命令：nvcc -o reduction 05_reduction.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

#define BLOCK_SIZE 256

// ==================== 版本1: 朴素全局内存归约 ====================
__global__ void reduceNaive(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 使用全局内存进行归约
    // 注意：这是低效的实现，仅作对比基准
    
    if (idx < n) {
        // 使用原子操作累加到全局结果
        atomicAdd(output, input[idx]);
    }
}

// ==================== 版本2: 共享内存归约 ====================
__global__ void reduceShared(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 只有线程0写入结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ==================== 版本3: 优化共享内存归约 ====================
/**
 * 优化点：
 * 1. 循环展开
 * 2. 避免bank conflict
 * 3. 减少同步次数
 */
template <int blockSize>
__device__ void warpReduce(volatile float *sdata, int tid) {
    // 最后一个warp不需要同步
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)  sdata[tid] += sdata[tid + 1];
}

template <int blockSize>
__global__ void reduceOptimized(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    
    // 每个线程处理2个元素
    float mySum = (idx < n) ? input[idx] : 0.0f;
    if (idx + blockSize < n) {
        mySum += input[idx + blockSize];
    }
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // 循环展开的归约
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    
    // 最后一个warp使用warpReduce
    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ==================== 版本4: Warp Shuffle 归约 (CUDA >= 9.0) ====================
__global__ void reduceWarpShuffle(const float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp级别的归约
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // 只有每个warp的第一个线程写入共享内存
    __shared__ float warpSums[BLOCK_SIZE / warpSize];
    
    if (tid % warpSize == 0) {
        warpSums[tid / warpSize] = val;
    }
    __syncthreads();
    
    // 第一个warp归约warpSums
    if (tid < warpSize) {
        val = (tid < BLOCK_SIZE / warpSize) ? warpSums[tid] : 0.0f;
        
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    if (tid == 0) {
        atomicAdd(output, val);
    }
}

// ==================== 版本5: Cooperative Groups (CUDA >= 9.0) ====================
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__global__ void reduceCooperativeGroups(const float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp级归约
    val = cg::reduce(warp, val, cg::plus<float>());
    
    // Block级归约
    __shared__ float warpSums[BLOCK_SIZE / 32];
    
    if (warp.thread_rank() == 0) {
        warpSums[warp.meta_group_rank()] = val;
    }
    
    cg::sync(block);
    
    if (warp.meta_group_rank() == 0) {
        val = warp.thread_rank() < warp.meta_group_size() 
              ? warpSums[warp.thread_rank()] : 0.0f;
        val = cg::reduce(warp, val, cg::plus<float>());
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

// ==================== 主机辅助函数 ====================
void initArray(float *arr, int n, float seed) {
    for (int i = 0; i < n; i++) {
        arr[i] = seed * ((float)rand() / RAND_MAX);
    }
}

float cpuReduce(const float *arr, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// ==================== 主函数 ====================
int main(int argc, char **argv) {
    int n = 1 << 24;  // 16M 元素
    size_t bytes = n * sizeof(float);
    
    printf("=== CUDA Reduction Benchmark ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", n, (float)bytes / 1024 / 1024);
    
    // 分配主机内存
    float *h_input = (float *)malloc(bytes);
    float h_output = 0.0f;
    
    // 初始化数据
    srand(42);
    initArray(h_input, n, 1.0f);
    
    // CPU参考结果
    float cpuSum = cpuReduce(h_input, n);
    printf("CPU Reference Sum: %.6f\n\n", cpuSum);
    
    // 分配设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
    
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("Performance Results:\n");
    printf("--------------------------------------------------\n");
    
    // 测试函数
    auto testKernel = [&](const char* name, auto kernel, bool extraParams = false) {
        // 清零输出
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        
        // Warmup
        if (extraParams) {
            kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        } else {
            kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        int iter = 50;
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iter; i++) {
            CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
            kernel<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iter;
        
        // 获取结果
        CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        
        double bandwidth = (bytes / 1e9) / (ms / 1000.0);
        printf("%-25s: %6.3f ms, %6.2f GB/s, %.6f %s\n", 
               name, ms, bandwidth, h_output, 
               fabs(h_output - cpuSum) < 1e-3 ? "✓" : "✗");
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };
    
    testKernel("Naive (Atomic)", reduceNaive);
    testKernel("Shared Memory", reduceShared);
    testKernel("Optimized", reduceOptimized<BLOCK_SIZE>);
    testKernel("Warp Shuffle", reduceWarpShuffle);
    testKernel("Cooperative Groups", reduceCooperativeGroups);
    
    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    
    printf("\nDone.\n");
    return 0;
}