/**
 * CUDA 矩阵乘法示例 - 多版本优化对比
 * 
 * 演示内容：
 * 1. 朴素实现（baseline）
 * 2. 共享内存优化
 * 3. 矩阵分块（Tiling）技术
 * 
 * 编译命令：nvcc -o matrix_mul 02_matrix_mul.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ==================== 版本1: 朴素实现 ====================
/**
 * 每个线程计算输出矩阵的一个元素
 * 全局内存访问，无优化
 */
__global__ void matrixMulNaive(const float *A, const float *B, float *C, 
                                int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M维度
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N维度
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ==================== 版本2: 共享内存优化 ====================
#define TILE_SIZE 16

/**
 * 使用共享内存和分块技术
 * 显著减少全局内存访问次数
 */
__global__ void matrixMulShared(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    // 共享内存声明
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 全局索引
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 遍历所有分块
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载分块到共享内存（带边界检查）
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // 同步：确保所有线程都加载完毕
        __syncthreads();
        
        // 在共享内存中计算部分结果
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // 同步：确保计算完成后再加载下一个分块
        __syncthreads();
    }
    
    // 写回全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== 版本3: 优化版（向量化读取 + 循环展开） ====================
/**
 * 进一步优化：
 * 1. 向量化内存访问
 * 2. 循环展开
 * 3. 寄存器使用优化
 */
__global__ void matrixMulOptimized(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];  // +1避免bank conflict
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // 使用寄存器存储累加结果
    float cValue = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载分块
        As[ty][tx] = (row < M && t * TILE_SIZE + tx < K) 
                     ? A[row * K + t * TILE_SIZE + tx] : 0.0f;
        Bs[ty][tx] = (t * TILE_SIZE + ty < K && col < N) 
                     ? B[(t * TILE_SIZE + ty) * N + col] : 0.0f;
        
        __syncthreads();
        
        // 循环展开优化
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            cValue += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// ==================== 辅助函数 ====================
void initMatrix(float *mat, int rows, int cols, float seed) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = seed * ((float)rand() / RAND_MAX);
    }
}

void cpuMatrixMul(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verifyResult(const float *ref, const float *gpu, int size, float eps = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(ref[i] - gpu[i]) > eps) {
            printf("Mismatch at index %d: CPU=%.4f, GPU=%.4f\n", i, ref[i], gpu[i]);
            return false;
        }
    }
    return true;
}

// ==================== 性能测试 ====================
void benchmark(const char *name, void (*kernel)(), dim3 grid, dim3 block,
               const float *d_A, const float *d_B, float *d_C, 
               int M, int N, int K, int warmup = 5, int iter = 20) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iter; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iter;
    
    double gflops = 2.0 * M * N * K / (ms * 1e6);
    printf("%-25s: %7.3f ms, %8.2f GFLOPS\n", name, ms, gflops);
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv) {
    // 矩阵维度 (M x K) * (K x N) = (M x N)
    int M = 1024, K = 1024, N = 1024;
    
    printf("=== Matrix Multiplication Benchmark ===\n");
    printf("Matrix A: %d x %d\n", M, K);
    printf("Matrix B: %d x %d\n", K, N);
    printf("Matrix C: %d x %d\n\n", M, N);
    
    // 分配主机内存
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    
    float *h_A = (float *)malloc(bytesA);
    float *h_B = (float *)malloc(bytesB);
    float *h_C = (float *)malloc(bytesC);
    float *h_ref = (float *)malloc(bytesC);
    
    // 初始化数据
    srand(1234);
    initMatrix(h_A, M, K, 1.0f);
    initMatrix(h_B, K, N, 1.0f);
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    
    // 数据传输到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    
    // 配置Kernel参数
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Kernel Configuration:\n");
    printf("  Grid:  (%d, %d)\n", grid.x, grid.y);
    printf("  Block: (%d, %d)\n\n", block.x, block.y);
    
    printf("Performance Results:\n");
    printf("----------------------------------------\n");
    
    // 测试各版本
    benchmark("Naive", matrixMulNaive, grid, block, d_A, d_B, d_C, M, N, K);
    benchmark("Shared Memory", matrixMulShared, grid, block, d_A, d_B, d_C, M, N, K);
    benchmark("Optimized", matrixMulOptimized, grid, block, d_A, d_B, d_C, M, N, K);
    
    // 验证正确性
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    cpuMatrixMul(h_A, h_B, h_ref, M, N, K);
    
    printf("\nVerification: %s\n", 
           verifyResult(h_ref, h_C, M * N) ? "PASSED ✓" : "FAILED ✗");
    
    // 清理
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    
    return 0;
}