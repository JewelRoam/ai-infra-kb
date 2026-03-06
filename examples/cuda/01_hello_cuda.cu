/**
 * CUDA 入门示例：向量加法
 * 
 * 演示内容：
 * 1. CUDA程序基本结构
 * 2. 内存分配与数据传输
 * 3. Kernel函数编写
 * 4. 错误处理宏
 * 
 * 编译命令：nvcc -o vector_add 01_hello_cuda.cu
 * 运行命令：./vector_add
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ==================== 错误处理宏 ====================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// ==================== Kernel 函数 ====================
/**
 * 向量加法 Kernel
 * 每个线程处理一个元素
 * 
 * @param a 输入向量A
 * @param b 输入向量B
 * @param c 输出向量C = A + B
 * @param n 向量长度
 */
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ==================== 辅助函数 ====================
/**
 * 初始化向量数据
 */
void initVector(float *vec, int n, float value) {
    for (int i = 0; i < n; i++) {
        vec[i] = value * i;
    }
}

/**
 * 验证结果正确性
 */
bool verifyResult(const float *a, const float *b, const float *c, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5) {
            printf("Verification failed at index %d: expected %.2f, got %.2f\n",
                   i, a[i] + b[i], c[i]);
            return false;
        }
    }
    return true;
}

// ==================== 主函数 ====================
int main(int argc, char **argv) {
    // 参数设置
    int n = 1 << 20;  // 1M 元素
    size_t bytes = n * sizeof(float);
    
    printf("=== CUDA Vector Addition Demo ===\n");
    printf("Vector size: %d elements (%.2f MB)\n\n", n, (float)bytes / 1024 / 1024);
    
    // ==================== 1. 分配主机内存 ====================
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    
    // 初始化数据
    initVector(h_a, n, 0.01f);
    initVector(h_b, n, 0.02f);
    
    // ==================== 2. 分配设备内存 ====================
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // ==================== 3. 数据传输：Host -> Device ====================
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    // ==================== 4. 配置Kernel启动参数 ====================
    int blockSize = 256;  // 每个block的线程数
    int gridSize = (n + blockSize - 1) / blockSize;  // block数量（向上取整）
    
    printf("Grid size: %d blocks\n", gridSize);
    printf("Block size: %d threads\n", blockSize);
    printf("Total threads: %d\n\n", gridSize * blockSize);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // ==================== 5. 启动Kernel ====================
    CUDA_CHECK(cudaEventRecord(start));
    
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // 检查Kernel执行错误
    CUDA_CHECK(cudaGetLastError());
    
    // 计算执行时间
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);
    printf("Effective bandwidth: %.2f GB/s\n", 
           (3.0 * bytes / 1e9) / (ms / 1000.0));  // 读a, 读b, 写c
    
    // ==================== 6. 数据传输：Device -> Host ====================
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    // ==================== 7. 验证结果 ====================
    if (verifyResult(h_a, h_b, h_c, n)) {
        printf("\n✓ Verification passed!\n");
    } else {
        printf("\n✗ Verification failed!\n");
    }
    
    // ==================== 8. 清理资源 ====================
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\nDone.\n");
    return 0;
}