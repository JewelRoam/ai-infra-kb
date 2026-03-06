"""
TensorRT 模型优化与推理示例

演示内容：
1. ONNX 转换为 TensorRT Engine
2. FP16/INT8 量化
3. 动态 batch size 支持
4. 推理性能测试

依赖：
    pip install tensorrt onnxruntime-gpu
"""

import os
import argparse
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np


# ==================== TensorRT Builder 配置 ====================
def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    min_batch_size: int = 1,
    opt_batch_size: int = 8,
    max_workspace_size: int = 4 << 30,  # 4GB
    int8_calib_dataset: Optional[str] = None,
) -> dict:
    """
    将 ONNX 模型转换为 TensorRT Engine
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出 engine 路径
        precision: 精度模式 ("fp32", "fp16", "int8")
        max_batch_size: 最大 batch size
        min_batch_size: 最小 batch size（动态 batch）
        opt_batch_size: 优化目标 batch size
        max_workspace_size: 最大 GPU 工作空间大小
        int8_calib_dataset: INT8 校准数据集路径
    
    Returns:
        构建信息字典
    """
    import tensorrt as trt
    
    # 创建 logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 创建 builder
    builder = trt.Builder(TRT_LOGGER)
    builder.max_workspace_size = max_workspace_size
    
    # 创建 network
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建 config
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    
    # 解析 ONNX 模型
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")
    
    # 获取输入信息
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape
    print(f"Input: {input_name}, shape: {input_shape}")
    
    # 配置动态 batch（如果需要）
    if -1 in input_shape or max_batch_size > 1:
        profile = builder.create_optimization_profile()
        min_shape = list(input_shape)
        opt_shape = list(input_shape)
        max_shape = list(input_shape)
        
        # 设置动态维度
        if min_shape[0] == -1:
            min_shape[0] = min_batch_size
            opt_shape[0] = opt_batch_size
            max_shape[0] = max_batch_size
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        print(f"Dynamic batch: min={min_shape[0]}, opt={opt_shape[0]}, max={max_shape[0]}")
    
    # 配置精度
    build_info = {
        "onnx_path": onnx_path,
        "engine_path": engine_path,
        "precision": precision,
        "max_batch_size": max_batch_size,
    }
    
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 mode enabled")
        else:
            print("⚠ FP16 not supported on this platform")
            precision = "fp32"
    
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ INT8 mode enabled")
            
            # INT8 校准器（需要实现）
            if int8_calib_dataset:
                calibrator = Int8Calibrator(int8_calib_dataset)
                config.int8_calibrator = calibrator
                print("✓ INT8 calibrator configured")
        else:
            print("⚠ INT8 not supported on this platform")
            precision = "fp32"
    
    # 构建引擎
    print("Building TensorRT engine...")
    start_time = time.time()
    
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    build_time = time.time() - start_time
    print(f"✓ Engine built in {build_time:.2f} seconds")
    
    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())
    print(f"✓ Engine saved to: {engine_path}")
    
    build_info["build_time_sec"] = build_time
    build_info["engine_size_mb"] = os.path.getsize(engine_path) / (1024 * 1024)
    
    return build_info


# ==================== INT8 校准器 ====================
class Int8Calibrator(trt.IInt8MinMaxCalibrator):
    """INT8 校准器实现"""
    
    def __init__(self, calibration_data: np.ndarray, batch_size: int = 1):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_batch(self, names: list) -> list:
        if self.current_index >= len(self.calibration_data):
            return None
        
        batch = self.calibration_data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size
        
        return [batch.astype(np.float32)]
    
    def read_calibration_cache(self) -> bytes:
        return None
    
    def write_calibration_cache(self, cache: bytes) -> None:
        pass


# ==================== TensorRT 推理 ====================
class TensorRTInference:
    """TensorRT 推理封装"""
    
    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        
        # 加载引擎
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # 获取输入输出信息
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            if self.engine.binding_is_input(i):
                self.input_name = name
                self.input_shape = shape
                self.input_dtype = dtype
                print(f"Input: {name}, shape: {shape}, dtype: {dtype}")
            else:
                self.output_name = name
                self.output_shape = shape
                self.output_dtype = dtype
                print(f"Output: {name}, shape: {shape}, dtype: {dtype}")
        
        # 分配 GPU 内存
        self.stream = cuda.Stream()
    
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """执行推理"""
        import pycuda.driver as cuda
        
        # 处理动态 batch
        if self.input_shape[0] == -1:
            self.context.set_binding_shape(0, input_data.shape)
            output_shape = list(self.output_shape)
            output_shape[0] = input_data.shape[0]
        else:
            output_shape = self.output_shape
        
        # 分配内存
        input_size = input_data.size * input_data.itemsize
        output_size = int(np.prod(output_shape)) * np.dtype(self.output_dtype).itemsize
        
        # GPU 内存
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        # 拷贝输入
        cuda.memcpy_htod_async(d_input, input_data.astype(self.input_dtype), self.stream)
        
        # 执行
        bindings = [int(d_input), int(d_output)]
        self.context.execute_async_v2(bindings, self.stream.handle)
        
        # 拷贝输出
        output = np.empty(output_shape, dtype=self.output_dtype)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        
        # 同步
        self.stream.synchronize()
        
        return output
    
    def benchmark(
        self,
        batch_size: int,
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        """性能测试"""
        # 创建测试数据
        input_shape = list(self.input_shape)
        if input_shape[0] == -1:
            input_shape[0] = batch_size
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            self(dummy_input)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            self(dummy_input)
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / iterations * 1000
        throughput = iterations * batch_size / (end - start)
        
        return {
            "batch_size": batch_size,
            "avg_time_ms": avg_time_ms,
            "throughput_qps": throughput,
        }


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="TensorRT Optimization")
    parser.add_argument("--mode", type=str, default="build", choices=["build", "benchmark"])
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--engine-path", type=str, default="model.trt")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--benchmark-batch-sizes", type=str, default="1,8,16,32")
    args = parser.parse_args()
    
    if args.mode == "build":
        info = build_tensorrt_engine(
            args.onnx_path,
            args.engine_path,
            precision=args.precision,
            max_batch_size=args.max_batch,
        )
        print(f"\nBuild Info: {info}")
    
    elif args.mode == "benchmark":
        trt_infer = TensorRTInference(args.engine_path)
        
        batch_sizes = [int(x) for x in args.benchmark_batch_sizes.split(",")]
        
        print("\nBenchmark Results:")
        print("-" * 50)
        for bs in batch_sizes:
            result = trt_infer.benchmark(bs)
            print(f"Batch {bs:3d}: {result['avg_time_ms']:.3f} ms, "
                  f"{result['throughput_qps']:.1f} QPS")


if __name__ == "__main__":
    main()