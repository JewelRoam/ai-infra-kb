"""
ONNX 模型导出与优化示例

演示内容：
1. PyTorch 模型导出为 ONNX
2. ONNX 模型优化
3. ONNX Runtime 推理
4. 动态输入尺寸处理

依赖：
    pip install onnx onnxruntime onnxoptimizer
"""

import os
import argparse
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ==================== 示例模型 ====================
class SimpleModel(nn.Module):
    """简单模型用于演示"""
    def __init__(self, input_size: int = 512, hidden_size: int = 1024, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ==================== ONNX 导出 ====================
def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 512),
    dynamic_batch: bool = False,
    opset_version: int = 14,
    simplify: bool = True,
) -> dict:
    """
    将 PyTorch 模型导出为 ONNX 格式
    
    Args:
        model: PyTorch 模型
        output_path: 输出路径
        input_shape: 输入张量形状 (不含 batch 维度的动态)
        dynamic_batch: 是否支持动态 batch size
        opset_version: ONNX opset 版本
        simplify: 是否使用 onnx-simplifier 简化模型
    
    Returns:
        导出信息字典
    """
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(input_shape)
    
    # 动态轴配置
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"ONNX model exported to: {output_path}")
    
    # 验证导出的模型
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation passed")
    
    # 获取模型信息
    model_info = {
        "input_shape": input_shape,
        "dynamic_batch": dynamic_batch,
        "opset_version": opset_version,
        "file_size_mb": os.path.getsize(output_path) / (1024 * 1024),
    }
    
    # 简化模型 (可选)
    if simplify:
        try:
            import onnxsim
            simplified_path = output_path.replace(".onnx", "_simplified.onnx")
            onnxsim.simplify(output_path, simplified_path)
            print(f"✓ Simplified model saved to: {simplified_path}")
            model_info["simplified_path"] = simplified_path
        except ImportError:
            print("Warning: onnxsim not installed, skipping simplification")
    
    return model_info


# ==================== ONNX 优化 ====================
def optimize_onnx_model(
    input_path: str,
    output_path: str,
) -> None:
    """
    使用 onnxoptimizer 优化 ONNX 模型
    
    优化包括：
    - 常量折叠
    - 死代码消除
    - 算子融合
    """
    import onnx
    from onnx import optimizer
    
    # 加载模型
    model = onnx.load(input_path)
    
    # 获取所有可用的优化 passes
    all_passes = optimizer.get_available_passes()
    print(f"Available optimization passes: {len(all_passes)}")
    
    # 选择要应用的优化
    passes = [
        "eliminate_unused_initializer",
        "eliminate_identity",
        "eliminate_nop_transpose",
        "eliminate_nop_pad",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
    ]
    
    # 应用优化
    optimized_model = optimizer.optimize(model, passes)
    
    # 验证
    onnx.checker.check_model(optimized_model)
    
    # 保存
    onnx.save(optimized_model, output_path)
    print(f"Optimized model saved to: {output_path}")


# ==================== ONNX Runtime 推理 ====================
class ONNXInferenceSession:
    """ONNX Runtime 推理封装"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_threads: int = 4,
    ):
        import onnxruntime as ort
        
        # 配置 providers
        providers = []
        if device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")  # 作为后备
        
        # Session 选项
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 创建 session
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"ONNX Session initialized:")
        print(f"  Input: {self.input_name} {self.input_shape}")
        print(f"  Output: {self.output_name}")
        print(f"  Providers: {self.session.get_providers()}")
    
    def __call__(self, input_data) -> any:
        """执行推理"""
        return self.session.run(
            [self.output_name],
            {self.input_name: input_data},
        )[0]
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        warmup: int = 10,
        iterations: int = 100,
    ) -> dict:
        """性能测试"""
        import numpy as np
        
        # 创建输入数据
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
        throughput = iterations / (end - start)
        
        return {
            "avg_time_ms": avg_time_ms,
            "throughput_qps": throughput,
            "input_shape": input_shape,
        }


# ==================== PyTorch vs ONNX 对比 ====================
def compare_inference(
    torch_model: nn.Module,
    onnx_session: ONNXInferenceSession,
    input_shape: Tuple[int, ...],
    num_samples: int = 100,
) -> dict:
    """对比 PyTorch 和 ONNX 推理"""
    import numpy as np
    
    torch_model.eval()
    device = next(torch_model.parameters()).device
    
    # 随机生成测试数据
    test_inputs = [torch.randn(input_shape) for _ in range(num_samples)]
    
    # PyTorch 推理
    with torch.no_grad():
        torch_start = time.perf_counter()
        torch_outputs = [torch_model(inp.to(device)).cpu().numpy() for inp in test_inputs]
        torch_time = time.perf_counter() - torch_start
    
    # ONNX 推理
    onnx_inputs = [inp.numpy() for inp in test_inputs]
    onnx_start = time.perf_counter()
    onnx_outputs = [onnx_session(inp) for inp in onnx_inputs]
    onnx_time = time.perf_counter() - onnx_start
    
    # 计算误差
    max_diff = 0.0
    avg_diff = 0.0
    for torch_out, onnx_out in zip(torch_outputs, onnx_outputs):
        diff = np.abs(torch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
        avg_diff += diff
    avg_diff /= num_samples
    
    return {
        "torch_time_ms": torch_time / num_samples * 1000,
        "onnx_time_ms": onnx_time / num_samples * 1000,
        "speedup": torch_time / onnx_time,
        "max_diff": max_diff,
        "avg_diff": avg_diff,
        "outputs_match": max_diff < 1e-5,
    }


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="ONNX Export and Optimization")
    parser.add_argument("--mode", type=str, default="export", 
                        choices=["export", "optimize", "benchmark", "compare"])
    parser.add_argument("--model-path", type=str, default="model.onnx")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=10)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    if args.mode == "export":
        # 创建模型
        model = SimpleModel(args.input_size, args.hidden_size, args.output_size)
        model.eval()
        
        # 导出
        info = export_to_onnx(
            model,
            args.model_path,
            input_shape=(1, args.input_size),
            dynamic_batch=args.dynamic_batch,
        )
        print(f"\nExport Info: {info}")
    
    elif args.mode == "optimize":
        optimize_onnx_model(args.model_path, args.model_path.replace(".onnx", "_optimized.onnx"))
    
    elif args.mode == "benchmark":
        session = ONNXInferenceSession(args.model_path, args.device)
        results = session.benchmark((1, args.input_size))
        print(f"\nBenchmark Results: {results}")
    
    elif args.mode == "compare":
        # 创建 PyTorch 模型
        model = SimpleModel(args.input_size, args.hidden_size, args.output_size)
        if args.device == "cuda":
            model = model.cuda()
        model.eval()
        
        # 创建 ONNX session
        session = ONNXInferenceSession(args.model_path, args.device)
        
        # 对比
        results = compare_inference(model, session, (1, args.input_size))
        print(f"\nComparison Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()