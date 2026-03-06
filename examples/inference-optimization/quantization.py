"""
模型量化示例

演示内容：
1. 动态量化 (Dynamic Quantization)
2. 静态量化 (Static Quantization)
3. 量化感知训练 (Quantization Aware Training)
4. INT8/FP16 混合精度

依赖：
    pip install torch torchvision
"""

import os
import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub


# ==================== 量化模型定义 ====================
class QuantizableModel(nn.Module):
    """可量化模型"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 1024, output_size: int = 10):
        super().__init__()
        
        # 量化桩
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 模型层
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """融合层（量化前准备）"""
        torch.quantization.fuse_modules(self.layers, ['0', '1'], inplace=True)
        torch.quantization.fuse_modules(self.layers, ['2', '3'], inplace=True)


# ==================== 动态量化 ====================
def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    动态量化
    
    特点：
    - 权重被量化为 INT8
    - 激活值在推理时动态量化
    - 适用于 LSTM, Transformer 等
    - 不需要校准数据
    """
    model.eval()
    
    # 动态量化配置
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},  # 要量化的层类型
        dtype=torch.qint8,
    )
    
    return quantized_model


# ==================== 静态量化 ====================
def apply_static_quantization(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    num_calibration_batches: int = 100,
) -> nn.Module:
    """
    静态量化
    
    特点：
    - 权重和激活值都被量化
    - 需要校准数据确定量化范围
    - 推理速度最快
    """
    model.eval()
    
    # 融合层
    model.fuse_model()
    
    # 设置量化配置
    model.qconfig = quant.get_default_qconfig('fbgemm')  # x86 CPU
    
    # 准备量化
    quant.prepare(model, inplace=True)
    
    # 校准
    print("Running calibration...")
    with torch.no_grad():
        for i, (data, _) in enumerate(calibration_loader):
            if i >= num_calibration_batches:
                break
            model(data)
    
    # 转换为量化模型
    quant.convert(model, inplace=True)
    
    return model


# ==================== 量化感知训练 ====================
class QuantizationAwareTrainer:
    """量化感知训练"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-5,
        num_epochs: int = 5,
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
    
    def prepare_model(self):
        """准备量化感知训练"""
        self.model.train()
        self.model.fuse_model()
        self.model.qconfig = quant.get_default_qat_qconfig('fbgemm')
        quant.prepare_qat(self.model, inplace=True)
    
    def train(self):
        """训练"""
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
    
    def get_quantized_model(self) -> nn.Module:
        """获取量化后的模型"""
        self.model.eval()
        quantized_model = quant.convert(self.model)
        return quantized_model


# ==================== 量化精度对比 ====================
def compare_quantization_precision(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> dict:
    """对比原始模型和量化模型的精度"""
    original_model.eval()
    quantized_model.eval()
    
    original_model.to(device)
    quantized_model.to(device)
    
    def evaluate(model, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        return 100. * correct / total
    
    original_acc = evaluate(original_model, test_loader)
    quantized_acc = evaluate(quantized_model, test_loader)
    
    return {
        "original_accuracy": original_acc,
        "quantized_accuracy": quantized_acc,
        "accuracy_drop": original_acc - quantized_acc,
    }


# ==================== 模型大小和速度对比 ====================
def compare_model_stats(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_shape: tuple,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """对比模型大小和推理速度"""
    
    def get_model_size(model):
        torch.save(model.state_dict(), "temp_model.pt")
        size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
        os.remove("temp_model.pt")
        return size_mb
    
    def benchmark(model, shape):
        model.eval()
        dummy_input = torch.randn(shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy_input)
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                model(dummy_input)
        end = time.perf_counter()
        
        return (end - start) / iterations * 1000  # ms
    
    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    
    original_time = benchmark(original_model, input_shape)
    quantized_time = benchmark(quantized_model, input_shape)
    
    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": original_size / quantized_size,
        "original_time_ms": original_time,
        "quantized_time_ms": quantized_time,
        "speedup": original_time / quantized_time,
    }


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Model Quantization")
    parser.add_argument("--mode", type=str, default="dynamic", 
                        choices=["dynamic", "static", "qat", "compare"])
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--output-size", type=int, default=10)
    parser.add_argument("--calibration-batches", type=int, default=100)
    args = parser.parse_args()
    
    # 创建模型
    model = QuantizableModel(args.input_size, args.hidden_size, args.output_size)
    
    # 模拟数据加载器
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000, input_size=512, output_size=10):
            self.data = torch.randn(size, input_size)
            self.targets = torch.randint(0, output_size, (size,))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
    
    dummy_loader = torch.utils.data.DataLoader(
        DummyDataset(1000, args.input_size, args.output_size),
        batch_size=32,
    )
    
    if args.mode == "dynamic":
        print("Applying dynamic quantization...")
        quantized_model = apply_dynamic_quantization(model)
        print(f"✓ Dynamic quantization complete")
        
    elif args.mode == "static":
        print("Applying static quantization...")
        quantized_model = apply_static_quantization(
            model, dummy_loader, args.calibration_batches
        )
        print(f"✓ Static quantization complete")
    
    elif args.mode == "qat":
        print("Running quantization-aware training...")
        trainer = QuantizationAwareTrainer(model, dummy_loader)
        trainer.prepare_model()
        trainer.train()
        quantized_model = trainer.get_quantized_model()
        print(f"✓ QAT complete")
    
    elif args.mode == "compare":
        print("Comparing quantization methods...")
        
        # 原始模型
        original_model = QuantizableModel(args.input_size, args.hidden_size, args.output_size)
        
        # 动态量化
        dynamic_model = apply_dynamic_quantization(
            QuantizableModel(args.input_size, args.hidden_size, args.output_size)
        )
        
        stats = compare_model_stats(
            original_model, dynamic_model,
            (1, args.input_size)
        )
        
        print("\nComparison Results:")
        print("-" * 50)
        print(f"Original Model Size: {stats['original_size_mb']:.2f} MB")
        print(f"Quantized Model Size: {stats['quantized_size_mb']:.2f} MB")
        print(f"Compression Ratio: {stats['compression_ratio']:.2f}x")
        print(f"Original Inference Time: {stats['original_time_ms']:.3f} ms")
        print(f"Quantized Inference Time: {stats['quantized_time_ms']:.3f} ms")
        print(f"Speedup: {stats['speedup']:.2f}x")


if __name__ == "__main__":
    main()