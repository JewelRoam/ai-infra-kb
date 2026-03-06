"""
DeepSpeed 训练脚本示例

演示内容：
1. DeepSpeed 初始化
2. ZeRO 优化器配置
3. 混合精度训练
4. 梯度检查点
5. 模型保存与加载

启动命令：
    deepspeed train_deepspeed.py --deepspeed_config deepspeed_config.json
    或
    torchrun --nproc_per_node=4 train_deepspeed.py --deepspeed_config deepspeed_config.json
"""

import os
import argparse
import json
from typing import Optional

import torch
import torch.nn as nn
import deepspeed
from deepspeed.accelerator import get_accelerator


# ==================== 模型定义 ====================
class TransformerBlock(nn.Module):
    """Transformer 块"""
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """简单 Transformer 模型用于演示"""
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {"loss": loss, "logits": logits}


# ==================== DeepSpeed 工具函数 ====================
def get_ds_config(config_path: str) -> dict:
    """加载 DeepSpeed 配置"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def print_trainable_parameters(model: nn.Module):
    """打印模型可训练参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {100 * trainable_params / all_param:.2f}%")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Training Example")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    parser.add_argument("--deepspeed_config", type=str, required=True, help="DeepSpeed config path")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    
    # 初始化 DeepSpeed
    deepspeed.init_distributed()
    
    # 加载配置
    ds_config = get_ds_config(args.deepspeed_config)
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
    )
    
    print("=" * 60)
    print("DeepSpeed Training Configuration:")
    print(f"  ZeRO Stage: {ds_config.get('zero_optimization', {}).get('stage', 'N/A')}")
    print(f"  FP16 Enabled: {ds_config.get('fp16', {}).get('enabled', False)}")
    print(f"  BF16 Enabled: {ds_config.get('bf16', {}).get('enabled', False)}")
    print("=" * 60)
    print_trainable_parameters(model)
    
    # DeepSpeed 初始化
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    
    device = get_accelerator().device_name(model_engine.local_rank)
    
    # 模拟数据加载器
    def get_batch():
        batch_size = ds_config["train_micro_batch_size_per_gpu"]
        seq_len = args.max_seq_len
        input_ids = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        return input_ids.to(device), labels.to(device)
    
    # 训练循环
    num_steps = ds_config["scheduler"]["params"]["total_num_steps"]
    
    for step in range(num_steps):
        # 前向传播
        input_ids, labels = get_batch()
        outputs = model_engine(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # 反向传播
        model_engine.backward(loss)
        
        # 参数更新
        model_engine.step()
        
        # 日志
        if step % 100 == 0 and model_engine.local_rank == 0:
            print(f"Step [{step}/{num_steps}] Loss: {loss.item():.4f}")
        
        # 定期保存
        if (step + 1) % 1000 == 0 and model_engine.local_rank == 0:
            save_dir = os.path.join(args.output_dir, f"checkpoint_{step+1}")
            model_engine.save_checkpoint(save_dir)
            print(f"Checkpoint saved: {save_dir}")
    
    # 最终保存
    if model_engine.local_rank == 0:
        final_dir = os.path.join(args.output_dir, "final_model")
        model_engine.save_checkpoint(final_dir)
        print(f"Final model saved: {final_dir}")


if __name__ == "__main__":
    main()