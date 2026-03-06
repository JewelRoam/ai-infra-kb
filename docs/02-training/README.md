# AI 训练系统

> 本章介绍分布式训练技术、加速框架和优化方法

## 目录

- [2.1 分布式训练基础](#21-分布式训练基础)
  - [数据并行](#数据并行)
  - [模型并行](#模型并行)
  - [流水并行](#流水并行)
- [2.2 训练框架](#22-训练框架)
  - [PyTorch DDP](#pytorch-ddp)
  - [PyTorch FSDP](#pytorch-fsdp)
  - [DeepSpeed](#deepspeed)
  - [Megatron-LM](#megatron-lm)
- [2.3 训练优化技术](#23-训练优化技术)
  - [混合精度训练](#混合精度训练)
  - [梯度累积](#梯度累积)
  - [Checkpointing](#checkpointing)
- [2.4 通信优化](#24-通信优化)

---

## 2.1 分布式训练基础

### 数据并行 (Data Parallelism)

每个节点拥有完整的模型副本，处理不同的数据batch。

```
┌─────────────────────────────────────────────────────────┐
│                    Data Parallel                         │
├─────────────────────────────────────────────────────────┤
│  GPU 0          GPU 1          GPU 2          GPU 3    │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐ │
│  │Model│        │Model│        │Model│        │Model│ │
│  │Copy │        │Copy │        │Copy │        │Copy │ │
│  └─────┘        └─────┘        └─────┘        └─────┘ │
│     ↓              ↓              ↓              ↓      │
│  Forward       Forward        Forward        Forward   │
│     ↓              ↓              ↓              ↓      │
│  Loss          Loss          Loss          Loss        │
│     ↓              ↓              ↓              ↓      │
│  Backward     Backward      Backward      Backward     │
│     ↓              ↓              ↓              ↓      │
│  ────────── All-Reduce (梯度同步) ──────────           │
│     ↓              ↓              ↓              ↓      │
│  Optimizer Update (相同)                            │
└─────────────────────────────────────────────────────────┘
```

**优点**: 实现简单，扩展性好
**缺点**: 显存占用高（每个GPU都需要完整模型）

**适用场景**: 单卡能放下模型的场景

### 模型并行 (Model Parallelism)

将模型切分到多个GPU，每个GPU只负责部分层。

```
┌─────────────────────────────────────────────────────────┐
│                    Model Parallel                        │
├─────────────────────────────────────────────────────────┤
│  GPU 0          GPU 1          GPU 2          GPU 3    │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐ │
│  │L0-L4│  ──▶  │L5-L9│  ──▶  │L10-14│ ──▶   │L15-19│ │
│  │     │        │     │        │     │        │     │ │
│  └─────┘        └─────┘        └─────┘        └─────┘ │
│                    Activations Forward                  │
│                    Gradients Backward                   │
└─────────────────────────────────────────────────────────┘
```

**优点**: 可以训练超大模型
**缺点**: 通信开销大，GPU利用率可能低

**适用场景**: 模型过大，单卡无法加载

### 流水并行 (Pipeline Parallelism)

将模型按层分组，形成流水线式的执行。

```
┌─────────────────────────────────────────────────────────┐
│                  Pipeline Parallel                       │
├─────────────────────────────────────────────────────────┤
│  Stage 0       Stage 1       Stage 2       Stage 3     │
│  ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐ │
│  │L0-L4│  ──▶  │L5-L9│  ──▶  │L10-14│ ──▶   │L15-19│ │
│  └─────┘        └─────┘        └─────┘        └─────┘ │
│                                                          │
│  Time:  T0    T1    T2    T3    T4    T5    T6        │
│  GPU0: [F0]   [F1]  [B1]  [B0]                        │
│  GPU1:  -    [F0]  [F1]  [B1]  [B0]                   │
│  GPU2:  -     -    [F0]  [F1]  [B1]  [B0]             │
│  GPU3:  -     -     -    [F0]  [F1]  [B1] [B0]        │
└─────────────────────────────────────────────────────────┘
F = Forward, B = Backward
```

**优点**: 减少流水线气泡，提高效率
**缺点**: 需要调度优化

---

## 2.2 训练框架

### PyTorch DDP (DistributedDataParallel)

最常用的数据并行训练方式。

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

# 初始化进程组
dist.init_process_group(backend="nccl")

# 模型设置
model = MyModel().cuda()
model = DDP(model, device_ids=[local_rank])

# 数据采样器
train_sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, sampler=train_sampler)

# 训练循环
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for batch in train_loader:
        inputs, targets = batch.cuda(), targets.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**关键配置**:
```bash
# 启动命令
torchrun --nproc_per_node=8 train.py

# 环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### PyTorch FSDP (FullyShardedDataParallel)

参数分片存储，适合大模型训练。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 自动包装Transformer层
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer}
)

# 混合精度
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# FSDP 包装
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device(),
)
```

**优点**:
- 显存效率高（参数分片到多卡）
- 支持超大模型训练

**缺点**:
- 通信开销比DDP大
- 配置复杂

### DeepSpeed

微软开源的分布式训练库，特别优化了ZeRO（Zero Redundancy Optimizer）。

```python
import deepspeed

# DeepSpeed 配置
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "steps_per_print": 10,
}

# 初始化
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    model.backward(loss)
    model.step()
```

**ZeRO 优化阶段**:
- Stage 1: Optimizer State 分片
- Stage 2: + Gradient 分片
- Stage 3: + Parameter 分片

### Megatron-LM

NVIDIA 的大模型训练框架，针对Transformer模型优化。

```python
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.mlp import MLPSubmodule

# _tensor_model_parallel_size: 张量并行度
# _pipeline_model_parallel_size: 流水并行度
# _data_parallel_size: 数据并行度

# 模型构建
model = GPTModel(
    num_layers=transformer_config.num_layers,
    vocab_size=vocab_size,
    max_sequence_length=seq_len,
    parallel_output=True,
)
```

---

## 2.3 训练优化技术

### 混合精度训练

使用FP16/BF16加速训练，减少显存占用。

```python
# PyTorch AMP (Automatic Mixed Precision)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast(dtype=torch.float16):
        outputs = model(batch.input)
        loss = criterion(outputs, batch.target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**精度选择**:
- **FP16**: 兼容性好，但需要loss scaling防止下溢
- **BF16**: 动态范围更大，不需要loss scaling，推荐使用

### 梯度累积

当batch过大时，分多次计算梯度后合并。

```python
# 梯度累积
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps  # 归一化
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Checkpointing

选择性保存激活值，节省显存。

```python
from torch.utils.checkpoint import checkpoint_sequential

# 分段保存激活值
model = nn.Sequential(*layers)
segments = 4  # 分成4段

checkpointed_model = checkpoint_sequential(
    model, 
    segments, 
    input
)
```

---

## 2.4 通信优化

### NCCL 优化

```bash
# 环境变量优化
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0          # 启用InfiniBand
export NCCL_NET_GDR_LEVEL=2       # GPUDirect RDMA
export NCCL_SOCKET_IFNAME=eth0    # 指定网卡
export NCCL_IB_TIMEOUT=20
export NCCL_IB_RETRY_CNT=10
```

### 通信算法选择

| 算法 | 描述 | 适用场景 |
|------|------|----------|
| Ring AllReduce | 环形通信，数据中心 | 小规模集群 |
| Tree AllReduce | 树形通信 | 大规模集群 |
| NCCL | NVIDIA优化库 | GPU训练 |

---

## 本章小结

- **数据并行** (DDP) 适合单卡能放下模型的场景
- **模型并行** 适合超大模型，需要手动切分
- **FSDP/ZeRO** 是大模型训练的主流方案
- **混合精度** 可提升性能并节省显存

---

## 下一步

- [03-inference](../03-inference/README.md) - 学习推理优化
- [../examples/distributed-training/](../../examples/distributed-training/README.md) - 查看代码示例