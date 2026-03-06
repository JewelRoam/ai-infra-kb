# AI 硬件基础

> 本章介绍 AI 基础设施所需的硬件知识

## 目录

- [1.1 AI芯片](#11-ai芯片)
  - [GPU](#gpu)
  - [NPU](#npu)
  - [TPU](#tpu)
- [1.2 服务器](#12-服务器)
- [1.3 存储](#13-存储)
- [1.4 网络](#14-网络)

---

## 1.1 AI芯片

### GPU (Graphics Processing Unit)

GPU 是目前 AI 训练和推理的主流硬件。

#### NVIDIA 产品线

| 型号 | 定位 | 显存 | 算力 (FP16) | 适用场景 |
|------|------|------|-------------|----------|
| H100 | 数据中心 | 80GB HBM3 | 51 TFLOPS | 大模型训练 |
| A100 | 数据中心 | 40/80GB HBM2 | 19.5 TFLOPS | 训练/推理 |
| RTX 4090 | 消费级 | 24GB GDDR6X | 82.6 TFLOPS | 个人开发/小规模训练 |
| L40S | 数据中心 | 48GB GDDR6X | 46 TFLOPS | 推理/可视化 |
| H200 | 数据中心 | 141GB HBM3e | 51 TFLOPS | 大模型推理 |

> 注: 算力为稀疏(sparse)算力

#### AMD 产品线

| 型号 | 定位 | 显存 | 算力 (FP16) |
|------|------|------|-------------|
| MI300X | 数据中心 | 192GB HBM3 | 97.2 TFLOPS |
| MI250X | 数据中心 | 80GB HBM2 | 47.9 TFLOPS |

### NPU (Neural Processing Unit)

NPU 专为神经网络推理设计，适合端侧和边缘 AI 场景。

#### Intel NPU (Meteor Lake / Arrow Lake)

- **定位**: 端侧 AI 推理
- **算力**: 10-15 TOPS (Int8)
- **特点**: 低功耗、集成在 CPU 中
- **支持**: OpenVINO, DirectML

#### AMD NPU (Ryzen AI)

- **定位**: 笔记本电脑 AI 加速
- **算力**: 10-16 TOPS (Int8)
- **特点**: XDNA 架构
- **支持**: ONNX Runtime, TensorFlow Lite

#### 华为昇腾 NPU

| 型号 | 定位 | 算力 | 显存 |
|------|------|------|------|
| 910B | 数据中心 | 400 TFLOPS (FP16) | 64GB HBM |
| 310 | 边缘 | 8 TFLOPS (FP16) | - |

### TPU (Tensor Processing Unit)

Google 自研的 AI 加速器，主要用于云端。

| 型号 | 矩阵单元 | 峰值算力 | 显存 |
|------|----------|----------|------|
| TPU v5p | 8960 | 459 TFLOPS (BF16) | 95GB HBM |
| TPU v4 | 4096 | 275 TFLOPS (BF16) | 32GB HBM |

---

## 1.2 服务器

### GPU 服务器规格

| 规格 | GPU数量 | 网络 | 电源 | 典型型号 |
|------|---------|------|------|----------|
| 1U 4卡 | 4x H100 | 200Gb IB | 2000W | NVIDIA DGX H100 |
| 2U 8卡 | 8x H100 | 400Gb IB | 3500W | NVIDIA DGX H100 |
| 4U 8卡 | 8x H100 | 400Gb IB | 5000W | Supermicro 4U |

### 典型配置

```yaml
# 8卡 GPU 服务器配置示例
server:
  cpu: 2x Intel Xeon Gold 6448Y (32核/64线程)
  memory: 1TB DDR5
  gpu: 8x NVIDIA H100 80GB
  network:
    - 2x 100GbE (数据网络)
    - 2x 200Gb InfiniBand (集群网络)
  storage:
    - 2x 7.68TB NVMe (系统盘)
    - 4x 15.36TB NVMe (数据盘, RAID 0)
  power: 3500W (双电源)
  cooling: 液冷/风冷
```

---

## 1.3 存储

### 类型对比

| 类型 | 带宽 | 延迟 | 容量 | 用途 |
|------|------|------|------|------|
| HBM | 2-3.5 TB/s | ~100ns | 80-192 GB | GPU 显存 |
| NVMe SSD | 7 GB/s | ~100μs | 2-16 TB | 训练数据缓存 |
| SATA SSD | 0.5 GB/s | ~100μs | 2-8 TB | 轻量存储 |
| 分布式存储 | 10-100 GB/s | 1-10ms | PB级 | 大规模训练数据 |

### 推荐配置

```yaml
# 训练集群存储配置
storage:
  # 本地 NVMe 缓存
  local_cache:
    type: NVMe SSD
    capacity: 15 TB
    raid: 0
    
  # 分布式存储
  distributed:
    type: Lustre / GPFS / CephFS
    bandwidth: 100 GB/s
    capacity: 10 PB
    
  # 对象存储
  object_storage:
    type: MinIO / S3
    capacity: 1 PB
```

---

## 1.4 网络

### 网络技术

| 技术 | 带宽 | 延迟 | 成本 | 适用场景 |
|------|------|------|------|----------|
| InfiniBand HDR | 200 Gbps | <1μs | 高 | 数据中心训练 |
| RoCE v2 | 100-200 Gbps | <5μs | 中 | 云端训练 |
| 100GbE | 100 Gbps | ~10μs | 中 | 推理服务 |
| 25GbE | 25 Gbps | ~50μs | 低 | 开发测试 |

### 多网卡配置

```yaml
# 训练节点网络配置
network:
  # 数据网络 - 用于数据传输
  data:
    type: Ethernet
    speed: 100GbE
    bonding: LACP
    
  # 集群网络 - 用于梯度同步
  cluster:
    type: InfiniBand HDR
    topology: Fat-tree
    
  # 管理网络
  mgmt:
    type: 10GbE
```

---

## 本章小结

- **GPU** 是 AI 训练主力，H100/A100 为数据中心主流
- **NPU** 适合端侧推理，Intel Ultra/AMD XDNA 是消费级选择
- **NVMe SSD** 是训练数据缓存的最佳选择
- **InfiniBand** 是大规模训练的首选网络

---

## 下一步

- [02-training](../02-training/README.md) - 学习分布式训练
- [03-inference](../03-inference/README.md) - 了解推理优化