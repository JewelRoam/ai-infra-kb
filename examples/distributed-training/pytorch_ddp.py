"""
PyTorch 分布式数据并行 (DDP) 训练示例

演示内容：
1. DDP 初始化与配置
2. 多进程训练循环
3. 梯度同步机制
4. Checkpoint 保存与加载
5. 多节点训练支持

启动命令：
    单机多卡: torchrun --nproc_per_node=4 pytorch_ddp.py
    多机多卡: torchrun --nnodes=2 --nproc_per_node=4 \
              --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
              pytorch_ddp.py
"""

import os
import sys
import argparse
import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ==================== 模型定义 ====================
class SimpleCNN(nn.Module):
    """示例CNN模型"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== 分布式工具函数 ====================
def setup_distributed(backend: str = "nccl"):
    """
    初始化分布式环境
    
    环境变量（由torchrun自动设置）：
    - WORLD_SIZE: 总进程数
    - LOCAL_RANK: 当前节点内的进程排名
    - RANK: 全局进程排名
    - MASTER_ADDR: 主节点地址
    - MASTER_PORT: 主节点端口
    """
    # 获取分布式信息
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(minutes=30)
    )
    
    # 设置设备
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """是否为主进程"""
    return dist.get_rank() == 0


# ==================== 数据加载 ====================
def create_dataloader(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int
) -> tuple:
    """
    创建分布式数据加载器
    
    关键点：
    1. 使用 DistributedSampler 确保数据分片
    2. 每个 rank 获取不同的数据子集
    3. epoch 开始时调用 sampler.set_epoch() 保证随机性
    """
    import torchvision
    import torchvision.transforms as transforms
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集（示例使用CIFAR-10）
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 确保每个 batch 大小一致
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader, train_sampler


# ==================== 训练函数 ====================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选，防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 日志（仅在主进程）
        if is_main_process() and batch_idx % 100 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
    
    # TensorBoard 记录
    if writer is not None and is_main_process():
        writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", 100. * correct / total, epoch)
    
    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    # 聚合所有进程的结果
    total_loss_tensor = torch.tensor(total_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = total_loss_tensor.item() / dist.get_world_size()
    accuracy = 100. * correct_tensor.item() / total_tensor.item()
    
    return avg_loss, accuracy


# ==================== Checkpoint 管理 ====================
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: str,
    extra_info: Optional[dict] = None
):
    """
    保存 Checkpoint
    
    注意：
    - DDP 模型需要保存 model.module.state_dict()
    - 只在主进程保存
    """
    if not is_main_process():
        return
    
    # 获取原始模型（去掉 DDP 包装）
    if isinstance(model, DDP):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    path: str,
    device: torch.device
) -> int:
    """
    加载 Checkpoint
    
    返回：
    - epoch: 保存的 epoch 数
    """
    checkpoint = torch.load(path, map_location=device)
    
    # 处理 DDP 模型
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"]


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP Training Example")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    args = parser.parse_args()
    
    # 初始化分布式
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f"=" * 60)
        print(f"Distributed Training Configuration:")
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local Rank: {local_rank}")
        print(f"  Device: {device}")
        print(f"=" * 60)
    
    # 创建输出目录
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据加载器
    train_loader, test_loader, train_sampler = create_dataloader(
        args.dataset, args.batch_size, args.num_workers, rank, world_size
    )
    
    # 创建模型
    model = SimpleCNN(num_classes=10).to(device)
    
    # 用 DDP 包装模型
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # 设为 True 如果有未使用的参数
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr * world_size,  # 线性缩放学习率
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # TensorBoard（仅主进程）
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # 加载 Checkpoint
    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, device)
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    best_accuracy = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # 设置 epoch（确保每个 epoch 数据分片不同）
        train_sampler.set_epoch(epoch)
        
        if is_main_process():
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")
            print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device, writer
        )
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if is_main_process():
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # TensorBoard 记录
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
            
            # 保存最佳模型
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                save_checkpoint(
                    model, optimizer, epoch,
                    os.path.join(args.output_dir, "best_model.pth"),
                    {"best_accuracy": best_accuracy}
                )
            
            # 定期保存 Checkpoint
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    model, optimizer, epoch,
                    os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                )
        
        # 更新学习率
        scheduler.step()
    
    # 清理
    if writer is not None:
        writer.close()
    cleanup_distributed()
    
    if is_main_process():
        print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()