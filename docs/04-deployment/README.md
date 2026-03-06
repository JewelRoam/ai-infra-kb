# AI 部署与运维

> 本章介绍容器化部署、Kubernetes 调度和 CI/CD 流程

## 目录

- [4.1 容器化](#41-容器化)
  - [Dockerfile 最佳实践](#dockerfile-最佳实践)
  - [多阶段构建](#多阶段构建)
- [4.2 Kubernetes 部署](#42-kubernetes-部署)
  - [GPU 调度](#gpu-调度)
  - [模型服务部署](#模型服务部署)
- [4.3 CI/CD](#43-cicd)
- [4.4 GPU 共享调度](#44-gpu-共享调度)

---

## 4.1 容器化

### Dockerfile 最佳实践

```dockerfile
# 基础镜像选择
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    pip \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 使用 non-root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]
```

### 多阶段构建

```dockerfile
# 构建阶段
FROM python:3.11-slim AS builder
RUN pip install --user torch torchvision

# 运行阶段
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "main.py"]
```

### 训练和推理镜像

```dockerfile
# 训练镜像
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# 安装 PyTorch, DeepSpeed, NCCL 等
RUN pip install torch deepspeed transformers

# 推理镜像
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# 轻量级，只安装推理依赖
RUN pip install torch torchvision tensorrt
```

---

## 4.2 Kubernetes 部署

### GPU 调度

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1  # 请求1个GPU
```

### 推理服务部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-service
  template:
    metadata:
      labels:
        app: inference-service
    spec:
      containers:
      - name: inference
        image: myregistry/inference-service:v1
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          requests:
            memory: "8Gi"
            cpu: "2"
        env:
        - name: MODEL_PATH
          value: "/models/llm"
---
apiVersion: v1
kind: Service
metadata:
  name: inference-service
spec:
  selector:
    app: inference-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### GPU 共享 (TimeSlicing)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4  # 每个GPU虚拟为4个
```

### MIG (Multi-Instance GPU)

```yaml
# 需要 A100 80GB
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod
spec:
  containers:
  - name: mig-container
    resources:
      limits:
        nvidia.com/gpu: "3g.20gb"  # 3个MIG实例，每个20GB
```

---

## 4.3 CI/CD

### GitHub Actions

```yaml
name: AI Training CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  train:
    runs-on: gpu-runner
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run training
      run: |
        python train.py --config configs/default.yaml
        
    - name: Upload model
      uses: actions/upload-artifact@v4
      with:
        name: model-checkpoint
        path: outputs/checkpoint.pt
```

### MLflow 集成

```python
import mlflow

mlflow.set_experiment("llm-training")

with mlflow.start_run():
    # 记录参数
    mlflow.log_params({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
    })
    
    # 训练
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, data)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
    
    # 保存模型
    mlflow.pytorch.log_model(model, "model")
```

---

## 4.4 资源监控

### Prometheus + Grafana

```yaml
# prometheus-config.yaml
scrape_configs:
- job_name: 'inference-service'
  static_configs:
  - targets: ['inference-service:8000']
```

```yaml
# grafana-dashboard.json
{
  "panels": [
    {
      "title": "GPU Utilization",
      "type": "graph",
      "targets": [
        {
          "expr": "gpu_utilization{job='inference-service'}"
        }
      ]
    },
    {
      "title": "Inference Latency",
      "type": "graph", 
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))"
        }
      ]
    }
  ]
}
```

---

## 本章小结

- **多阶段构建** 可显著减小镜像体积
- **Kubernetes** 支持 GPU 调度和自动扩缩容
- **TimeSlicing** 可提高 GPU 利用率
- **MLflow** 便于实验追踪和模型管理

---

## 下一步

- [05-mlops](../05-mlops/README.md) - 学习 MLOps 实践
- [../examples/deployment/](../../examples/deployment/README.md) - 查看部署配置