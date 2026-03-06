# 部署配置示例

本目录包含 Docker 和 Kubernetes 部署配置模板。

## 示例列表

| 文件 | 说明 |
|------|------|
| `Dockerfile.train` | 训练环境 Dockerfile |
| `Dockerfile.inference` | 推理服务 Dockerfile |
| `docker-compose.yml` | 多服务编排配置 |
| `k8s-training-job.yaml` | Kubernetes 训练任务 |
| `k8s-inference-deployment.yaml` | Kubernetes 推理服务部署 |
| `k8s-gpu-config.yaml` | GPU 资源配置 |
| `helm-chart/` | Helm Chart 模板 |