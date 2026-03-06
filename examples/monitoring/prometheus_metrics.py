"""
AI Inference Prometheus Metrics Exporter
AI 推理服务 Prometheus 指标导出示例
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import FastAPI, Request
import time
import torch

app = FastAPI()

# ============================================
# 定义指标
# ============================================

# 推理请求计数器
inference_requests_total = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model', 'status']
)

# 推理延迟直方图
inference_latency_seconds = Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['model'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# GPU 利用率 gauge
gpu_utilization = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device']
)

# GPU 显存使用
gpu_memory_used = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['device']
)

# 模型在加载 Gauge
model_loaded = Gauge(
    'model_loaded',
    'Model loaded status (1=loaded, 0=not loaded)',
    ['model']
)

# ============================================
# 中间件：自动记录指标
# ============================================

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    
    # 获取模型名称
    model_name = request.query_params.get('model', 'default')
    
    try:
        response = await call_next(request)
        status = "success"
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        
        # 记录指标
        inference_requests_total.labels(
            model=model_name,
            status=status
        ).inc()
        
        inference_latency_seconds.labels(
            model=model_name
        ).observe(duration)
    
    return response

# ============================================
# GPU 监控端点
# ============================================

@app.get("/metrics/gpu")
async def get_gpu_metrics():
    """获取 GPU 指标"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    metrics = []
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        memory_allocated = torch.cuda.memory_allocated(i)
        memory_reserved = torch.cuda.memory_reserved(i)
        
        metrics.append({
            "device": device_name,
            "memory_allocated_mb": memory_allocated / 1024**2,
            "memory_reserved_mb": memory_reserved / 1024**2
        })
        
        # 更新 Prometheus metrics
        gpu_memory_used.labels(device=f"gpu{i}").set(memory_allocated)
    
    return {"devices": metrics}

# ============================================
# Prometheus 指标端点
# ============================================

@app.get("/metrics")
async def metrics():
    """Prometheus 拉取端点"""
    return generate_latest()

# ============================================
# 健康检查
# ============================================

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)