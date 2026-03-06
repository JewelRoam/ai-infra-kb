# MLOps 实践

> 本章介绍机器学习运维流程、实验管理和模型监控

## 目录

- [5.1 实验管理](#51-实验管理)
  - [MLflow](#mlflow)
  - [Weights & Biases](#weights--biases)
- [5.2 特征工程平台](#52-特征工程平台)
  - [Feast](#feast)
- [5.3 模型监控](#53-模型监控)
  - [Prometheus + Grafana](#prometheus--grafana)
  - [Evidently AI](#evidently-ai)
- [5.4 编排工具](#54-编排工具)
  - [Kubeflow](#kubeflow)
  - [Airflow](#airflow)

---

## 5.1 实验管理

### MLflow

开源的 ML 生命周期管理平台。

```python
import mlflow
from mlflow.tracking import MlflowClient

# 设置跟踪服务器
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("my-experiment")

# 记录实验
with mlflow.start_run(run_name="baseline"):
    # 记录参数
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "transformer"
    })
    
    # 训练
    model = train_model(...)
    
    # 记录指标
    mlflow.log_metrics({
        "accuracy": 0.95,
        "loss": 0.05,
        "f1": 0.93
    })
    
    # 记录Artifacts
    mlflow.log_artifact("model.pt")
    mlflow.log_artifact("confusion_matrix.png")
    
    # 记录模型
    mlflow.pytorch.log_model(model, "model")
```

```bash
# 启动 MLflow 服务器
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlflow \
    --host 0.0.0.0
```

### Weights & Biases

云端实验追踪工具。

```python
import wandb

# 初始化
wandb.init(
    project="my-project",
    entity="my-team",
    name="experiment-001"
)

# 记录
wandb.config.lr = 0.001
wandb.log({"loss": 0.5, "accuracy": 0.9})

# 保存模型
wandb.save("model.pt")

wandb.finish()
```

---

## 5.2 特征工程平台

### Feast

开源特征存储系统。

```python
# 定义特征
feast_entity = Entity(name="user_id", description="User ID")
feast_feature = Feature(
    name="user_total_purchases",
    dtype=Float32,
    description="Total purchases"
)

# 注册特征
from feast import FileSource
source = FileSource(
    path="s3://bucket/features/user_features.parquet",
    timestamp_field="event_timestamp"
)

# 获取特征
from feast import FeatureService
feature_store = FeatureStore(repo_path=".")
features = feature_store.get_online_features(
    features=["user_features:total_purchases"],
    entity_rows=[{"user_id": "user_123"}]
)
```

---

## 5.3 模型监控

### Prometheus + Grafana

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI

app = FastAPI()

# 定义指标
inference_requests = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['model', 'status']
)

inference_latency = Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['model']
)

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    inference_requests.labels(
        model="llm",
        status=response.status_code
    ).inc()
    
    inference_latency.labels(model="llm").observe(duration)
    
    return response
```

### Evidently AI

数据漂移和模型性能监控。

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, RegressionPreset

# 数据漂移检测
report = Report(metrics=[
    DataDriftTab(),
])

report.run(
    reference_data=reference_df,
    current_data=current_df,
    column_mapping=column_mapping
)

report.save_html("drift_report.html")
```

---

## 5.4 编排工具

### Kubeflow

Kubernetes 上的 ML 平台。

```yaml
# Kubeflow Pipeline
apiVersion: kubeflow.org/v1beta1
kind: Pipeline
metadata:
  name: ml-pipeline
spec:
  pipelineSpec:
    tasks:
    - name: preprocess
      container:
        image: preprocess:latest
        command: ["python", "preprocess.py"]
        
    - name: train
      container:
        image: train:latest
        command: ["python", "train.py"]
      dependencies: [preprocess]
      
    - name: evaluate
      container:
        image: evaluate:latest
        command: ["python", "evaluate.py"]
      dependencies: [train]
```

### Airflow

通用工作流编排。

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
)

t1 = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='train',
    python_callable=train_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id='deploy',
    python_callable=deploy_model,
    dag=dag,
)

t1 >> t2 >> t3
```

---

## 本章小结

- **MLflow** 是开源实验管理的首选
- **Feast** 提供特征共享和复用
- **Evidently AI** 便于监控数据漂移
- **Kubeflow/Airflow** 实现工作流自动化

---

## 下一步

- [06-agent-infra](../06-agent-infra/README.md) - 学习 Agent 基础设施
- [../examples/monitoring/](../../examples/monitoring/README.md) - 查看监控示例