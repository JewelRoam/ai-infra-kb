# AI 推理系统

> 本章介绍模型推理优化、推理引擎和服务化部署

## 目录

- [3.1 推理优化技术](#31-推理优化技术)
  - [模型量化](#模型量化)
  - [算子融合](#算子融合)
  - [剪枝](#剪枝)
- [3.2 推理引擎](#32-推理引擎)
  - [TensorRT](#tensorrt)
  - [ONNX Runtime](#onnx-runtime)
  - [vLLM](#vllm)
- [3.3 推理服务化](#33-推理服务化)
  - [Triton Inference Server](#triton-inference-server)
  - [FastAPI](#fastapi)
  - [批量推理](#批量推理)

---

## 3.1 推理优化技术

### 模型量化

将模型权重从 FP32/FP16 转换为低精度格式。

| 精度 | 存储 | 性能 | 精度损失 |
|------|------|------|----------|
| FP32 | 32bit | 1x | 无 |
| FP16 | 16bit | ~1x | 很小 |
| BF16 | 16bit | ~1x | 很小 |
| INT8 | 8bit | 2-4x | 1-2% |
| INT4 | 4bit | 4-8x | 3-5% |

#### INT8 量化

```python
import torch
from torch.quantization import quantize_dynamic

# 动态量化 (后训练量化)
model_int8 = quantize_dynamic(
    model, 
    {torch.nn.Linear},  # 只量化Linear层
    dtype=torch.qint8
)

# 静态量化 (需要校准数据)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.prepare(model)
# 校准
model = torch.quantization.convert(model)
```

#### GPTQ 量化

```python
from autoquant import GPTQQuantizer

quantizer = GPTQQuantizer(
    model=model,
    bits=4,
    group_size=128,
    desc=False
)
quantized_model = quantizer.quantize()
```

### 算子融合

将多个算子合并为一个，减少kernel启动开销和内存访问。

```python
# 原始: Linear + ReLU
# 融合后: Linear + ReLU (一个kernel)

# TensorRT 自动融合
# Conv + BN + ReLU → ConvBatchNormReLU
# MatMul + Add → FusedAttention
```

### 剪枝

移除不重要的权重或神经元。

```python
import torch.nn.utils.prune as prune

# 结构性剪枝 - 移除整个神经元
prune.l1_unstructured(model.linear, name='weight', amount=0.5)

# 非结构性剪枝 - 随机置零
prune.random_unstructured(model.linear, name='weight', amount=0.5)

# LLM 剪枝 (SparseGPT)
from sparsegpt import SparseGPT
sparsegpt = SparseGPT(model)
sparsegpt.prune(percentage=50)
```

---

## 3.2 推理引擎

### TensorRT

NVIDIA 高性能推理引擎，支持 GPU 加速。

```python
import tensorrt as trt

# 构建引擎
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# 创建网络
parser = trt.OnnxParser(network, logger)
parser.parse_from_file('model.onnx')

# 构建引擎
engine = builder.build_serialized_network(network, config)

# 推理
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine)
context = engine.create_execution_context()

# 分配内存
inputs, outputs, bindings = [], [], []
for i in range(engine.num_binding_io):
    dtype = trt.nptype(engine.get_binding_dtype(i))
    size = trt.volume(engine.get_binding_shape(i))
    cuda_mem = cuda.calloc(size * 4)  # FP32
    bindings.append(int(cuda_mem))
    if engine.binding_is_input(i):
        inputs.append(cuda_mem)
    else:
        outputs.append(cuda_mem)
```

#### TensorRT 优化技巧

1. **选择正确的精度**: FP16 > INT8 > FP32
2. **启用CUDA图**: 减少kernel launch开销
3. **使用TensorRT LLM**: 针对LLM优化

### ONNX Runtime

跨平台推理引擎，支持 CPU/GPU/NPU。

```python
import onnxruntime as ort

# 创建会话
sess = ort.InferenceSession(
    'model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 推理
outputs = sess.run(None, {'input': input_data})

# 优化选项
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_profiling = True
```

#### ONNX 优化

```python
from onnxruntime.transformers import optimizer

# 自动优化
optimized_model = optimizer.optimize_model(
    'model.onnx',
    num_heads=12,
    hidden_size=768,
    optimization_options={'attention': True}
)
optimized_model.save_model_to_file('model_optimized.onnx')
```

### vLLM

针对 LLM 推理的高性能引擎，支持 Continuous Batching。

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # 多卡
    dtype="half",            # FP16
    max_model_len=4096,
)

# 推理
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=256,
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

#### vLLM 特性

- **Continuous Batching**: 动态批处理，大幅提升吞吐量
- **PagedAttention**: 内存优化，支持更长上下文
- **AWQ/GPTQ 量化**: 支持 INT4 量化

---

## 3.3 推理服务化

### Triton Inference Server

NVIDIA 开源推理服务框架。

```yaml
# config.pbtxt
name: "llm_model"
platform: "tensorrtllm_backend"
max_batch_size: 8

input [
  {
    name: "prompt"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_STRING
    dims: [1]
  }
]

parameters: {
  key: "max_tokens"
    value: {string_value: "256"}
}
```

```bash
# 启动服务
tritonserver --model-repository=/models --http-port=8000
```

### FastAPI

轻量级 Python Web 框架。

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

@app.post("/predict")
async def predict(request: InferenceRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=request.max_tokens)
    return {"result": tokenizer.decode(outputs[0])}
```

```bash
# 运行
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 批量推理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 同步批量
def batch_inference(prompts, model, batch_size=32):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        results.extend(model.predict(batch))
    return results

# 异步批量
async def async_batch_inference(prompts, model, batch_size=32):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, model.predict, batch)
            for batch in chunks(prompts, batch_size)
        ]
        results = await asyncio.gather(*futures)
    return results
```

---

## 本章小结

- **量化** 可显著提升推理速度 (2-4x)
- **TensorRT** 是 NVIDIA GPU 推理的首选
- **vLLM** 是 LLM 推理的高性能方案
- **Continuous Batching** 可大幅提升服务吞吐量

---

## 下一步

- [04-deployment](../04-deployment/README.md) - 学习部署运维
- [../examples/inference-optimization/](../../examples/inference-optimization/README.md) - 查看代码示例