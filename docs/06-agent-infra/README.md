# Agent 基础设施

> 本章介绍 Agent 的基础设施架构，包括本地 LLM 部署、工具调用、RAG 和多模态

## 目录

- [6.1 Agent 架构](#61-agent-架构)
  - [ReAct 模式](#react-模式)
  - [Agent 类型](#agent-类型)
- [6.2 本地 LLM 部署](#62-本地-llm-部署)
  - [Ollama](#ollama)
  - [LM Studio](#lm-studio)
- [6.3 工具调用](#63-工具调用)
  - [Function Calling](#function-calling)
  - [MCP (Model Context Protocol)](#mcp-model-context-protocol)
- [6.4 RAG 系统](#64-rag-系统)
  - [向量数据库](#向量数据库)
  - [Embedding 模型](#embedding-模型)
- [6.5 多模态感知](#65-多模态感知)
  - [视觉理解](#视觉理解)
  - [语音交互](#语音交互)

---

## 6.1 Agent 架构

### ReAct 模式

Reasoning + Acting：推理后行动，行动后反思。

```
┌─────────────────────────────────────────────────────────┐
│                    ReAct 循环                            │
├─────────────────────────────────────────────────────────┤
│  1. Thought: 需要计算 123 × 456                          │
│  2. Action: 调用计算器工具                                │
│  3. Observation: 56088                                   │
│  4. Thought: 乘法结果正确，可以继续                       │
│  5. Action: 生成最终回复                                 │
│  6. Final Answer: 123 × 456 = 56088                      │
└─────────────────────────────────────────────────────────┘
```

```python
# ReAct 实现示例
class ReActAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        
    def run(self, prompt):
        thought = prompt
        for _ in range(max_iterations):
            # 推理
            response = self.model.generate(thought)
            
            # 解析动作
            if "Action:" in response:
                action_name = parse_action(response)
                action_input = parse_input(response)
                
                # 执行动作
                observation = self.tools[action_name](action_input)
                thought = f"Observation: {observation}\n"
            else:
                return response  # 最终答案
```

### Agent 类型

| 类型 | 描述 | 例子 |
|------|------|------|
| **单一 Agent** | 单个 LLM + 工具 | ChatGPT + 插件 |
| **多 Agent** | 多个 Agent 协作 | AutoGen, CrewAI |
| **Agentic Workflow** | 预设工作流 + Agent | AI 程序员, 自动化测试 |
| **Multi-Modal Agent** | 多模态感知 + 行动 | 视觉助手, 机器人 |

---

## 6.2 本地 LLM 部署

### Ollama

开源本地 LLM 部署工具。

```bash
# 安装
brew install ollama  # macOS
curl -fsSL https://ollama.com/install.sh | sh  # Linux

# 拉取模型
ollama pull llama2
ollama pull mistral
ollama pull codellama

# 运行
ollama serve
ollama run llama2
```

```python
# Python API
import ollama

response = ollama.chat(
    model='llama2',
    messages=[
        {'role': 'user', 'content': 'Explain quantum computing'}
    ]
)
print(response['message']['content'])
```

### LM Studio

桌面端 LLM 客户端。

```bash
# 下载: https://lmstudio.ai/
# 支持: Hugging Face 模型, GGUF 格式
# 功能: 聊天, API 服务, 模型管理
```

---

## 6.3 工具调用

### Function Calling

让 LLM 调用外部函数。

```python
# OpenAI Function Calling
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称"}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [{"role": "user", "content": "北京天气怎么样?"}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# 执行函数
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "get_weather":
        args = json.loads(tool_call.function.arguments)
        weather = get_weather(args["location"])
```

### MCP (Model Context Protocol)

Anthropic 推出的 Agent 通信协议。

```python
# MCP Server 示例
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("weather-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="获取城市天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        weather = fetch_weather(arguments["city"])
        return [TextContent(type="text", text=weather)]
```

---

## 6.4 RAG 系统

### 向量数据库

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Chroma** | 轻量, Python原生 | 开发/原型 |
| **Qdrant** | 高性能, Rust | 生产环境 |
| **Milvus** | 分布式, 大规模 | 大规模向量检索 |
| **Pinecone** | 云托管 | SaaS |
| **Weaviate** | GraphQL, 多模态 | 复杂查询 |

```python
# Chroma 示例
import chroma
from chroma.config import Settings

client = chroma.Client(Settings(anonymized_telemetry=False))

# 创建集合
collection = client.create_collection("documents")

# 添加文档
collection.add(
    documents=["AI 是未来", "机器学习是 AI 的分支"],
    ids=["doc1", "doc2"],
    embeddings=embedding_model.encode(["AI 是未来", "机器学习是 AI 的分支"])
)

# 检索
results = collection.query(
    query_texts=["关于人工智能"],
    n_results=2
)
```

### Embedding 模型

| 模型 | 维度 | 特点 |
|------|------|------|
| **text-embedding-3-small** | 1536 | OpenAI, 高效 |
| **text-embedding-3-large** | 3072 | OpenAI, 高精度 |
| **bge-base-zh-v1.5** | 768 | 中文, 开源 |
| **bge-large-zh-v1.5** | 1024 | 中文, 高精度 |
| **mxbai-embed-large** | 1024 | 多语言, 高效 |

```python
# 使用 BGE Embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-zh-v1.5')

embeddings = model.encode([
    "人工智能改变世界",
    "机器学习和深度学习"
])
```

---

## 6.5 多模态感知

### 视觉理解

```python
# GPT-4V 图像理解
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)

# 本地模型: LLaVA, BakLLaVA
# pip install llava
from llava.model import LlavaForConditionalGeneration
from llava.mm_utils import process_images
```

### 语音交互

```python
# Whisper 语音识别
import openai

audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)
print(transcript.text)

# TTS 语音合成
response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input="Hello, I'm your AI assistant."
)
response.stream_to_file("output.mp3")
```

---

## 本章小结

- **ReAct** 是 Agent 推理的基础模式
- **Ollama/LM Studio** 实现本地 LLM 部署
- **Function Calling** 让 Agent 调用外部工具
- **RAG** 扩展 Agent 知识边界
- **多模态** 实现更丰富的交互

---

## 相关资源

- [Ollama](https://ollama.ai)
- [LangChain](https://langchain.dev) - Agent 开发框架
- [AutoGen](https://microsoft.github.io/autogen) - 多 Agent 框架
- [CrewAI](https://crewai.com) - 多 Agent 编排

---

*返回 [README](../README.md)*