# Agent Study

从零开始学习 AI Agent 开发的完整实践项目。不依赖 LangChain 等重框架，只用 OpenAI SDK 手写核心逻辑，逐步进阶。

通过五个阶段，从最基础的 ReAct 循环，逐步掌握工程化、RAG、反思、Plan-Execute、多 Agent 协作、流式输出、长期记忆、工具并行、Human-in-the-loop、MCP、记忆压缩、记忆淘汰、工具依赖图、A2A 协议等生产级能力。

## 项目结构

```
agent-study/
├── stage1/                   # 阶段1：ReAct 基础循环
│   └── agent.py
├── stage2/                   # 阶段2：工程化 + RAG + 反思
│   └── agent_v2.py
├── stage3/                   # 阶段3：Plan-Execute + 多Agent + 流式
│   └── agent_v3.py
├── stage4/                   # 阶段4：长期记忆 + 并行 + HITL + MCP
│   ├── agent_v4.py           #   主循环
│   ├── v4_memory.py          #   长期记忆模块
│   └── v4_mcp_server.py      #   MCP 服务器
├── stage5/                   # 阶段5：记忆压缩 + 淘汰 + DAG + A2A
│   ├── agent_v5.py           #   主循环
│   ├── v5_memory.py          #   记忆系统（压缩+淘汰）
│   ├── v5_dag.py             #   工具依赖图 DAG 调度器
│   ├── a2a_server.py         #   A2A Remote Agent（服务端）
│   └── a2a_client.py         #   A2A Client（客户端）
├── config.example.json       # 配置模板（提交）
├── config.json               # 本地配置（已 gitignore，含真实 key）
├── requirements.txt          # 依赖
├── knowledge_base/           # RAG 知识库数据（各阶段共享）
│   ├── product_faq.txt       #   产品 FAQ 示例
│   └── company_info.txt      #   公司信息示例
├── docs/                     # 学习笔记
│   ├── notes-01-agent-fundamentals.md          # 阶段1笔记
│   ├── notes-02-engineering-rag-reflection.md # 阶段2笔记
│   ├── notes-03-plan-multiagent-streaming.md  # 阶段3笔记
│   ├── notes-03-streaming-deep-dive.md        # 流式输出专题
│   ├── notes-04-memory-parallel-hitl-mcp.md   # 阶段4笔记
│   └── notes-05-memory-dag-a2a.md             # 阶段5笔记
└── .vscode/
    └── launch.json           # 调试配置
```

## 五个阶段

| 阶段 | 文件 | 核心内容 |
|------|------|---------|
| 阶段1 | `stage1/agent.py` | ReAct 循环、Function Calling、工具定义 |
| 阶段2 | `stage2/agent_v2.py` | max_steps、错误重试、RAG 知识检索、反思机制 |
| 阶段3 | `stage3/agent_v3.py` | Plan-Execute、多 Agent 协作、流式输出 |
| 阶段4 | `stage4/agent_v4.py` | 长期记忆、工具并行、Human-in-the-loop、MCP |
| 阶段5 | `stage5/agent_v5.py` | 记忆压缩、记忆淘汰、工具依赖图、A2A 协议 |

对应的学习笔记在 `docs/` 目录，图文并茂，配合代码食用。

## 快速开始

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd agent-study

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp config.example.json config.json
# 编辑 config.json，填入你的 api_key

# 4. 运行
python stage1/agent.py               # 阶段1
python stage2/agent_v2.py            # 阶段2
python stage3/agent_v3.py --mode multi   # 阶段3：多Agent协作（默认）
python stage3/agent_v3.py --mode plan    # 阶段3：Plan-Execute
python stage3/agent_v3.py --mode stream  # 阶段3：流式ReAct
python stage4/agent_v4.py            # 阶段4：记忆+并行+HITL+MCP
python stage5/agent_v5.py            # 阶段5：记忆压缩+淘汰+DAG
python stage5/v5_memory.py           # 阶段5：记忆系统独立测试
python stage5/v5_dag.py              # 阶段5：DAG依赖图独立测试
# 阶段5 A2A（需要两个终端，先服务端后客户端）
python stage5/a2a_server.py
python stage5/a2a_client.py
```

## 配置说明

`config.json` 字段：

| 字段 | 说明 | 示例 |
|------|------|------|
| `base_url` | 模型 API 地址 | `https://api.deepseek.com` |
| `api_key` | API 密钥 | `sk-xxxx` |
| `model` | 模型名 | `deepseek-v4-flash` |
| `max_steps` | 最大循环步数 | `10` |
| `enable_reflection` | 是否开启反思 | `true` |
| `max_replans` | 最大重规划次数（v3） | `3` |
| `stream_output` | 是否流式输出（v3） | `true` |
| `parallel_tools` | 是否并行工具（v4） | `true` |
| `enable_hitl` | 是否开启人工确认（v4） | `true` |
| `enable_mcp` | 是否使用 MCP 工具（v4） | `true` |

> ⚠️ `config.json` 包含真实 API Key，已被 `.gitignore` 忽略。提交前请确认不会泄露。团队协作请使用 `config.example.json` 作为模板。

## 常见模型配置

| 模型 | base_url | model |
|------|----------|-------|
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

## 调试

VSCode / CodeBuddy 打开项目，按 `F5` 调试。`.vscode/launch.json` 已配置好五个阶段对应的调试入口，在调试面板下拉选择。

## 学习路径

1. 读 `docs/notes-01-agent-fundamentals.md`，理解 ReAct 循环的本质
2. 跑 `stage1/agent.py`，断点观察 LLM 怎么决策
3. 读 `docs/notes-02-engineering-rag-reflection.md`，理解工程化与 RAG
4. 跑 `stage2/agent_v2.py`，对比 v1 的健壮性差异
5. 读 `docs/notes-03-plan-multiagent-streaming.md`，理解三种高级模式
6. 跑 `stage3/agent_v3.py`，体验 Plan-Execute / 多 Agent / 流式
7. 读 `docs/notes-04-memory-parallel-hitl-mcp.md`，理解四个进阶能力
8. 跑 `stage4/agent_v4.py`，体验长期记忆 / 并行 / HITL / MCP
9. 读 `docs/notes-05-memory-dag-a2a.md`，理解记忆维护 + 依赖图 + A2A
10. 跑 `stage5/` 各文件，体验记忆压缩 / 淘汰 / DAG / A2A
