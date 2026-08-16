# Agent 开发第一阶段学习笔记

> 学习目标：从零理解 Agent 的本质，能跑通一个最小可运行的 Agent，并看懂它的每一行在干什么。

---

## 一、核心认知：Agent 到底是什么

### 1.1 最本质的一句话

> **Agent = 一个 while 循环 + LLM 做决策 + 代码执行工具**

Agent 的"智能"来自 LLM 的实时规划，代码只负责三件事：**准备上下文、调用模型、执行模型决定要做的事**。代码里没有 if-else 业务逻辑判断，所有"下一步该干嘛"都是 LLM 现想出来的。

### 1.2 代码和 LLM 的分工

| 代码负责的（基础设施）| LLM 负责的（思考决策）|
|---------------------|---------------------|
| 维护对话历史 / 上下文 | 分析当前状态 |
| 定义工具的接口和描述 | 决定调用哪个工具 |
| 执行工具（调 API、查数据库等）| 决定传什么参数 |
| 循环控制（什么时候停）| 判断任务是否完成 |
| Prompt 模板组装 | 生成最终回答 |

### 1.3 关键澄清

很多人以为 Agent 的规划步骤是"写代码用算法分析出来的"——**不是**。规划步骤是 LLM 在每一轮循环中实时生成的，代码只是把 LLM 的决策执行下去。

---

## 二、Agent 的核心运行模式：ReAct 循环

ReAct = **Reasoning（推理）+ Acting（行动）**，是当前 90% Agent 的底层骨架。

### 2.1 循环逻辑

```
循环开始:
  1. 把 [用户任务 + 历史步骤 + 可用工具描述] 拼成 prompt
  2. 发给 LLM
  3. LLM 返回: "调用工具 X"（行动）+ 可能的思考
  4. 代码执行工具 X，拿到结果
  5. 把结果塞回历史上下文
  6. 判断: LLM 是否还要调工具？
     - 要 → 回到第 1 步
     - 不要 → LLM 给最终回答，循环结束
```

### 2.2 最小伪代码

```python
def run_agent(task):
    messages = [
        {"role": "system", "content": "你是一个助手，可以调用工具..."},
        {"role": "user", "content": task}
    ]

    while True:
        # ⭐ 这一步才是真正的"规划+思考"，LLM 在这里决策
        response = llm.chat(messages, tools=TOOL_DEFINITIONS)

        if response.has_tool_call():
            # 代码负责执行工具
            result = execute_tool(response.tool_call)
            messages.append(response)
            messages.append({"role": "tool", "content": result})
        else:
            # LLM 认为任务完成了，返回最终回答
            return response.text
```

**整段代码里，`llm.chat()` 这一行就是 Agent 的"大脑"**，往上是准备上下文，往下是执行决策。

---

## 三、两种主流规划策略

### 3.1 ReAct（边想边做）—— 最常见

LLM 每一步只规划"下一步"，做完看结果再决定下下步。OpenAI Function Calling、Claude Tool Use、各种框架默认都是这个路子。

```
LLM: 用户要查天气 → call get_weather("北京")
代码: 执行，返回 "28度，晴"
LLM: 拿到天气了，用户还问要不要带伞 → 输出回答
```

**优点**：灵活，能根据中间结果调整方向。
**缺点**：每步只看眼前，缺乏全局视野。

### 3.2 Plan-and-Execute（先规划再执行）

先让 LLM 一次性把所有步骤列出来，再逐步执行：

```
第一轮 LLM 调用（规划阶段）:
  输入: "请把这个任务拆成步骤"
  输出: ["步骤1: 搜索...", "步骤2: 读取...", "步骤3: 总结..."]

后续 LLM 调用（执行阶段）:
  逐步执行，每步可能还会根据结果重新调整计划
```

**优点**：规划更系统，适合复杂任务。
**缺点**：计划容易赶不上变化，需要中途重规划能力。

---

## 四、工具定义：LLM 怎么知道有哪些工具

工具以 JSON Schema 格式描述给 LLM，核心三要素：**名字、描述、参数定义**。

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",                    # 工具名
            "description": "查询某个城市的天气",        # ⭐ 最关键，LLM 靠它判断要不要调
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，如 '北京'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**关键认知**：`description` 字段决定了 LLM 会不会调这个工具。描述写得模糊，LLM 就不知道该用；描述写得好，LLM 就能准确判断什么时候调。

---

## 五、Agent 何时结束：finish_reason 的秘密

### 5.1 两种 finish_reason

```
finish_reason: tool_calls   ← LLM 还要调工具，循环继续
finish_reason: stop          ← LLM 觉得搞定了，循环结束
```

### 5.2 结束的判断逻辑

```python
if msg.tool_calls:
    # 情况A：LLM 想调工具 → 执行工具，continue 继续循环
    ...
    continue

# 情况B：LLM 没返回 tool_calls → 任务完成，返回最终回答
return msg.content
```

**Agent 什么时候停，完全由 LLM 决定**，代码只是"看见 LLM 没调工具就退出"。

### 5.3 两个常见坑

1. **该停不停**：LLM 陷入死循环，反复调同一工具 → 需要加 `max_steps` 上限
2. **不该停却停了**：LLM 漏了步骤就提前结束 → 靠改 system prompt 缓解（如"必须完成所有要求后再结束"）

```python
# 生产级 Agent 必须加的保护
while step < MAX_STEPS:  # 防死循环
    try:
        response = client.chat.completions.create(...)
    except Exception as e:
        # 重试 / 降级 / 兜底
        ...
```

---

## 六、上下文累积：Agent 的"记忆"本质

每轮循环，`messages` 列表都在变长。这就是 Agent 的短期记忆——**把所有历史塞进下一轮的 prompt**。

```
第1轮发给 LLM 的 messages: [system, user]                                          (2条)
第2轮发给 LLM 的 messages: [system, user, assistant(调工具), tool(结果)]            (4条)
第3轮发给 LLM 的 messages: [system, user, assistant, tool, assistant, tool]         (6条)
...
```

**LLM 每次都看到完整历史**，这就是为什么：
- Agent 能"记住"前面做过什么
- 对话越长，token 消耗越大（成本问题）
- 需要长上下文窗口的模型

---

## 七、最小可运行 Agent 的完整代码

完整代码见仓库 `agent.py`，核心结构如下：

```python
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ===== 读配置 =====
with open("config.json") as f:
    config = json.load(f)
client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])

# ===== 工具实现（代码干活）=====
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate(expression: str) -> str:
    result = eval(expression, {"__builtins__": {}}, {})
    return f"{expression} = {result}"

def get_weather(city: str) -> str:
    fake = {"北京": "28°C晴", "上海": "26°C多云"}
    return fake.get(city, "无数据")

# ===== 工具描述（给 LLM 看）=====
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"]
            }
        }
    },
    # ... 其他工具
]

TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}

# ===== Agent 主循环 =====
def run_agent(task: str, model: str = config["model"]):
    messages = [
        {"role": "system", "content": "你是一个能调用工具的助手..."},
        {"role": "user", "content": task},
    ]

    step = 0
    while True:
        step += 1
        # ⭐⭐⭐ LLM 在这里规划下一步 ⭐⭐⭐
        response = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # 情况A：LLM 要调工具 → 代码执行
            messages.append(msg)
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = TOOL_MAP[name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            continue

        # 情况B：LLM 不调工具了 → 任务完成
        return msg.content

if __name__ == "__main__":
    run_agent("现在几点？算一下 (123+456)*2，查北京天气，最后总结")
```

---

## 八、调试技巧

### 8.1 配置文件 `config.json`

把 URL、token、模型名外置，不写死在代码里：

```json
{
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "sk-xxxx",
    "model": "deepseek-chat"
}
```

常见模型配置：

| 模型 | base_url | model |
|------|----------|-------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

### 8.2 VSCode / CodeBuddy 断点调试

`.vscode/launch.json`：

```json
{
    "version": "0.2.0",
    "configurations": [{
        "name": "调试 Agent",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/agent.py",
        "console": "integratedTerminal"
    }]
}
```

**重点断点位置**：
- `response = client.chat.completions.create(...)` 返回后 —— 看 LLM 这一步想了什么
- `result = TOOL_MAP[name](**args)` 执行前 —— 看 LLM 决定调什么工具、传什么参数

**重点观察变量**：
- `messages` —— 对话历史列表，每轮变长，能看到上下文累积
- `msg.tool_calls` —— LLM 这一步决定调哪些工具
- `response.choices[0].finish_reason` —— 判断是继续还是结束

### 8.3 关键日志打印

每轮打印两块信息有助于理解：

```python
# 打印发给 LLM 的完整上下文
print(f"📤 发给 LLM 的 messages: 共 {len(messages)} 条")
for i, m in enumerate(messages):
    print(f"  [{i}] {m['role']}: ...")

# 打印 LLM 的原始返回
print(f"📥 LLM 返回:")
print(f"  finish_reason: {choice.finish_reason}")
print(f"  tool_calls: {msg.tool_calls}")
print(f"  content: {msg.content}")
```

**盯住 `finish_reason`**：从 `tool_calls` 变成 `stop` 的那一刻，就是 LLM 决定"收工"的瞬间。

---

## 九、这个例子的局限性（重要！）

这个最小例子只是 Agent 的**骨架**，不能代表生产级 Agent。在骨架之上，真实 Agent 还多了这些东西：

### 9.1 规划方式更多样

| 能力 | 本例子 | 生产 Agent |
|------|--------|-----------|
| ReAct（边想边做）| ✅ | ✅ |
| Plan-and-Execute（先规划再执行）| ❌ | ✅ |
| Tree of Thought（多思路分支）| ❌ | ✅ |
| Reflection（自我反思纠错）| ❌ | ✅ |

### 9.2 记忆系统更完整

- **短期记忆**：本例的 `messages`（跑完就没了）
- **长期记忆**：存向量数据库，跨会话回忆（如 MemGPT）
- **工作记忆**：临时 scratchpad，记中间结果但不塞满历史（省 token）

### 9.3 工具系统更复杂

- **动态工具**：从上百个工具池里按需选
- **MCP 协议**：工具即服务，可远程挂载
- **工具检索**：工具太多时先用 RAG 检索相关工具

### 9.4 多 Agent 协作

单 Agent → 多 Agent 演进：

```
用户任务
  ├─ Planner Agent（拆任务）
  ├─ Researcher Agent（查资料）
  ├─ Coder Agent（写代码）
  └─ Reviewer Agent（审查产出）
```

代表作：AutoGen、CrewAI、MetaGPT。

### 9.5 RAG 检索增强

```
LLM 回答前 → 先去知识库检索相关文档 → 塞进 prompt → 再回答
```

这是让 Agent "知道"私有数据的关键。

### 9.6 工程化必需品

| 能力 | 本例子 | 生产 Agent |
|------|--------|-----------|
| 错误处理 / 重试 | ❌ | ✅ |
| max_steps 防死循环 | ❌ | ✅ |
| 流式输出 | ❌ | ✅ |
| 工具并行调用 | ❌ | ✅ |
| 成本 / token 控制 | ❌ | ✅ |
| 全链路 trace | ❌ | ✅（LangSmith / Langfuse）|
| Human-in-the-loop | ❌ | ✅ |
| 状态持久化 / 恢复 | ❌ | ✅ |

### 9.7 但骨架的意义不可替代

**上面所有复杂东西，剥开外壳，里面都是这个 while 循环：**

- 多 Agent？= 多个 while 循环互相发消息
- Plan-Execute？= 先跑一次 LLM 生成计划，再跑 while 循环执行
- 反思？= while 循环里多加一轮"自我批评"的 LLM 调用
- RAG？= `client.chat.completions.create` 之前多插一步检索

**理解了这个循环，就有了看懂所有 Agent 框架源码的钥匙。**

---

## 十、下一阶段学习路径

在当前骨架上按顺序加东西，每一步都学到真东西：

1. **加 `max_steps` 和错误处理** → 感受工程化
2. **加一个 RAG 工具**（读本地文档回答）→ 感受知识增强
3. **加反思机制**（每步后让 LLM 自评"这步对吗"）→ 感受自我纠错
4. **改成 Plan-Execute**（先出完整计划再执行）→ 感受不同规划策略
5. **拆成两个 Agent**（一个规划一个执行，互相传消息）→ 感受多 Agent 协作

---

## 附录：关键术语速查

| 术语 | 含义 |
|------|------|
| Agent | 能自主规划+调用工具完成任务的 AI 程序 |
| ReAct | Reasoning + Acting，边推理边行动的循环模式 |
| Function Calling / Tool Use | LLM 返回"要调用哪个工具"的结构化能力 |
| Tool / Function | Agent 能调用的具体能力（查天气、算数等）|
| finish_reason | LLM 返回的结束标志：`tool_calls`（继续）或 `stop`（结束）|
| messages | 对话历史列表，Agent 的短期记忆 |
| Plan-and-Execute | 先一次性规划全部步骤，再逐步执行 |
| RAG | 检索增强生成，从知识库找资料塞进 prompt |
| MCP | Model Context Protocol，工具即服务的标准协议 |
| Multi-Agent | 多个分工不同的 Agent 协作完成任务 |
