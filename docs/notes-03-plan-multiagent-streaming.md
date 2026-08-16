# Agent 开发第三阶段学习笔记：Plan-Execute + 多Agent协作 + 流式输出

> 学习目标：掌握 2026 年工业界三大主流 Agent 高级模式——先规划再执行、多 Agent 协同、流式交互。
>
> 对应代码：`agent_v3.py` ｜ 运行方式：`python agent_v3.py --mode [plan|multi|stream]`

---

## 一、全景总览：三个模式的定位

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent 模式演进路线图                               │
│                                                                     │
│  阶段1: ReAct (agent.py)                                             │
│    └─ 边想边做，一步一LLM                                            │
│                                                                     │
│  阶段2: ReAct + 工程化 (agent_v2.py)                                 │
│    └─ 加 RAG / 反思 / 重试 / max_steps                              │
│                                                                     │
│  阶段3: 高级模式 (agent_v3.py)                                       │
│    ├─ Plan-Execute     → 先规划后执行，快3.6倍，省Token              │
│    ├─ Multi-Agent      → 多Agent分工协作，各司其职                   │
│    └─ Streaming        → 边生成边输出，告别干等                      │
│                                                                     │
│  未来: 长期记忆 / 工具并行 / Human-in-the-loop ...                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 三种模式的关系

| 维度 | ReAct (v1/v2) | Plan-Execute | Multi-Agent | Streaming |
|------|--------------|--------------|-------------|-----------|
| 解决问题 | 基础 Agent 能力 | 复杂多步任务的效率 | 单一 Agent 能力瓶颈 | 用户体验 |
| 核心思想 | 一步一想 | 先想全部再执行 | 拆任务分给不同 Agent | 边生成边推送 |
| 与 ReAct 关系 | 本身 | 替代品 | 扩展品 | 增强品 |
| 可组合？ | 可以 | Plan内每步可走ReAct | Worker内可走ReAct | 任何模式都能套 |

**关键认知**：三种模式不是互斥的，可以组合使用。比如：Multi-Agent 模式下，每个 Worker 内部走 ReAct，最终回答用 Streaming 输出。

---

## 二、Plan-Execute 模式

### 2.1 什么是 Plan-Execute

```
ReAct (边走边看):                    Plan-Execute (先想清楚):
                                    
  用户问 → LLM想 → 调工具           用户问 → Planner → 全局计划
            ↑       ↓                        ↓
            └── 看结果 ←┘              [步骤1, 步骤2, 步骤3, ...]
                                              ↓
  ❌ 每步都调LLM                     Executor → 逐步执行（不用每步调LLM）
  ❌ 没有全局视角                           ↓
  ❌ Token消耗大                     Replanner → 检查是否完成
                                              ↓
                                     ✅ 一次规划，多次执行
                                     ✅ 全局视角
                                     ✅ Token消耗低
```

### 2.2 三阶段架构图

```
                    ┌─────────────────────────┐
                    │       用户任务           │
                    └───────────┬─────────────┘
                                │
            ╔═══════════════════▼═══════════════════╗
            ║        阶段1：规划 (Planner)           ║
            ║                                       ║
            ║  调用一次 LLM → 输出步骤列表           ║
            ║  [{step:1, tool:weather, ...},        ║
            ║   {step:2, tool:calculate, ...},      ║
            ║   {step:3, tool:null, desc:"总结"}]   ║
            ║                                       ║
            ║  Token消耗：1次 LLM调用               ║
            ╚═══════════════════╦═══════════════════╝
                                │
            ╔═══════════════════▼═══════════════════╗
            ║        阶段2：执行 (Executor)          ║
            ║                                       ║
            ║  遍历计划，逐步骤执行：                 ║
            ║    - 有 tool → 代码调工具（不调LLM）   ║
            ║    - 无 tool → 调LLM生成文字           ║
            ║                                       ║
            ║  Token消耗：0（有工具）或 1次（无工具）║
            ╚═══════════════════╦═══════════════════╝
                                │
            ╔═══════════════════▼═══════════════════╗
            ║      阶段3：重规划 (Replanner)         ║
            ║                                       ║
            ║  检查结果 → 完成了？                    ║
            ║    ├─ COMPLETE → 结束                  ║
            ║    └─ REPLAN   → 生成补充计划          ║
            ║                  → 回到阶段2            ║
            ║                                       ║
            ║  Token消耗：1次 LLM调用               ║
            ╚═══════════════════════════════════════╝
```

### 2.3 与 ReAct 的成本对比

```
场景：用户要求 "查北京天气 + 算3道数学题 + 搜2条知识库信息 + 总结"

ReAct 模式：
  每步调LLM = 至少 6次 LLM 调用
  每次调用都带完整历史 → Token 指数增长
  
Plan-Execute 模式：
  规划: 1次 LLM调用
  执行: 5次工具调用（不调LLM）+ 1次总结（调LLM）
  检查: 1次 LLM调用
  总计: 3次 LLM调用（节省50%）
```

**2026年实测数据（来源：学术论文）：完成率 92%，比 ReAct 快 3.6 倍。**

### 2.4 核心代码：Planner（规划器）

```python
class Planner:
    """规划器：把用户任务拆成可执行的步骤列表"""

    @staticmethod
    def generate_plan(task: str, model: str) -> list[dict]:
        # 把可用工具列给 LLM，让它知道能做什么
        tools_desc = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in TOOLS
        )

        plan_prompt = f"""你是任务规划专家。请将以下任务拆解为执行步骤。

【可用工具】
{tools_desc}

【任务】
{task}

【要求】
1. 每个步骤包含：step编号、action描述、tool工具名、params参数
2. 如果某步骤不需要工具（如"总结"），tool填null
3. 步骤顺序合理，无遗漏
4. 输出纯 JSON 数组：
[
  {{"step": 1, "action": "查北京天气", "tool": "get_weather", "params": {{"city": "北京"}}}},
  {{"step": 2, "action": "总结以上信息", "tool": null, "params": {{}}}}
]"""

        # 规划用非流式调用（规划是内部决策，不需要用户看到过程）
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": plan_prompt}],
        )
        return json.loads(response.choices[0].message.content)
```

### 2.5 核心代码：Executor（执行器）

```python
class Executor:
    """执行器：按计划逐步执行，区分"调工具"和"调LLM""""

    @staticmethod
    def execute_plan(plan: list[dict], task: str, model: str) -> list[dict]:
        results = []
        for step in plan:
            tool_name = step.get("tool")
            params = step.get("params", {})

            if tool_name and tool_name in TOOL_MAP:
                # ★ 有工具：代码直接调工具，不调 LLM（省 Token）
                result = execute_tool_safely(tool_name, params)
            else:
                # ★ 无工具（如"总结"）：调 LLM 用上下文生成回答
                result = llm_call(step["action"], context=results)
            
            results.append({"step": step["step"], "result": result})
        return results
```

**关键设计**：Executor 区分两种步骤——有工具的直接代码执行（0 Token），无工具的调 LLM（1 Token），这是 Plan-Execute 省成本的核心。

### 2.6 核心代码：Replanner（重规划器）

```python
class Replanner:
    """重规划器：检查执行结果，决定是否重新规划"""

    @staticmethod
    def check_and_replan(task, plan, results, model) -> tuple[bool, Optional[list]]:
        # 让 LLM 评估：所有子任务都完成了吗？结果可信吗？
        check_prompt = f"""你是质量审查员。检查以下任务执行情况：
【原始任务】{task}
【原计划】{json.dumps(plan)}
【执行结果】{json.dumps(results)}

如果完全完成，回复: COMPLETE
如果有遗漏，回复: REPLAN，并给出补充步骤（JSON数组）"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": check_prompt}],
        )
        verdict = response.choices[0].message.content

        if "COMPLETE" in verdict.upper():
            return True, None      # 完成，不需要重规划
        else:
            new_plan = extract_json(verdict)
            return False, new_plan  # 没完成，返回补充计划
```

**为什么需要 Replanner？** Plan-Execute 最致命的弱点就是"计划赶不上变化"。如果第一步的结果完全出乎意料（比如天气API挂了），后面的计划就全废了。Replanner 在执行完一轮后检查，发现问题可以补充新步骤。

### 2.7 Plan-Execute 适用场景

| 适合 | 不适合 |
|------|--------|
| 步骤多且结构清晰的任务 | 高度探索性的任务（目标模糊） |
| 工具调用密集的任务 | 只有1-2步的简单任务 |
| 对成本敏感的场景 | 每步结果会极大改变后续路径 |
| 代码生成、数据分析、API编排 | 开放式聊天、头脑风暴 |

---

## 三、多 Agent 协作

### 3.1 为什么需要多 Agent

单一 Agent 有三个致命瓶颈：

```
问题1: System Prompt 过长
  一个 Agent 既要会规划、又要会执行、还要会总结
  → prompt 塞满指令 → LLM 容易搞混 → 输出质量下降

问题2: 工具集冲突
  工具太多时 LLM 选择困难
  比如 50 个工具全注册给一个 Agent → 容易选错

问题3: "人格"冲突
  规划需要"全局思维"，执行需要"专注细节"，审查需要"批判性"
  一个 prompt 不可能同时扮演三种角色
```

**多 Agent 的方案**：拆成多个 Agent，每个有独立的 prompt 和工具集，通过 Manager 协调。

### 3.2 本实现架构：Orchestrator-Worker（星型拓扑）

```
                          ┌──────────────┐
                          │  用户任务     │
                          └──────┬───────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Manager Agent       │
                    │     (总调度器)           │
                    │                         │
                    │  职责：                  │
                    │  1. 委托Planner拆任务    │
                    │  2. 分配Worker执行       │
                    │  3. 委托Reviewer审查     │
                    │  4. 流式输出最终结果      │
                    └──┬────────┬─────────┬───┘
                       │        │         │
          ┌────────────▼─┐ ┌───▼──────┐ ┌▼──────────┐
          │   Planner    │ │ToolWorker│ │Reviewer   │
          │   (规划)      │ │(执行工具) │ │(审查汇总)  │
          │              │ │          │ │           │
          │ 职责: 拆任务  │ │ 职责: 调 │ │ 职责: 检查│
          │ 工具: 无      │ │ get_weather│ │ 汇总输出 │
          │              │ │ calculate │ │           │
          └──────────────┘ │ search_kb │ └───────────┘
                           └──────────┘
```

这是 2026 年工业界最主流的多 Agent 架构（~70% 企业采用）。核心优势：控制性强、故障隔离好、扩展方便。

### 3.3 Agent 基类设计

```python
class Agent:
    """
    每个 Agent 有独立的：
      - name: 名称（用于日志和调度）
      - system_prompt: 角色定位（决定"人格"和职责）
      - tools: 可用工具列表（None 表示纯语言Agent）

    Agent 之间不直接通信，通过 Manager 中转。
    这是 Orchestrator-Worker 的标准做法。
    """

    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []

    def run(self, task: str, model: str, context: str = "") -> str:
        """处理一个子任务（内部走 ReAct 循环）"""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"【上下文】{context}"})
        messages.append({"role": "user", "content": task})

        # ReAct 循环（单Worker任务简单，不需要Plan-Execute）
        while step < 5:
            response = call_llm_with_retry(messages, self.tools, model)
            msg = response.choices[0].message
            if msg.tool_calls:
                # 执行工具...
                continue
            return msg.content
```

### 3.4 三个 Agent 的角色对比

| 维度 | Planner | ToolWorker | Reviewer |
|------|---------|------------|----------|
| system_prompt | "你是任务规划专家，把任务拆成子任务" | "你是工具执行专家，调工具拿数据" | "你是质量审查员，检查+汇总" |
| 有工具？ | ❌ 无（纯思考） | ✅ 有（get_weather等4个） | ❌ 无（纯语言） |
| 输入 | 用户原始任务 | 子任务描述 + 上下文 | 全部执行结果 |
| 输出 | JSON 子任务列表 | 工具执行结果 | 审查意见/最终回答 |
| 调用次数 | 1次/任务 | 每个tool子任务1次 | 1次/任务 |

### 3.5 ManagerAgent 完整流程

```python
class ManagerAgent:
    """Manager（总调度器）：不做具体工作，只做协调"""

    def __init__(self):
        # 注册三个子Agent，各自独立prompt和工具
        self.planner = Agent("Planner", "你是规划专家...", tools=None)
        self.tool_worker = Agent("ToolWorker", "你是执行专家...", tools=TOOLS)
        self.reviewer = Agent("Reviewer", "你是审查专家...", tools=None)

    def run(self, task: str) -> str:
        # ── 阶段1：Planner 拆任务 ──
        plan_text = self.planner.run(task)
        subtasks = json.loads(plan_text)
        # → [{"type":"tool","desc":"查天气"}, {"type":"text","desc":"总结"}]

        # ── 阶段2：分配 Worker 执行 ──
        worker_results = []
        for subtask in subtasks:
            # 根据类型选 Worker
            worker = self.tool_worker if subtask["type"] == "tool" else self.text_worker
            # 传递上下文（已完成的结果）
            result = worker.run(subtask["desc"], context=worker_results)
            worker_results.append(result)

        # ── 阶段3：Reviewer 审查汇总 ──
        review_input = f"原始任务: {task}\n执行结果: {worker_results}"
        final_answer = self.reviewer.run(review_input)

        return final_answer
```

### 3.6 多 Agent 协作的工程实践

**原则1：先简单后复杂（工业共识）**

> 能用单Agent+好prompt解决的，不要上多Agent。
> 复杂度是负债，不是资产。

**原则2：成本是一等架构约束**

Anthropic 实测 multi-agent token 消耗是单 Agent 的 ~15x。所以：
- 贵模型做编排（Planner/Reviewer 用强模型）
- 便宜模型做执行（Worker 用小模型）

**原则3：Agent 间不要直接通信**

通过 Manager 中转，好处：
- 故障隔离：一个 Worker 挂了不影响其他
- 可观测：Manager 能看到所有通信
- 可控制：Manager 可以随时调整策略

**原则4：可观测性先于复杂性**

在增加 Agent 数量前，先确保能看到每个 Agent 的行为、成本和失败原因。

### 3.7 多 Agent 架构对比

| 架构模式 | 结构 | 适用场景 | 本实现 |
|---------|------|---------|--------|
| Orchestrator-Worker | 星型 | 任务可拆解、步骤独立 | ✅ |
| Chain/Pipeline | 链式 | 固定流程（A→B→C） | ❌ |
| Peer-to-Peer | Mesh | 辩论、头脑风暴 | ❌ |
| Hierarchical | 树型 | 超复杂多级任务 | ❌ |
| Dynamic/Adaptive | 动态 | 任务类型多变 | ❌ |

---

## 四、流式输出（Streaming）

### 4.1 为什么需要流式输出

```
非流式（普通调用）:
  用户："查北京天气，算 888*333，查产品定价"
  → [等待 8 秒... 屏幕一片空白...]
  → 一次性出现全部回答
  → 体验：像在等一个慢网站加载

流式输出（SSE）:
  用户："查北京天气，算 888*333，查产品定价"
  → 北 → 京 → 天 → 气 → 2 → 8 → ° → C → ...
  → 像打字一样逐字出现
  → 体验：AI 在"思考"，有参与感
```

### 4.2 SSE 协议原理

```
客户端                            OpenAI 服务端
  │                                    │
  │  POST /chat/completions            │
  │  {"stream": true, ...}             │
  │ ─────────────────────────────────→ │
  │                                    │
  │  data: {"choices":[{"delta":       │
  │    {"content":"北"}}]}             │
  │ ←────────────────────────────────  │
  │  data: {"choices":[{"delta":       │
  │    {"content":"京"}}]}             │
  │ ←────────────────────────────────  │
  │  data: {"choices":[{"delta":       │
  │    {"content":"天"}}]}             │
  │ ←────────────────────────────────  │
  │  ...                               │
  │  data: [DONE]                      │
  │ ←────────────────────────────────  │
```

### 4.3 流式 + Function Calling 的复杂性

这是流式输出最棘手的问题：

```
问题：LLM 返回 tool_calls 时，参数是分片来的

chunk1: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather"}}]}}]}
chunk2: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}
chunk3: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"北京\"}"}}]}}]}

必须自己把 arguments 拼起来：{"city":"北京"}
```

### 4.4 核心代码：流式调用引擎

```python
class StreamingLLM:
    """流式LLM调用引擎"""

    @staticmethod
    def stream_text(messages, model, on_token=None):
        """流式输出纯文本（规划、审查、总结等场景）"""
        full_text = ""
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True  # ← 核心开关
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                print(token, end="", flush=True)  # 实时打印，不换行
        return full_text

    @staticmethod
    def stream_with_tools(messages, tools, model):
        """
        流式调用 + Function Calling 支持。

        处理分散的 tool_calls delta：
          - content delta → 流式打印
          - tool_calls delta → 按 index 拼接 name 和 arguments
          - finish_reason == "tool_calls" → 返回工具调用列表
          - finish_reason == "stop" → 返回完整文本
        """
        tool_calls_acc = {}  # {index: {id, function: {name, arguments}}}
        full_text = ""
        finish_reason = None

        stream = client.chat.completions.create(
            model=model, messages=messages, tools=tools, stream=True
        )

        for chunk in stream:
            delta = chunk.choices[0].delta

            # 处理纯文本
            if delta.content:
                full_text += delta.content
                print(delta.content, end="", flush=True)

            # ★ 处理 tool_calls：按 index 拼接分散的 delta
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "", "function": {"name": "", "arguments": ""}
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        # ★ arguments 分片拼接
                        tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments

            finish_reason = chunk.choices[0].finish_reason or finish_reason

        # 判断返回类型
        if finish_reason == "tool_calls":
            return {"type": "tool_calls", "calls": list(tool_calls_acc.values())}
        return {"type": "text", "content": full_text}
```

### 4.5 三种流式场景的适配

| 场景 | 使用的方法 | 为什么 |
|------|-----------|--------|
| Planner 输出计划 | `stream_text` | 纯文本，不需要 tools |
| Worker 执行工具 | `stream_with_tools` | 需要 Function Calling |
| Reviewer 最终回答 | `stream_text` | 纯文本总结 |
| ReAct 循环 | `stream_with_tools` | 边想边调工具 |

---

## 五、运行方式

### 5.1 三种模式

```bash
# 模式1：Plan-Execute（先规划再执行）
python agent_v3.py --mode plan

# 模式2：多Agent协作（默认）
python agent_v3.py --mode multi

# 模式3：流式ReAct（边想边做+流式输出）
python agent_v3.py --mode stream

# 自定义任务
python agent_v3.py --mode multi --task "查深圳天气，算 999*888，总结"
```

### 5.2 VSCode 调试

`.vscode/launch.json` 已配好 5 个配置：

| 配置名 | 对应模式 |
|--------|---------|
| v3-多Agent协作 (默认) | `--mode multi` |
| v3-Plan-Execute模式 | `--mode plan` |
| v3-流式ReAct模式 | `--mode stream` |

在调试面板下拉选择，按 F5 启动。

### 5.3 断点建议

| 位置 | 看什么 |
|------|--------|
| `Planner.generate_plan()` 返回处 | 计划 JSON 长什么样 |
| `Executor.execute_plan()` 循环内 | 每步是调工具还是调 LLM |
| `Replanner.check_and_replan()` 返回处 | 什么时候触发重规划 |
| `ManagerAgent.run()` 各阶段 | Planner→Worker→Reviewer 怎么传递 |
| `StreamingLLM.stream_with_tools()` 的 `for chunk` | SSE 事件流长什么样 |
| `tool_calls_acc` 变量 | 分散的 delta 怎么拼接 |

---

## 六、核心认知总结

### 6.1 三种模式的本质

```
Plan-Execute = 给 Agent 加"规划大脑"
  → 不是改变"怎么执行"，而是改变"什么时候想"
  → 从"每步都想"变成"先想完再干"

Multi-Agent = 把 Agent 拆成"专业团队"
  → 不是让一个Agent更强，而是让多个Agent各司其职
  → 从"全科医生"变成"专科门诊"

Streaming = 给 Agent 加"实时表达能力"
  → 不是改变 Agent 的逻辑，而是改变输出的方式
  → 从"憋完了再说"变成"边想边说"
```

### 6.2 模式选择的决策树

```
你的任务是什么？
│
├─ 1-2步简单任务？
│   └─ 用 ReAct（v1/v2），别搞复杂
│
├─ 3+步结构清晰的任务？
│   └─ 用 Plan-Execute（v3 --mode plan）
│
├─ 需要不同"人格"或不同工具集？
│   └─ 用 Multi-Agent（v3 --mode multi）
│
├─ 任何模式 + 用户需要实时反馈？
│   └─ 加 Streaming（v3 默认开启）
│
└─ 超复杂任务？
    └─ Plan-Execute + Multi-Agent + Streaming 组合使用
```

### 6.3 与 v2 的关系

v2 的所有东西（工程化、RAG、反思）v3 全部复用：
- `call_llm_with_retry` → v3 的 `StreamingLLM` 和 `Agent.run` 里都用
- `execute_tool_safely` → v3 的 `Executor` 和 `Agent.run` 里都用
- `reflect_on_step` → v3 可以加，但为了聚焦核心模式暂时没加
- `KnowledgeBase` → v3 的 `search_knowledge` 工具复用

**v3 是在 v2 基础上的"架构升级"，不是重写。**

---

## 七、下一阶段方向

| 方向 | 内容 | 难度 |
|------|------|------|
| 长期记忆 | 对话存入向量数据库，跨会话回忆 | ★★★ |
| 工具并行 | 多个无依赖工具同时执行（DAG调度） | ★★★ |
| Human-in-the-loop | 关键步骤暂停等人确认 | ★★☆ |
| MCP 协议 | 标准化的工具即服务 | ★★★ |
| A2A 协议 | Agent间标准化通信 | ★★★★ |
| 动态路由 | 根据任务类型自动选择 Agent 架构 | ★★★★ |

---

## 附录：第三阶段新增术语

| 术语 | 含义 |
|------|------|
| Plan-Execute | 先规划后执行，一次规划全局步骤，逐步执行 |
| Planner | 规划器，负责把任务拆成步骤列表 |
| Executor | 执行器，按计划逐步执行 |
| Replanner | 重规划器，检查结果决定是否重新规划 |
| Multi-Agent | 多个分工不同的 Agent 协作完成任务 |
| Orchestrator-Worker | 星型多Agent架构，Manager协调Workers（70%企业采用）|
| Agent基类 | 每个Agent有独立name/system_prompt/tools |
| ManagerAgent | 总调度器，不做具体工作只做协调 |
| SSE | Server-Sent Events，服务端推送技术 |
| Streaming | 流式输出，边生成边推送token |
| delta | 流式返回的增量数据块 |
| finish_reason | 流结束原因：stop(完成) / tool_calls(要调工具) |
| tool_calls 拼接 | 流式下 tool_calls 数据分散在多个 delta，需自行拼接 |
| 星型拓扑 | Manager 中心 + Worker 外围的架构模式 |
| Token 消耗 | multi-agent 约 15x 于单 agent |
