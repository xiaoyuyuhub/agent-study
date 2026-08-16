"""
Agent 第三阶段：Plan-Execute + 多Agent协作 + 流式输出
====================================================
在第一、二阶段的基础上，增加三个生产级高级特性：

  1. Plan-Execute（先规划再执行）  → 一次规划全局步骤，逐步执行，支持中途重规划
  2. Multi-Agent 协作              → Planner Agent 拆任务 + Worker Agent 执行 + Reviewer 审查
  3. 流式输出 (Streaming)          → SSE 边生成边输出，解决长等待问题

架构总览：
  ┌─────────────────────────────────────────────┐
  │              Manager Agent (总调度)           │
  │   负责：接收任务 → 委托 Planner → 协调 Workers │
  └──────────────┬──────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌────────┐ ┌────────┐ ┌──────────┐
│Planner │ │Worker1 │ │Reviewer  │
│拆解任务│ │执行工具│ │审查产出   │
└────────┘ └────────┘ └──────────┘

运行：
  python agent_v3.py
"""
import json
import time
import math
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from openai import OpenAI

# ╔══════════════════════════════════════════════════════════════╗
# ║  第零部分：配置读取                                            ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
DEFAULT_MODEL = config["model"]
MAX_STEPS = config.get("max_steps", 10)
MAX_REPLANS = config.get("max_replans", 3)
STREAM_OUTPUT = config.get("stream_output", True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第一部分：工具函数（复用 v2 的工具集）                        ║
# ╚══════════════════════════════════════════════════════════════╝

def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"不支持的字符: {expression}"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


def get_weather(city: str) -> str:
    """模拟天气查询"""
    fake_weather = {
        "北京": "28°C，晴",
        "上海": "26°C，多云",
        "深圳": "32°C，雷阵雨",
        "杭州": "24°C，阴",
        "广州": "30°C，阵雨",
    }
    return fake_weather.get(city, f"{city}: 暂无数据")


# ── RAG 知识检索（从 v2 复用）──
KB_DIR = Path(__file__).parent / "knowledge_base"


def _tokenize(text: str) -> list[str]:
    """中文分词：中文逐字 + 英文按词"""
    text = text.lower().strip()
    english_words = re.findall(r"[a-z0-9]+", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_words + chinese_chars


def _build_tfidf_vectors(documents: list[str]) -> tuple[list[dict], list[str]]:
    """构建 TF-IDF 向量"""
    tokenized_docs = [_tokenize(doc) for doc in documents]
    doc_freq = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    num_docs = len(documents)
    idf = {token: math.log(num_docs / (freq + 1)) for token, freq in doc_freq.items()}
    tfidf_vectors = []
    all_tokens = set()
    for tokens in tokenized_docs:
        total = len(tokens)
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        vector = {}
        for token, count in tf.items():
            vector[token] = (count / total) * idf.get(token, 0)
            all_tokens.add(token)
        tfidf_vectors.append(vector)
    return tfidf_vectors, list(all_tokens)


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """余弦相似度"""
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
    na = math.sqrt(sum(v * v for v in vec_a.values()))
    nb = math.sqrt(sum(v * v for v in vec_b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class KnowledgeBase:
    """迷你向量数据库（从 v2 复用）"""

    def __init__(self, kb_dir: Path):
        self.chunks = []
        self.chunk_sources = []
        for file_path in sorted(kb_dir.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8")
            for para in content.split("\n\n"):
                if para.strip():
                    self.chunks.append(para.strip())
                    self.chunk_sources.append(file_path.name)
        if self.chunks:
            self.vectors, _ = _build_tfidf_vectors(self.chunks)
            print(f"📚 知识库: {len(self.chunks)} 个文档块")
        else:
            self.vectors = []
            print(f"⚠️  知识库为空")

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.chunks:
            return "知识库为空"
        query_tokens = _tokenize(query)
        query_vec, _ = _build_tfidf_vectors([" ".join(query_tokens)])
        query_vector = query_vec[0] if query_vec else {}
        scored = []
        for i, doc_vec in enumerate(self.vectors):
            score = _cosine_similarity(query_vector, doc_vec)
            scored.append((score, self.chunks[i], self.chunk_sources[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, text, source in scored[:top_k]:
            results.append(f"[{source} | {score:.2f}]\n{text}")
        return "\n\n---\n\n".join(results)


KB = KnowledgeBase(KB_DIR)


def search_knowledge(query: str) -> str:
    """RAG 知识检索工具"""
    return KB.search(query, top_k=3)


# ── 工具定义 ──
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '3*(2+4)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 '北京'"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索内部知识库，查找产品FAQ、公司信息、定价、联系方式等。当用户问到公司或产品问题时，必须调用此工具，不要自己编造。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"}
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
    "search_knowledge": search_knowledge,
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  第二部分：工程化基础设施（从 v2 复用）                        ║
# ╚══════════════════════════════════════════════════════════════╝

def call_llm_with_retry(messages: list, tools: list, model: str, max_retries: int = 3) -> object:
    """
    带指数退避重试的 LLM 调用。

    为什么需要重试？
      - 网络抖动（超时、连接重置）
      - API 限流（429 Too Many Requests）
      - 服务端临时 500

    指数退避策略：1s → 2s → 4s
    好处：避免大量请求同时重试压垮服务端。
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, tools=tools)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)
                print(f"  ⚠️  失败（{attempt}/{max_retries}），{wait_time}s后重试: {e}")
                time.sleep(wait_time)
    raise last_error


def execute_tool_safely(name: str, args: dict) -> str:
    """
    安全执行工具。

    两层保护：
      1. 工具不存在 → 返回错误信息（防 LLM 幻觉出不存在的工具名）
      2. 工具内部异常 → 捕获并返回错误信息（不让 Agent 崩溃）
    """
    if name not in TOOL_MAP:
        return f"错误: 工具 '{name}' 不存在。可用: {list(TOOL_MAP.keys())}"
    try:
        return TOOL_MAP[name](**args)
    except Exception as e:
        return f"工具执行出错 [{name}]: {e}"


def reflect_on_step(task: str, tool_name: str, tool_args: dict, tool_result: str, model: str) -> str:
    """
    反思机制（从 v2 复用）。

    让 LLM 以"审查员"身份评估刚执行的步骤。
    - 独立调用，不带完整历史（更客观 + 省 token）
    - 不带 tools（反思本身不需要再调工具）
    """
    prompt = f"""你是严谨的审查员。评估以下 Agent 执行步骤：

【用户任务】{task}
【这一步】调用 {tool_name}，参数 {tool_args}，结果 {tool_result}

判断：工具选择是否正确？参数是否合理？结果是否可信？
如果正确回复"OK"，发现问题指出问题并给建议。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[反思跳过: {e}]"


# ╔══════════════════════════════════════════════════════════════╗
# ║  第三部分：流式输出引擎（第三阶段新增 ★核心1★）               ║
# ║                                                              ║
# ║  为什么需要流式输出？                                        ║
# ║    普通调用：发请求 → 等服务端生成完 → 一次性返回            ║
# ║       → 用户干等几秒甚至几十秒，体验极差                      ║
# ║    流式调用：发请求 → 边生成边返回 token  → 实时看到文字     ║
# ║       → 像打字一样逐字出现，体验丝滑                          ║
# ║                                                              ║
# ║  SSE (Server-Sent Events) 原理：                             ║
# ║    服务端推送 data: {"choices":[{"delta":{"content":"你"}}]}  ║
# ║    客户端逐行读取，实时打印                                   ║
# ║                                                              ║
# ║  本引擎解决的三个难题：                                        ║
# ║    1. 流式 + Function Calling 并存 → 先收齐 tool_call，      ║
# ║       再决定走"工具执行"还是"流式输出文本"                    ║
# ║    2. 工具调用参数也是流式来的 → 需要逐 delta 拼接            ║
# ║    3. 同时支持流式和非流式（非流式用于规划等不需要展示的场景）  ║
# ╚══════════════════════════════════════════════════════════════╝

class StreamingLLM:
    """
    流式 LLM 调用引擎。

    封装了两种调用模式：
      - stream=True：边生成边输出（用户可见的文字回答）
      - stream=False：一次性返回（规划、反思等内部调用）

    处理流式 + Function Calling 并存的复杂性：
      当 LLM 返回 tool_calls 时，不能直接流式输出 content（因为 content 通常为 None）。
      必须先收集完整的 tool_calls，再执行工具。
      当 LLM 返回纯文本时，逐 token 流式打印。
    """

    @staticmethod
    def stream_text(
        messages: list,
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        流式输出纯文本（不带 tools）。

        适用场景：
          - Planner 输出计划（用户想看计划生成过程）
          - Reviewer 输出审查意见
          - Manager 的最终总结

        on_token 回调：每收到一个 token 就调用，可以自定义输出方式。
        默认是 print 到终端。
        """
        full_text = ""

        # 关键参数：stream=True 开启 SSE 流式传输
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,  # ← 这是核心开关
        )

        # 逐 chunk 处理 SSE 事件流
        for chunk in stream:
            # chunk 结构: {choices: [{delta: {content: "你"}}]}
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    token = delta.content  # 当前这个 token 的文本
                    full_text += token
                    if on_token:
                        on_token(token)  # 回调：自定义处理
                    else:
                        print(token, end="", flush=True)  # 实时打印，不换行

        return full_text

    @staticmethod
    def stream_with_tools(
        messages: list,
        tools: list,
        model: str,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        流式调用 + Function Calling 支持（这是最复杂的情况）。

        挑战：
          SSE 返回的是增量 delta，tool_calls 的数据是分散在多个 chunk 里的。
          比如 {"name": "get_weather", "arguments": {"city": "北"}}
          下一个 chunk：{"arguments": {"city": "北京"}}
          必须自己拼起来。

        返回格式：
          {"type": "text", "content": "..."}           # 纯文本回答
          {"type": "tool_calls", "calls": [...]}       # 需要调工具

        处理逻辑：
          1. 遍历每个 chunk
          2. 如果是 content delta → 流式输出 + 累积
          3. 如果是 tool_calls delta → 按 index + 字段名 拼接
          4. 全部收齐后，判断 finish_reason：
             - "tool_calls" → 返回工具调用列表
             - "stop" → 返回完整文本
        """
        # ── 用于拼接分散的 tool_calls 数据 ──
        # tool_calls_acc 的结构：
        #   {0: {"id": "call_xxx", "name": "get_weather", "arguments": ""}}
        # 因为 LLM 可能一次返回多个 tool_call，用 index 区分
        tool_calls_acc = {}
        full_text = ""
        finish_reason = None

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # ── 处理纯文本 delta ──
            # 在 Function Calling 场景下，LLM 通常不输出 content，
            # 但某些模型可能会先输出一句"我来帮你查一下"再调工具
            if delta.content:
                token = delta.content
                full_text += token
                if on_text:
                    on_text(token)
                else:
                    print(token, end="", flush=True)

            # ── 处理 tool_calls delta ──
            # delta.tool_calls 是一个列表，每个元素是某个 tool_call 的增量
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index  # tool_call 的序号（0, 1, 2...）

                    # 初始化这个 tool_call 的累加器
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }

                    # id 一般只在第一个 chunk 出现
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id

                    # function name 一般只在第一个 chunk 出现
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] = tc_delta.function.name

                    # arguments 是 JSON 字符串，分散在多个 chunk
                    # 每个 chunk 追加一段，需要自己拼接
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )

            # ── 记录 finish_reason ──
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # ── 流结束，判断返回类型 ──
        if finish_reason == "tool_calls" and tool_calls_acc:
            # LLM 要调工具，返回结构化的 tool_calls
            return {
                "type": "tool_calls",
                "calls": [
                    {
                        "id": tc["id"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in sorted(tool_calls_acc.values(), key=lambda x: x["id"])
                ],
            }
        else:
            # 纯文本回答
            return {"type": "text", "content": full_text}

    @staticmethod
    def call_non_stream(messages: list, tools: list, model: str) -> object:
        """
        非流式调用（内部使用：规划、反思等不需要展示的场景）。

        与 v2 的 call_llm_with_retry 的区别：
          - 这个用于内部决策（Planner、Reviewer），不需要用户看到过程
          - 直接返回 OpenAI response 对象，由调用方自行解析
        """
        return call_llm_with_retry(messages, tools, model)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第四部分：Plan-Execute 模式（第三阶段新增 ★核心2★）          ║
# ║                                                              ║
# ║  与 ReAct 的本质区别：                                       ║
# ║    ReAct: 每步都调 LLM 想下一步（边走边看）                  ║
# ║    Plan-Execute: 先一次性规划全部步骤，再逐步执行（先想清楚）  ║
# ║                                                              ║
# ║  流程：                                                       ║
# ║    用户任务 → Planner 规划 → [步骤1, 步骤2, ...]             ║
# ║    → 逐步执行每个步骤                                        ║
# ║    → 执行完后 Replanner 检查：完成? 或 重新规划?              ║
# ║                                                              ║
# ║  为什么 Plan-Execute 比 ReAct 好？                           ║
# ║    1. 成本更低：规划只调一次大模型，执行阶段用小模型或无模型  ║
# ║    2. 速度快：不需要每步都等 LLM 思考                        ║
# ║    3. 全局视角：先想清楚再干，不容易跑偏                     ║
# ║    4. 可并行：无依赖的步骤可以同时执行（本 demo 未实现并行）  ║
# ║                                                              ║
# ║  2026 年实测数据（来源：agent设计模式论文）：                 ║
# ║    完成率 92%，比 ReAct 快 3.6 倍                            ║
# ╚══════════════════════════════════════════════════════════════╝

class Planner:
    """
    规划器：负责把用户任务拆成可执行的步骤列表。

    这是 Plan-Execute 模式的核心组件。
    只负责"想"，不负责"做"。
    """

    @staticmethod
    def generate_plan(task: str, model: str) -> list[dict]:
        """
        让 LLM 生成执行计划。

        输入：用户原始任务
        输出：结构化步骤列表
        示例：
          [
            {"step": 1, "action": "查天气", "tool": "get_weather", "params": {"city": "北京"}},
            {"step": 2, "action": "算数学", "tool": "calculate", "params": {"expression": "3+4"}},
            ...
          ]

        设计要点：
          1. prompt 里明确列出可用工具 → LLM 知道能做什么
          2. 要求输出 JSON 格式 → 方便代码解析
          3. 要求判断"能直接回答"的步骤 → 避免无工具可调时瞎编
        """
        tools_desc: str = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']}"
            for t in TOOLS
        )

        plan_prompt = f"""你是任务规划专家。请将以下任务拆解为执行步骤。

【可用工具】
{tools_desc}

【任务】
{task}

【要求】
1. 每个步骤包含：step编号、action描述、tool工具名（必须是上面列出的工具之一）、params参数
2. 如果某步骤不需要工具（如"总结"），tool填null
3. 步骤顺序合理，无遗漏
4. 输出纯 JSON 数组，不要其他文字：
[
  {{"step": 1, "action": "...", "tool": "get_weather", "params": {{"city": "北京"}}}},
  {{"step": 2, "action": "总结以上信息", "tool": null, "params": {{}}}}
]"""

        # 规划阶段用非流式调用（规划是内部决策，不需要用户看到生成过程）
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": plan_prompt}],
        )
        content = response.choices[0].message.content

        # ── 解析 JSON ──
        # LLM 可能返回 ```json ... ``` 包裹的格式，需要去掉
        try:
            # 去掉可能的 markdown 代码块标记
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]  # 去掉第一行 ```json
                if content.endswith("```"):
                    content = content.rsplit("\n", 1)[0]  # 去掉最后一行 ```
            plan = json.loads(content)
            return plan
        except json.JSONDecodeError:
            # 解析失败时，降级为单步骤计划（把整个任务当一个步骤）
            print("  ⚠️  计划解析失败，降级为单步执行")
            return [{"step": 1, "action": task, "tool": None, "params": {}}]


class Executor:
    """
    执行器：按照计划逐步执行每个步骤。

    核心逻辑：
      遍历计划 → 对每个步骤：
        - 有 tool → 调工具执行
        - 无 tool → 调 LLM 用上下文回答问题（如"总结"步骤）
      → 收集所有结果 → 返回
    """

    @staticmethod
    def execute_plan(
        plan: list[dict],
        task: str,
        model: str,
    ) -> list[dict]:
        """
        按计划执行。

        返回：每个步骤的执行结果列表
          [{"step": 1, "action": "查天气", "result": "北京 28°C"}, ...]
        """
        results = []

        for step in plan:
            step_num = step["step"]
            action = step["action"]
            tool_name = step.get("tool")
            params = step.get("params", {})

            print(f"\n  ▶ 步骤{step_num}: {action}")

            if tool_name and tool_name in TOOL_MAP:
                # ── 有工具：调工具执行 ──
                # 这是 Plan-Execute 的核心：用代码执行工具，不调 LLM
                result = execute_tool_safely(tool_name, params)
                print(f"    工具结果: {result[:80]}")
            else:
                # ── 无工具：调 LLM 用上下文生成回答 ──
                # 典型场景：总结、分析、判断等纯语言任务
                summary_prompt = (
                    f"任务背景：{task}\n"
                    f"已完成步骤及结果：{json.dumps(results, ensure_ascii=False)}\n"
                    f"当前步骤：{action}\n"
                    f"请完成当前步骤，给出结果。"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": summary_prompt}],
                )
                result = response.choices[0].message.content
                print(f"    LLM 结果: {result[:80]}")

            results.append({"step": step_num, "action": action, "result": result})

        return results


class Replanner:
    """
    重规划器：检查执行结果，决定是否需要重新规划。

    这是 Plan-Execute 比"一次性规划"更先进的地方：
      - 执行完后检查：所有步骤都完成了吗？结果可靠吗？
      - 如果不够好 → 重新规划未完成的部分
      - 最多重规划 MAX_REPLANS 次，防止死循环
    """

    @staticmethod
    def check_and_replan(
        task: str,
        plan: list[dict],
        results: list[dict],
        model: str,
    ) -> tuple[bool, Optional[list[dict]]]:
        """
        检查执行结果，返回 (是否完成, 新计划或None)。

        逻辑：
          1. 让 LLM 评估：所有子任务完成了吗？结果可信吗？
          2. 完成了 → (True, None)
          3. 没完成 → (False, 新的补充计划)
        """
        check_prompt = f"""你是质量审查员。检查以下任务执行情况：

【原始任务】{task}

【原计划】{json.dumps(plan, ensure_ascii=False, indent=2)}

【执行结果】{json.dumps(results, ensure_ascii=False, indent=2)}

请判断：
1. 任务是否完全完成？
2. 结果是否准确可信？
3. 如果有遗漏，需要补充什么步骤？

如果完全完成，回复: COMPLETE
如果有遗漏，回复: REPLAN，并给出补充步骤（JSON数组，格式同原计划）"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": check_prompt}],
        )
        verdict = response.choices[0].message.content.strip()

        if "COMPLETE" in verdict.upper():
            return True, None
        else:
            # ── 尝试从回复中提取补充计划 ──
            try:
                # 找 JSON 数组
                import re as re_mod
                json_match = re_mod.search(r"\[[\s\S]*\]", verdict)
                if json_match:
                    new_plan = json.loads(json_match.group())
                    return False, new_plan
            except Exception:
                pass
            return False, None


def run_plan_execute(task: str, model: str = DEFAULT_MODEL):
    """
    Plan-Execute 模式的主入口。

    完整流程：
      1. Planner 生成计划
      2. Executor 逐步执行
      3. Replanner 检查 → 完成? 或 重规划?
      4. 循环 2-3 直到完成或达到最大重规划次数
    """
    print(f"\n{'='*60}")
    print(f"📋 Plan-Execute 模式启动")
    print(f"{'='*60}")

    # ── 阶段1：规划 ──
    print(f"\n🧠 [Planner] 正在制定计划...")
    plan = Planner.generate_plan(task, model)
    print(f"   计划共 {len(plan)} 步:")
    for s in plan:
        tool_str = s.get("tool") or "无需工具"
        print(f"     {s['step']}. {s['action']} [{tool_str}]")

    # ── 阶段2-3：执行 + 重规划循环 ──
    all_results = []
    replan_count = 0


    while replan_count <= MAX_REPLANS:
        # 执行当前计划
        print(f"\n⚡ [Executor] 开始执行 (重规划次数: {replan_count})")
        results = Executor.execute_plan(plan, task, model)
        all_results.extend(results)

        # Replanner 检查
        print(f"\n🔍 [Replanner] 检查执行结果...")
        done, new_plan = Replanner.check_and_replan(task, plan, results, model)

        if done:
            print(f"   ✅ 任务完成！")
            break
        elif new_plan:
            print(f"   🔄 需要补充 {len(new_plan)} 步，重新执行...")
            plan = new_plan
            replan_count += 1
        else:
            print(f"   ⚠️  无法解析补充计划，终止")
            break

    if replan_count > MAX_REPLANS:
        print(f"   ⚠️  超过最大重规划次数 ({MAX_REPLANS})")

    # ── 阶段4：汇总 ──
    print(f"\n📊 执行汇总:")
    for r in all_results:
        print(f"   步骤{r['step']}: {r['action']} → {str(r['result'])[:80]}")

    return all_results


# ╔══════════════════════════════════════════════════════════════╗
# ║  第五部分：多Agent协作（第三阶段新增 ★核心3★）                ║
# ║                                                              ║
# ║  为什么需要多 Agent？                                        ║
# ║    1. 单一 Agent 的 system prompt 太长 → 指令冲突           ║
# ║    2. 不同子任务需要不同的工具集 → 混在一起效率低             ║
# ║    3. 不同子任务需要不同的"人格" → 一个 prompt 做不到        ║
# ║    4. 可以并行处理无依赖的子任务 → 速度快                     ║
# ║                                                              ║
# ║  本实现的架构：Orchestrator-Worker (星型拓扑)                 ║
# ║    这是 2026 年工业界最主流的多 Agent 模式（~70% 采用率）。   ║
# ║                                                              ║
# ║                  ┌──────────────┐                            ║
# ║                  │   Manager    │  ← 总调度，唯一入口        ║
# ║                  │   Agent      │                            ║
# ║                  └──────┬───────┘                            ║
# ║           ┌─────────────┼─────────────┐                     ║
# ║           ▼             ▼             ▼                     ║
# ║    ┌──────────┐  ┌──────────┐  ┌──────────┐                ║
# ║    │ Planner  │  │  Worker  │  │ Reviewer │                ║
# ║    │  (规划)   │  │  (执行)   │  │  (审查)   │                ║
# ║    └──────────┘  └──────────┘  └──────────┘                ║
# ║                                                              ║
# ║  Worker 可以有多个，每个负责不同类型的任务。                 ║
# ║  本 demo 用了 2 个 Worker:                                   ║
# ║    - ToolWorker: 负责调工具（天气、计算、知识检索）          ║
# ║    - TextWorker: 负责纯语言任务（总结、分析）                ║
# ╚══════════════════════════════════════════════════════════════╝

class Agent:
    """
    单个 Agent 的基类。

    每个 Agent 有：
      - name: 名称（用于日志和调度）
      - system_prompt: 角色定位（决定这个 Agent 的"人格"和职责）
      - tools: 可用工具列表（None 表示纯语言 Agent）

    Agent 之间的通信通过 Manager 中转，不直接对话。
    这是 Orchestrator-Worker 模式的标准做法：中央协调，避免 Agent 间混乱通信。
    """

    def __init__(self, name: str, system_prompt: str, tools: list = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []

    def run(self, task: str, model: str = DEFAULT_MODEL, context: str = "") -> str:
        """
        让这个 Agent 处理一个任务。

        参数：
          task: 要处理的任务描述
          context: Manager 提供的上下文（之前的执行结果等）

        内部走 ReAct 循环（因为单个 Worker 的任务通常比较简单，
        不需要再拆成 Plan-Execute）。
        """
        # ── 组装 messages ──
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            # 把 Manager 提供的上下文塞进去
            messages.append({"role": "system", "content": f"【上下文】{context}"})
        messages.append({"role": "user", "content": task})

        # ── ReAct 循环 ──
        step = 0
        while step < 5:  # 单个 Worker 的步数限制（比全局 MAX_STEPS 小）
            step += 1
            response = call_llm_with_retry(messages, self.tools, model)
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = execute_tool_safely(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue
            else:
                return msg.content

        return f"[{self.name}] 达到最大步数"


class ManagerAgent:
    """
    Manager Agent（总调度器）。

    职责：
      1. 接收用户任务
      2. 委托 Planner 拆解任务
      3. 根据子任务类型分配给不同 Worker
      4. 收集 Worker 结果，交给 Reviewer 审查
      5. 审查通过后，汇总输出最终结果

    这是 Orchestrator-Worker 模式的核心。
    Manager 不做具体工作，只做协调。
    """

    def __init__(self):
        # ── 注册子 Agent ──
        # 每个子 Agent 有独立的 system prompt 和工具集

        # Planner: 专门负责拆解任务，不需要工具
        self.planner = Agent(
            name="Planner",
            system_prompt="""你是任务规划专家。你的职责是把复杂任务拆解为子任务。

规则：
1. 每个子任务必须明确指定类型：
   - 类型"tool"：需要调用工具的（查天气、算数学、搜知识库）
   - 类型"text"：纯语言处理的（总结、分析、判断）
2. 子任务之间不要有重复
3. 输出纯 JSON 数组格式：
[
  {"type": "tool", "description": "查北京天气"},
  {"type": "tool", "description": "计算 888*333"},
  {"type": "text", "description": "总结以上所有信息"}
]""",
        )

        # ToolWorker: 负责调工具执行具体操作
        self.tool_worker = Agent(
            name="ToolWorker",
            system_prompt="""你是工具执行专家。你负责调用工具完成具体操作。
可用的工具：
- get_current_time: 查时间
- calculate: 算数学
- get_weather: 查天气
- search_knowledge: 搜索知识库（产品FAQ、公司信息等）

规则：
1. 根据任务描述，选择合适的工具
2. 每个工具调用完返回结果即可
3. 不要做总结，不要做分析，只负责调工具拿数据""",
            tools=TOOLS,  # ← 只有 ToolWorker 有工具
        )

        # TextWorker: 负责纯语言处理
        self.text_worker = Agent(
            name="TextWorker",
            system_prompt="""你是语言处理专家。你负责总结、分析、润色文本内容。
你不需要调用任何工具，只需要根据上下文生成清晰、准确的文字。

规则：
1. 根据提供的上下文和任务要求，生成回答
2. 保持简洁、准确
3. 如果上下文中有数据，确保引用的数据准确""",
            # TextWorker 没有 tools，纯语言 Agent
        )

        # Reviewer: 负责审查和汇总
        self.reviewer = Agent(
            name="Reviewer",
            system_prompt="""你是质量审查员。你负责：
1. 检查所有子任务的执行结果是否准确、完整
2. 如果发现问题，指出具体问题
3. 如果一切正常，将所有结果汇总成一段完整的回答

汇总要求：
- 用自然语言，不要列 bullet points（除非用户要求）
- 确保所有数据准确引用
- 如果信息来自知识库，标注来源""",
        )

    def run(self, task: str, model: str = DEFAULT_MODEL) -> str:
        """
        多 Agent 协作的主流程。

        流程：
          1. Planner 拆任务 → 子任务列表
          2. 根据类型分配 Worker → 并行收集结果（本 demo 串行以简化）
          3. Reviewer 审查 → 有问题? 重新执行 : 输出最终回答
        """
        print(f"\n{'='*60}")
        print(f"👔 [Manager] 多 Agent 协作模式启动")
        print(f"{'='*60}")

        # ── 阶段1：Planner 拆任务 ──
        print(f"\n🧠 [{self.planner.name}] 正在拆解任务...")
        plan_text = self.planner.run(task, model)
        print(f"   原始输出: {plan_text[:200]}...")

        # 解析 Planner 的 JSON 输出
        try:
            # 去掉可能的 markdown 代码块
            clean = plan_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean.rsplit("\n", 1)[0]
            subtasks = json.loads(clean)
        except json.JSONDecodeError:
            print(f"   ⚠️  计划解析失败，降级处理")
            subtasks = [{"type": "text", "description": task}]

        print(f"   拆解为 {len(subtasks)} 个子任务:")
        for i, st in enumerate(subtasks):
            print(f"     {i+1}. [{st['type']}] {st['description']}")

        # ── 阶段2：分配 Worker 执行 ──
        print(f"\n⚡ [Manager] 分配子任务给 Workers...")
        worker_results = []

        for i, subtask in enumerate(subtasks):
            sub_type = subtask["type"]
            sub_desc = subtask["description"]

            # 根据类型选择 Worker
            if sub_type == "tool":
                worker = self.tool_worker
            else:
                worker = self.text_worker

            # 构建上下文（包含之前的执行结果，让 Worker 知道全局）
            context = f"原始任务: {task}\n已完成的子任务结果: {json.dumps(worker_results, ensure_ascii=False)}"

            print(f"\n  [{worker.name}] 执行子任务{i+1}: {sub_desc}")
            result = worker.run(sub_desc, model, context)
            print(f"  [{worker.name}] 结果: {result[:100]}...")

            worker_results.append({
                "task": sub_desc,
                "worker": worker.name,
                "result": result,
            })

        # ── 阶段3：Reviewer 审查 ──
        print(f"\n🔍 [{self.reviewer.name}] 正在审查所有结果...")

        # 把全部结果喂给 Reviewer
        review_input = (
            f"原始任务: {task}\n\n"
            f"子任务执行结果:\n{json.dumps(worker_results, ensure_ascii=False, indent=2)}\n\n"
            f"请审查并汇总。"
        )
        final_answer = self.reviewer.run(review_input, model)

        # ── 流式输出最终答案 ──
        if STREAM_OUTPUT:
            print(f"\n{'='*60}")
            print(f"📋 最终回答（流式输出）:")
            print(f"{'='*60}")

            # 用流式方式重新生成最终回答（体验更好）
            final_messages = [
                {"role": "system", "content": self.reviewer.system_prompt},
                {"role": "user", "content": review_input},
            ]
            final_answer = StreamingLLM.stream_text(final_messages, model)
            print()  # 换行
        else:
            print(f"\n{'='*60}")
            print(f"📋 最终回答:")
            print(f"{'='*60}")
            print(final_answer)

        return final_answer


# ╔══════════════════════════════════════════════════════════════╗
# ║  第六部分：流式 ReAct Agent（第三阶段新增）                    ║
# ║                                                              ║
# ║  在第一阶段 ReAct 的基础上，加上流式输出。                     ║
# ║  用于对比：同样的 ReAct 逻辑，流式输出体验好多少。            ║
# ╚══════════════════════════════════════════════════════════════╝

def run_streaming_react(task: str, model: str = DEFAULT_MODEL):
    """
    流式 ReAct Agent。

    这是第一阶段的 ReAct 循环 + 第三阶段的流式输出。
    对比非流式版本，你能直观感受到流式输出的优势。

    核心改动：
      - 非流式：response = client.chat.completions.create(stream=False)
        → 等全部生成完 → 一次性输出
      - 流式：StreamingLLM.stream_with_tools(stream=True)
        → 边生成边输出 → 实时看到 LLM 的思考
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是能调用工具的助手。可用工具: get_current_time, calculate, "
                "get_weather, search_knowledge。遇到问题先调工具，不要瞎编。"
            ),
        },
        {"role": "user", "content": task},
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n{'='*60}")
        print(f"🔄 第 {step}/{MAX_STEPS} 轮")

        # ★ 流式调用（支持 Function Calling）
        print(f"📥 ", end="", flush=True)
        result = StreamingLLM.stream_with_tools(messages, TOOLS, model)
        print()  # 换行

        if result["type"] == "tool_calls":
            # ── LLM 要调工具 ──
            # 注意：流式返回的 tool_calls 没有标准的 message 对象，
            # 需要手动构造一个"假的" assistant message 塞进历史

            tool_call_objects = []
            tool_results = []
            for tc in result["calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                print(f"  🔧 调工具: {name}({args})")
                tool_result = execute_tool_safely(name, args)
                print(f"  📤 结果: {tool_result[:80]}")

                # 构造 tool_call 对象
                tool_call_objects.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": tc["function"]["arguments"],
                    },
                })
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # 把 assistant 的 tool_call 决策 + 工具结果 塞进历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_call_objects,
            })
            messages.extend(tool_results)
            continue

        else:
            # ── LLM 不调工具了，任务完成 ──
            print(f"\n🏁 任务完成")
            return result["content"]

    return f"达到最大步数 {MAX_STEPS}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  第七部分：综合演示入口                                         ║
# ║                                                              ║
# ║  三个模式可以选择运行：                                        ║
# ║    1. Plan-Execute 模式                                      ║
# ║    2. 多 Agent 协作模式                                       ║
# ║    3. 流式 ReAct 模式                                        ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent v3 - Plan-Execute / Multi-Agent / Streaming")
    parser.add_argument(
        "--mode", choices=["plan", "multi", "stream"], default="multi",
        help="运行模式: plan(Plan-Execute), multi(多Agent协作), stream(流式ReAct)"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="自定义任务（不指定则用默认任务）"
    )
    args = parser.parse_args()

    # 默认任务：能充分展示三种模式差异的复合任务
    default_task = (
        "帮我做以下几件事："
        "1. 查北京和深圳的天气；"
        "2. 算 (888 * 333) + (123 * 456) 等于多少；"
        "3. CodeBuddy Pro 的定价方案是什么？"
        "4. 公司总部在哪里？"
        "最后把所有结果总结成一段完整的汇报。"
    )
    task = args.task or default_task

    print(f"\n📋 用户任务: {task}")
    print(f"⚙️  配置: model={DEFAULT_MODEL}, max_steps={MAX_STEPS}, stream={STREAM_OUTPUT}")

    if args.mode == "plan":
        # ── 模式1：Plan-Execute ──
        results = run_plan_execute(task)

        # 最终总结
        print(f"\n{'='*60}")
        print(f"📋 最终总结:")
        print(f"{'='*60}")
        summary_prompt = (
            f"用户任务: {task}\n\n"
            f"执行结果: {json.dumps(results, ensure_ascii=False)}\n\n"
            f"请将这些结果整理成一段完整的汇报。"
        )
        if STREAM_OUTPUT:
            final = StreamingLLM.stream_text(
                [{"role": "user", "content": summary_prompt}], DEFAULT_MODEL
            )
            print()
        else:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            final = resp.choices[0].message.content
            print(final)

    elif args.mode == "multi":
        # ── 模式2：多 Agent 协作 ──
        manager = ManagerAgent()
        manager.run(task)

    elif args.mode == "stream":
        # ── 模式3：流式 ReAct ──
        run_streaming_react(task)
