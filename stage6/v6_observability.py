"""
Agent V6 方向6：可观测性（全链路追踪 Trace/Span）
===================================================
给 Agent 加上「可观测性」，出问题时能定位「哪一步、花了多久、为什么错」。

【为什么需要可观测性？】

随着阶段推进，Agent 越来越复杂：
  - 一次任务可能调用多次 LLM（规划、执行、反思、汇总）
  - 每次 LLM 又可能调用多个工具
  - 多 Agent 编排下，一次请求会跨越多个 Agent

一旦结果不对，你面对的是一个「黑盒」：不知道是 LLM 决策错了、
工具调用错了、还是某个步骤超时了。

可观测性就是给这个黑盒装上「监控摄像头」。

【核心概念：Trace 和 Span】

  可观测性领域（借鉴分布式追踪 OpenTelemetry）有两个核心概念：

  Trace（追踪）= 一次完整请求的完整记录
  Span（片段）= Trace 里的一个最小单元（比如一次 LLM 调用、一次工具调用）

  关系：
    Trace
     ├── Span: LLM 规划（200ms, 150 tokens）
     ├── Span: 工具 get_weather（5ms）
     │        └── Span: 工具 calculate（3ms）   ← 嵌套的 Span
     └── Span: LLM 汇总（180ms, 120 tokens）

  Span 可以嵌套（父 Span 包含子 Span），形成一棵「调用树」。
  每个 Span 记录：名字、类型、开始/结束时间、输入、输出、标签。

【生产方案：LangSmith / Langfuse】

  工业界有成熟的可观测平台：
    LangSmith（LangChain 生态）—— 记录每次 LLM/工具调用的全链路
    Langfuse（开源）—— 类似，可自托管
  它们都能可视化地展示 Trace/Span 树、token 消耗、耗时、成本。

  这些平台需要注册 + API key。为了让「零依赖能跑 + 理解本质」，
  本文件实现一个轻量 Tracer，用同样的 Trace/Span 概念，
  输出文本版调用树。理解了它，再上手 LangSmith/Langfuse 会非常快。

【本文件实现】

  1. Span —— 一个调用单元（记录耗时、输入输出、标签）
  2. Tracer —— 管理 Span 栈，用 context manager 自动计时
  3. report —— 生成文本版调用树 + 统计摘要
  4. traced_llm_call / traced_tool_call —— 演示如何包装真实调用

运行：
  python v6_observability.py
"""
import time
from contextlib import contextmanager
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  第一部分：Span（调用片段）
# ═══════════════════════════════════════════════════════════════

class Span:
    """一个调用单元，记录一次操作的完整信息"""

    def __init__(self, name: str, span_type: str):
        self.name = name            # 名称，如 "LLM:规划"
        self.span_type = span_type  # 类型：llm / tool / agent
        self.start = time.time()    # 开始时间（Unix 时间戳）
        self.end: Optional[float] = None  # 结束时间
        self.input = None           # 输入
        self.output = None          # 输出
        self.tags = {}              # 标签（token 数、model 名等）
        self.children: list["Span"] = []  # 子 Span

    @property
    def duration_ms(self) -> float:
        """耗时（毫秒）"""
        if self.end is None:
            return 0.0
        return (self.end - self.start) * 1000

    def __repr__(self):
        return f"<Span {self.name} {self.duration_ms:.1f}ms>"


# ═══════════════════════════════════════════════════════════════
#  第二部分：Tracer（追踪器）
# ═══════════════════════════════════════════════════════════════

class Tracer:
    """
    追踪器：记录一次任务里的所有 Span，并生成报告。

    用法（核心是 context manager，自动计时）：

        tracer = Tracer()
        with tracer.span("LLM调用", "llm", model="gpt-4") as s:
            s.input = messages
            ...  # 这里调用 LLM
            s.output = "回答"

    退出 with 块时自动记录结束时间，并挂到当前父 Span 下。
    """

    def __init__(self):
        self.root_spans: list[Span] = []  # 顶层 Span
        self._stack: list[Span] = []      # 当前 Span 栈（用于嵌套）

    @contextmanager
    def span(self, name: str, span_type: str, **tags):
        """创建一个 Span 上下文，自动计时 + 自动挂载"""
        s = Span(name, span_type)
        s.tags.update(tags)

        # 挂到父 Span（如果有）或顶层
        if self._stack:
            self._stack[-1].children.append(s)
        else:
            self.root_spans.append(s)

        self._stack.append(s)
        try:
            yield s  # 把 Span 交给调用方，让它填 input/output
        finally:
            s.end = time.time()
            self._stack.pop()

    # ── 报告生成 ──────────────────────────────────────────

    def report(self) -> str:
        """生成文本版调用树 + 统计摘要"""
        lines = []
        lines.append("=" * 64)
        lines.append("📊 全链路追踪报告（Trace）")
        lines.append("=" * 64)

        for span in self.root_spans:
            self._render_span(span, lines, indent="", is_last=True)

        lines.append("─" * 64)
        lines.append(self._summary())
        return "\n".join(lines)

    def _render_span(self, span: Span, lines: list, indent: str, is_last: bool):
        """递归渲染一棵 Span 树（带分支符号）"""
        branch = "└─ " if is_last else "├─ "
        type_icon = {"llm": "🧠", "tool": "🔧", "agent": "🤖"}.get(span.span_type, "•")
        tag_str = ""
        if span.tags:
            tag_str = " [" + ", ".join(f"{k}={v}" for k, v in span.tags.items()) + "]"
        lines.append(f"{indent}{branch}{type_icon} {span.name} ({span.duration_ms:.1f}ms){tag_str}")

        child_indent = indent + ("   " if is_last else "│  ")
        for i, child in enumerate(span.children):
            self._render_span(child, lines, child_indent, is_last=(i == len(span.children) - 1))

    def _summary(self) -> str:
        """统计摘要：总耗时、各类 Span 数量、总 token 数"""
        all_spans = self._flatten()
        total_ms = sum(s.duration_ms for s in all_spans)
        llm_count = sum(1 for s in all_spans if s.span_type == "llm")
        tool_count = sum(1 for s in all_spans if s.span_type == "tool")
        total_tokens = sum(s.tags.get("tokens", 0) for s in all_spans)
        return (
            f"📈 汇总：共 {len(all_spans)} 个 Span，"
            f"LLM 调用 {llm_count} 次，工具调用 {tool_count} 次，"
            f"累计耗时 {total_ms:.1f}ms，累计 token {total_tokens}"
        )

    def _flatten(self) -> list:
        """把所有 Span 拍平成一维列表"""
        result = []
        def walk(spans):
            for s in spans:
                result.append(s)
                walk(s.children)
        walk(self.root_spans)
        return result


# ═══════════════════════════════════════════════════════════════
#  第三部分：包装真实调用的示例
# ═══════════════════════════════════════════════════════════════

def traced_llm_call(tracer: Tracer, name: str, model: str, messages, tokens: int = 0):
    """
    演示如何把一次 LLM 调用纳入追踪。

    真实场景：在 client.chat.completions.create() 外面套一层 span，
    调用结束后把 token 数写进 tags。
    """
    with tracer.span(name, "llm", model=model) as s:
        s.input = messages
        time.sleep(0.1)  # 模拟 LLM 推理耗时
        s.output = "（模拟 LLM 返回内容）"
        if tokens:
            s.tags["tokens"] = tokens
        return s.output


def traced_tool_call(tracer: Tracer, tool_name: str, args):
    """演示如何把一次工具调用纳入追踪"""
    with tracer.span(f"工具:{tool_name}", "tool") as s:
        s.input = args
        time.sleep(0.03)  # 模拟工具执行耗时
        s.output = f"（{tool_name} 的结果）"
        return s.output


# ═══════════════════════════════════════════════════════════════
#  第四部分：独立测试入口
# ═══════════════════════════════════════════════════════════════

def simulate_agent_run():
    """
    模拟一次 Agent 任务的完整执行，全程用 tracer 追踪。

    结构（和真实 Agent 的执行一致）：
      顶层 Span: 处理任务
        ├─ LLM: 规划（产生多个工具调用）
        ├─ 工具: get_weather
        ├─ 工具: calculate
        └─ LLM: 汇总
    """
    tracer = Tracer()

    with tracer.span("处理任务", "agent"):
        # ① LLM 规划
        plan = traced_llm_call(
            tracer, "LLM:规划", "gpt-4o-mini",
            messages=["帮我查天气和算数学"], tokens=150,
        )
        # ② 两个工具调用（无依赖，并行）
        w = traced_tool_call(tracer, "get_weather", {"city": "北京"})
        c = traced_tool_call(tracer, "calculate", {"expression": "6*7"})
        # ③ LLM 汇总
        final = traced_llm_call(
            tracer, "LLM:汇总", "gpt-4o-mini",
            messages=[plan, w, c], tokens=120,
        )

    return tracer


if __name__ == "__main__":
    print("=" * 64)
    print("📊 方向6：可观测性 —— 全链路追踪演示")
    print("=" * 64)

    print("\n【模拟一次 Agent 任务，全程记录 Trace】")
    tracer = simulate_agent_run()

    print("\n【输出追踪报告】")
    print(tracer.report())

    print("\n✅ 演示完成："
          "每个 Span 记录了「谁、花了多久、输入输出、token 数」，\n"
          "   出问题时能快速定位到具体是哪一步慢、哪一步错。\n"
          "   生产环境把这份数据上报到 LangSmith / Langfuse，就有可视化面板。")
