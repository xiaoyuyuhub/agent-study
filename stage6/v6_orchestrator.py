"""
Agent V6 方向5：多 Agent 编排（Orchestrator + 流水线组网）
===========================================================
把「单个 Agent」升级为「多个 Agent 组网协作」。

【为什么需要多 Agent 编排？】

前面学的都是「一个 Agent 干所有事」。但真实业务里，一个复杂任务
往往需要多种不同能力协作，而一个 Agent 很难同时做好所有事：

  - 一个「研究员」擅长查资料、理解需求
  - 一个「程序员」擅长写代码
  - 一个「审查员」擅长挑毛病

如果塞进一个 Agent，system prompt 会又长又冲突，工具也混在一起。

【编排（Orchestration）是什么？】

编排 = 一个「指挥」把复杂任务拆解，按顺序（或并行）派给多个
「专家 Agent」，再汇总结果。指挥本身不干活，只负责调度。

  ┌─────────────────────────────────────────────┐
  │              Orchestrator（指挥）             │
  │   拆任务 → 找合适的 Agent → 派活 → 汇总       │
  └──────┬──────────────┬──────────────┬─────────┘
         ▼              ▼              ▼
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │ Researcher │ │   Coder    │ │  Reviewer  │
  │  (研究员)   │ │  (程序员)   │ │  (审查员)   │
  └────────────┘ └────────────┘ └────────────┘

【两种编排模式】

  1. 流水线（Pipeline）：上游输出 → 下游输入，串行
       研究员理解需求 → 程序员写代码 → 审查员审查

  2. 能力路由（Routing）：根据任务类型，选一个最合适的 Agent
       "查资料" → 研究员；"写代码" → 程序员

【Agent Card 能力发现（复用 V5 的 A2A 概念）】

  每个 Agent 通过「能力名片」声明自己会干什么。Orchestrator 靠
  这张名片决定「该找谁」。这复用 V5 A2A 里的 Agent Card 思想。

【本文件的「迷你 A2A 总线」】

  生产环境：每个 Agent 是独立进程/服务，通过真实 A2A HTTP 端点通信。
  本文件为了「单文件零依赖能跑」，用一个进程内的 A2ABus 类来模拟
  「Agent 组网 + 能力发现 + 派发」的完整逻辑，概念完全一致。
  真实场景把每个 RemoteAgent 换成 A2AClient（连到独立服务）即可。

运行：
  python v6_orchestrator.py
"""


# ═══════════════════════════════════════════════════════════════
#  第一部分：Agent Card（能力名片）
# ═══════════════════════════════════════════════════════════════

class AgentCard:
    """
    能力名片：声明一个 Agent 的身份和能力。

    和 V5 的 Agent Card 概念一致，只是简化成 Python 对象。
    真实 A2A 里这是一份 JSON（GET /.well-known/agent.json）。
    """

    def __init__(self, name: str, description: str, capabilities: list):
        self.name = name
        self.description = description
        self.capabilities = capabilities  # 能力标签列表，如 ["coding", "review"]

    def __repr__(self):
        return f"{self.name}({', '.join(self.capabilities)})"


# ═══════════════════════════════════════════════════════════════
#  第二部分：Remote Agent（专家 Agent）
# ═══════════════════════════════════════════════════════════════

class RemoteAgent:
    """
    一个「专家 Agent」。

    card:     能力名片
    handler:  处理函数，输入任务字符串，输出结果字符串
              （真实场景这里内部调用 LLM，本 demo 用规则模拟）
    """

    def __init__(self, card: AgentCard, handler):
        self.card = card
        self.handler = handler

    def handle(self, task: str) -> str:
        """处理一个任务（真实场景调 LLM）"""
        return self.handler(task)


# ── 三个专家 Agent 的处理逻辑（规则模拟，真实场景换成 LLM）──────

def researcher_handler(task: str) -> str:
    """研究员：理解需求、产出「需求分析」"""
    return f"[研究员] 需求分析：需要实现一个「{task}」的功能，并保证代码安全可靠。"


def coder_handler(task: str) -> str:
    """程序员：根据需求写代码"""
    return (
        "[程序员] 已实现代码：\n"
        "```python\n"
        "def calculate_price(quantity, price):\n"
        "    if quantity < 0 or price < 0:\n"
        "        raise ValueError('数量价格不能为负')\n"
        "    return quantity * price\n"
        "```"
    )


def reviewer_handler(task: str) -> str:
    """审查员：审查代码/结果"""
    if "quantity < 0" in task:
        return "[审查员] 审查通过 ✅：代码已处理负数边界，逻辑正确。"
    return "[审查员] 审查意见 ⚠️：缺少输入校验，建议补充边界检查。"


# ═══════════════════════════════════════════════════════════════
#  第三部分：A2A Bus（迷你组网总线）
# ═══════════════════════════════════════════════════════════════

class A2ABus:
    """
    迷你 A2A 总线：负责 Agent 的「注册 / 发现 / 派发」。

    这是「组网」的核心设施——所有 Agent 都注册到这里，
    Orchestrator 通过总线找到并调用它们，而不是直接硬编码。
    """

    def __init__(self):
        self.agents = []  # 已注册的 Agent 列表

    def register(self, agent: RemoteAgent):
        """注册一个 Agent 到总线"""
        self.agents.append(agent)
        print(f"  📇 注册 Agent: {agent.card}")

    def discover(self, capability: str) -> RemoteAgent:
        """
        能力发现：根据能力标签找到第一个匹配的 Agent。

        这模拟了 V5 A2A 里「GET Agent Card → 匹配能力」的过程。
        """
        for agent in self.agents:
            if capability in agent.card.capabilities:
                return agent
        raise ValueError(f"没有找到具备 [{capability}] 能力的 Agent")

    def send(self, agent: RemoteAgent, task: str) -> str:
        """给指定 Agent 派活（模拟 POST /tasks/send）"""
        print(f"  📤 派活给 {agent.card.name}: {task[:40]}...")
        return agent.handle(task)


# ═══════════════════════════════════════════════════════════════
#  第四部分：Orchestrator（指挥）
# ═══════════════════════════════════════════════════════════════

class Orchestrator:
    """
    编排器：拆任务 + 找 Agent + 派活 + 汇总。

    它自己不干活，只做调度——这是「编排」和「自己做」的本质区别。
    """

    def __init__(self, bus: A2ABus):
        self.bus = bus

    def run_pipeline(self, task: str) -> str:
        """
        流水线编排：研究员 → 程序员 → 审查员，串行执行。
        每一步的输出作为下一步的输入（上下文传递）。
        """
        print("\n🧩 [Orchestrator] 流水线编排：研究员 → 程序员 → 审查员")
        print("─" * 64)

        # ① 研究员理解需求
        researcher = self.bus.discover("research")
        analysis = self.bus.send(researcher, task)
        print(f"     ↳ {analysis}\n")

        # ② 程序员写代码（输入是研究员的需求分析）
        coder = self.bus.discover("coding")
        code = self.bus.send(coder, analysis)
        print(f"     ↳ {code}\n")

        # ③ 审查员审查（输入是程序员写的代码）
        reviewer = self.bus.discover("review")
        review = self.bus.send(reviewer, code)
        print(f"     ↳ {review}\n")

        return f"{analysis}\n{code}\n{review}"

    def route(self, task: str, capability: str) -> str:
        """
        能力路由：根据能力标签，把任务直接派给最合适的 Agent。
        适合「单一明确」的任务（不需要多步协作）。
        """
        agent = self.bus.discover(capability)
        return self.bus.send(agent, task)


# ═══════════════════════════════════════════════════════════════
#  第五部分：独立测试入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("🧩 方向5：多 Agent 编排 —— Orchestrator 组网协作演示")
    print("=" * 64)

    # ── 1. 创建总线，注册三个专家 Agent ──
    print("\n【组网】创建 A2A 总线并注册三个专家 Agent：")
    bus = A2ABus()
    bus.register(RemoteAgent(
        AgentCard("Researcher", "需求研究员", ["research"]),
        researcher_handler,
    ))
    bus.register(RemoteAgent(
        AgentCard("Coder", "代码程序员", ["coding"]),
        coder_handler,
    ))
    bus.register(RemoteAgent(
        AgentCard("Reviewer", "代码审查员", ["review"]),
        reviewer_handler,
    ))

    orchestrator = Orchestrator(bus)

    # ── 2. 流水线编排 ──
    print("\n" + "=" * 64)
    print("场景A：流水线编排（一个复杂任务，多 Agent 协作）")
    print("=" * 64)
    final = orchestrator.run_pipeline("计算商品总价")

    print("=" * 64)
    print("📦 流水线最终产出：")
    print("=" * 64)
    print(final)

    # ── 3. 能力路由 ──
    print("\n" + "=" * 64)
    print("场景B：能力路由（单一任务，直接找最合适的 Agent）")
    print("=" * 64)
    result = orchestrator.route("实现一个排序函数", capability="coding")
    print(f"     ↳ {result}")

    print("\n✅ 演示完成："
          "Orchestrator 通过「能力发现」调度多个专家 Agent 协作，"
          "而非一个 Agent 单打独斗。")
