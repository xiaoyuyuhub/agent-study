"""
Agent V4：长期记忆 + 工具并行 + Human-in-the-loop + MCP
========================================================
在第三阶段的基础上，整合四个高级特性：

  1. 长期记忆（Long-term Memory）  → 跨会话记住用户信息
  2. 工具并行（Parallel Tools）    → 无依赖工具同时执行，提速
  3. Human-in-the-loop（HITL）     → 关键操作暂停等人确认
  4. MCP 协议                      → 标准化工具，动态发现、热插拔

════════════════════════════════════════════════════════════════
  架构总览
════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────┐
  │                    Agent V4 主循环                      │
  │                                                       │
  │  会话开始：                                            │
  │    ① 从长期记忆 recall 相关记忆 → 注入 system prompt  │
  │                                                       │
  │  主循环 while step < max_steps：                       │
  │    ② 调用 LLM（工具列表 = 本地工具 + MCP工具）         │
  │    ③ LLM 返回多个工具调用？                           │
  │       ├─ 是 → ④ 并行执行（无依赖）                    │
  │       └─ 否 → 任务完成                                │
  │    ⑤ 工具执行前检查是否需人工确认（HITL）             │
  │    ⑥ 结果塞回 messages                                │
  │                                                       │
  │  会话结束：                                            │
  │    ⑦ 提取重要信息 → remember 到长期记忆               │
  └───────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════
  四个特性的文件分布
════════════════════════════════════════════════════════════════
  v4_memory.py       → 长期记忆模块（独立）
  v4_mcp_server.py   → MCP 服务器（独立进程）
  本文件 agent_v4.py → 主循环 + 工具并行 + HITL + MCP客户端

运行：
  python agent_v4.py
"""
import os
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── 尝试导入 MCP（如果没安装，MCP 功能降级，不影响其他功能）──
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# 导入长期记忆模块
from v4_memory import LongTermMemory


# ═══════════════════════════════════════════════════════════════
#  第零部分：配置读取
# ═══════════════════════════════════════════════════════════════
# 配置文件在项目根目录（本文件在 stage4/ 子目录，所以要向上找一级）
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
DEFAULT_MODEL = config["model"]
MAX_STEPS = config.get("max_steps", 10)
PARALLEL_TOOLS = config.get("parallel_tools", False)      # 是否并行执行工具
ENABLE_HITL = config.get("enable_hitl", True)             # 是否开启人工确认
ENABLE_MCP = config.get("enable_mcp", True)               # 是否使用 MCP 工具


# ═══════════════════════════════════════════════════════════════
#  第一部分：本地工具定义（硬编码在 Agent 里的工具）
# ═══════════════════════════════════════════════════════════════
# 对比：本地工具 vs MCP 工具
#   本地工具：直接写在本文件里，最简单，但每个 Agent 都要重复写
#   MCP 工具：在 v4_mcp_server.py 里定义，通过协议动态发现
# V4 同时保留两种，让你对比它们的差异

def get_weather(city: str) -> str:
    """查询城市天气"""
    fake_weather = {
        "北京": "28°C，晴",
        "上海": "26°C，多云",
        "深圳": "32°C，雷阵雨",
    }
    return fake_weather.get(city, f"{city}: 暂无数据")


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


def send_email(to: str, subject: str, body: str) -> str:
    """
    发送邮件（模拟危险操作，需要人工确认）。

    这是专门用来演示 Human-in-the-loop 的工具。
    发送邮件是「不可逆」操作，发错了收不回来，所以执行前要人确认。
    """
    return f"✅ 邮件已发送给 {to}，主题「{subject}」"


# ── 本地工具 → OpenAI 格式描述 ──
LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "表达式"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送邮件。直接调用此工具即可，发送前的确认流程由系统自动处理，不要在对话里口头询问用户是否发送。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人"},
                    "subject": {"type": "string", "description": "主题"},
                    "body": {"type": "string", "description": "正文"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

# 本地工具名 → 函数 的映射
LOCAL_TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "send_email": send_email,
}

# ── 需要人工确认的工具集合（HITL）──
# 凡是「不可逆」或「高风险」的操作，都应该加到这个集合里
APPROVAL_REQUIRED_TOOLS = {"send_email"}


# ═══════════════════════════════════════════════════════════════
#  第二部分：MCP 客户端（动态发现并调用远程工具）
# ═══════════════════════════════════════════════════════════════
class MCPClient:
    """
    MCP 客户端：连接 v4_mcp_server.py，动态发现并调用它的工具。

    核心价值：
      本地工具写死在代码里，MCP 工具在独立服务器里。
      通过 MCP，工具可以「热插拔」——服务器加了新工具，客户端
      下次连接自动发现，不用改 Agent 代码。

    工作流程：
      1. connect()：启动 MCP 服务器子进程，获取工具列表
      2. 把 MCP 工具转成 OpenAI 格式，合并进 tools 列表
      3. call()：调用某个 MCP 工具
    """

    def __init__(self):
        self.tools = []          # MCP 工具（OpenAI 格式）
        self.tool_names = set()  # MCP 工具名集合（用于判断工具来源）
        self.available = MCP_AVAILABLE

    def connect(self):
        """
        连接 MCP 服务器，获取工具列表（程序启动时调用一次）。

        如果没装 mcp 包，或者连接失败，降级为「只用本地工具」，
        不影响主流程。
        """
        if not MCP_AVAILABLE:
            print("⚠️  未安装 mcp 包，MCP 功能不可用（pip install mcp）")
            return

        try:
            import asyncio
            asyncio.run(self._async_connect())
            print(f"🔌 MCP 连接成功，发现 {len(self.tools)} 个工具")
        except Exception as e:
            print(f"⚠️  MCP 连接失败（降级为只用本地工具）: {e}")

    async def _async_connect(self):
        """异步连接 MCP 服务器并列出工具"""
        # stdio 传输：把 v4_mcp_server.py 作为子进程启动，通过 stdin/stdout 通信
        # ★ 用 sys.executable 而不是硬编码 "python"：
        #   因为不同电脑上 python 可能叫 python / python3，也可能在虚拟环境里。
        #   sys.executable 永远指向「当前正在运行的这个解释器」的绝对路径，
        #   保证 MCP 子进程和主进程用同一个 Python 环境（mcp 包一定在里面）。
        params = StdioServerParameters(
            command=sys.executable,         # 当前解释器路径（如 .venv/bin/python）
            args=[str(Path(__file__).parent / "v4_mcp_server.py")],  # 子进程文件（绝对路径，保证从任何目录运行都能找到）
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                # 初始化会话（MCP 握手）
                await session.initialize()

                # 列出服务器提供的所有工具（动态发现！）
                mcp_tools = await session.list_tools()

                # 把 MCP 工具转成 OpenAI 的 tools 格式
                # 注意 mcp 2.x 字段用 snake_case：input_schema（不是 inputSchema）
                self.tools = []
                self.tool_names = set()
                for tool in mcp_tools.tools:
                    self.tool_names.add(tool.name)
                    self.tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    })

    def call(self, name: str, args: dict) -> str:
        """
        调用 MCP 工具。

        注意：每次调用都重新建立连接（教学简化版）。
        生产环境应该保持长连接复用，避免反复启动子进程的开销。
        """
        import asyncio
        return asyncio.run(self._async_call(name, args))

    async def _async_call(self, name: str, args: dict) -> str:
        """异步调用 MCP 工具"""
        params = StdioServerParameters(
            command=sys.executable,          # 同 _async_connect，用当前解释器
            args=[str(Path(__file__).parent / "v4_mcp_server.py")],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                # 结果可能是多个 content，取第一个文本
                if result.content:
                    return result.content[0].text
                return str(result)


# ═══════════════════════════════════════════════════════════════
#  第三部分：工具执行（含 HITL + 并行）
# ═══════════════════════════════════════════════════════════════

def _execute_single(name: str, args: dict, mcp_client: MCPClient) -> str:
    """
    执行单个工具（判断工具来源 + HITL 确认）。

    执行顺序：
      1. 判断是本地工具还是 MCP 工具
      2. 如果是「需确认」的工具，暂停等人确认（HITL）
      3. 执行并返回结果
    """
    # ── 步骤1：HITL 人工确认 ──
    # 某些操作（发邮件、删数据、扣款）是不可逆的，执行前要人拍板
    if ENABLE_HITL and name in APPROVAL_REQUIRED_TOOLS:
        print(f"\n  🛑 即将执行【{name}】，这是不可逆操作")
        print(f"     参数: {args}")
        try:
            answer = input("     确认执行吗？输入 y 继续，其他任意键取消: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # 非交互环境（比如 PyCharm 的 Run 窗口没开「模拟终端」）
            # 读不到输入时会抛 EOFError。此时默认「取消」，避免崩溃。
            print("     ⚠️ 未检测到交互输入，默认取消该操作")
            return "❌ 用户取消了该操作（无交互输入）"
        if answer != "y":
            return "❌ 用户取消了该操作"
        print("     ✅ 用户已确认")

    # ── 步骤2：判断工具来源并执行 ──
    if name in LOCAL_TOOL_MAP:
        # 本地工具：直接调用本地函数
        try:
            return LOCAL_TOOL_MAP[name](**args)
        except Exception as e:
            return f"工具执行出错 [{name}]: {e}"
    elif mcp_client and name in mcp_client.tool_names:
        # MCP 工具：通过 MCP 客户端调用
        return mcp_client.call(name, args)
    else:
        # 工具不存在（LLM 幻觉）
        return f"错误: 工具 '{name}' 不存在"


def _execute_tool_calls(tool_calls: list, mcp_client: MCPClient) -> list:
    """
    执行一批工具调用（支持并行）。

    参数：
      tool_calls: LLM 返回的多个工具调用

    并行逻辑：
      - 如果 PARALLEL_TOOLS=True 且工具数 > 1，用线程池并行执行
      - 否则串行执行

    为什么能并行？
      多个工具调用之间如果没有依赖（一个的输出不是另一个的输入），
      就可以同时执行，总耗时 ≈ 最慢的那个，而不是所有耗时之和。

    例：查 4 个城市的天气
      串行：4 次调用，耗时 4×1秒 = 4秒
      并行：4 次同时，耗时 ≈ 1秒
    """
    results = []

    if PARALLEL_TOOLS and len(tool_calls) > 1:
        # ── 并行执行 ──
        print(f"  ⚡ 并行执行 {len(tool_calls)} 个工具...")
        # ThreadPoolExecutor：线程池，让多个任务同时跑
        with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            # 提交所有任务，拿到 future 对象（还没执行完的结果）
            futures = {}
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                future = executor.submit(_execute_single, name, args, mcp_client)
                futures[future] = tc

            # 按完成顺序收集结果
            for future in as_completed(futures):
                tc = futures[future]
                name = tc.function.name
                result = future.result()
                results.append({"name": name, "result": result, "id": tc.id})
                print(f"    ✅ {name} → {result[:60]}")
    else:
        # ── 串行执行 ──
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"  🔧 调用工具: {name}({args})")
            result = _execute_single(name, args, mcp_client)
            results.append({"name": name, "result": result, "id": tc.id})
            print(f"    📤 结果: {result[:60]}")

    return results


# ═══════════════════════════════════════════════════════════════
#  第四部分：带重试的 LLM 调用
# ═══════════════════════════════════════════════════════════════

def call_llm(messages: list, tools: list, model: str, max_retries: int = 3):
    """带指数退避重试的 LLM 调用"""
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=tools
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    raise last_error


# ═══════════════════════════════════════════════════════════════
#  第五部分：主循环（整合四个特性）
# ═══════════════════════════════════════════════════════════════

def run_agent(task: str, model: str = DEFAULT_MODEL):
    """
    Agent V4 主循环，整合四个特性。

    完整流程：
      ① 会话开始：recall 长期记忆 → 注入 system prompt
      ② 主循环：LLM 决策 → 工具执行（并行 + HITL）→ 结果塞回
      ③ 会话结束：提取重要信息 → remember 到长期记忆
    """
    print(f"\n{'='*60}")
    print(f"🤖 Agent V4 启动")
    print(f"{'='*60}")
    print(f"⚙️  配置: model={model}, 并行={PARALLEL_TOOLS}, HITL={ENABLE_HITL}, MCP={ENABLE_MCP}")

    # ── ① 会话开始：回忆长期记忆 ──
    memory = LongTermMemory()  # 用默认路径（stage4/ 下的 memory_store.json）
    recalled = memory.recall(task, top_k=3)
    memory_context = ""
    if recalled:
        memory_context = "以下是关于用户的历史记忆，回答时可以参考：\n"
        for r in recalled:
            memory_context += f"- {r['content']}\n"
        print(f"\n🧠 回忆到 {len(recalled)} 条相关记忆:")
        for r in recalled:
            print(f"    [相关度 {r['score']}] {r['content']}")

    # ── MCP 连接（发现远程工具）──
    mcp_client = None
    if ENABLE_MCP:
        mcp_client = MCPClient()
        mcp_client.connect()
        # 合并工具列表 = 本地工具 + MCP 工具（按名字去重）
        # ★ 为什么去重：OpenAI API 要求工具名唯一。
        #   如果 MCP 服务器提供的工具和本地重名，直接相加会报
        #   "Tool names must be unique" 错误。
        all_tools = LOCAL_TOOLS.copy()
        seen_names = {t["function"]["name"] for t in LOCAL_TOOLS}
        for t in mcp_client.tools:
            name = t["function"]["name"]
            if name not in seen_names:
                all_tools.append(t)
                seen_names.add(name)
    else:
        all_tools = LOCAL_TOOLS

    # ── 组装 system prompt（含记忆上下文）──
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能调用工具的助手。\n"
                f"{memory_context}\n"  # 注入长期记忆
                "规则：\n"
                "1. 遇到查天气、算数学、查股票，调用对应工具\n"
                "2. 遇到发邮件的需求，【直接调用 send_email 工具】，"
                "不要先在对话里口头询问用户，确认流程由系统自动处理\n"
                "3. 完成所有子任务后总结\n"
            ),
        },
        {"role": "user", "content": task},
    ]

    # ── ② 主循环 ──
    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n{'='*60}")
        print(f"🔄 第 {step}/{MAX_STEPS} 轮")
        print(f"{'='*60}")

        # 调用 LLM
        response = call_llm(messages, all_tools, model)
        msg = response.choices[0].message

        # ── 分支A：LLM 要调工具 ──
        if msg.tool_calls:
            messages.append(msg)

            # ③④ 执行工具（并行 + HITL）
            results = _execute_tool_calls(msg.tool_calls, mcp_client)

            # 把结果塞回上下文
            for r in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": r["id"],
                    "content": r["result"],
                })
            continue

        # ── 分支B：任务完成 ──
        final_answer = msg.content
        print(f"\n{'='*60}")
        print(f"🏁 任务完成（第 {step} 轮）")
        print(f"{'='*60}")
        print(f"\n📋 最终回答:\n{'─'*60}")
        print(final_answer)
        print(f"{'─'*60}")

        # ── ③ 会话结束：提取重要信息，写入长期记忆 ──
        _save_memory(memory, task, final_answer, model)

        return final_answer

    print(f"\n⚠️ 达到最大步数 {MAX_STEPS}")
    return f"Agent 达到最大步数 {MAX_STEPS} 仍未完成"


def _save_memory(memory: LongTermMemory, task: str, answer: str, model: str):
    """
    会话结束时，从本次对话提取值得记住的信息，写入长期记忆。

    为什么不是「无脑全部记住」？
      - 对话内容很多是废话（寒暄、中间过程）
      - 只有「关于用户的稳定事实」值得记住（姓名、偏好、项目）
      - 所以让 LLM 提取「值得长期记住的事实」，再写入

    这是记忆管理的关键：不是存全部，而是存精华。
    """
    print(f"\n💾 正在提取值得记住的信息...")
    extract_prompt = f"""
从以下对话中，提取值得长期记住的「用户个人信息或偏好」。

【用户任务】{task}
【Agent 回答】{answer}

提取规则：
1. 只提取关于用户的稳定事实（如姓名、职业、偏好、在做的事）
2. 不要提取一次性信息（如当天的天气、临时计算）
3. 如果对话中没有值得记住的信息，回复「无」
4. 每条信息单独一行，格式：- 用户xxx

输出示例：
- 用户叫小明，在北京工作
- 用户喜欢喝咖啡
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extract_prompt}],
        )
        extracted = response.choices[0].message.content

        # 逐行提取，跳过「无」和空行
        if extracted and "无" != extracted.strip():
            for line in extracted.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    memory.remember(line[2:], importance=4)
    except Exception as e:
        print(f"  ⚠️ 记忆提取失败: {e}")


# ═══════════════════════════════════════════════════════════════
#  第六部分：入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 默认任务：演示四个特性
    #   1. 查 3 个城市天气 → 工具并行
    #   2. 算数学 → 本地工具
    #   3. 查股票 → MCP 工具（get_stock_price 是 MCP 服务器提供的）
    #   4. 发邮件 → HITL（会暂停等确认）
    task = (
        "帮我做几件事："
        "1. 查一下北京、上海、深圳三个城市的天气；"
        "2. 算一下 999 * 888 等于多少；"
        "3. 查一下苹果公司（AAPL）和特斯拉（TSLA）的股票价格；"
        "4. 给张三发一封邮件，主题是「项目进展」，内容是「明天下午3点开会」；"
        "最后总结一下。"
    )

    print(f"📋 用户任务: {task}")
    run_agent(task)
