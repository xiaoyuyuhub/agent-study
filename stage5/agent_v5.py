"""
Agent V5：记忆压缩 + 记忆淘汰 + 工具依赖图
===========================================
在 V4 的基础上，增加三个生产级能力：

  1. 记忆压缩（Consolidation）：记忆太多时自动合并相关记忆
  2. 记忆淘汰（Eviction）：自动清理过时/不重要的记忆
  3. 工具依赖图（DAG）：有依赖的工具自动按序并行执行

════════════════════════════════════════════════════════════════
  架构总览
════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────┐
  │              会话开始                          │
  │  ① recall 记忆 → 注入 system prompt           │
  └──────────────────┬───────────────────────────┘
                     ↓
  ┌──────────────────────────────────────────────┐
  │              主循环（ReAct）                   │
  │  ② 调 LLM 决策                                │
  │  ③ 执行工具（可 DAG 并行）                    │
  │  ④ 结果塞回 messages                          │
  └──────────────────┬───────────────────────────┘
                     ↓
  ┌──────────────────────────────────────────────┐
  │              会话结束（记忆维护）              │
  │  ⑤ remember：提取新记忆                       │
  │  ⑥ consolidate：压缩相关记忆  ← ★新增        │
  │  ⑦ evict：淘汰过时记忆        ← ★新增        │
  └──────────────────────────────────────────────┘

运行：
  python agent_v5.py             # 主循环（含记忆管理）
  python agent_v5.py --mode dag  # 工具依赖图演示
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# 导入记忆系统和 DAG 执行器
from v5_memory import LongTermMemory
from v5_dag import DAGNode, DAGExecutor


# ═══════════════════════════════════════════════════════════════
#  第零部分：配置读取
# ═══════════════════════════════════════════════════════════════
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
DEFAULT_MODEL = config["model"]
MAX_STEPS = config.get("max_steps", 10)

# 记忆相关配置
MEMORY_MAX_ENTRIES = config.get("memory_max_entries", 15)   # 记忆容量上限
MEMORY_MAX_AGE_DAYS = config.get("memory_max_age_days", 30) # 时间淘汰阈值


# ═══════════════════════════════════════════════════════════════
#  第一部分：工具定义
# ═══════════════════════════════════════════════════════════════

def get_weather(city: str) -> str:
    """查询城市天气"""
    fake = {"北京": "28°C晴", "上海": "26°C多云", "深圳": "32°C雷阵雨"}
    return fake.get(city, f"{city}: 暂无数据")


def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"不支持的字符: {expression}"
        return f"{expression} = {eval(expression, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"计算失败: {e}"


def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工具描述 + 映射
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
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
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# ═══════════════════════════════════════════════════════════════
#  第二部分：LLM 调用（带重试）
# ═══════════════════════════════════════════════════════════════

def call_llm(messages: list, tools: list, model: str, max_retries: int = 3):
    """带重试的 LLM 调用"""
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


def llm_summarize(contents: list) -> str:
    """
    用 LLM 把多条相关记忆合并成一条摘要。

    这是「记忆压缩」的核心——让 LLM 理解多条记忆，提炼成一句。
    比如：
      输入 ["用户喜欢咖啡", "用户喜欢美式", "用户不加糖"]
      输出 "用户喜欢喝无糖美式咖啡"
    """
    prompt = (
        "把以下几条相关的用户记忆，合并成一条简洁准确的摘要（保留所有关键信息）：\n"
        + "\n".join(f"- {c}" for c in contents)
    )
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # 摘要失败就返回第一条（不崩溃）
        return contents[0]


# ═══════════════════════════════════════════════════════════════
#  第三部分：主循环（整合记忆压缩 + 淘汰）
# ═══════════════════════════════════════════════════════════════

def run_agent(task: str, model: str = DEFAULT_MODEL):
    """
    Agent V5 主循环。

    相比 V4 的新增点：
      会话结束时，不仅 remember 新记忆，还自动触发：
        - consolidate：压缩相关记忆
        - evict：淘汰过时/不重要记忆
      这两个操作让记忆库保持「小而精」，不会无限膨胀。
    """
    print(f"\n{'='*60}")
    print(f"🤖 Agent V5 启动（记忆压缩 + 淘汰 + DAG）")
    print(f"{'='*60}")

    # ── ① 会话开始：回忆记忆 ──
    memory = LongTermMemory(
        max_entries=MEMORY_MAX_ENTRIES,
        max_age_days=MEMORY_MAX_AGE_DAYS,
    )
    recalled = memory.recall(task, top_k=3)
    memory_context = ""
    if recalled:
        memory_context = "以下是关于用户的历史记忆：\n"
        for r in recalled:
            memory_context += f"- {r['content']}\n"
        print(f"\n🧠 回忆到 {len(recalled)} 条记忆")

    # ── 组装 messages ──
    messages = [
        {
            "role": "system",
            "content": (
                "你是能调用工具的助手。\n"
                f"{memory_context}\n"
                "规则：遇到查天气、算数学、查时间，调用对应工具。\n"
            ),
        },
        {"role": "user", "content": task},
    ]

    # ── ② 主循环 ──
    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n🔄 第 {step}/{MAX_STEPS} 轮")

        response = call_llm(messages, TOOLS, model)
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                print(f"  🔧 调工具: {name}({args})")
                result = TOOL_MAP[name](**args) if name in TOOL_MAP else f"工具 {name} 不存在"
                print(f"  📤 结果: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        # 任务完成
        final_answer = msg.content
        print(f"\n🏁 任务完成")
        print(f"📋 最终回答: {final_answer[:200]}...")

        # ── ③ 会话结束：记忆维护（V5 核心新增）──
        _save_memory(memory, task, final_answer, model)

        # ★ 记忆压缩：把相关记忆合并
        print(f"\n📦 记忆压缩（合并相关记忆）...")
        removed = memory.consolidate(llm_summarize)
        print(f"   压缩掉 {removed} 条记忆，当前 {len(memory.entries)} 条")

        # ★ 记忆淘汰：清理过时/不重要记忆
        print(f"\n🗑️ 记忆淘汰（清理低价值记忆）...")
        evicted = memory.evict()
        print(f"   淘汰掉 {evicted} 条记忆，当前 {len(memory.entries)} 条")

        return final_answer

    print(f"\n⚠️ 达到最大步数 {MAX_STEPS}")
    return f"达到最大步数 {MAX_STEPS}"


def _save_memory(memory: LongTermMemory, task: str, answer: str, model: str):
    """会话结束，用 LLM 提取值得记住的信息，写入记忆"""
    print(f"\n💾 提取值得记住的信息...")
    extract_prompt = f"""
从对话中提取值得长期记住的「用户个人信息或偏好」。
【用户任务】{task}
【Agent回答】{answer}
规则：只提取稳定事实（姓名、偏好、在做的事），一次性信息不要。没有就回复「无」。
每条一行，格式：- 用户xxx
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": extract_prompt}],
        )
        extracted = response.choices[0].message.content
        if extracted and extracted.strip() != "无":
            for line in extracted.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    memory.remember(line[2:], importance=4)
    except Exception as e:
        print(f"  ⚠️ 记忆提取失败: {e}")


# ═══════════════════════════════════════════════════════════════
#  第四部分：工具依赖图演示
# ═══════════════════════════════════════════════════════════════

def run_dag_demo():
    """
    工具依赖图演示：展示有依赖的工具如何按序并行执行。

    场景：查天气 + 算数学（可并行）→ 生成建议（依赖前两者）
    """
    print(f"\n{'='*60}")
    print(f"📊 工具依赖图 DAG 演示")
    print(f"{'='*60}")

    def generate_advice(weather: str, calc: str) -> str:
        return f"根据「{weather}」和「{calc}」，建议今天出门散步"

    dag_tool_map = {**TOOL_MAP, "generate_advice": generate_advice}

    # 构建 DAG：weather 和 calc 并行，advice 依赖它们
    nodes = [
        DAGNode("weather", "get_weather", {"city": "北京"}, deps=[]),
        DAGNode("calc", "calculate", {"expression": "6*7"}, deps=[]),
        DAGNode(
            "advice",
            "generate_advice",
            {"weather": "{weather}", "calc": "{calc}"},
            deps=["weather", "calc"],
        ),
    ]

    print("\nDAG 结构：")
    print("  weather(查天气) ─┐")
    print("                    ├─→ advice(生成建议)")
    print("  calc(算数学) ─────┘")

    executor = DAGExecutor(nodes, dag_tool_map)
    results = executor.execute()

    print("\n最终结果：")
    for nid, result in results.items():
        print(f"  {nid}: {result}")


# ═══════════════════════════════════════════════════════════════
#  第五部分：入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent V5 - 记忆压缩/淘汰/DAG")
    parser.add_argument("--mode", choices=["agent", "dag"], default="agent",
                        help="agent(主循环含记忆管理) 或 dag(工具依赖图演示)")
    args = parser.parse_args()

    if args.mode == "dag":
        run_dag_demo()
    else:
        # 默认任务：让 Agent 透露个人信息，触发记忆的完整生命周期
        task = (
            "帮我做几件事："
            "1. 查一下北京的天气；"
            "2. 算一下 123 * 456 等于多少；"
            "3. 现在几点了？"
            "顺便告诉你，我叫小明，喜欢喝咖啡，目前在学 AI Agent 开发。"
            "最后总结一下。"
        )
        print(f"📋 用户任务: {task}")
        run_agent(task)
