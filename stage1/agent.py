"""
最小可运行 Agent 示例
=====================
不依赖 LangChain 等框架，只用 OpenAI SDK 的 function calling，
完整展示 Agent 的 ReAct 循环：思考 -> 调工具 -> 再思考 -> ... -> 完成

运行前：
  pip install openai
  把你的 base_url / api_key / model 填进 config.json

运行：
  python agent.py
  或在 VSCode 里按 F5 断点调试
"""
import os
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI


# ============================================================
# 第零部分：读配置（url 和 token 从 config.json 里读，不写死在代码里）
# ============================================================
# 配置文件在项目根目录（本文件在 stage1/ 子目录，所以要向上找一级）
CONFIG_PATH = Path(__file__).parent.parent / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

client = OpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
)
DEFAULT_MODEL = config["model"]

# ============================================================
# 第一部分：工具定义（代码负责的部分 —— 真正"干活"的函数）
# ============================================================

def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """安全地计算数学表达式，如 '3 * (2 + 4)'"""
    try:
        # 只允许数字和运算符，防止注入
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"不支持的字符: {expression}"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


def get_weather(city: str) -> str:
    """模拟天气查询（这里返回假数据，真实场景换成调天气API）"""
    fake_weather = {
        "北京": "28°C，晴",
        "上海": "26°C，多云",
        "深圳": "32°C，雷阵雨",
    }
    return fake_weather.get(city, f"{city}: 暂无数据（这是mock，假装查到了）")


# 把工具登记成 LLM 能看懂的格式（名字 + 描述 + 参数定义）
# ⭐ 这里的 description 非常关键，LLM 就是靠它判断"该不该调这个工具"
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
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '3 * (2 + 4)'",
                    }
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
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 '北京'"}
                },
                "required": ["city"],
            },
        },
    },
]

# 工具名 -> Python 函数 的映射表，代码执行时靠它找到真正要跑的函数
TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}


# ============================================================
# 第二部分：Agent 主循环（核心！看清楚 LLM 和代码各干什么）
# ============================================================
def _dump_messages(messages):
    """把发给 LLM 的完整 messages 漂亮地打印出来"""
    print(f"\n📤 【发给 LLM 的完整上下文】共 {len(messages)} 条消息：")
    print("─" * 60)
    for i, m in enumerate(messages):
        role = getattr(m, "role", m.get("role") if isinstance(m, dict) else "?")
        # 把 message 对象转成 dict 方便看
        if hasattr(m, "model_dump"):
            m_dict = m.model_dump()
        else:
            m_dict = m if isinstance(m, dict) else {}

        if role == "system":
            print(f"  [{i}] system: {m_dict.get('content', '')[:100]}...")
        elif role == "user":
            print(f"  [{i}] user: {m_dict.get('content', '')}")
        elif role == "assistant":
            tool_calls = m_dict.get("tool_calls") or []
            content = m_dict.get("content") or ""
            if tool_calls:
                calls = ", ".join(
                    f"{tc['function']['name']}({tc['function']['arguments']})"
                    for tc in tool_calls
                )
                print(f"  [{i}] assistant: [调工具] {calls}")
            else:
                print(f"  [{i}] assistant: {content[:80]}{'...' if len(content) > 80 else ''}")
        elif role == "tool":
            print(f"  [{i}] tool结果: {m_dict.get('content', '')}")
        else:
            print(f"  [{i}] {role}: {m_dict}")
    print("─" * 60)


def _dump_response(response):
    """把 LLM 的原始返回完整打印出来"""
    choice = response.choices[0]
    msg = choice.message
    print(f"\n📥 【LLM 原始返回】")
    print("─" * 60)
    print(f"  finish_reason: {choice.finish_reason}")
    print(f"  role: {msg.role}")
    print(f"  content: {msg.content!r}")  # !r 显示原始字符串，包含换行
    if msg.tool_calls:
        print(f"  tool_calls: (共 {len(msg.tool_calls)} 个)")
        for i, tc in enumerate(msg.tool_calls):
            print(f"    [{i}] id={tc.id}")
            print(f"        name={tc.function.name}")
            print(f"        arguments={tc.function.arguments}")
    else:
        print(f"  tool_calls: None  ← 没有调工具，说明 LLM 认为任务完成了")
    if response.usage:
        u = response.usage
        print(f"  usage: prompt={u.prompt_tokens}, completion={u.completion_tokens}, total={u.total_tokens}")
    print("─" * 60)


def run_agent(task: str, model: str = DEFAULT_MODEL):
    """
    这就是 Agent 的灵魂 —— 一个 while 循环：
      1. 把对话历史发给 LLM        ← 代码干的事
      2. LLM 决定"下一步做什么"    ← LLM 干的事（规划）
      3. 如果 LLM 要调工具，代码去执行 ← 代码干的事
      4. 把结果喂回 LLM，继续循环   ← 代码干的事
      5. LLM 觉得完成了就停         ← LLM 干的事（判断完成）
    """
    messages = [
        {
            "role": "system",
            "content": "你是一个能调用工具的助手。遇到需要计算、查时间、查天气的任务，主动调用对应工具，不要自己瞎编。",
        },
        {"role": "user", "content": task},
    ]

    step = 0
    while True:
        step += 1
        print(f"\n{'='*60}")
        print(f"🔄 第 {step} 轮：把历史发给 LLM，让它思考下一步")
        print(f"{'='*60}")

        # 打印发给 LLM 的完整上下文（关键信息！能看到 messages 是怎么累积的）
        _dump_messages(messages)

        # ⭐⭐⭐ 最关键的一行：这一步才是真正的"规划"，LLM 在这里思考 ⭐⭐⭐
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # 打印 LLM 的完整原始返回（关键信息！能看到 LLM 到底想了什么）
        _dump_response(response)

        # 情况A：LLM 想调用工具
        if msg.tool_calls:
            messages.append(msg)  # 先把 LLM 的"我要调工具"这个决定记进历史
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"🤖 LLM 决定调用工具: {name}({args})")

                # 代码负责真正执行工具
                result = TOOL_MAP[name](**args)
                print(f"🛠️  代码执行结果: {result}")

                # 把工具结果塞回历史，LLM 下一轮就能看到
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            # 继续循环，让 LLM 看着工具结果再思考下一步
            continue

        # 情况B：LLM 不调工具了，说明它觉得任务完成了，直接给最终回答
        print(f"\n{'='*60}")
        print(f"🏁 任务结束！这是最后一轮，LLM 没有再调工具，直接给了最终回答")
        print(f"{'='*60}")
        print(f"\n📋 最终回答原文:")
        print("─" * 60)
        print(msg.content)
        print("─" * 60)
        return msg.content


# ============================================================
# 第三部分：跑起来
# ============================================================
if __name__ == "__main__":
    # 这个任务会迫使 Agent 调用多个工具、做多步推理
    task = "现在几点了？帮我算一下 (123 + 456) * 2 等于多少，再查一下北京和上海的天气。最后总结一下。"

    print(f"📋 用户任务: {task}")
    run_agent(task)
