"""
Agent V6 方向3：LLMCompiler（LLM 自动生成 DAG + 动态重规划）
=============================================================
把 V5 的「手工构建 DAG」升级为「LLM 自动生成 DAG」。

【V5 的问题：DAG 是「手工画」的】

V5 的 DAG 需要人肉指定每个节点、依赖关系：

    nodes = [
        DAGNode("weather", "get_weather", {"city": "北京"}, deps=[]),
        DAGNode("calc",    "calculate",   {"expression": "6*7"}, deps=[]),
        DAGNode("advice",  "generate_advice", {"weather": "{weather}", "calc": "{calc}"},
                deps=["weather", "calc"]),
    ]

这在「任务固定」时没问题，但用户的问题是千变万化的，
不可能每次都为新问题手写一份 DAG。所以需要 LLM 来「自动画图」。

【LLMCompiler 是什么？】

LLMCompiler 是斯坦福 ICML 2024 论文提出的框架，核心思想一句话：

  「让 LLM 同时做两件事：拆任务 + 画依赖图，一次输出。」

它把 Agent 的执行分成三个角色：

  ┌────────────┐    ┌────────────┐    ┌────────────┐
  │  Planner   │ →  │  Executor  │ →  │   Joiner   │
  │  LLM 拆任务 │    │  按 DAG     │    │  LLM 汇总   │
  │  + 画依赖图 │    │  分层并行   │    │  最终答案   │
  └────────────┘    └────────────┘    └────────────┘
       ▲                                   │
       │         ┌────────────┐            │
       └──────── │ Replanner  │◄───────────┘
                 │ 失败时重规划 │
                 └────────────┘

【本文件实现】

  1. LLMCompilerPlanner —— 让 LLM 输出「带依赖关系的 DAG JSON」
  2. Executor —— 复用 stage5 的 DAGExecutor（分层并行执行）
  3. Joiner —— 让 LLM 把所有结果汇总成最终答案
  4. Replanner —— 某步骤失败时，动态重新规划

【零依赖可跑的 demo】

  真实 Planner/Joiner/Replanner 都依赖 LLM（需要 API key）。
  为了让你没 key 也能看懂完整流程，本文件提供 --demo 模式：
  用「模拟的 LLM 返回」代替真实 LLM，跑通完整链路，
  包括一次「失败 → 动态重规划」的完整演示。

运行：
  python v6_llm_compiler.py --demo     # 零依赖，模拟 LLM，看完整流程
  python v6_llm_compiler.py            # 真实 LLM（需 config.json + key）
"""
import json
import sys
from pathlib import Path

# ── 跨目录导入 stage5 的 DAG 执行器（分层并行是现成的，直接复用）──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage5"))
from v5_dag import DAGNode, DAGExecutor  # noqa: E402

# ── 惰性导入 OpenAI（真实 LLM 模式才需要，demo 模式完全不碰）──
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

CONFIG_PATH = Path(__file__).parent.parent / "config.json"


# ═══════════════════════════════════════════════════════════════
#  第一部分：工具定义（Executor 真正执行的东西）
# ═══════════════════════════════════════════════════════════════

def get_weather(city: str) -> str:
    return {"北京": "28°C晴", "上海": "26°C多云"}.get(city, f"{city}: 暂无数据")


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"失败: 不支持的字符 {expression}"
        return f"{expression} = {eval(expression, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"失败: {e}"


def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_advice(weather: str, calc: str) -> str:
    """生成出行建议（依赖 weather 和 calc 的结果）"""
    return f"根据「{weather}」和「{calc}」，建议今天出门散步"


# 工具描述（给 Planner 看，让它知道能用什么工具）
TOOLS_DESC = [
    {"name": "get_weather", "desc": "查询城市天气", "args": {"city": "城市名"}},
    {"name": "calculate", "desc": "计算数学表达式", "args": {"expression": "表达式"}},
    {"name": "get_current_time", "desc": "获取当前时间", "args": {}},
    {"name": "generate_advice", "desc": "根据天气和计算结果生成建议", "args": {"weather": "天气结果", "calc": "计算结果"}},
]

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_current_time": get_current_time,
    "generate_advice": generate_advice,
}


# ═══════════════════════════════════════════════════════════════
#  第二部分：Planner（LLM 拆任务 + 画依赖图）
# ═══════════════════════════════════════════════════════════════

def _parse_plan(content: str) -> list:
    """把 LLM 返回的文本解析成 DAG JSON 列表"""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("\n", 1)[0]
    return json.loads(content)


class LLMCompilerPlanner:
    """真实 LLM 规划器：一次输出「任务 + 依赖图」"""

    @staticmethod
    def plan(task: str, client, model: str) -> list:
        tools_desc = "\n".join(
            f"- {t['name']}: {t['desc']}，参数 {t['args']}" for t in TOOLS_DESC
        )
        prompt = f"""你是任务规划专家。请把下面的任务拆解成执行步骤，并标注步骤之间的依赖关系。

【可用工具】
{tools_desc}

【任务】
{task}

【输出要求】严格输出纯 JSON 数组，每个元素是一个步骤：
- id: 唯一编号（字符串，如 "1"、"2"）
- tool: 工具名（必须是上面列出的之一）
- args: 工具参数。如果某个参数的值依赖前一个步骤的结果，用 "{{步骤id}}" 引用
- deps: 依赖的步骤 id 列表（该步骤需要的所有前置步骤）

【判断依赖的关键】问自己：这一步需要上一步的输出吗？需要就写进 deps。

示例：
[
  {{"id": "1", "tool": "get_weather", "args": {{"city": "北京"}}, "deps": []}},
  {{"id": "2", "tool": "calculate", "args": {{"expression": "6*7"}}, "deps": []}},
  {{"id": "3", "tool": "generate_advice", "args": {{"weather": "{{1}}", "calc": "{{2}}"}}, "deps": ["1", "2"]}}
]"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return _parse_plan(resp.choices[0].message.content)
        except json.JSONDecodeError:
            return []


# ═══════════════════════════════════════════════════════════════
#  第三部分：Joiner + Replanner
# ═══════════════════════════════════════════════════════════════

def _find_failed(results: dict) -> list:
    """找出执行结果里「失败」的节点 id"""
    return [nid for nid, r in results.items() if str(r).startswith("失败")]


class LLMCompilerJoiner:
    """真实 LLM 汇总器：把所有结果汇总成最终答案"""

    @staticmethod
    def join(task: str, results: dict, client, model: str) -> str:
        prompt = f"""请根据执行结果，回答用户任务。

【用户任务】{task}
【各步骤结果】{json.dumps(results, ensure_ascii=False)}

请用自然语言给出完整、准确的回答。"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


class LLMCompilerReplanner:
    """真实 LLM 重规划器：失败时重新规划失败的步骤"""

    @staticmethod
    def replan(task: str, failed_ids: list, results: dict, client, model: str) -> list:
        prompt = f"""以下步骤执行失败了，请重新规划这些步骤（可改用其他工具或修正参数）。

【任务】{task}
【失败的步骤】{failed_ids}
【已有结果】{json.dumps(results, ensure_ascii=False)}

输出纯 JSON 数组，格式同原计划（id、tool、args、deps）。"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return _parse_plan(resp.choices[0].message.content)
        except json.JSONDecodeError:
            return []


# ═══════════════════════════════════════════════════════════════
#  第四部分：完整流程编排（Planner → Executor → Joiner ⇄ Replanner）
# ═══════════════════════════════════════════════════════════════

def _build_dag(plan: list) -> list:
    """把 LLM 输出的 JSON 计划，转换成 DAGExecutor 认识的 DAGNode 列表"""
    nodes = []
    for item in plan:
        nodes.append(DAGNode(
            node_id=item["id"],
            tool_name=item["tool"],
            args=item.get("args", {}),
            deps=item.get("deps", []),
        ))
    return nodes


def run_compiler(task, client=None, model=None, max_replans=2):
    """
    LLMCompiler 完整流程。

    client/model 为 None 时走 demo（模拟 LLM），否则走真实 LLM。
    """
    print("🧠 [Planner] LLM 拆解任务 + 生成依赖图...")
    if client is None:
        plan = _demo_plan()
        print("   （demo 模式：用模拟 LLM 返回一个带依赖的 DAG）")
    else:
        plan = LLMCompilerPlanner.plan(task, client, model)
    if not plan:
        print("   ⚠️ 计划解析失败，任务终止")
        return {}

    _print_dag(plan)

    # 累计所有结果（重规划可能补充新步骤）
    all_results = {}
    for replan_count in range(max_replans + 1):
        print(f"\n⚡ [Executor] 分层并行执行（第 {replan_count + 1} 轮）...")
        executor = DAGExecutor(_build_dag(plan), TOOL_MAP)
        results = executor.execute()
        all_results.update(results)

        # 检查是否有失败步骤
        failed = _find_failed(results)
        if not failed:
            break  # 全部成功

        print(f"\n🔍 [Joiner] 发现 {len(failed)} 个失败步骤: {failed}")
        if replan_count >= max_replans:
            print("   ⚠️ 达到最大重规划次数，停止")
            break

        print("🔄 [Replanner] 重新规划失败的步骤...")
        if client is None:
            plan = _demo_replan(failed)
        else:
            plan = LLMCompilerReplanner.replan(task, failed, all_results, client, model)
        if not plan:
            break
        _print_dag(plan)

    # ── Joiner 汇总 ──
    print("\n📝 [Joiner] 汇总最终答案...")
    if client is None:
        answer = _demo_join(task, all_results)
    else:
        answer = LLMCompilerJoiner.join(task, all_results, client, model)

    print("=" * 64)
    print("最终答案：")
    print(answer)
    print("=" * 64)
    return all_results


def _print_dag(plan: list):
    """打印 DAG 结构"""
    print("   生成的 DAG 计划：")
    for item in plan:
        deps = item.get("deps", [])
        dep_str = f" ← 依赖 {deps}" if deps else ""
        print(f"     节点[{item['id']}] {item['tool']}({item.get('args', {})}){dep_str}")


# ═══════════════════════════════════════════════════════════════
#  第五部分：demo 模式的「模拟 LLM」返回
# ═══════════════════════════════════════════════════════════════
# 这些函数返回的内容，和真实 LLM 返回的 JSON 结构完全一致，
# 只是把「LLM 的推理」替换成了「写死的答案」，方便离线观察流程。

def _demo_plan() -> list:
    """
    模拟 Planner：返回一个 DAG。
    故意包含一个会失败的节点 "3"（10/0），用于演示动态重规划。
    """
    return [
        {"id": "1", "tool": "get_weather", "args": {"city": "北京"}, "deps": []},
        {"id": "2", "tool": "calculate", "args": {"expression": "6*7"}, "deps": []},
        {"id": "3", "tool": "calculate", "args": {"expression": "10/0"}, "deps": []},
        {"id": "4", "tool": "generate_advice",
         "args": {"weather": "{1}", "calc": "{2}"}, "deps": ["1", "2"]},
    ]


def _demo_replan(failed: list) -> list:
    """模拟 Replanner：把失败的 calculate(10/0) 换成查当前时间"""
    return [
        {"id": "5", "tool": "get_current_time", "args": {}, "deps": []},
    ]


def _demo_join(task: str, results: dict) -> str:
    """模拟 Joiner：把结果拼成一句回答"""
    ok = {k: v for k, v in results.items() if not str(v).startswith("失败")}
    parts = "；".join(f"{k}: {v}" for k, v in ok.items())
    return f"任务「{task}」的执行结果汇总：{parts}"


# ═══════════════════════════════════════════════════════════════
#  第六部分：入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLMCompiler 完整实现")
    parser.add_argument("--demo", action="store_true", help="零依赖演示（模拟 LLM）")
    parser.add_argument("--task", type=str, default=None, help="自定义任务")
    args = parser.parse_args()

    task = args.task or "查一下北京天气，算一下 6*7，然后给我一个出行建议。"

    # ── demo 模式：零依赖 ──
    if args.demo:
        print("=" * 64)
        print("🚀 LLMCompiler 完整流程（demo 模式，模拟 LLM）")
        print("=" * 64)
        print(f"📋 任务: {task}")
        run_compiler(task, client=None, model=None)
        sys.exit(0)

    # ── 真实 LLM 模式 ──
    if not LLM_AVAILABLE:
        print("❌ 未安装 openai，请用 --demo 模式，或 pip install openai")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print("❌ 缺少 config.json，请先复制 config.example.json 并填 key")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    model = config["model"]

    print("=" * 64)
    print("🚀 LLMCompiler 完整流程（真实 LLM 模式）")
    print("=" * 64)
    print(f"📋 任务: {task}")
    run_compiler(task, client=client, model=model)
