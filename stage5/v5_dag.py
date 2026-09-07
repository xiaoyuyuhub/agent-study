"""
V5 工具依赖图 DAG 调度器（v5_dag.py）
=====================================
把「有依赖关系的工具调用」组织成 DAG，自动并行执行无依赖的部分。

【为什么需要工具依赖图？】

V4 的并行有个粗暴的假设：LLM 同一次返回的所有工具调用都是「无依赖」的，
可以全部并行。但真实场景中，工具之间常常有依赖：

  例：「查用户ID → 用ID查订单 → 根据订单生成报表」
    - 查订单 依赖 查用户ID 的结果
    - 生成报表 依赖 查订单 的结果
    - 这三步必须串行，不能并行

如果盲目并行，会拿到错误的结果（查订单时 ID 还没出来）。

【DAG 是什么？】

DAG = Directed Acyclic Graph（有向无环图）
  - 节点：一个工具调用
  - 边（有向）：依赖关系，A → B 表示「B 依赖 A 的输出」
  - 无环：不能有循环依赖（A 依赖 B，B 又依赖 A 是错的）

【核心思想：分层并行】

  把 DAG 分层，同一层的节点（依赖都已满足）可以并行执行：

        ┌─────┐   ┌─────┐
  第0层  │ 查天气│   │ 算数学│    ← 无依赖，并行执行
        └──┬──┘   └──┬──┘
           │          │
           ▼          ▼
        ┌─────────────┐
  第1层  │  生成建议    │    ← 依赖上面两个结果，等它们完成
        └─────────────┘

  这样既保证了依赖顺序，又最大化了并行度。

【本模块实现】
  1. DAGNode：表示一个工具调用节点
  2. DAGExecutor：拓扑排序 + 分层并行执行
  3. 支持参数引用：节点的参数可以用 "{依赖节点id}" 引用前驱的结果

  这是 LLMCompiler（斯坦福 ICML 2024）的核心思想。
"""
import json
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ═══════════════════════════════════════════════════════════════
#  第一部分：DAG 节点
# ═══════════════════════════════════════════════════════════════

class DAGNode:
    """
    DAG 的一个节点 = 一个工具调用。

    属性：
      id: 节点唯一标识（如 "weather"、"calc"）
      tool_name: 要调用的工具名
      args: 工具参数，支持引用前驱结果：
            用 "{前驱节点id}" 占位，执行时会替换成前驱节点的实际结果
      deps: 依赖的节点 id 列表（这些节点必须先执行完）

    例子：
      节点 advice 依赖 weather 和 calc 的结果：
        DAGNode(
          id="advice",
          tool_name="generate_advice",
          args={"天气": "{weather}", "计算结果": "{calc}"},  # 引用前驱
          deps=["weather", "calc"],
        )
    """

    def __init__(self, node_id: str, tool_name: str, args: dict, deps: list = None):
        self.id = node_id
        self.tool_name = tool_name
        self.args = args
        self.deps = deps or []   # 依赖的节点 id 列表
        self.result = None       # 执行结果（执行后填充）

    def __repr__(self):
        return f"DAGNode({self.id}: {self.tool_name}, deps={self.deps})"


# ═══════════════════════════════════════════════════════════════
#  第二部分：DAG 执行器
# ═══════════════════════════════════════════════════════════════

class DAGExecutor:
    """
    DAG 执行器：按依赖关系分层并行执行所有节点。

    核心算法（拓扑排序 + 分层并行）：
      1. 找出所有「依赖已满足」的节点（入度为 0）
      2. 这些节点并行执行
      3. 执行完的节点，释放它的后继节点
      4. 重复，直到所有节点执行完

    依赖检测：
      如果节点数对不上（有环），会报错。
    """

    def __init__(self, nodes: List[DAGNode], tool_map: Dict[str, callable]):
        """
        参数：
          nodes: DAG 节点列表
          tool_map: 工具名 → 函数的映射
        """
        self.nodes = {n.id: n for n in nodes}  # id → 节点
        self.tool_map = tool_map

    def execute(self) -> Dict[str, str]:
        """
        执行整个 DAG，返回 {节点id: 结果}。

        流程：
          1. 拓扑排序分层
          2. 逐层并行执行
          3. 返回所有结果
        """
        # ── 第一步：计算每个节点的「未满足依赖数」（入度）──
        # 入度 = 还有多少个依赖节点没执行完
        in_degree = {nid: len(node.deps) for nid, node in self.nodes.items()}

        # 构建依赖的反向关系：dep → [依赖它的节点们]
        dependents = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"节点 {nid} 依赖了不存在的节点 {dep}")
                dependents[dep].append(nid)

        # ── 第二步：分层并行执行 ──
        # ready = 当前「依赖都满足」可以执行的节点
        ready = [nid for nid, deg in in_degree.items() if deg == 0]

        executed_count = 0
        while ready:
            print(f"\n  ⚡ 并行执行 {len(ready)} 个节点: {ready}")

            # 这一层的所有节点并行执行
            layer_results = {}
            with ThreadPoolExecutor(max_workers=len(ready)) as executor:
                futures = {}
                for nid in ready:
                    node = self.nodes[nid]
                    future = executor.submit(self._execute_node, node)
                    futures[future] = nid

                for future in as_completed(futures):
                    nid = futures[future]
                    result = future.result()
                    layer_results[nid] = result
                    self.nodes[nid].result = result
                    print(f"    ✅ {nid} → {str(result)[:60]}")

            executed_count += len(ready)

            # ── 释放后继节点 ──
            # 这一层执行完，把依赖它的节点入度减 1
            next_ready = []
            for nid in ready:
                for dependent in dependents[nid]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_ready.append(dependent)

            ready = next_ready

        # ── 检查是否有环（执行不完）──
        if executed_count != len(self.nodes):
            unfinished = [nid for nid in self.nodes if self.nodes[nid].result is None]
            raise ValueError(f"DAG 存在循环依赖或孤立节点，未执行: {unfinished}")

        return {nid: self.nodes[nid].result for nid in self.nodes}

    def _execute_node(self, node: DAGNode) -> str:
        """
        执行单个节点。

        关键：把 args 里的 "{依赖节点id}" 引用替换成前驱节点的实际结果。
        这样节点就能拿到它依赖的数据。
        """
        # ── 替换参数引用 ──
        # args 里可能有 "{weather}" 这样的占位符，要替换成 weather 节点的结果
        resolved_args = {}
        for key, value in node.args.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                dep_id = value[1:-1]  # 去掉花括号，得到依赖节点 id
                # 用依赖节点的结果替换
                resolved_args[key] = self.nodes[dep_id].result
            else:
                resolved_args[key] = value

        # ── 执行工具 ──
        if node.tool_name not in self.tool_map:
            return f"错误: 工具 '{node.tool_name}' 不存在"

        try:
            return str(self.tool_map[node.tool_name](**resolved_args))
        except Exception as e:
            return f"工具执行出错 [{node.tool_name}]: {e}"


# ═══════════════════════════════════════════════════════════════
#  第三部分：独立测试入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("📊 工具依赖图 DAG 独立测试")
    print("=" * 60)

    # ── 定义工具函数 ──
    def get_weather(city: str) -> str:
        return f"{city}: 28°C，晴"

    def calculate(expression: str) -> str:
        return f"{expression} = {eval(expression)}"

    def generate_advice(weather: str, calc: str) -> str:
        """根据天气和计算结果生成建议（依赖前两个）"""
        return f"根据「{weather}」和「{calc}」，建议今天出门散步"

    tool_map = {
        "get_weather": get_weather,
        "calculate": calculate,
        "generate_advice": generate_advice,
    }

    # ── 构建 DAG ──
    # 三个节点：
    #   weather 和 calc 无依赖（可并行）
    #   advice 依赖 weather 和 calc（要等它们完成）
    nodes = [
        DAGNode(
            node_id="weather",
            tool_name="get_weather",
            args={"city": "北京"},
            deps=[],  # 无依赖
        ),
        DAGNode(
            node_id="calc",
            tool_name="calculate",
            args={"expression": "6*7"},
            deps=[],  # 无依赖
        ),
        DAGNode(
            node_id="advice",
            tool_name="generate_advice",
            args={
                "weather": "{weather}",   # 引用 weather 节点的结果
                "calc": "{calc}",         # 引用 calc 节点的结果
            },
            deps=["weather", "calc"],     # 依赖这两个节点
        ),
    ]

    # ── 执行 ──
    print("\nDAG 结构：")
    print("  weather(查天气) ─┐")
    print("                    ├─→ advice(生成建议)")
    print("  calc(算数学) ─────┘")
    print("  其中 weather 和 calc 可并行，advice 等它们完成")

    executor = DAGExecutor(nodes, tool_map)
    results = executor.execute()

    print("\n最终结果：")
    for nid, result in results.items():
        print(f"  {nid}: {result}")

    print("\n✅ 测试完成：weather 和 calc 并行执行，advice 等它们完成后才执行")
