"""
V4 MCP 工具服务器（v4_mcp_server.py）
=====================================
用 Model Context Protocol (MCP) 标准暴露工具，实现「工具即服务」。

【MCP 是什么？】
MCP（Model Context Protocol，模型上下文协议）是 Anthropic 在 2024 年底
提出、2026 年已成为行业事实标准的「工具标准化协议」。

它解决的核心痛点：
  - 传统方式：每个 Agent 都要自己写工具调用代码，工具写死在 Agent 里
  - MCP 方式：工具由「独立的 MCP Server」提供，任何 Agent 都能通过
    「MCP Client」动态发现并调用，一次编写，处处复用

【类比理解】
  传统 Function Calling ≈ 每家餐馆自己雇厨师
  MCP                ≈ 统一的外卖平台（商家上架菜品，任何App都能点）

【MCP 的四层架构】

  ┌───────────────────────────────────┐
  │ 应用层：Client / Server / Agent    │
  ├───────────────────────────────────┤
  │ 协议层：Tools / Resources /        │
  │          Prompts / Sampling        │
  ├───────────────────────────────────┤
  │ 消息层：JSON-RPC 2.0              │
  ├───────────────────────────────────┤
  │ 传输层：stdio / HTTP+SSE / WS      │
  └───────────────────────────────────┘

【本文件是什么】
本文件是一个「MCP Server」，用官方 SDK 的 FastMCP 高层 API 定义工具。
它独立于 agent_v4.py 运行，agent_v4.py 通过 MCP Client 连接它。

【运行方式】
  本文件通常不直接手动运行，而是由 MCP Client（在 agent_v4.py 里）
  作为子进程启动。但你也可以单独测试：
    python v4_mcp_server.py

【依赖】
  pip install mcp>=2.0.0   （mcp 2.x，2026年7月后的新版）
  注意：mcp 1.x 用 FastMCP，mcp 2.x 已改名为 MCPServer

【版本注意】
  mcp 2.x（2026-07-28 发布）相比 1.x 有重大变化：
    - FastMCP → MCPServer（从 mcp.server.mcpserver 导入）
    - 客户端字段 camelCase → snake_case（inputSchema → input_schema）
  本文件用的是 mcp 2.x 的新 API。
"""
from mcp.server.mcpserver import MCPServer

# ═══════════════════════════════════════════════════════════════
#  创建 MCP 服务器实例
# ═══════════════════════════════════════════════════════════════
# MCPServer("名字") 创建一个服务器，名字用于标识
mcp = MCPServer("agent-tools")


# ═══════════════════════════════════════════════════════════════
#  定义工具（用装饰器 @mcp.tool() 注册）
# ═══════════════════════════════════════════════════════════════
# 对比传统 Function Calling 的定义方式：
#   传统：在 Agent 代码里写一个 JSON Schema 描述工具
#   MCP：直接在函数上打 @mcp.tool() 装饰器，SDK 自动生成描述
# 区别：MCP 的工具定义和 Agent 完全解耦，可以独立部署、热插拔

# ═══════════════════════════════════════════════════════════════
#  定义工具（用装饰器 @mcp.tool() 注册）
# ═══════════════════════════════════════════════════════════════
# ⭐ 重要设计：MCP 工具应该是「本地没有的外部服务工具」。
#   如果 MCP 服务器提供的工具和本地工具重名（比如都有 get_weather），
#   合并工具列表时会报错 "Tool names must be unique"。
#   所以这里只提供本地 Agent 没有的「外部服务」：查股票、查时间。

@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    查询股票价格（模拟外部金融数据服务）。

    这是一个「本地 Agent 没有」的工具，由 MCP 服务器动态提供。
    体现 MCP 的核心价值：外部服务即插即用，Agent 不用自己写。

    :param symbol: 股票代码，如 "AAPL"（苹果）、"TSLA"（特斯拉）
    :return: 股票价格描述
    """
    fake_prices = {
        "AAPL": "苹果公司 $235.20（+1.2%）",
        "TSLA": "特斯拉 $310.45（-0.8%）",
        "NVDA": "英伟达 $128.90（+3.5%）",
        "MSFT": "微软 $415.30（+0.5%）",
    }
    return fake_prices.get(symbol.upper(), f"{symbol}: 暂无数据（模拟）")


@mcp.tool()
def get_current_time() -> str:
    """
    获取当前的日期和时间。

    :return: 格式化的时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════
#  服务器入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 用 stdio 传输启动服务器
    # stdio = 标准输入输出，适合本地进程间通信（零网络配置）
    # MCP Client 会作为子进程启动本文件，通过 stdin/stdout 通信
    # 注意 mcp 2.x 中，传输参数在 run() 里传，而不是构造函数里
    mcp.run(transport="stdio")
