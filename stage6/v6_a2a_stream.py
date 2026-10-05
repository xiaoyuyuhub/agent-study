"""
Agent V6 方向4：A2A 流式响应（SSE 实时推送任务进度）
=====================================================
把 V5 的 A2A「轮询」升级为「SSE 流式推送」。

【V5 的问题：轮询是「反复敲门问」】

V5 的 A2A 客户端用轮询（polling）拿结果：

    while True:
        task = get_task(task_id)   # GET /tasks/{id}
        if task["status"] == "completed":
            return task["result"]
        time.sleep(0.5)            # 每隔 0.5 秒问一次「好了吗？」

轮询的问题：
  1. 低效：Client 要反复发请求，即使任务还在跑，也在空问
  2. 有延迟：最多要等一个轮询周期才知道结果出来了
  3. 拿不到「中间进度」：只能看到 working → completed，看不到细节

【SSE 是什么？】

SSE = Server-Sent Events（服务器推送事件）。
它是 HTTP 的一种用法：服务端保持连接不关闭，持续往客户端「推」数据。

  轮询  = 客户端反复问「好了吗？」（客户端主动拉）
  SSE   = 服务端主动说「我进行到 X 了」（服务端主动推）

【SSE 的报文格式】

  服务端持续发送这样的文本块（用两个换行分隔事件）：

    data: {"event": "progress", "step": "正在检查 eval 风险"}

    data: {"event": "progress", "step": "正在检查硬编码密码"}

    data: {"event": "done", "result": "审查完成..."}

  客户端用流式 HTTP 请求逐行读取，实时收到每一条推送。

【本文件实现】

  一个文件搞定两端，用 --mode 切换：
    python v6_a2a_stream.py --mode server   # 启动 SSE 服务端
    python v6_a2a_stream.py --mode client   # 启动 SSE 客户端（另开终端）

【和 MCP 的关系】

  V4 学过 MCP（Agent 连工具），V5 学过 A2A（Agent 连 Agent）。
  本方向在 A2A 之上，把「拿结果的方式」从轮询升级为流式推送。

运行（两个终端）：
  终端1：python v6_a2a_stream.py --mode server
  终端2：python v6_a2a_stream.py --mode client
"""
import json
import uuid
from datetime import datetime

# ── 惰性导入：只有 server 模式才需要 FastAPI，client 模式只需要 requests ──


# ═══════════════════════════════════════════════════════════════
#  第一部分：共享工具函数（两端都会用到）
# ═══════════════════════════════════════════════════════════════

def sse_event(event: str, data: dict) -> str:
    """
    把一个事件封装成 SSE 报文。

    SSE 报文格式：`data: <JSON字符串>\n\n`
    （两个换行符标志一个事件的结束）
    """
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False)}\n\n"


def fake_code_review(code: str) -> list:
    """
    模拟代码审查，返回「审查步骤列表」。
    每个元素是 (步骤描述, 是否有问题)。
    真实场景这里会逐步调用 LLM 分析，这里用规则模拟，逐步产出进度。
    """
    steps = [
        ("解析代码结构", True),
        ("检查安全风险（eval）", "eval" in code),
        ("检查硬编码敏感信息", ("password" in code or "secret" in code)),
        ("检查可读性", len(code.strip()) > 0),
        ("生成审查报告", True),
    ]
    return steps


def build_review_result(code: str) -> str:
    """根据审查步骤生成最终的审查意见"""
    lines = ["代码审查结果："]
    if "eval" in code:
        lines.append("  ⚠️ 安全风险：使用了 eval()，存在注入风险")
    if "password" in code or "secret" in code:
        lines.append("  ⚠️ 安全隐患：硬编码了敏感信息")
    if not code.strip():
        lines.append("  ❌ 代码为空")
    lines.append("  ✅ 整体可读性良好（模拟审查）")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  第二部分：服务端（FastAPI + SSE）
# ═══════════════════════════════════════════════════════════════

def run_server():
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import asyncio
    import uvicorn

    app = FastAPI(title="Code Review Agent (SSE)")

    class TaskRequest(BaseModel):
        message: str = ""

    # 任务存储（内存字典）
    tasks = {}

    @app.post("/tasks/send")
    def send_task(request: TaskRequest):
        """创建任务，返回 task_id（之后用 SSE 流式获取进度）"""
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "id": task_id,
            "message": request.message,
            "status": "working",
            "created_at": datetime.now().isoformat(),
        }
        return {"id": task_id, "status": "working"}

    @app.get("/tasks/{task_id}/stream")
    async def stream_task(task_id: str):
        """
        SSE 端点：持续推送任务进度。

        这是本方向的核心。对比 V5 的轮询端点（客户端反复 GET），
        这里服务端主动、持续地把中间进度推给客户端。
        """
        if task_id not in tasks:
            return {"error": "任务不存在"}

        code = tasks[task_id]["message"]

        async def event_generator():
            """异步生成器：逐步产出 SSE 事件"""
            # 边审查边推送进度（每个步骤之间 sleep，模拟审查耗时）
            for step_desc, _ in fake_code_review(code):
                await asyncio.sleep(0.6)
                yield sse_event("progress", {"step": step_desc})

            # 最后推送最终结果
            result = build_review_result(code)
            tasks[task_id]["status"] = "completed"
            yield sse_event("done", {"result": result})

        # media_type="text/event-stream" 是 SSE 的关键标识
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    print("启动 A2A 流式服务端（SSE）...")
    print("  SSE 端点示例: GET http://localhost:8000/tasks/<id>/stream")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ═══════════════════════════════════════════════════════════════
#  第三部分：客户端（SSE 消费）
# ═══════════════════════════════════════════════════════════════

def run_client():
    import requests

    base = "http://localhost:8000"

    # 要审查的代码（故意带 eval，让审查结果有内容）
    code = """
def calculate_price(quantity, price):
    result = eval(f"{quantity} * {price}")
    return result
"""

    print("=" * 64)
    print("📡 SSE 客户端：给代码审查 Agent 派任务，流式接收进度")
    print("=" * 64)
    print("\n① 发送任务...")
    resp = requests.post(f"{base}/tasks/send", json={"message": code})
    task_id = resp.json()["id"]
    print(f"   任务已提交，task_id={task_id}")

    print("\n② 建立 SSE 连接，实时接收服务端推送...")
    print("-" * 64)

    # ★ 关键：stream=True 让 requests 以流式方式读取，边收边处理
    with requests.get(f"{base}/tasks/{task_id}/stream", stream=True) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            # 去掉 "data: " 前缀，解析 JSON
            payload = json.loads(line[5:].strip())
            event = payload["event"]
            if event == "progress":
                print(f"   📶 进度: {payload['step']}")
            elif event == "done":
                print("-" * 64)
                print("   ✅ 收到最终结果：")
                print(payload["result"])
                break

    print("\n③ 完成！对比 V5 的轮询：客户端不需要反复问，服务端主动推。")


# ═══════════════════════════════════════════════════════════════
#  第四部分：入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A2A 流式响应（SSE）")
    parser.add_argument("--mode", choices=["server", "client"], required=True,
                        help="server=启动 SSE 服务端, client=启动 SSE 客户端")
    args = parser.parse_args()

    if args.mode == "server":
        run_server()
    else:
        run_client()
