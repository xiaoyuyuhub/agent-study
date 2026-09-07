"""
A2A Remote Agent 服务端（a2a_server.py）
========================================
用 A2A 协议（Agent-to-Agent Protocol）暴露一个「代码审查 Agent」。

【A2A 协议是什么？】

A2A 是 Google 在 2026 年 3 月发布的标准，解决「Agent 之间怎么通信协作」。
与 MCP（Agent 连接工具）不同，A2A 解决的是「Agent A 给 Agent B 分配任务」。

  一句话区分：
    MCP = Agent 的「手」→ 连接外部工具干活
    A2A = Agent 的「嘴」→ 和其他 Agent 对话协作

【A2A 的 3 个实体】

  ┌─────────────┐          ┌──────────────┐
  │ Client Agent │ ─任务──► │ Remote Agent │
  │ （发起者）    │ ◄─结果── │ （执行者）    │
  └─────────────┘          └──────────────┘
         双方围绕一个「Task（任务）」协作

【A2A 的 4 个 HTTP 端点】

  GET  /.well-known/agent.json  → 获取 Agent Card（能力名片）
  POST /tasks/send              → 发送任务
  GET  /tasks/{id}              → 查询任务状态
  POST /tasks/{id}/cancel       → 取消任务

【任务状态机】

  submitted → working → completed
                     ↘ failed
                     ↘ cancelled

【本文件是什么】

本文件是一个「Remote Agent」——一个代码审查专家。
它用 FastAPI 暴露上面 4 个端点，等待 Client Agent 给它派任务。

【运行方式】

  uvicorn a2a_server:app --port 8000
  （或 python a2a_server.py）
"""
import time
import threading
import uuid
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ── FastAPI 应用 ──
app = FastAPI(title="Code Review Agent", description="A2A 协议的代码审查 Remote Agent")


# ═══════════════════════════════════════════════════════════════
#  数据模型（Pydantic）
# ═══════════════════════════════════════════════════════════════

class AgentCapability(BaseModel):
    """Agent 的一项能力"""
    id: str
    name: str
    description: str


class AgentCard(BaseModel):
    """Agent Card（数字名片）：向外界声明身份和能力"""
    name: str
    description: str
    version: str
    capabilities: list
    endpoint: str


class TaskRequest(BaseModel):
    """Client 发送的任务请求"""
    message: str = ""          # 要审查的代码
    context: dict = {}         # 可选上下文


# ═══════════════════════════════════════════════════════════════
#  Agent Card 定义
# ═══════════════════════════════════════════════════════════════

AGENT_CARD = AgentCard(
    name="Code Review Agent",
    description="专业的代码审查专家，能分析代码质量、找出 bug 和安全隐患",
    version="1.0.0",
    capabilities=[
        AgentCapability(
            id="code_review",
            name="代码审查",
            description="审查代码，找出 bug、安全隐患、可读性问题",
        ),
    ],
    endpoint="http://localhost:8000",
)


# ═══════════════════════════════════════════════════════════════
#  任务存储（内存字典，生产环境用数据库）
# ═══════════════════════════════════════════════════════════════
# 结构：{task_id: {"status": ..., "message": ..., "result": ..., "created_at": ...}}
tasks = {}


def _fake_code_review(code: str) -> str:
    """
    模拟代码审查（真实场景这里会调用 LLM 做审查）。

    这里用简单的规则模拟，返回审查意见。
    """
    review_lines = ["代码审查结果："]
    if "eval" in code:
        review_lines.append("  ⚠️ 安全风险：使用了 eval()，存在注入风险")
    if "password" in code or "secret" in code:
        review_lines.append("  ⚠️ 安全隐患：硬编码了敏感信息")
    if len(code.strip()) == 0:
        review_lines.append("  ❌ 代码为空")
    if not any(x in code for x in ["def ", "class ", "function"]):
        review_lines.append("  ℹ️ 未发现函数定义")
    review_lines.append("  ✅ 整体可读性良好（模拟审查）")
    return "\n".join(review_lines)


def _process_task(task_id: str):
    """
    后台处理任务（模拟异步审查过程）。

    真实场景：这里调用 LLM 做代码审查。
    这里：sleep 2 秒模拟「审查需要时间」，让客户端能观察到轮询过程。
    """
    time.sleep(2)  # 模拟审查耗时

    task = tasks[task_id]
    try:
        result = _fake_code_review(task["message"])
        task["status"] = "completed"
        task["result"] = result
    except Exception as e:
        task["status"] = "failed"
        task["result"] = str(e)


# ═══════════════════════════════════════════════════════════════
#  A2A 的 4 个端点
# ═══════════════════════════════════════════════════════════════

@app.get("/.well-known/agent.json")
def get_agent_card():
    """
    端点1：Agent Card（能力发现）

    Client 通过这个端点了解「这个 Remote Agent 能干什么」。
    相当于 Agent 的「数字名片」。
    """
    return AGENT_CARD.model_dump()


@app.post("/tasks/send")
def send_task(request: TaskRequest):
    """
    端点2：发送任务

    Client 提交任务，Remote Agent 返回 task_id 和初始状态。
    任务在后台异步处理，Client 需要轮询查询进度。
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "status": "working",        # 初始状态：处理中
        "message": request.message,
        "result": None,
        "created_at": datetime.now().isoformat(),
    }

    # 后台线程处理任务（模拟异步）
    thread = threading.Thread(target=_process_task, args=(task_id,))
    thread.start()

    return {"id": task_id, "status": "working"}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """
    端点3：查询任务状态

    Client 轮询这个端点，直到 status 变成 completed/failed。
    """
    if task_id not in tasks:
        return {"error": "任务不存在"}
    task = tasks[task_id]
    return {
        "id": task_id,
        "status": task["status"],
        "result": task["result"],
    }


@app.post("/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    """
    端点4：取消任务

    Client 可以取消一个还在处理中的任务。
    """
    if task_id not in tasks:
        return {"error": "任务不存在"}
    if tasks[task_id]["status"] == "working":
        tasks[task_id]["status"] = "cancelled"
        return {"id": task_id, "status": "cancelled"}
    return {"id": task_id, "status": tasks[task_id]["status"]}


# ═══════════════════════════════════════════════════════════════
#  直接运行入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("启动 A2A Remote Agent（代码审查专家）...")
    print("Agent Card 地址: http://localhost:8000/.well-known/agent.json")
    uvicorn.run(app, host="0.0.0.0", port=8000)
