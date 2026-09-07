"""
A2A Client Agent 客户端（a2a_client.py）
========================================
作为 Client Agent，通过 A2A 协议给 Remote Agent 派任务。

【Client Agent 的完整流程】

  ① discover()    → GET Agent Card，了解 Remote Agent 能力
  ② send_task()   → POST 发送任务，拿到 task_id
  ③ get_task()    → 轮询查询任务状态
  ④ execute()     → 完整流程：发现 → 发送 → 轮询 → 返回结果

【时序图】

  Client                          Remote Agent
    │                                  │
    │ ① GET /.well-known/agent.json    │
    │─────────────────────────────────►│
    │ ◄────────── Agent Card ──────────│  了解对方能力
    │                                  │
    │ ② POST /tasks/send               │
    │─────────────────────────────────►│
    │ ◄────── {id, status:working} ────│  任务已提交
    │                                  │
    │ ③ GET /tasks/{id}（轮询）        │
    │─────────────────────────────────►│
    │ ◄── {status:working} ────────────│  还在处理
    │  ...重复轮询...                   │
    │ ◄── {status:completed} ──────────│  处理完成
    │                                  │

【运行方式】
  先启动服务端：python a2a_server.py
  再运行客户端：python a2a_client.py
"""
import time
import requests


class A2AClient:
    """
    A2A 客户端：负责发现 Remote Agent 并给它派任务。
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        参数：
          base_url: Remote Agent 的地址
        """
        self.base_url = base_url
        self.agent_card = None  # 缓存的 Agent Card

    # ── ① 发现：获取 Agent Card ─────────────────────────────

    def discover(self) -> dict:
        """
        获取 Remote Agent 的 Agent Card（能力名片）。

        这一步就像「递名片」——先了解对方是谁、能干什么，
        再决定要不要把任务交给它。
        """
        url = f"{self.base_url}/.well-known/agent.json"
        response = requests.get(url)
        response.raise_for_status()
        self.agent_card = response.json()
        return self.agent_card

    # ── ② 发送任务 ─────────────────────────────────────────

    def send_task(self, message: str) -> dict:
        """
        给 Remote Agent 发送一个任务。

        参数：
          message: 任务内容（比如要审查的代码）

        返回：
          {"id": task_id, "status": "working"}
        """
        url = f"{self.base_url}/tasks/send"
        response = requests.post(url, json={"message": message})
        response.raise_for_status()
        return response.json()

    # ── ③ 查询任务状态 ─────────────────────────────────────

    def get_task(self, task_id: str) -> dict:
        """查询任务状态（轮询用）"""
        url = f"{self.base_url}/tasks/{task_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    # ── ④ 完整流程：发现 → 发送 → 轮询 → 返回 ──────────────

    def execute(self, message: str, poll_interval: float = 0.5) -> str:
        """
        完整执行流程：把任务派给 Remote Agent，轮询直到完成。

        参数：
          message: 任务内容
          poll_interval: 轮询间隔（秒）

        返回：
          最终结果字符串
        """
        # ① 发现能力
        print(f"① 发现 Remote Agent 能力...")
        card = self.discover()
        print(f"   发现: {card['name']} - {card['description']}")

        # ② 发送任务
        print(f"② 发送任务...")
        response = self.send_task(message)
        task_id = response["id"]
        print(f"   任务已提交，task_id={task_id}，状态={response['status']}")

        # ③ 轮询状态
        print(f"③ 轮询任务状态...")
        while True:
            task = self.get_task(task_id)
            status = task["status"]
            print(f"   当前状态: {status}")

            if status == "completed":
                print(f"④ 任务完成！")
                return task["result"]
            elif status in ("failed", "cancelled"):
                return f"任务{status}: {task['result']}"

            time.sleep(poll_interval)  # 等一下再查


# ═══════════════════════════════════════════════════════════════
#  演示入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🤝 A2A 协议演示：Client Agent → 代码审查 Remote Agent")
    print("=" * 60)

    # 创建客户端，连接 Remote Agent
    client = A2AClient("http://localhost:8000")

    # 要审查的代码（故意带点问题，让审查结果有内容）
    code_to_review = """
def calculate_price(quantity, price):
    result = eval(f"{quantity} * {price}")
    return result
"""

    print(f"\n📤 派发任务：审查以下代码")
    print("-" * 40)
    print(code_to_review)
    print("-" * 40)

    # 执行完整 A2A 流程
    result = client.execute(code_to_review)

    print(f"\n📥 收到审查结果：")
    print("-" * 40)
    print(result)
    print("-" * 40)
    print("\n✅ A2A 演示完成")
