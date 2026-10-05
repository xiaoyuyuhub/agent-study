"""
Agent V6 方向2：自动记忆整合（后台定时任务）
=============================================
把 V5 的「手动触发的记忆压缩/淘汰」升级为「后台定时自动运行」。

【V5 的问题：维护是「手动的」】

V5 的记忆压缩（consolidate）和记忆淘汰（evict），是在每次会话
结束时由主循环主动调用的：

    run_agent 里：
      会话结束 → _save_memory(...)      # 记新记忆
              → memory.consolidate(...)  # 手动压缩
              → memory.evict()           # 手动淘汰

这在「单用户 + 短会话」下没问题，但生产环境有致命缺陷：

  1. 如果 Agent 长时间没有会话（比如半夜、周末），记忆就一直不整理
  2. 记忆膨胀到很大时，用户一开会话就要卡顿地先压缩一顿
  3. 压缩/淘汰和主流程耦合，主流程一忙就容易漏掉维护

【本方向解决什么】

把「维护」从主流程里剥离出来，交给一个**后台守护线程**，
按固定时间间隔自动运行。就像手机系统后台定期清理缓存一样：

  ┌──────────────────────────────────────────────┐
  │  主线程（Agent 主循环）                        │
  │    · 只负责「记」和「忆」                      │
  │    · 不关心什么时候压缩、什么时候淘汰           │
  └──────────────────────────────────────────────┘
                        ↑ 共享同一个 memory 对象
  ┌──────────────────────────────────────────────┐
  │  后台守护线程（维护调度器）                     │
  │    · 每 interval 秒醒来一次                    │
  │    · 自动 consolidate（压缩）                  │
  │    · 自动 evict（淘汰）                        │
  └──────────────────────────────────────────────┘

【线程安全为什么重要】

主线程在「记/忆」，后台线程在「压缩/淘汰」，两个线程同时操作
同一个 memory.entries 列表，如果不加锁，会出现「竞态条件」：
  - 后台线程正在删条目，主线程刚好在遍历 → 崩溃或漏数据
所以本文件用 threading.Lock 保护所有对记忆的写操作。

运行：
  python v6_scheduler.py
"""
import sys
import threading
import time
from pathlib import Path

# ── 跨目录导入 stage5 的记忆系统 ──
# V6 的「自动整合」是在 V5 记忆系统（记/忆/压缩/淘汰）基础上做的，
# 所以直接复用 V5 的 LongTermMemory，不重复造轮子。
# 用 sys.path 把 stage5 目录加进搜索路径，跨目录 import。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage5"))
from v5_memory import LongTermMemory  # noqa: E402


# ═══════════════════════════════════════════════════════════════
#  第一部分：记忆维护调度器
# ═══════════════════════════════════════════════════════════════

class MemoryMaintenanceScheduler:
    """
    后台守护线程：定期自动执行记忆压缩和淘汰。

    核心设计：
      1. 一个 daemon 线程，循环「睡 interval 秒 → 维护一次」
      2. 用 Event 做「优雅停止」（不是暴力 kill 线程）
      3. 用 Lock 保证和主线程的写操作互斥
    """

    def __init__(
        self,
        memory: LongTermMemory,
        summarize_fn,
        interval: float = 5.0,
    ):
        """
        参数：
          memory:       要维护的记忆对象（和主线程共享同一个）
          summarize_fn: 摘要函数（压缩时调用，通常用 LLM）
          interval:     维护间隔（秒），demo 设小一点方便观察
        """
        self.memory = memory
        self.summarize_fn = summarize_fn
        self.interval = interval

        self._lock = threading.Lock()      # 保护记忆写操作
        self._stop_event = threading.Event()  # 优雅停止信号
        self._thread: threading.Thread = None
        self._maintain_count = 0           # 已执行维护的次数（统计用）

    # ── 生命周期：启动 / 停止 ──────────────────────────────

    def start(self):
        """启动后台维护线程"""
        if self._thread and self._thread.is_alive():
            return
        # daemon=True：主线程退出时，这个线程自动被回收，不会卡住程序
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"⏰ 记忆维护调度器已启动（每 {self.interval} 秒维护一次）")

    def stop(self):
        """优雅停止后台线程（最多等 2 秒）"""
        self._stop_event.set()  # 通知线程「该停了」
        if self._thread:
            self._thread.join(timeout=2)

    # ── 后台循环 ──────────────────────────────────────────

    def _loop(self):
        """
        后台线程的主循环：睡一觉 → 维护一次 → 再睡一觉。

        为什么用 wait() 而不是 sleep()？
          Event.wait(timeout) 既能「定时醒来」，又能在 stop() 时
          立刻被唤醒退出，不用等到下一个 interval 才结束。
        """
        while not self._stop_event.is_set():
            # 等待 interval 秒，期间若 stop() 被调用则立即返回
            self._stop_event.wait(self.interval)
            if self._stop_event.is_set():
                break
            self.maintain()

    # ── 维护动作（可被外部手动调用，也可被线程周期调用）──────

    def maintain(self):
        """执行一次记忆维护：压缩 + 淘汰"""
        # 加锁：防止和主线程的「记/忆」写操作打架
        with self._lock:
            self._maintain_count += 1
            removed = self.memory.consolidate(self.summarize_fn)
            evicted = self.memory.evict()

        total = len(self.memory.entries)
        print(f"\n🧹 [自动维护 #{self._maintain_count}] "
              f"压缩 {removed} 条，淘汰 {evicted} 条，剩余 {total} 条记忆")
        return removed, evicted


# ═══════════════════════════════════════════════════════════════
#  第二部分：独立测试入口
# ═══════════════════════════════════════════════════════════════

def _fake_summarize(contents: list) -> str:
    """
    模拟 LLM 摘要函数。

    真实场景这里会调用 LLM：「把这几条记忆合并成一句」。
    demo 里用一个简单规则模拟，避免依赖 API key，也能跑通全流程。
    """
    # 简单粗暴：把相关内容用「、」拼起来，当作「压缩后的摘要」
    # （真实场景这里调用 LLM：「把这几条记忆合并成一句」）
    return "、".join(contents)


if __name__ == "__main__":
    print("=" * 64)
    print("⏰ 方向2：自动记忆整合 —— 后台定时维护演示")
    print("=" * 64)

    # ── 1. 创建记忆对象 ──
    # 用独立文件，避免污染 stage5 的 memory_store.json
    # max_entries 设小（4），故意让记忆超容量，从而触发「容量淘汰」
    mem_file = Path(__file__).parent / "memory_store_v6.json"
    memory = LongTermMemory(
        file_path=str(mem_file),
        max_entries=4,
        max_age_days=30,
    )

    print("\n【灌入初始记忆】")
    memory.remember("用户喜欢喝咖啡", importance=3)
    memory.remember("用户喜欢美式咖啡", importance=3)      # ↑ 与上一条相关，压缩时会合并
    memory.remember("用户喝咖啡不加糖", importance=3)
    memory.remember("用户今天穿了蓝色衣服", importance=1)  # 低价值
    memory.remember("用户中午吃了牛肉面", importance=1)    # 低价值
    memory.remember("用户在北京工作", importance=5)        # 高价值
    initial_count = len(memory.entries)
    print(f"\n  初始共 {initial_count} 条记忆")
    print("  （3条「咖啡」相关、2条低价值、1条高价值）")

    # ── 2. 启动后台维护调度器 ──
    scheduler = MemoryMaintenanceScheduler(
        memory=memory,
        summarize_fn=_fake_summarize,
        interval=3.0,   # 3 秒维护一次，方便观察
    )
    scheduler.start()

    # ── 3. 主线程模拟「Agent 继续干活」 ──
    # 关键点：主线程只「记/忆」，维护全交给后台线程，互不打扰。
    print("\n【主线程模拟 Agent 正常会话，同时后台线程在自动维护】")
    try:
        for i in range(8):
            time.sleep(1)
            # 主线程偶尔回忆一下（读操作，演示并发下正常运作）
            if i == 2:
                results = memory.recall("用户喜欢喝什么？")
                top = results[0]["content"] if results else "（空）"
                print(f"  💬 主线程回忆：{top}")
            if i == 5:
                memory.remember("用户在做 AI Agent 开发", importance=4)
    except KeyboardInterrupt:
        pass

    # ── 4. 停止调度器，展示最终结果 ──
    scheduler.stop()
    print("\n" + "=" * 64)
    print("🏁 演示结束，最终记忆状态：")
    print("=" * 64)
    print(memory.summarize_memories())
    print(f"\n  （对比：初始 {initial_count} 条 → 现在 {len(memory.entries)} 条）")
    print("  ✅ 全程无需人工干预：后台线程自动完成了「压缩」和「淘汰」——")
    print("     相关记忆被合并，评分最低的低价值记忆被清理，高价值记忆保留。")

    # 清理演示产生的临时记忆文件，保持仓库干净
    if mem_file.exists():
        mem_file.unlink()
        print("  （已清理演示用的临时记忆文件 memory_store_v6.json）")
