"""
V5 记忆系统（v5_memory.py）：记忆压缩 + 记忆淘汰
=================================================
在 V4 长期记忆（记/忆/忘）的基础上，增加两个生产级能力：

  1. 记忆压缩（Consolidation）：记忆太多时，把相关记忆合并成摘要
  2. 记忆淘汰（Eviction）：自动清理过时/不重要的记忆

【为什么需要压缩和淘汰？】

V4 的记忆有个致命问题：只增不减。用户用久了，记忆文件会无限膨胀：
  - 记忆越来越多 → 检索越来越慢
  - 大量过时/重复的记忆 → 检索精度下降（召回一堆废话）
  - 超过 LLM 上下文 → 无法全部注入

类比：你的笔记本记满后，要么把零散笔记整理成目录（压缩），
要么撕掉没用的页（淘汰）。V4 只会一直写，不会整理。

【记忆压缩 vs 记忆淘汰】

  记忆压缩 = 把「多条相关记忆」合并成「一条摘要」
    例：["用户喜欢咖啡", "用户喜欢美式", "用户讨厌加糖"]
       → 压缩成 "用户喜欢喝无糖美式咖啡"
    结果：数量变少，信息保留

  记忆淘汰 = 把「没价值的记忆」直接删除
    例："用户今天穿了蓝色衣服"（一次性信息，没长期价值）
       → 直接删除
    结果：数量变少，垃圾被清理

【记忆的完整生命周期】

  记住(remember) → 回忆(recall) → 压缩(consolidate) → 淘汰(evict)
      ↑                                              ↓
      └──────────────── 下一次会话 ←──────────────────┘

【本模块实现（零依赖教学版）】

  压缩：需要外部提供 summarize 函数（用 LLM 做摘要），本模块只负责调度
  淘汰：纯代码实现（时间衰减 + 重要度 + 访问频率三个维度评分）
  存储：JSON 文件（生产环境用向量数据库 + 后台定时任务）
"""
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════
#  第一部分：文本向量化工具（复用 V4 的 TF-IDF 思路）
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    """中文分词（简化版）：中文逐字 + 英文按词"""
    text = text.lower().strip()
    return re.findall(r"[a-z0-9]+", text) + re.findall(r"[\u4e00-\u9fff]", text)


def _build_tfidf(documents: list) -> list:
    """构建 TF-IDF 向量"""
    tokenized = [_tokenize(d) for d in documents]
    doc_freq = {}
    for tokens in tokenized:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    num_docs = len(documents)
    idf = {t: math.log(num_docs / (f + 1)) for t, f in doc_freq.items()}
    vectors = []
    for tokens in tokenized:
        total = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vectors.append({t: (c / total) * idf[t] for t, c in tf.items()})
    return vectors


def _cosine(a: dict, b: dict) -> float:
    """余弦相似度"""
    dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ═══════════════════════════════════════════════════════════════
#  第二部分：记忆系统类
# ═══════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    长期记忆系统（V5 增强版），在 V4 基础上增加压缩和淘汰。

    记忆条目结构（比 V4 多了 access_count 和 last_access）：
      {
        "id": "唯一标识",
        "content": "记忆内容",
        "timestamp": 创建时间戳,
        "importance": 重要度 1~5,
        "access_count": 被回忆（recall）命中的次数,   ← 新增，用于淘汰
        "last_access": 最后被回忆的时间戳              ← 新增，用于淘汰
      }
    """

    def __init__(
        self,
        file_path: str = None,
        max_entries: int = 20,       # 记忆容量上限，超过就触发压缩/淘汰
        max_age_days: int = 30,      # 超过这个天数没被访问的记忆视为过时
    ):
        """
        初始化。

        参数：
          file_path: 记忆文件路径（默认本模块所在目录）
          max_entries: 记忆容量上限，超过就压缩/淘汰
          max_age_days: 时间淘汰阈值（天）
        """
        if file_path is None:
            file_path = Path(__file__).parent / "memory_store.json"
        self.file_path = Path(file_path)
        self.max_entries = max_entries
        self.max_age_days = max_age_days
        self.entries = self._load()
        print(f"🧠 记忆系统已加载: {len(self.entries)} 条记忆")

    # ── 磁盘读写 ──────────────────────────────────────────────

    def _load(self) -> list:
        """从磁盘加载记忆"""
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def _save(self):
        """保存记忆到磁盘"""
        self.file_path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 基础操作：记 / 忆 / 忘 ───────────────────────────────

    def remember(self, content: str, importance: float = 1.0) -> str:
        """记住一条信息"""
        entry = {
            "id": uuid.uuid4().hex[:8],
            "content": content,
            "timestamp": time.time(),
            "importance": importance,
            "access_count": 0,          # 初始访问次数 0
            "last_access": time.time(),
        }
        self.entries.append(entry)
        self._save()
        print(f"  💾 记住: {content}")
        return entry["id"]

    def recall(self, query: str, top_k: int = 3) -> list:
        """回忆相关信息（同时更新访问统计，供淘汰使用）"""
        if not self.entries:
            return []

        contents = [e["content"] for e in self.entries]
        all_vecs = _build_tfidf(contents + [query])
        mem_vecs = all_vecs[:-1]
        query_vec = all_vecs[-1]

        scored = []
        for i, vec in enumerate(mem_vecs):
            score = _cosine(query_vec, vec)
            scored.append((score, self.entries[i]))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for s, e in scored[:top_k]:
            # ★ 更新访问统计：这条记忆被回忆了，说明它有用，更不该淘汰
            e["access_count"] = e.get("access_count", 0) + 1
            e["last_access"] = time.time()
            results.append({"score": round(s, 3), "content": e["content"], "id": e["id"]})

        self._save()  # 访问统计变了，保存
        return results

    def forget(self, entry_id: str) -> bool:
        """遗忘一条记忆"""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) < before:
            self._save()
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    #  ★ 新增能力1：记忆压缩（Consolidation）
    # ═══════════════════════════════════════════════════════════

    def consolidate(self, summarize_fn: Callable[[list], str]) -> int:
        """
        压缩记忆：把相关记忆合并成摘要。

        参数：
          summarize_fn: 摘要函数，输入多条记忆内容 list[str]，
                        输出合并后的一条摘要 str。
                        通常用 LLM 实现（"把这几条合并成一句"）。

        流程：
          1. 按相似度聚类（把相关的记忆分到一组）
          2. 每组调用 summarize_fn 合并成一条摘要
          3. 用摘要替换原来的多条记忆

        返回：压缩后减少的记忆条数

        为什么需要压缩？
          记忆越积越多，大量记忆其实是相关的（比如关于同一个主题的
          零散信息），可以合并成一条，既减少数量又保留信息。
        """
        if len(self.entries) < 3:
            return 0  # 记忆太少，没必要压缩

        # ── 第一步：按相似度聚类 ──
        # 把内容相近的记忆分到同一组（简单贪心聚类）
        contents = [e["content"] for e in self.entries]
        vectors = _build_tfidf(contents)

        clusters = []  # 每个元素是一组记忆的索引列表
        used = set()
        for i in range(len(self.entries)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(self.entries)):
                if j not in used and _cosine(vectors[i], vectors[j]) > 0.3:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)

        # ── 第二步：每组合并成摘要 ──
        new_entries = []
        removed = 0
        for cluster in clusters:
            if len(cluster) == 1:
                # 只有一条，不用合并，直接保留
                new_entries.append(self.entries[cluster[0]])
            else:
                # 多条相关记忆，调用 summarize_fn 合并
                group_contents = [self.entries[idx]["content"] for idx in cluster]
                try:
                    summary = summarize_fn(group_contents)
                except Exception:
                    # 摘要失败就保留第一条（不崩溃）
                    summary = group_contents[0]

                # 合并后的记忆：重要度取组内最高，访问次数累加
                max_imp = max(self.entries[idx].get("importance", 1) for idx in cluster)
                total_access = sum(self.entries[idx].get("access_count", 0) for idx in cluster)
                new_entries.append({
                    "id": uuid.uuid4().hex[:8],
                    "content": summary,
                    "timestamp": time.time(),
                    "importance": max_imp,
                    "access_count": total_access,
                    "last_access": time.time(),
                })
                removed += len(cluster) - 1  # N 条变成 1 条，减少 N-1 条

        self.entries = new_entries
        self._save()
        return removed

    # ═══════════════════════════════════════════════════════════
    #  ★ 新增能力2：记忆淘汰（Eviction）
    # ═══════════════════════════════════════════════════════════

    def evict(self) -> int:
        """
        淘汰记忆：删除过时/不重要的记忆。

        综合三个维度评分（评分越低越先被淘汰）：
          1. 重要度：importance 越低越容易淘汰
          2. 时间新鲜度：越久没被访问越容易淘汰（时间衰减）
          3. 访问频率：access_count 越低越容易淘汰

        淘汰触发条件：
          - 记忆数量超过容量上限 max_entries
          - 或者某条记忆超过 max_age_days 没被访问且重要度低

        返回：淘汰的记忆条数
        """
        now = time.time()
        removed = 0
        kept = []

        for entry in self.entries:
            age_days = (now - entry.get("last_access", entry["timestamp"])) / 86400
            importance = entry.get("importance", 1)

            # ── 条件1：时间淘汰 ──
            # 超过 max_age_days 没被访问，且重要度不高（<4），直接淘汰
            if age_days > self.max_age_days and importance < 4:
                print(f"  🗑️ 淘汰(过时): {entry['content']}（{int(age_days)}天未访问）")
                removed += 1
                continue
            kept.append(entry)

        # ── 条件2：容量淘汰 ──
        # 如果还是超过容量上限，按综合评分排序，淘汰评分最低的
        if len(kept) > self.max_entries:
            # 计算每条记忆的保留价值评分
            scored = []
            for entry in kept:
                score = self._retention_score(entry, now)
                scored.append((score, entry))
            scored.sort(key=lambda x: x[0])  # 评分低的排前面（先淘汰）

            # 淘汰评分最低的，直到数量降到容量内
            overflow = len(kept) - self.max_entries
            for score, entry in scored[:overflow]:
                print(f"  🗑️ 淘汰(容量): {entry['content']}（评分 {score:.2f}）")
                removed += 1
            kept = [entry for score, entry in scored[overflow:]]

        self.entries = kept
        self._save()
        return removed

    def _retention_score(self, entry: dict, now: float) -> float:
        """
        计算一条记忆的「保留价值」评分。

        评分越高越值得保留，越低越容易被淘汰。

        评分 = 重要度 × 2 + 时间新鲜度 + 访问频率

          - 重要度（importance 1~5）：权重最大，×2
          - 时间新鲜度：越新越高，用指数衰减（越旧越接近 0）
          - 访问频率：被回忆次数越多越高

        这个公式体现了记忆淘汰的核心直觉：
          重要的、新的、常用的记忆，才值得长期保留。
        """
        importance = entry.get("importance", 1)
        age_days = (now - entry.get("last_access", entry["timestamp"])) / 86400
        access_count = entry.get("access_count", 0)

        # 时间新鲜度：指数衰减，越旧越接近 0
        freshness = math.exp(-age_days / 10)  # 10 天衰减到 e^-1 ≈ 0.37

        # 访问频率：访问次数开根号，避免访问次数主导评分
        access_weight = math.sqrt(access_count)

        return importance * 2 + freshness * 3 + access_weight

    # ── 展示所有记忆 ──────────────────────────────────────────

    def summarize_memories(self) -> str:
        """把所有记忆格式化成文本"""
        if not self.entries:
            return "（暂无记忆）"
        lines = ["以下是关于用户的长期记忆："]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"{i}. {e['content']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  第三部分：独立测试入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 记忆系统 V5 独立测试（压缩 + 淘汰）")
    print("=" * 60)

    mem = LongTermMemory(max_entries=5, max_age_days=30)

    # 1. 记住几条「相关」的记忆（会被压缩）
    print("\n【记住相关记忆】这些内容相关，压缩时会合并")
    mem.remember("用户喜欢喝咖啡", importance=3)
    mem.remember("用户喜欢美式咖啡", importance=3)
    mem.remember("用户喝咖啡不加糖", importance=3)

    # 2. 记住几条「无关」的一次性记忆（会被淘汰）
    print("\n【记住一次性记忆】这些没长期价值，淘汰时会删除")
    mem.remember("用户今天穿了蓝色衣服", importance=1)
    mem.remember("用户中午吃了牛肉面", importance=1)

    print("\n【压缩前】共", len(mem.entries), "条记忆")
    print(mem.summarize_memories())

    # 3. 压缩：把相关记忆合并（这里用简单的拼接模拟 LLM 摘要）
    print("\n【记忆压缩】把相关记忆合并")
    def fake_summarize(contents):
        return "用户喜欢喝无糖美式咖啡"  # 模拟 LLM 摘要结果
    removed = mem.consolidate(fake_summarize)
    print(f"   压缩了 {removed} 条记忆")

    print("\n【压缩后】共", len(mem.entries), "条记忆")
    print(mem.summarize_memories())

    # 4. 淘汰：清理低价值记忆
    print("\n【记忆淘汰】清理过时/不重要记忆")
    evicted = mem.evict()
    print(f"   淘汰了 {evicted} 条记忆")

    print("\n【淘汰后】共", len(mem.entries), "条记忆")
    print(mem.summarize_memories())

    print("\n✅ 测试完成")
