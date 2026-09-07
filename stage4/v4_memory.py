"""
V4 长期记忆模块（v4_memory.py）
================================
让 Agent 拥有「跨会话记忆」，告别金鱼记忆。

【背景：LLM 是无状态的】
LLM 每次对话都从零开始，不记得之前的对话。
第一到第三阶段的 Agent，关掉程序就「失忆」了。

【长期记忆解决什么问题】
  会话 A（周一）：用户说「我叫小明，我在北京工作，喜欢喝咖啡」
  会话 B（周五）：用户问「我喜欢喝什么？」
  期望：Agent 回答「你喜欢喝咖啡」  ← 这就是长期记忆的价值

【多层记忆架构（2026 工业标准）】

  ┌───────────────────────────────┐
  │ 短期记忆（工作记忆）            │
  │  = 当前对话的 messages 列表     │
  │  · 只存在于本次会话             │
  │  · 会话结束就没了               │
  │  · 上下文窗口有限，塞不下全部    │
  ├───────────────────────────────┤
  │ 长期记忆                       │
  │  · 持久化到磁盘，跨会话保存      │
  │  · 记住重要信息，下次能回忆      │
  │  · 通过检索（而非全量塞入）召回  │
  └───────────────────────────────┘

【本模块实现（零依赖教学版）】
  - 存储：JSON 文件（生产环境用向量数据库，如 Qdrant / Milvus）
  - 检索：TF-IDF + 余弦相似度（生产环境用 embedding 模型）
  - 三个核心操作：
      remember(content, importance)  → 记住一条信息
      recall(query, top_k)           → 回忆相关信息
      forget(entry_id)               → 遗忘一条信息

【和 v2 RAG 的关系】
  v2 的 KnowledgeBase = 「只读的外部知识」（产品FAQ、公司文档）
  v4 的 LongTermMemory = 「可读写的记忆」（用户偏好、历史事实）
  两者底层检索逻辑一样（TF-IDF + 余弦相似度），但语义不同：
  RAG 回答「知识」，记忆回答「关于你的知识」。
"""
import json
import math
import re
import time
import uuid
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  第一部分：文本向量化工具（复用 v2 的 TF-IDF 思路）
# ═══════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list:
    """
    中文分词（简化版）：中文逐字切，英文按词切。

    真实场景用 jieba 等专业分词库，这里为了零依赖用正则简单处理。
    例："小明喜欢咖啡" → ["小","明","喜","欢","咖","啡"]
    """
    text = text.lower().strip()
    english_words = re.findall(r"[a-z0-9]+", text)          # 英文单词
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)    # 中文字符
    return english_words + chinese_chars


def _build_tfidf(documents: list) -> list:
    """
    给一批文档构建 TF-IDF 向量。

    TF-IDF 直觉：
      - TF（词频）：词在文档里出现越多越重要
      - IDF（逆文档频率）：词在所有文档都出现就不重要（如"的""是"）
      - 两者相乘 = 词在这篇文档的权重
    """
    tokenized = [_tokenize(d) for d in documents]

    # 统计每个词出现在几篇文档里
    doc_freq = {}
    for tokens in tokenized:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # 算 IDF
    num_docs = len(documents)
    idf = {t: math.log(num_docs / (f + 1)) for t, f in doc_freq.items()}

    # 算 TF-IDF 向量
    vectors = []
    for tokens in tokenized:
        total = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vectors.append({t: (c / total) * idf[t] for t, c in tf.items()})
    return vectors


def _cosine(a: dict, b: dict) -> float:
    """余弦相似度，值域 [-1, 1]，越接近 1 越相似"""
    dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ═══════════════════════════════════════════════════════════════
#  第二部分：长期记忆类
# ═══════════════════════════════════════════════════════════════

class LongTermMemory:
    """
    长期记忆：把重要信息持久化到磁盘，跨会话保存。

    数据结构（每条记忆）：
      {
        "id": "唯一标识",
        "content": "记忆内容，如「用户叫小明」",
        "timestamp": 记录时间戳,
        "importance": 重要度（1~5，越高越不容易被遗忘）
      }

    一个真实的记忆文件长这样：
      [
        {"id": "a1b2", "content": "用户叫小明，在北京工作", "timestamp": 1750000000, "importance": 5},
        {"id": "c3d4", "content": "用户喜欢喝咖啡", "timestamp": 1750000100, "importance": 3}
      ]
    """

    def __init__(self, file_path: str = None):
        """
        初始化：加载磁盘上已有的记忆。

        参数：
          file_path: 记忆文件路径。默认存在本文件所在目录（stage4/），
                     这样无论从哪里运行，记忆文件位置都固定。
        """
        if file_path is None:
            # 默认：记忆文件放在本模块所在目录，保证位置稳定
            file_path = Path(__file__).parent / "memory_store.json"
        self.file_path = Path(file_path)
        self.entries = self._load()   # 从磁盘加载记忆列表
        print(f"🧠 长期记忆已加载: {len(self.entries)} 条记忆")

    # ── 内部方法：磁盘读写 ──────────────────────────────────────

    def _load(self) -> list:
        """
        从 JSON 文件加载记忆。

        为什么需要这个方法？
          程序重启后，内存里的记忆没了，必须从磁盘读回来。
          这就是「持久化」的意义——数据不随进程结束而消失。
        """
        if self.file_path.exists():
            try:
                return json.loads(self.file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # 文件损坏时返回空列表，不让程序崩溃
                return []
        return []  # 文件不存在 = 第一次运行 = 没有记忆

    def _save(self):
        """
        把记忆写回磁盘。

        每次 remember / forget 后都要调用，否则数据只在内存里，
        程序一关就丢了。
        """
        self.file_path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 三个核心操作：记 / 忆 / 忘 ─────────────────────────────

    def remember(self, content: str, importance: float = 1.0) -> str:
        """
        记住一条信息。

        参数：
          content: 记忆内容，如「用户叫小明，在北京工作」
          importance: 重要度 1~5，越高越重要（决定淘汰时的优先级）

        流程：
          1. 生成唯一 id
          2. 构造记忆条目
          3. 追加到列表
          4. 保存到磁盘
        """
        entry = {
            "id": uuid.uuid4().hex[:8],      # 8位随机唯一id
            "content": content,
            "timestamp": time.time(),        # 记录时间（用于时间衰减）
            "importance": importance,
        }
        self.entries.append(entry)
        self._save()
        print(f"  💾 记住: {content}")
        return entry["id"]

    def recall(self, query: str, top_k: int = 3) -> list:
        """
        回忆相关信息。

        参数：
          query: 查询问题，如「用户喜欢什么？」
          top_k: 返回最相关的几条

        原理（和 RAG 检索一模一样）：
          1. 把 query 变成 TF-IDF 向量
          2. 和每条记忆算余弦相似度
          3. 按相似度排序，取 top_k
        """
        if not self.entries:
            return []

        # 所有记忆的文本 + query 一起向量化（保证词表一致）
        contents = [e["content"] for e in self.entries]
        all_vecs = _build_tfidf(contents + [query])
        mem_vecs = all_vecs[:-1]   # 记忆向量
        query_vec = all_vecs[-1]   # 查询向量

        # 逐条算相似度
        scored = []
        for i, vec in enumerate(mem_vecs):
            score = _cosine(query_vec, vec)
            scored.append((score, self.entries[i]))

        # 排序取 top_k
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {"score": round(s, 3), "content": e["content"], "id": e["id"]}
            for s, e in scored[:top_k]
        ]

    def forget(self, entry_id: str) -> bool:
        """
        遗忘一条记忆。

        参数：
          entry_id: 要删除的记忆 id

        返回：
          是否成功删除

        什么时候用？
          - 用户要求删除某条信息（隐私合规）
          - 记忆过时了需要清理
        """
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["id"] != entry_id]
        if len(self.entries) < before:
            self._save()
            return True
        return False

    def summarize_memories(self) -> str:
        """
        把所有记忆格式化成文本，用于注入 prompt。

        会话开始时，把相关记忆召回后，拼成一段文字，
        塞进 system prompt，让 LLM「知道」用户是谁。

        例：
          「以下是关于用户的长期记忆：
           1. 用户叫小明，在北京工作
           2. 用户喜欢喝咖啡」
        """
        if not self.entries:
            return "（暂无长期记忆）"

        lines = ["以下是关于用户的长期记忆："]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"{i}. {e['content']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  第三部分：独立测试入口（不依赖 agent_v4.py 也能单独跑）
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 这个测试演示了长期记忆的完整生命周期
    print("=" * 60)
    print("🧠 长期记忆独立测试")
    print("=" * 60)

    mem = LongTermMemory()  # 用默认路径（stage4/ 下的 memory_store.json）

    # 1. 记住几条信息（模拟用户第一次会话）
    print("\n【第一次会话】用户透露个人信息")
    mem.remember("用户叫小明，在北京工作", importance=5)
    mem.remember("用户喜欢喝咖啡，不喜欢喝茶", importance=3)
    mem.remember("用户在做 AI Agent 开发项目", importance=4)

    # 2. 回忆（模拟第二次会话开始，检索相关记忆）
    print("\n【第二次会话】用户问问题，Agent 回忆")
    results = mem.recall("用户喜欢喝什么？")
    print(f"  查询「用户喜欢喝什么？」召回 {len(results)} 条:")
    for r in results:
        print(f"    [相关度 {r['score']}] {r['content']}")

    # 3. 展示全部记忆
    print("\n【全部记忆】")
    print(mem.summarize_memories())

    # 4. 遗忘一条
    print("\n【遗忘】删除「喜欢喝茶」相关记忆")
    mem.remember("临时测试记忆，稍后删除", importance=1)
    # 找到刚才加的那条
    for e in mem.entries:
        if "临时测试" in e["content"]:
            mem.forget(e["id"])
            print(f"  已删除: {e['content']}")
            break

    print("\n✅ 测试完成，记忆已持久化到 memory_store.json")
