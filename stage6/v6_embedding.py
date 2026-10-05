"""
Agent V6 方向1：记忆向量化（Embedding 语义检索）
=================================================
把 V5 的 TF-IDF「字面检索」升级为 embedding「语义检索」。

【为什么 TF-IDF 不够用？】

V5 的记忆检索用 TF-IDF + 余弦相似度。它有个致命局限：
只认识「字面相同」的词，不认识「语义相近」的词。

  例：记忆库里存的是「用户爱喝咖啡」
      用户这次问「你喜欢喝什么饮料？」

  TF-IDF 的视角：
    查询分词 → [你, 喜, 欢, 喝, 什, 么, 饮, 料]
    记忆分词 → [用, 户, 爱, 喝, 咖, 啡]
    只有「喝」一个字重合 → 相似度极低 → 检索不到 ❌

  Embedding 的视角：
    「用户爱喝咖啡」→ 向量 [0.12, -0.45, 0.83, ...]
    「喜欢喝什么饮料」→ 向量 [0.10, -0.41, 0.79, ...]
    两个向量方向相近 → 相似度高 → 能检索到 ✅

【Embedding 是什么？】

Embedding = 把一段文字变成一串数字（向量）。
核心特性：语义相近的文字，向量也相近（在向量空间里离得近）。

  比如在一个训练好的 embedding 模型里：
    「咖啡」和「拿铁」的向量很接近
    「咖啡」和「数学」的向量很远
    甚至能做向量运算：国王 - 男人 + 女人 ≈ 王后

【向量数据库是什么？】

向量数据库 = 专门存向量 + 快速找「最相似的 N 个向量」的数据库。
本文件实现一个极简版 VectorStore（暴力搜索），生产环境用
FAISS / Chroma / Milvus / Qdrant 等。

【本文件提供什么（零依赖能跑）】

  1. EmbeddingProvider 抽象基类 —— 统一「文本 → 向量」接口
  2. HashNGramEmbedding —— 字符 n-gram 哈希，零依赖本地 embedding
     （教学演示用，能体现语义检索的思路，但不如真 embedding 精准）
  3. OpenAIEmbedding —— 真实 embedding（接 OpenAI 兼容 API）
  4. VectorStore —— 极简向量数据库（add / search）

【零依赖设计说明】

  本文件不 import openai，因此可以完全离线运行。
  OpenAIEmbedding 接收「由调用方传入的 client」，做到解耦：
  你想接 OpenAI / 通义 / 本地模型，只需传对应的 client 和 model 名。

运行：
  python v6_embedding.py
"""
import hashlib
import math
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
#  第一部分：EmbeddingProvider 抽象
# ═══════════════════════════════════════════════════════════════

class EmbeddingProvider:
    """
    Embedding 抽象基类。

    统一「文本 → 向量」的接口，让上层（VectorStore、记忆系统）
    不用关心底层用的是真 embedding 还是本地哈希 embedding。

    这是典型的「依赖倒置」：
      上层依赖抽象（EmbeddingProvider），不依赖具体实现。
      想换 embedding 方案，只换注入的实例，不改上层代码。
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量把多个文本变成向量"""
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        """把单个文本变成向量（默认调用 embed 取第一个）"""
        return self.embed([text])[0]


# ═══════════════════════════════════════════════════════════════
#  第二部分：HashNGramEmbedding（零依赖本地 embedding）
# ═══════════════════════════════════════════════════════════════

class HashNGramEmbedding(EmbeddingProvider):
    """
    字符 n-gram 哈希 embedding（零依赖的教学实现）。

    【原理】

    1. 把文本切成连续的 n 个字符的片段（n-gram）
       "咖啡" (n=3 时) → ["咖", "啡"] 不够 3，会补边界空格
       实际上 "我爱咖啡" 的 tri-gram 是：
         "  我"、" 我爱"、"我爱咖"、"爱咖啡"、"咖啡 "、"啡  "

    2. 每个 n-gram 通过哈希函数映射到一个固定维度的桶（向量下标）

    3. 在对应桶里累加计数 → 得到这条文本的「指纹向量」

    4. 做 L2 归一化，让不同长度的文本也能公平比较

    【为什么它能做语义检索？】

    语义相近的文本会共享大量 n-gram（比如「咖啡」「拿铁」都属于
    「饮品」语境时，会有一些共同的相邻字符模式），所以向量会
    在若干维度上重合，余弦相似度偏高。

    当然它远不如真正训练过的 embedding 模型，但胜在零依赖、
    可离线、能直观展示「向量化 + 相似度检索」的完整链路。

    【重要细节：为什么用 hashlib 而不是 hash()？】

    Python 内置 hash() 对字符串的结果在「每次启动程序」时都不同
    （因为 PYTHONHASHSEED 随机化，为了防止哈希碰撞攻击）。
    如果用了 hash()，同一个词这次存的向量和下次查的向量对不上，
    检索就失效了。hashlib.md5 是确定性算法，永远稳定。
    """

    def __init__(self, dim: int = 256, n: int = 3):
        """
        参数：
          dim: 向量维度（桶的数量），越大越不容易「撞桶」，但也越稀疏
          n:   n-gram 的 n，中文建议 2~3，英文建议 3~5
        """
        self.dim = dim
        self.n = n

    def _ngrams(self, text: str) -> List[str]:
        """把文本切成 n-gram 片段"""
        text = text.lower()
        # 前后补空格，让开头和结尾的字符也能组成完整的 n-gram
        padded = " " * (self.n - 1) + text + " " * (self.n - 1)
        return [padded[i:i + self.n] for i in range(len(padded) - self.n + 1)]

    def _hash_to_bucket(self, ngram: str) -> int:
        """把一个 n-gram 稳定地映射到 [0, dim) 区间"""
        digest = hashlib.md5(ngram.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for ng in self._ngrams(text):
            vec[self._hash_to_bucket(ng)] += 1.0

        # L2 归一化：让向量长度变成 1，消除文本长度带来的偏差
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ═══════════════════════════════════════════════════════════════
#  第三部分：OpenAIEmbedding（真实 embedding）
# ═══════════════════════════════════════════════════════════════

class OpenAIEmbedding(EmbeddingProvider):
    """
    真实 embedding（接 OpenAI 兼容 API）。

    用法（需要 config.json 里有 api_key，模型需支持 embeddings）：
        from openai import OpenAI
        client = OpenAI(base_url=..., api_key=...)
        emb = OpenAIEmbedding(client, model="text-embedding-3-small")
        vec = emb.embed_one("用户爱喝咖啡")  # → 1536 维向量

    本类不在此 import openai，保持文件零依赖可离线运行，
    由调用方传入已经建好的 client 对象。
    """

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        # 按输入顺序取回向量
        return [d.embedding for d in resp.data]


# ═══════════════════════════════════════════════════════════════
#  第四部分：VectorStore（极简向量数据库）
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    """
    极简向量数据库。

    只做两件事：
      add(content)     → 存一条（先向量化再存）
      search(query)    → 找最相似的 top_k 条

    生产环境用 FAISS/Chroma/Milvus 替代（它们有倒排索引、HNSW 图等
    加速结构，能在百万级向量里毫秒级检索；本实现是暴力搜索，只适合
    教学和几百条以内的数据）。
    """

    def __init__(self, embedding: EmbeddingProvider):
        self.embedding = embedding
        self.entries = []  # [{"id": ..., "content": ..., "vector": [...]}]

    def add(self, entry_id: str, content: str):
        """存入一条内容（内部自动向量化）"""
        self.entries.append({
            "id": entry_id,
            "content": content,
            "vector": self.embedding.embed_one(content),
        })

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """检索最相似的 top_k 条"""
        q = self.embedding.embed_one(query)
        scored = []
        for e in self.entries:
            score = self._cosine(q, e["vector"])
            scored.append({"score": round(score, 3), "id": e["id"], "content": e["content"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        """余弦相似度，越接近 1 越相似"""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ═══════════════════════════════════════════════════════════════
#  第五部分：附 —— TF-IDF 参考实现（用于对比理解「词袋模型」的局限）
# ═══════════════════════════════════════════════════════════════
# 这里放一个精简的 TF-IDF 检索器（复用 V5 的思路），方便你对比理解
# 两种「字符级」向量化方式的差异：
#
#   TF-IDF = 单字词频 + IDF 加权，是「词袋模型」（bag-of-words）
#            → 只关心"字有没有出现"，不关心"字和字是否挨在一起"
#            → 所以「上海」和「海上」在它眼里字面完全一样
#   n-gram = 相邻字符组合
#            → 关心"哪些字挨在一起"，即保留了局部顺序信息
#            → 所以能区分「上海」和「海上」
#
# 说明：demo 主流程没有直接调用本函数做对比。原因是 TF-IDF 的 IDF
# 机制在「很短的中文句子」上会把高频字（如"上""海"）权重惩罚到 0，
# 反而让对比结果失真、更难读懂。这里保留实现，供你断点研究其原理。

def _tfidf_search(documents: List[str], query: str, top_k: int = 3) -> List[dict]:
    """
    精简版 TF-IDF 检索（单字词频 + IDF 加权，词袋模型）。

    注意：TF-IDF 把每个字独立看待，忽略字的顺序，
    所以「上海」和「海上」在它眼里字面完全一样。
    """
    import re

    def tokenize(text):
        text = text.lower()
        return re.findall(r"[a-z0-9]+", text) + re.findall(r"[\u4e00-\u9fff]", text)

    docs_tokens = [tokenize(d) for d in documents]
    query_tokens = tokenize(query)

    # 算 IDF
    doc_freq = {}
    for tokens in docs_tokens:
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    n = len(documents)
    idf = {t: math.log(n / (f + 1)) for t, f in doc_freq.items()}

    # 算每个文档的 TF-IDF 向量
    vecs = []
    for tokens in docs_tokens:
        total = len(tokens)
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vecs.append({t: (c / total) * idf.get(t, 0) for t, c in tf.items()})

    # 查询向量
    total_q = len(query_tokens)
    qtf = {}
    for t in query_tokens:
        qtf[t] = qtf.get(t, 0) + 1
    qvec = {t: (c / total_q) * idf.get(t, 0) for t, c in qtf.items()}

    def cosine(a, b):
        dot = sum(a.get(t, 0) * b.get(t, 0) for t in a)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    scored = []
    for i, v in enumerate(vecs):
        scored.append({"score": round(cosine(qvec, v), 3), "id": str(i), "content": documents[i]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ═══════════════════════════════════════════════════════════════
#  第六部分：独立测试入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("🧬 方向1：记忆向量化 —— Embedding 语义检索演示")
    print("=" * 64)

    # 用 bi-gram（n=2）：中文双字组合，比单字更能体现「词」的概念
    emb = HashNGramEmbedding(dim=256, n=2)

    # ── 场景：记忆库里「上海」和「海上」字相同但顺序不同 ──
    memories = [
        ("m1", "上海是直辖市"),
        ("m2", "船在海上航行"),
        ("m3", "喜欢喝咖啡"),
    ]
    query = "上海"

    print("\n【记忆库】")
    for mid, m in memories:
        print(f"  {mid}: {m}")
    print(f"\n【查询】{query}")
    print("  （关键：「上海」和「海上」由相同的两个字组成，只是顺序不同）")

    # ── 1. 链路演示：VectorStore 完整跑通（文本 → 向量 → 检索）──
    print("\n" + "─" * 64)
    print("① 字符 n-gram Embedding 检索（bi-gram，看「相邻双字」）：")
    print("─" * 64)
    store = VectorStore(emb)
    for mid, m in memories:
        store.add(mid, m)
    hits = store.search(query, top_k=3)
    for h in hits:
        print(f"   相关度 {h['score']}  {h['content']}")
    print("  ✅ 只有「上海是直辖市」命中，「船在海上航行」被正确排除")

    # ── 2. n-gram 的顺序敏感性 ──
    print("\n" + "─" * 64)
    print("② n-gram 为什么能区分「上海」和「海上」：")
    print("─" * 64)
    print("  n-gram 保留「字的相邻顺序」，顺序一变，向量就完全不同：")
    for a, b in [("上海", "海上"), ("网球", "球网")]:
        va = emb.embed_one(a)
        vb = emb.embed_one(b)
        sim = VectorStore._cosine(va, vb)
        print(f"  「{a}」vs「{b}」 → 相似度 {sim:.3f}")

    # ── 3. 诚实说明边界 ──
    print("\n" + "─" * 64)
    print("③ 诚实说明：字符 n-gram 的边界")
    print("─" * 64)
    print("  字符 n-gram 和 TF-IDF 一样，都是「字符级」近似方案：")
    print("   · n-gram 保留了字的相邻顺序 → 能区分「上海」和「海上」")
    print("   · 但它做不到「咖啡 ≈ 拿铁」这种跨词义的真正语义关联")
    print("  （字符 n-gram 没有「咖啡是一种饮品」这样的知识）")
    print()
    print("  真正的语义检索需要「训练过的」embedding 模型：")
    print("   · OpenAIEmbedding（接 text-embedding-3-small），本文件已提供代码")
    print("   · 把 VectorStore 里的 HashNGramEmbedding 换成 OpenAIEmbedding")
    print("     即可获得真正的语义能力（向量由大模型生成，内含语义知识）")

    print("\n✅ 演示完成：理解了「文本→向量→相似度检索」的完整链路，")
    print("   以及不同向量化方式（TF-IDF / n-gram / 真 embedding）的能力差异。")
