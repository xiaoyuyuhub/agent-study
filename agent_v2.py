"""
Agent 第二阶段：工程化 + RAG + 反思
=====================================
在第一阶段的 ReAct 骨架之上，增加三个生产级特性：

  1. max_steps + 错误处理  → 防死循环、防崩溃、可重试（工程化）
  2. RAG 知识检索工具       → 让 Agent 能"读取"本地文档回答问题（知识增强）
  3. 反思机制              → 每步执行后让 LLM 自评"这步对吗"（自我纠错）

运行前：
  pip install -r requirements.txt
  把 base_url / api_key / model 填进 config.json
  确保 knowledge_base/ 文件夹里有文档（已自带两个示例）

运行：
  python agent_v2.py
"""
import os
import json
import time
import math
import re
from pathlib import Path
from datetime import datetime
from openai import OpenAI


# ╔══════════════════════════════════════════════════════════════╗
# ║  第零部分：读配置                                              ║
# ║  把 url / token / model / 参数都从 config.json 读，不写死     ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

# 创建 OpenAI 客户端，兼容所有 OpenAI 协议的模型服务（DeepSeek/通义/Kimi等）
client = OpenAI(
    base_url=config["base_url"],
    api_key=config["api_key"],
)
DEFAULT_MODEL = config["model"]

# ── 第二阶段新增的配置项 ──────────────────────────────────────
# max_steps：Agent 最多循环多少轮，防止 LLM 陷入死循环无限调工具
# 生产环境一般设 10~20，demo 设小一点方便观察
MAX_STEPS = config.get("max_steps", 10)

# ENABLE_REFLECTION：是否开启反思机制（方便对比开关前后的效果）
ENABLE_REFLECTION = config.get("enable_reflection", True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第一部分：RAG 知识检索引擎（第二阶段新增）                    ║
# ║                                                                ║
# ║  RAG 的核心流程：                                              ║
# ║    文档 → 分块 → 向量化 → 存起来                               ║
# ║    查询 → 向量化 → 算相似度 → 返回最相关的几段                  ║
# ║                                                                ║
# ║  真实生产环境会用 embedding 模型（如 text-embedding-3-small）  ║
# ║  把文本变成高维向量，再存进向量数据库（如 Pinecone/Milvus）。   ║
# ║  但那需要额外的 API 和依赖。                                    ║
# ║                                                                ║
# ║  这里为了"零依赖能跑"，用纯 Python 实现一个简化版：             ║
# ║    用 TF-IDF（词频-逆文档频率）做向量化                        ║
# ║    用余弦相似度做检索                                          ║
# ║  流程和真实 RAG 完全一致，只是向量化方法更简单。                ║
# ╚══════════════════════════════════════════════════════════════╝

# 知识库文件夹路径，里面放 .txt 文档
KB_DIR = Path(__file__).parent / "knowledge_base"


def _tokenize(text: str) -> list[str]:
    """
    中文分词（简化版）。

    真实场景会用 jieba 这类专业分词库。
    这里用最简单的方案：中文按字切，英文按单词切。
    虽然粗糙，但对于演示 RAG 流程足够了。

    例："CodeBuddy支持Python" → ['codebuddy', '支', '持', 'python']
    """
    text = text.lower().strip()
    # 英文单词：提取连续的字母数字
    english_words = re.findall(r"[a-z0-9]+", text)
    # 中文字符：逐字切分（匹配 CJK 统一汉字范围）
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_words + chinese_chars


def _build_tfidf_vectors(documents: list[str]) -> tuple[list[dict], list[str]]:
    """
    给一批文档构建 TF-IDF 向量。

    TF-IDF 的直觉：
      - TF（词频）：一个词在文档里出现越多，越重要
      - IDF（逆文档频率）：一个词在所有文档里都出现，就不重要（如"的"、"是"）
      - 两者相乘 = 这个词在这篇文档的权重

    返回：
      - 每篇文档的词权重字典 [{词: 权重}, ...]
      - 所有不重复的词表
    """
    # ── 第一步：给每篇文档分词 ──
    tokenized_docs = [_tokenize(doc) for doc in documents]

    # ── 第二步：统计每个词出现在了多少篇文档里（DF，文档频率）──
    # 这是为了算 IDF：出现文档越多，这个词的区分度越低
    doc_freq = {}  # {词: 出现在了几篇文档里}
    for tokens in tokenized_docs:
        unique_tokens = set(tokens)  # 每篇文档里每个词只算一次
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # ── 第三步：算 IDF ──
    # 公式：idf = ln(文档总数 / (该词出现的文档数 + 1))
    # +1 是为了防止除以 0。ln 让权重变化更平滑。
    num_docs = len(documents)
    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log(num_docs / (freq + 1))

    # ── 第四步：给每篇文档算 TF-IDF 向量 ──
    # tf = 该词在文档里出现的次数 / 文档总词数
    # tfidf = tf * idf
    tfidf_vectors = []
    all_tokens = set()  # 收集词表
    for tokens in tokenized_docs:
        total = len(tokens)
        tf = {}  # 词频
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # 算每个词的 tfidf 权重
        vector = {}
        for token, count in tf.items():
            vector[token] = (count / total) * idf.get(token, 0)
            all_tokens.add(token)
        tfidf_vectors.append(vector)

    return tfidf_vectors, list(all_tokens)


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """
    计算两个向量字典的余弦相似度。

    余弦相似度 = (A·B) / (|A| × |B|)
    值域 [-1, 1]，越接近 1 表示越相似。

    直觉：两个向量"方向"越一致，相似度越高。
    这是向量检索最常用的相似度度量。
    """
    # 点积：相同词的权重相乘再求和
    dot_product = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)

    # 向量的模（长度）
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    # 防止除以 0
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class KnowledgeBase:
    """
    知识库：负责加载文档、分块、向量化、检索。

    这就是一个迷你版向量数据库。
    初始化时把所有文档读进来、分好块、算好向量，
    后续查询时直接算相似度就行。
    """

    def __init__(self, kb_dir: Path):
        """加载知识库文件夹里的所有 .txt 文件"""
        self.chunks = []       # 所有文档块（文本）
        self.chunk_sources = []  # 每个块来自哪个文件（溯源用）

        # 遍历知识库文件夹
        for file_path in sorted(kb_dir.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8")

            # ── 分块（Chunking）──
            # RAG 不是把整篇文档丢给 LLM，而是切成小块再检索。
            # 块太大 → 检索不精准；块太小 → 丢失上下文。
            # 这里按"双换行"（即段落）来切，是最简单的分块策略。
            # 真实场景会用滑动窗口（如每 500 字，重叠 100 字）。
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            for para in paragraphs:
                self.chunks.append(para)
                self.chunk_sources.append(file_path.name)

        # ── 向量化 ──
        # 把所有文档块一次性算成 TF-IDF 向量，后续查询时复用
        if self.chunks:
            self.vectors, _ = _build_tfidf_vectors(self.chunks)
            print(f"📚 知识库加载完成: {len(self.chunks)} 个文档块")
        else:
            self.vectors = []
            print(f"⚠️  知识库为空，请在 {kb_dir} 放一些 .txt 文件")

    def search(self, query: str, top_k: int = 3) -> str:
        """
        检索：给一个查询，返回最相关的 top_k 个文档块。

        流程：
          1. 把 query 也变成 TF-IDF 向量
          2. 跟所有文档块算余弦相似度
          3. 按相似度排序，取前 top_k 个
          4. 拼成文本返回给 LLM

        这就是 RAG 的"检索"部分。
        检索到的内容会被 LLM 用来生成回答（"生成"部分由 LLM 完成）。
        """
        if not self.chunks:
            return "知识库为空，没有可检索的内容。"

        # 把查询向量化
        query_tokens = _tokenize(query)
        query_vec, _ = _build_tfidf_vectors([" ".join(query_tokens)])
        query_vector = query_vec[0] if query_vec else {}

        # 跟每个文档块算相似度
        scored = []  # [(相似度, 块文本, 来源文件), ...]
        for i, doc_vec in enumerate(self.vectors):
            score = _cosine_similarity(query_vector, doc_vec)
            scored.append((score, self.chunks[i], self.chunk_sources[i]))

        # 按相似度从高到低排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 取 top_k 个，拼成结果
        results = []
        for score, text, source in scored[:top_k]:
            results.append(
                f"[来源: {source} | 相关度: {score:.2f}]\n{text}"
            )

        return "\n\n---\n\n".join(results)


# 初始化知识库（程序启动时加载一次）
KB = KnowledgeBase(KB_DIR)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第二部分：工具函数（代码真正干活的地方）                      ║
# ║  这些是 Agent 能调用的"手和脚"                                ║
# ╚══════════════════════════════════════════════════════════════╝

def get_current_time() -> str:
    """工具1：获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """工具2：安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return f"不支持的字符: {expression}"
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败: {e}"


def get_weather(city: str) -> str:
    """工具3：模拟天气查询（mock 数据）"""
    fake_weather = {
        "北京": "28°C，晴",
        "上海": "26°C，多云",
        "深圳": "32°C，雷阵雨",
    }
    return fake_weather.get(city, f"{city}: 暂无数据")


def search_knowledge(query: str) -> str:
    """
    工具4（第二阶段新增）：RAG 知识检索。

    这是第二阶段的核心新增工具。
    当用户问关于"产品FAQ"、"公司信息"等知识库内的问题时，
    LLM 应该调这个工具去检索，而不是瞎编。

    这就是 RAG 在 Agent 中的形态：作为一个"工具"被 LLM 调用。
    LLM 不直接知道知识库的内容，它通过调这个工具来"查资料"。
    """
    return KB.search(query, top_k=3)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第三部分：工具描述（给 LLM 看的"菜单"）                      ║
# ║                                                                ║
# ║  ⭐ description 是灵魂！                                       ║
# ║  LLM 完全靠这段描述来决定"什么时候该调这个工具"。              ║
# ║  描述写不好，LLM 就该调不调、不该调乱调。                      ║
# ╚══════════════════════════════════════════════════════════════╝
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '3 * (2 + 4)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 '北京'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        # 第二阶段新增的 RAG 工具
        # 注意 description 写得很具体，明确告诉 LLM 什么时候该用它
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索内部知识库，查找产品FAQ、公司信息、定价、联系方式等内部资料。当用户问到关于公司或产品的问题时，应该调用这个工具查找答案，不要自己编造。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键 词或问题，如 'CodeBuddy支持哪些语言' 或 '公司融资情况'",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → Python 函数 的映射
TOOL_MAP = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
    "search_knowledge": search_knowledge,
}


# ╔══════════════════════════════════════════════════════════════╗
# ║  第四部分：反思机制（第二阶段新增）                            ║
# ║                                                                ║
# ║  反思 = 让 LLM 回头看自己刚做的一步，评估对不对。              ║
# ║                                                                ║
# ║  为什么需要反思？                                              ║
# ║    LLM 有时候会犯傻：                                          ║
# ║      - 调了错误的工具                                          ║
# ║      - 传了错误的参数                                          ║
# ║      - 拿到结果后理解错了                                      ║
# ║    反思机制让 LLM 自己检查自己，发现问题可以纠正。             ║
# ║                                                                ║
# ║  实现：每次工具执行完，额外调一次 LLM 做评估。                 ║
# ║  如果发现问题，把反思结论塞进 messages，主循环的 LLM 能看到。  ║
# ║                                                                ║
# ║  代价：每步多花一次 LLM 调用（成本翻倍），所以可配置开关。     ║
# ╚══════════════════════════════════════════════════════════════╝

def reflect_on_step(
    task: str,
    tool_name: str,
    tool_args: dict,
    tool_result: str,
    model: str,
) -> str:
    """
    让 LLM 反思刚执行的一步。

    输入：用户任务 + 这一步调了什么工具、传了什么参数、拿到什么结果
    输出：LLM 的评估（对/错/需要修正）

    注意：反思用的是一次"独立的" LLM 调用，不携带完整对话历史。
    这样做的好处是反思更客观（不受之前上下文影响），也更省 token。
    """
    # 构造反思专用的 prompt
    # 关键：告诉 LLM 它的角色是"审查员"，不是"执行者"
    reflect_prompt = f"""你是一个严谨的审查员。请评估以下 Agent 执行步骤是否正确。

【用户原始任务】{task}

【这一步执行了什么】
- 调用工具: {tool_name}
- 参数: {tool_args}
- 结果: {tool_result}

请判断：
1. 这一步的工具选择是否正确？
2. 参数是否合理？
3. 结果是否可信？
4. 有没有遗漏或错误？

请简短回复（2-3句话）。如果一切正确，回复"OK"。
如果发现问题，指出问题并给出建议。"""

    # 反思也是一次 LLM 调用，但不带 tools（纯文本评估，不需要调工具）
    # 注意这里加了错误处理：反思失败不应该影响主流程
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": reflect_prompt}],
            # 不传 tools，反思本身不需要再调工具
        )
        reflection = response.choices[0].message.content
        return reflection
    except Exception as e:
        # 反思失败不阻断主流程，返回一个"跳过"的标记
        return f"[反思失败，跳过: {e}]"


# ╔══════════════════════════════════════════════════════════════╗
# ║  第五部分：工程化工具函数（第二阶段新增）                      ║
# ║                                                                ║
# ║  这些函数让 Agent 更"健壮"：                                   ║
# ║    - LLM 调用能重试（网络抖动不怕）                            ║
# ║    - 工具执行有保护（工具崩了不让整个 Agent 崩）               ║
# ║    - 有详细日志（出问题能排查）                                ║
# ╚══════════════════════════════════════════════════════════════╝

def call_llm_with_retry(
    messages: list,
    tools: list,
    model: str,
    max_retries: int = 3,
) -> object:
    """
    带 retry 的 LLM 调用。

    为什么要 retry？
      - 网络可能抖动（超时、连接重置）
      - API 可能限流（429 Too Many Requests）
      - 服务端可能临时 500

    第一阶段的代码直接 client.chat.completions.create()，
    一旦网络出问题整个 Agent 就崩了。
    生产环境必须有重试机制。

    策略：失败后等一下再试，每次等的时间翻倍（指数退避）。
    指数退避的好处：避免大量请求同时重试压垮服务端。
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            return response

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # 指数退避：第1次等1秒，第2次等2秒，第3次等4秒
                wait_time = 2 ** (attempt - 1)
                print(f"  ⚠️  LLM 调用失败（第{attempt}次），{wait_time}秒后重试: {e}")
                time.sleep(wait_time)
            else:
                print(f"  ❌ LLM 调用彻底失败（重试{max_retries}次后）: {e}")

    # 重试全部失败，抛出最后的错误
    raise last_error


def execute_tool_safely(name: str, args: dict) -> str:
    """
    安全执行工具，不让工具的异常搞崩整个 Agent。

    第一阶段：result = TOOL_MAP[name](**args)
    如果工具内部报错，整个程序就崩了。
    第二阶段：包一层 try/except，工具崩了返回错误信息，Agent 继续。

    还处理了"工具不存在"的情况（LLM 幻觉出不存在的工具名）。
    """
    # 检查工具是否存在（LLM 有时会幻觉出不存在的工具名）
    if name not in TOOL_MAP:
        return f"错误: 工具 '{name}' 不存在。可用的工具: {list(TOOL_MAP.keys())}"

    try:
        result = TOOL_MAP[name](**args)
        return result
    except Exception as e:
        # 工具执行出错，返回错误信息而不是崩掉
        # LLM 下一轮看到这个错误，可以自己调整策略
        return f"工具执行出错 [{name}]: {e}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  第六部分：日志打印函数（复用第一阶段的）                      ║
# ╚══════════════════════════════════════════════════════════════╝

def _dump_messages(messages):
    """打印发给 LLM 的完整上下文"""
    print(f"\n📤 【发给 LLM 的上下文】共 {len(messages)} 条消息：")
    print("─" * 60)
    for i, m in enumerate(messages):
        role = getattr(m, "role", m.get("role") if isinstance(m, dict) else "?")
        if hasattr(m, "model_dump"):
            m_dict = m.model_dump()
        else:
            m_dict = m if isinstance(m, dict) else {}

        if role == "system":
            print(f"  [{i}] system: {m_dict.get('content', '')[:80]}...")
        elif role == "user":
            print(f"  [{i}] user: {m_dict.get('content', '')}")
        elif role == "assistant":
            tool_calls = m_dict.get("tool_calls") or []
            content = m_dict.get("content") or ""
            if tool_calls:
                calls = ", ".join(
                    f"{tc['function']['name']}({tc['function']['arguments']})"
                    for tc in tool_calls
                )
                print(f"  [{i}] assistant: [调工具] {calls}")
            else:
                print(f"  [{i}] assistant: {content[:60]}...")
        elif role == "tool":
            print(f"  [{i}] tool结果: {m_dict.get('content', '')[:80]}...")
        elif role == "system_reminder":
            print(f"  [{i}] 💭反思: {m_dict.get('content', '')[:80]}...")
        else:
            print(f"  [{i}] {role}: ...")
    print("─" * 60)


def _dump_response(response):
    """打印 LLM 的原始返回"""
    choice = response.choices[0]
    msg = choice.message
    print(f"\n📥 【LLM 返回】 finish_reason={choice.finish_reason}")
    print("─" * 60)
    if msg.tool_calls:
        for i, tc in enumerate(msg.tool_calls):
            print(f"  tool_call[{i}]: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  content: {msg.content!r}")
        print(f"  tool_calls: None ← LLM 认为任务完成")
    if response.usage:
        u = response.usage
        print(f"  tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
    print("─" * 60)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第七部分：Agent 主循环（增强版）                              ║
# ║                                                                ║
# ║  对比第一阶段的改动：                                          ║
# ║    1. while True → while step < MAX_STEPS  （防死循环）       ║
# ║    2. 直接调 LLM → call_llm_with_retry      （错误重试）      ║
# ║    3. 直接执行工具 → execute_tool_safely     （异常保护）     ║
# ║    4. 工具执行后 → 加反思步骤                （自我纠错）     ║
# ╚══════════════════════════════════════════════════════════════╝

def run_agent(task: str, model: str = DEFAULT_MODEL):
    """
    增强版 Agent 主循环。

    完整流程（每一步都标注了是"代码干"还是"LLM干"）：
      1. [代码] 组tep <装初始上下文
      2. [代码] while s MAX_STEPS:        ← 防死循环
      3.   [代码] 打印上下文日志
      4.   [代码] call_llm_with_retry()         ← 带重试的 LLM 调用
      5.   [LLM]  LLM 决定下一步做什么           ← 核心规划
      6.   [代码] 打印 LLM 返回
      7.   [分支] 如果 LLM 要调工具:
      8.     [代码] execute_tool_safely()        ← 带保护的执行
      9.     [LLM]  reflect_on_step()            ← 反思这步对不对
      10.    [代码] 把结果和反思塞进上下文
      11.    [代码] continue 继续循环
      12.  [分支] 否则: 返回最终回答
    """
    # ── 初始化上下文 ──
    # system prompt 比 v1 更详细，明确告诉 LLM 有哪些工具、什么时候用
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个能调用工具的助手。你有以下工具：\n"
                "- get_current_time: 查当前时间\n"
                "- calculate: 算数学题\n"
                "- get_weather: 查城市天气\n"
                "- search_knowledge: 搜索内部知识库（产品FAQ、公司信息等）\n\n"
                "规则：\n"
                "1. 遇到需要计算、查时间、查天气的任务，主动调对应工具\n"
                "2. 遇到关于公司或产品的问题，必须调 search_knowledge 查资料，不要自己编\n"
                "3. 完成所有子任务后，再给出最终总结\n"
            ),
        },
        {"role": "user", "content": task},
    ]

    step = 0

    # ★★★ 第二阶段核心改动1：while True → while step < MAX_STEPS ★★★
    # 第一阶段是 while True，如果 LLM 一直调工具不结束，就死循环了。
    # 现在加了上限，超过 MAX_STEPS 就强制停止。
    while step < MAX_STEPS:
        step += 1
        print(f"\n{'='*60}")
        print(f"🔄 第 {step}/{MAX_STEPS} 轮")
        print(f"{'='*60}")

        # 打印完整上下文（观察 messages 怎么累积的）
        _dump_messages(messages)

        # ★★★ 第二阶段核心改动2：带 retry 的 LLM 调用 ★★★
        # 第一阶段：response = client.chat.completions.create(...)
        # 第二阶段：包了 retry，网络抖动不怕
        try:
            response = call_llm_with_retry(
                messages=messages,
                tools=TOOLS,
                model=model,
            )
        except Exception as e:
            # LLM 调用彻底失败（重试3次都不行），给出兜底回答
            print(f"\n❌ LLM 不可用，Agent 终止: {e}")
            return f"抱歉，AI 服务暂时不可用: {e}"

        msg = response.choices[0].message

        # 打印 LLM 返回
        _dump_response(response)

        # ── 分支A：LLM 要调工具 ──
        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                # LLM 返回的参数是 JSON 字符串，要解析成 dict
                # 这里也加了保护：JSON 解析失败不让 Agent 崩
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"🤖 调用工具: {name}({args})")

                # ★★★ 第二阶段核心改动3：安全的工具执行 ★★★
                # 第一阶段：result = TOOL_MAP[name](**args)
                # 第二阶段：包了 try/except + 工具存在性检查
                result = execute_tool_safely(name, args)
                print(f"🛠️  结果: {result[:100]}{'...' if len(result) > 100 else ''}")

                # ★★★ 第二阶段核心改动4：反思机制 ★★★
                # 每个工具执行完，让 LLM 反思这步对不对
                if ENABLE_REFLECTION:
                    print(f"\n💭 反思中...")
                    reflection = reflect_on_step(
                        task=task,
                        tool_name=name,
                        tool_args=args,
                        tool_result=result,
                        model=model,
                    )
                    print(f"   反思结论: {reflection}")

                    # 如果反思发现了问题（不是 OK），把反思结论追加到 tool 结果里
                    # ⚠️ 注意：不能新增一条单独的 message，因为 OpenAI API 只认
                    #    system / user / assistant / tool 这几种 role。
                    #    最干净的做法是把反思追加到 tool 结果的 content 里，
                    #    这样 LLM 下一轮在工具结果中就能看到反思提示。
                    if reflection.strip().upper() != "OK" and "OK" not in reflection[:5]:
                        result = result + f"\n\n[审查员反思] {reflection}"

                # 把工具结果（可能含反思）塞进上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue  # 继续下一轮循环

        # ── 分支B：LLM 不调工具了，任务完成 ──
        print(f"\n{'='*60}")
        print(f"🏁 任务完成（第 {step} 轮）")
        print(f"{'='*60}")
        print(f"\n📋 最终回答:\n{'─'*60}")
        print(msg.content)
        print(f"{'─'*60}")
        return msg.content

    # ★★★ 第二阶段核心改动1的兜底：达到 max_steps 还没完成 ★★★
    # 如果循环正常退出（不是因为 return），说明步数用尽了
    print(f"\n⚠️  达到最大步数 {MAX_STEPS}，Agent 被强制终止。")
    print(f"    这通常意味着 LLM 陷入了循环或任务太复杂。")
    return f"Agent 达到最大步数 {MAX_STEPS} 仍未完成任务。"


# ╔══════════════════════════════════════════════════════════════╗
# ║  第八部分：运行                                                ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    # 这个任务会触发所有新特性：
    # - 查天气 → 原有工具
    # - 算数学 → 原有工具
    # - 查产品定价 → RAG 知识检索（新增）
    # - 查公司融资 → RAG 知识检索（新增）
    # 每步之后会有反思（新增）
    task = (
        "帮我做几件事："
        "1. 查一下北京的天气；"
        "2. 算一下 888 * 333 等于多少；"
        "3. CodeBuddy Pro 有哪些定价方案？"
        "4. 公司融资情况怎么样？"
        "最后帮我总结一下。"
    )

    print(f"📋 用户任务: {task}")
    print(f"⚙️  配置: max_steps={MAX_STEPS}, reflection={'开' if ENABLE_REFLECTION else '关'}")

    run_agent(task)
