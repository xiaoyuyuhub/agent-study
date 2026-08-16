# Agent 开发第二阶段学习笔记：工程化 + RAG + 反思

> 学习目标：在第一阶段 ReAct 骨架之上，掌握生产级 Agent 的三大增强能力——工程化健壮性、RAG 知识增强、反思自我纠错。
>
> 对应代码：`agent_v2.py` ｜ 知识库：`knowledge_base/` ｜ 配置：`config.json`

---

## 一、全景架构：v2 相比 v1 多了什么

### 1.1 架构对比图

```
┌─────────────────────────────────────────────────────────────────┐
│                    第一阶段 (agent.py)                          │
│                                                                 │
│   while True:                                                   │
│     LLM 调用 (无保护) ──→ 执行工具 (无保护) ──→ 塞进上下文      │
│                                                                 │
│   问题：网络抖动就崩 / 工具报错就崩 / 死循环 / 无知识 / 无自检   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    第二阶段 (agent_v2.py)                       │
│                                                                 │
│   while step < MAX_STEPS:            ← ① 防死循环               │
│     ┌──────────────────────────┐                                │
│     │ call_llm_with_retry()    │ ← ② 指数退避重试               │
│     └───────────┬──────────────┘                                │
│                 ↓                                               │
│     ┌──────────────────────────┐                                │
│     │ execute_tool_safely()    │ ← ③ 异常保护 + 工具存在检查    │
│     └───────────┬──────────────┘                                │
│                 ↓                                               │
│     ┌──────────────────────────┐                                │
│     │ reflect_on_step()        │ ← ④ 反思自检（可配置开关）     │
│     └───────────┬──────────────┘                                │
│                 ↓                                               │
│     塞进上下文 (tool结果 + 反思)                                │
│                                                                 │
│   新增工具：search_knowledge()    ← ⑤ RAG 知识检索              │
│   新增配置：max_steps / enable_reflection                       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 三大增强能力一览

| 能力 | 解决什么问题 | 核心手段 |
|------|------------|---------|
| 工程化 | Agent 脆弱、容易崩、会死循环 | max_steps 上限 + 重试 + 异常保护 |
| RAG | LLM 不知道私有数据，会瞎编 | 本地文档分块 → 向量化 → 相似度检索 |
| 反思 | LLM 会犯错但不自知 | 每步执行后额外调 LLM 自评，发现问题塞回上下文 |

---

## 二、增强一：工程化（max_steps + 错误处理）

### 2.1 为什么需要工程化

第一阶段的代码有三个致命问题：

```
问题1: while True          → LLM 一直调工具不结束 → 死循环 → 烧钱
问题2: 直接 client.create() → 网络超时 → 整个 Agent 崩溃
问题3: 直接 TOOL_MAP[name]() → 工具内部报错 → 整个 Agent 崩溃
```

生产环境的 Agent 必须能应对：网络抖动、API 限流、工具异常、LLM 幻觉（调不存在的工具）、JSON 解析失败。

### 2.2 四层防护体系

```
            Agent 请求流
                │
    ┌───────────▼───────────┐
    │  第1层：max_steps 上限  │  防死循环，超时强制终止
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  第2层：LLM 调用重试    │  指数退避，最多重试3次
    │  (call_llm_with_retry) │  1s → 2s → 4s
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  第3层：工具安全执行    │  try/except + 工具存在性检查
    │ (execute_tool_safely)  │  工具崩了返回错误信息，不中断
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  第4层：JSON 解析保护   │  LLM 返回的参数解析失败不崩溃
    └───────────────────────┘
```

### 2.3 核心代码：带重试的 LLM 调用

```python
def call_llm_with_retry(messages, tools, model, max_retries=3):
    """
    指数退避重试策略：
      第1次失败 → 等 1 秒 → 重试
      第2次失败 → 等 2 秒 → 重试
      第3次失败 → 等 4 秒 → 重试
      全部失败 → 抛异常（由上层兜底）
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools
            )
            return response
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)  # 1, 2, 4 指数增长
                print(f"⚠️ LLM 调用失败（第{attempt}次），{wait_time}秒后重试")
                time.sleep(wait_time)
    raise last_error
```

### 2.4 核心代码：安全的工具执行

```python
def execute_tool_safely(name: str, args: dict) -> str:
    """工具崩了不让整个 Agent 崩，返回错误信息让 LLM 自己调整"""
    # 防御1：工具不存在（LLM 幻觉出不存在的工具名）
    if name not in TOOL_MAP:
        return f"错误: 工具 '{name}' 不存在。可用: {list(TOOL_MAP.keys())}"

    # 防御2：工具内部异常
    try:
        return TOOL_MAP[name](**args)
    except Exception as e:
        # 返回错误信息，LLM 下一轮看到后可以调整策略
        return f"工具执行出错 [{name}]: {e}"
```

### 2.5 核心代码：防死循环

```python
# v1: while True          ← 危险，可能无限循环
# v2: while step < MAX_STEPS  ← 安全，强制有上限

while step < MAX_STEPS:
    step += 1
    # ... LLM 调用 + 工具执行 ...

# 循环正常退出（非return），说明步数用尽
print(f"⚠️ 达到最大步数 {MAX_STEPS}，Agent 被强制终止")
```

### 2.6 工程化要点总结

| 防护点 | v1 做法 | v2 做法 | 原理 |
|--------|---------|---------|------|
| 循环控制 | `while True` | `while step < MAX_STEPS` | 防死循环 |
| LLM 调用 | 直接调 | `call_llm_with_retry`（3次+指数退避）| 防网络抖动 |
| 工具执行 | 直接调 | `execute_tool_safely`（try/except）| 防工具异常 |
| 工具不存在 | 崩溃 | 返回错误信息 | 防 LLM 幻觉 |
| JSON 解析 | 无保护 | try/except | 防参数格式错 |
| 彻底失败 | 崩溃 | 返回兜底回答 | 保证有响应 |

---

## 三、增强二：RAG 知识检索

### 3.1 RAG 是什么

RAG = **Retrieval-Augmented Generation（检索增强生成）**

核心思想：**LLM 不知道的数据，先从知识库检索出来，塞进 prompt，再让 LLM 回答。**

```
没有 RAG：
  用户："公司融资多少？" → LLM: "我不知道" / 瞎编一个

有 RAG：
  用户："公司融资多少？"
    → 先检索知识库，找到 "CodeBuddy 2024年3月完成A轮5000万美元融资..."
    → 把这段塞进 LLM 的上下文
    → LLM: "CodeBuddy 于2024年3月完成A轮融资5000万美元..."
```

### 3.2 RAG 在 Agent 中的形态

RAG 不是一个独立系统，而是**作为工具被 LLM 调用**：

```
用户提问
   │
   ▼
┌──────────┐     调用工具      ┌──────────────────┐
│   LLM    │ ──────────────→  │ search_knowledge │
│  (大脑)   │                  │   (RAG 工具)     │
└──────────┘                  └────────┬─────────┘
   ▲                                   │
   │  检索结果（相关文档段落）           │
   │ ←──────────────────────────────────┘
   │
   ▼
 基于检索结果生成回答
```

### 3.3 RAG 完整流程图

```
═══════════════════════════════════════════════════════
                   知识库初始化（启动时执行一次）
═══════════════════════════════════════════════════════

  knowledge_base/          ┌──────────────┐
  ├── product_faq.txt  ──→ │  读取 .txt    │
  └── company_info.txt ──→ │  文件内容     │
                           └──────┬───────┘
                                  │
                           ┌──────▼───────┐
                           │  分块         │  按段落（双换行）切
                           │  Chunking     │  "每段" = 一个块
                           └──────┬───────┘
                                  │
                           ┌──────▼───────┐
                           │  向量化       │  TF-IDF 算权重
                           │  (TF-IDF)    │  每块 → {词: 权重}
                           └──────┬───────┘
                                  │
                           ┌──────▼───────┐
                           │  存入内存     │  self.chunks
                           │              │  self.vectors
                           └──────────────┘

═══════════════════════════════════════════════════════
                   查询检索（每次调用执行）
═══════════════════════════════════════════════════════

  用户查询 "公司融资情况"
           │
    ┌──────▼───────┐
    │  查询分词     │  ["公司","融","资","情","况"]
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  查询向量化   │  TF-IDF → {词: 权重}
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  逐块算相似度 │  查询向量 vs 每个文档块向量
    │  (余弦相似度) │  score = cos(query, doc)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  排序 + 截断  │  按分数排序，取 top_k=3
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  拼成文本返回 │  "[来源: company_info.txt | 相关度: 0.82]
    │              │   CodeBuddy于2024年3月完成A轮融资..."
    └──────────────┘
```

### 3.4 向量化原理：TF-IDF

**TF（词频）**：一个词在文档里出现越多，越重要。

**IDF（逆文档频率）**：一个词在所有文档里都出现，就不重要（如"的"、"是"）。

**TF-IDF = TF × IDF**：兼顾"这个词在这篇文档多"和"这个词有区分度"。

```
示例：3篇文档
  doc1: "CodeBuddy 融资 5000万"
  doc2: "CodeBuddy 定价 39元"
  doc3: "CodeBuddy 支持 Python"

词 "codebuddy" 出现在3篇 → IDF 低（没有区分度）
词 "融资" 只出现在 doc1  → IDF 高（区分度高）
词 "5000"  只出现在 doc1  → IDF 高

查询 "融资多少"：
  跟 doc1 相似度最高（"融资"匹配 + IDF高）→ 排第一返回
```

> **注意**：真实 RAG 用 embedding 模型（如 `text-embedding-3-small`）把文本变成几百维的语义向量，存进向量数据库（Pinecone/Milvus）。本例用 TF-IDF 是为了**零依赖能跑**，流程完全一致，只是向量化方法更简单。

### 3.5 核心代码：知识库检索

```python
class KnowledgeBase:
    def __init__(self, kb_dir: Path):
        """启动时加载所有文档，分块 + 向量化"""
        self.chunks = []        # 所有文档块文本
        self.chunk_sources = [] # 每个块的来源文件

        for file_path in sorted(kb_dir.glob("*.txt")):
            content = file_path.read_text(encoding="utf-8")
            # 分块：按段落（双换行）切
            for para in content.split("\n\n"):
                if para.strip():
                    self.chunks.append(para.strip())
                    self.chunk_sources.append(file_path.name)

        # 一次性向量化所有块
        self.vectors, _ = _build_tfidf_vectors(self.chunks)

    def search(self, query: str, top_k: int = 3) -> str:
        """检索：查询 → 向量化 → 算相似度 → 返回top_k"""
        # 1. 查询向量化
        query_vec, _ = _build_tfidf_vectors([" ".join(_tokenize(query))])
        query_vector = query_vec[0] if query_vec else {}

        # 2. 逐块算余弦相似度
        scored = []
        for i, doc_vec in enumerate(self.vectors):
            score = _cosine_similarity(query_vector, doc_vec)
            scored.append((score, self.chunks[i], self.chunk_sources[i]))

        # 3. 排序，取top_k
        scored.sort(key=lambda x: x[0], reverse=True)

        # 4. 拼成文本返回给 LLM
        results = []
        for score, text, source in scored[:top_k]:
            results.append(f"[来源: {source} | 相关度: {score:.2f}]\n{text}")
        return "\n\n---\n\n".join(results)
```

### 3.6 RAG 工具注册

RAG 作为一个工具注册到 TOOLS 列表，LLM 通过 `description` 判断什么时候该调它：

```python
{
    "type": "function",
    "function": {
        "name": "search_knowledge",
        # ⭐ description 写得很具体，明确告诉 LLM 什么时候用
        "description": "搜索内部知识库，查找产品FAQ、公司信息、定价、"
                       "联系方式等内部资料。当用户问到关于公司或产品的"
                       "问题时，应该调用这个工具查找答案，不要自己编造。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题"
                }
            },
            "required": ["query"]
        }
    }
}
```

### 3.7 本例 RAG vs 生产 RAG 对比

| 环节 | 本例（教学版） | 生产版 |
|------|--------------|--------|
| 分词 | 中文逐字 + 英文按词 | jieba /HanLP 专业分词 |
| 向量化 | TF-IDF（词频统计） | embedding 模型（语义理解）|
| 存储 | Python 列表（内存） | 向量数据库（Pinecone/Milvus/Chroma）|
| 分块 | 按段落切 | 滑动窗口（如500字，重叠100字）|
| 检索 | 余弦相似度 brute-force | ANN 近似最近邻（HNSW/IVF）|
| 高级 | 无 | 重排序（rerank）、混合检索、查询改写 |

---

## 四、增强三：反思机制

### 4.1 为什么需要反思

LLM 在 Agent 循环中会犯三类错：

```
错误1: 调错工具      → 用户问天气，LLM 调了 calculate
错误2: 传错参数      → get_weather(city=123) 传了数字
错误3: 结果理解错    → 工具返回 "无数据"，LLM 却当成有数据
```

**反思 = 让 LLM 回头看自己刚做的一步，自评"这步对不对"。** 发现问题可以纠正，而不是一条路走到黑。

### 4.2 反思流程图

```
    工具执行完
        │
        ▼
 ┌──────────────────┐
 │ reflect_on_step  │  额外调一次 LLM（独立调用，不带历史）
 │                  │
 │  输入：           │
 │  - 用户原始任务   │
 │  - 调了什么工具   │
 │  - 传了什么参数   │
 │  - 拿到什么结果   │
 │                  │
 │  让 LLM 评估：    │
 │  - 工具选对了吗？ │
 │  - 参数合理吗？   │
 │  - 结果可信吗？   │
 └────────┬─────────┘
          │
          ▼
    ┌─────────┐
    │ 返回 OK │──→ 一切正常，不干预
    └─────────┘
          │
          ▼ （不是OK）
    ┌──────────────────┐
    │ 追加到 tool 结果  │
    │ result +=         │
    │  "[审查员反思]    │
    │   {reflection}"  │
    └────────┬─────────┘
             │
             ▼
    LLM 下一轮看到反思提示
    可能调整策略（换工具/改参数）
```

### 4.3 核心代码：反思函数

```python
def reflect_on_step(task, tool_name, tool_args, tool_result, model):
    """
    让 LLM 以"审查员"身份评估刚执行的一步。

    关键设计：
    - 独立调用，不带完整对话历史（更客观 + 省 token）
    - 不带 tools（反思本身不需要再调工具）
    - 失败不阻断主流程（try/except 兜底）
    """
    reflect_prompt = f"""你是一个严谨的审查员。请评估以下步骤是否正确。

【用户原始任务】{task}
【这一步执行了什么】
- 调用工具: {tool_name}
- 参数: {tool_args}
- 结果: {tool_result}

请判断：工具选择是否正确？参数是否合理？结果是否可信？
如果一切正确，回复"OK"。如果发现问题，指出问题并给出建议。"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": reflect_prompt}],
            # 不传 tools，反思本身不需要再调工具
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[反思失败，跳过: {e}]"
```

### 4.4 反思结果如何影响主循环

```python
# 工具执行后
result = execute_tool_safely(name, args)

# 反思
if ENABLE_REFLECTION:
    reflection = reflect_on_step(task, name, args, result, model)

    # 如果反思发现问题（不是OK），追加到 tool 结果
    if reflection.strip().upper() != "OK":
        result = result + f"\n\n[审查员反思] {reflection}"

# 把含反思的结果塞进上下文
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": result,  # ← 可能含反思提示
})
```

### 4.5 反思的代价与权衡

| 维度 | 不开反思 | 开反思 |
|------|---------|--------|
| LLM 调用次数 | N 次 | ~2N 次（每步多一次）|
| 成本 | 基准 | 约翻倍 |
| 准确性 | 靠 LLM 一次判断 | 有纠错机会 |
| 速度 | 快 | 慢（每步多等一轮）|

所以反思做成**可配置开关**（`config.json` 的 `enable_reflection`），生产环境按需开启。

### 4.6 踩坑记录：反思消息的 role 问题

> 这是一个实际踩到的坑，值得记录。

最初反思结果用了一条单独的消息塞进上下文：

```python
# ❌ 错误写法
messages.append({
    "role": "system_reminder",   # ← 这个 role API 不认识！
    "content": f"[反思提示] ..."
})
```

**问题现象**：第一轮正常，第二轮开始 API 调用全部失败。

**根因**：OpenAI API 只认 `system` / `user` / `assistant` / `tool` 这四种 role。`system_reminder` 是自定义的，API 收到非法 role 直接拒绝。

**为什么第一轮没事**：反思是在工具执行后才加的，第一轮调用 LLM 时 messages 里还没有反思消息，所以成功。第一轮工具执行后反思消息被塞入，第二轮调用时 API 看到非法 role 就报错。

**修复**：不再新增单独消息，把反思追加到 tool 结果的 content 里：

```python
# ✅ 正确写法
result = result + f"\n\n[审查员反思] {reflection}"
messages.append({
    "role": "tool",              # ← 合法的 role
    "tool_call_id": tool_call.id,
    "content": result,           # ← 含反思提示
})
```

**教训**：往 `messages` 里塞消息时，`role` 字段必须是 API 合法值，不能自创。

---

## 五、增强版主循环完整代码

这是第二阶段的核心，整合了所有增强能力：

```python
def run_agent(task: str, model: str = DEFAULT_MODEL):
    messages = [
        {"role": "system", "content": "你是一个能调用工具的助手..."},
        {"role": "user", "content": task},
    ]

    step = 0

    # ① 防死循环：有上限
    while step < MAX_STEPS:
        step += 1

        # ② 带重试的 LLM 调用
        try:
            response = call_llm_with_retry(messages, TOOLS, model)
        except Exception as e:
            return f"AI 服务不可用: {e}"  # 彻底失败有兜底

        msg = response.choices[0].message

        # 分支A：LLM 要调工具
        if msg.tool_calls:
            messages.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                # ③ JSON 解析保护
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # ④ 安全的工具执行
                result = execute_tool_safely(name, args)

                # ⑤ 反思机制（可配置）
                if ENABLE_REFLECTION:
                    reflection = reflect_on_step(task, name, args, result, model)
                    if reflection.strip().upper() != "OK":
                        result += f"\n\n[审查员反思] {reflection}"

                # 塞进上下文（含反思）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue  # 继续循环

        # 分支B：LLM 不调工具了，任务完成
        return msg.content

    # ① 的兜底：步数用尽
    return f"Agent 达到最大步数 {MAX_STEPS} 仍未完成"
```

---

## 六、配置体系

### 6.1 config.json

```json
{
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "sk-xxxx",
    "model": "deepseek-chat",
    "max_steps": 10,
    "enable_reflection": true
}
```

### 6.2 配置项说明

| 配置项 | 说明 | 建议值 |
|--------|------|--------|
| `base_url` | 模型 API 地址 | 按使用的模型填 |
| `api_key` | API 密钥 | 对应平台的 key |
| `model` | 模型名 | deepseek-chat / gpt-4o-mini 等 |
| `max_steps` | 最大循环步数 | demo: 10，生产: 15~20 |
| `enable_reflection` | 是否开反思 | true（学习时）/ false（省成本）|

---

## 七、知识库结构

```
knowledge_base/
├── product_faq.txt      ← 产品FAQ（定价、功能、安全等）
└── company_info.txt     ← 公司信息（团队、融资、联系方式等）
```

往这个文件夹放 `.txt` 文件，Agent 启动时自动加载。每个文件按段落（双换行）分块，不需要额外配置。

---

## 八、v1 vs v2 完整对比

| 维度 | v1 (agent.py) | v2 (agent_v2.py) |
|------|---------------|-------------------|
| 循环控制 | `while True`（可能死循环）| `while step < MAX_STEPS` |
| LLM 调用 | 直接调（网络抖动就崩）| `call_llm_with_retry`（3次重试+指数退避）|
| 工具执行 | 直接调（工具崩就崩）| `execute_tool_safely`（try/except+存在性检查）|
| JSON 解析 | 无保护 | try/except 兜底 |
| 彻底失败 | 崩溃 | 返回兜底回答 |
| 知识检索 | 无 | RAG（TF-IDF + 余弦相似度）|
| 自我纠错 | 无 | 反思机制（每步自评，可配置）|
| 工具数量 | 3 个 | 4 个（新增 search_knowledge）|
| 配置项 | 3 个 | 5 个（新增 max_steps / enable_reflection）|
| system prompt | 简单一句话 | 详细列出工具 + 使用规则 |
| 日志打印 | 基础 | 上下文 + 返回 + 反思全打印 |

---

## 九、调试要点

### 9.1 断点推荐位置

| 位置 | 看什么 |
|------|--------|
| `KB.search()` 返回处 | RAG 检索到的文档块和相似度分数 |
| `reflect_on_step()` 的 `reflection =` 行 | LLM 怎么自评的 |
| `execute_tool_safely()` 内部 | 工具执行的保护逻辑 |
| `messages.append({"role": "tool"...})` | 上下文怎么累积的 |

### 9.2 对比实验建议

1. **开反思 vs 关反思**：改 `config.json` 的 `enable_reflection`，对比输出差异
2. **改 max_steps**：设成 3 看看 Agent 被强制终止的效果
3. **问知识库外的问题**：问"比特币价格"，看 LLM 会不会瞎编
4. **往 knowledge_base 加文档**：加一个新 .txt，重启后看 Agent 能不能检索到

### 9.3 VSCode 调试配置

`.vscode/launch.json` 已配好两个配置，在调试面板下拉选择：
- "调试 Agent v1 (基础版)" → 跑 `agent.py`
- "调试 Agent v2 (工程化+RAG+反思)" → 跑 `agent_v2.py`

---

## 十、核心认知总结

### 10.1 三个增强的本质

```
工程化 = 给骨架加"安全气囊"
  → 不是改变 Agent 的逻辑，而是让它在出错时不崩、不卡死

RAG = 给 LLM 加"外挂记忆"
  → 不是让 LLM 更聪明，而是让它能"查到"它不知道的数据

反思 = 给 Agent 加"自我意识"
  → 不是替代 LLM 的决策，而是在决策后多一道"检查岗"
```

### 10.2 万变不离其宗

第二阶段加了这么多东西，但**核心还是第一阶段的那个 while 循环**：

- RAG？= 在 TOOLS 里多注册一个 `search_knowledge` 工具
- 反思？= 在工具执行后多调一次 LLM 做评估
- 重试？= 把 `client.create()` 包了一层 for 循环
- max_steps？= 把 `while True` 改成 `while step < N`

**所有增强都是在循环的各个环节"插桩"，不是推翻骨架。** 这就是为什么第一阶段的理解是地基——地基牢了，上面加什么都能看懂。

---

## 十一、下一阶段方向

| 方向 | 内容 | 难度 |
|------|------|------|
| Plan-Execute | 先一次性规划全部步骤，再逐步执行 | ★★☆ |
| 多 Agent 协作 | 拆成 Planner + Executor 两个 Agent 互传消息 | ★★★ |
| 流式输出 | SSE 边生成边返回，提升用户体验 | ★★☆ |
| 长期记忆 | 对话存入向量数据库，跨会话回忆 | ★★★ |
| 工具并行 | 多个无依赖的工具同时执行，提升速度 | ★★☆ |
| Human-in-the-loop | 关键步骤暂停等人确认 | ★★☆ |

---

## 附录：第二阶段新增术语

| 术语 | 含义 |
|------|------|
| TF-IDF | 词频-逆文档频率，衡量词在文档中的重要性 |
| 余弦相似度 | 衡量两个向量方向的相似程度，用于检索 |
| 分块 (Chunking) | 把长文档切成小块，便于精准检索 |
| 指数退避 | 重试时等待时间指数增长（1s→2s→4s），避免压垮服务端 |
| 反思 (Reflection) | Agent 执行后回头自评，发现问题可纠正 |
| Embedding | 用模型把文本变成高维语义向量（本例用 TF-IDF 替代）|
| 向量数据库 | 专门存储和检索向量的数据库（Pinecone/Milvus/Chroma）|
| top_k | 检索时返回相似度最高的 k 个结果 |
