# 增量更新策略：新数据场景分析

## 问题概述

在生产环境中，会遇到三种数据变化：
1. **新新闻加入** - 数据库新增新闻文章
2. **新用户注册** - 新用户加入系统
3. **用户行为更新** - 老用户的点击历史变化

**核心问题**：每次数据变化都需要重新训练整个模型吗？

---

## 一、当前架构分析

### 1.1 模型组成部分

```
LLMEnhancedRecommender
├─ 用户ID嵌入 (num_users × 128)        ← 固定用户数量
├─ 新闻ID嵌入 (num_news × 128)         ← 固定新闻数量
├─ LLM嵌入投影层 (1536 → 256)         ← 参数固定
├─ GNN模块 (GraphSAGE)                 ← 依赖图结构
├─ 注意力融合层                        ← 参数固定
└─ 用户历史注意力                      ← 推理时动态输入
```

### 1.2 数据依赖关系

| 组件 | 训练时依赖 | 推理时依赖 | 是否可扩展 |
|------|-----------|-----------|----------|
| **用户ID嵌入** | 用户数量固定 | 用户ID | ❌ 固定大小 |
| **新闻ID嵌入** | 新闻数量固定 | 新闻ID | ❌ 固定大小 |
| **LLM嵌入** | 预计算的.npy文件 | 新闻ID索引 | ✅ 可追加 |
| **GNN嵌入** | 知识图谱结构 | 图结构 | ⚠️ 需更新图 |
| **用户历史** | 无 | 动态输入 | ✅ 完全动态 |

---

## 二、三种场景详细分析

### 场景1：新新闻加入

#### 问题描述
```
数据库新增1000条新闻
├─ 新闻ID: 51283 ~ 52282
├─ 有标题、摘要、实体
└─ 需要被推荐给用户
```

#### 当前架构的限制

**❌ 问题1：ID嵌入越界**
```python
# 模型初始化时
self.id_embedding = nn.Embedding(
    num_embeddings=51282,  # 固定！
    embedding_dim=128
)

# 推理时
new_news_id = 51283  # 超出范围！
emb = self.id_embedding(new_news_id)  # ❌ IndexError
```

**❌ 问题2：LLM嵌入文件不包含新新闻**
```python
llm_embeddings = np.load('llm_embeddings.npy')  # shape: (51282, 1536)
new_emb = llm_embeddings[51283]  # ❌ IndexError
```

**❌ 问题3：GNN图不包含新新闻节点**
```python
graph_data = {
    'num_news': 51282,  # 固定！
    'node_features': tensor(78184, 100),  # 固定节点数
    'edge_index': tensor(2, 237738)
}
```

#### ✅ 解决方案

**方案A：零样本推理（Zero-Shot Inference）**

无需重新训练，仅依赖LLM+GNN：

```python
# 步骤1: 为新新闻生成LLM嵌入（实时调用API）
from openai import OpenAI
client = OpenAI(api_key="...")

def get_new_news_llm_embedding(news_text):
    """为新新闻生成LLM嵌入"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=news_text
    )
    return np.array(response.data[0].embedding)  # (1536,)

# 步骤2: 追加到LLM嵌入文件
old_embeddings = np.load('llm_embeddings.npy')  # (51282, 1536)
new_embedding = get_new_news_llm_embedding("Title: ... Abstract: ...")
updated_embeddings = np.vstack([old_embeddings, new_embedding])  # (51283, 1536)
np.save('llm_embeddings.npy', updated_embeddings)

# 步骤3: 更新知识图谱
from kg_utils import KnowledgeGraphBuilder
kg_builder = KnowledgeGraphBuilder(...)
kg_builder.add_news_node(
    news_id=51283,
    entities=['OpenAI', 'GPT-4']  # 从新新闻提取的实体
)
updated_graph = kg_builder.build_graph()

# 步骤4: 使用默认ID嵌入（或零向量）
# 方法1: 使用padding_idx (0号位置的嵌入)
default_id_emb = model.news_encoder.id_embedding(torch.tensor([0]))

# 方法2: 使用随机初始化（更好）
default_id_emb = torch.randn(1, 128) * 0.02  # 小方差初始化

# 步骤5: 推理
def predict_for_new_news(user_id, new_news_id):
    """为新新闻进行推理（零样本）"""
    # 获取LLM嵌入
    llm_emb = updated_embeddings[new_news_id]  # (1536,)

    # 获取GNN嵌入
    with torch.no_grad():
        gnn_emb = model.gnn(
            updated_graph['node_features'],
            updated_graph['edge_index']
        )[new_news_id]  # (128,)

    # 使用默认ID嵌入
    id_emb = default_id_emb  # (128,)

    # 融合推理（仅依赖LLM+GNN）
    # ID权重会自动降低，LLM+GNN权重提高
    score = model.predict(user_id, new_news_id, ...)

    return score
```

**性能预期**：
- ✅ 无需重新训练
- ✅ LLM提供高质量语义表示
- ✅ GNN利用实体关联
- ⚠️ ID嵌入缺失，协同过滤信号弱
- ⚠️ 预计性能下降 5-10%（可接受）

---

**方案B：增量微调（Incremental Fine-tuning）**

定期批量更新模型：

```python
# 每天/每周批量更新
def incremental_training(model, new_news_data):
    """增量训练新新闻的ID嵌入"""

    # 步骤1: 扩展ID嵌入层
    old_num_news = model.news_encoder.id_embedding.num_embeddings
    new_num_news = old_num_news + len(new_news_data)

    # 创建新的嵌入层
    new_id_embedding = nn.Embedding(new_num_news, 128, padding_idx=0)

    # 复制旧权重
    with torch.no_grad():
        new_id_embedding.weight[:old_num_news] = \
            model.news_encoder.id_embedding.weight

        # 新新闻用小方差初始化
        nn.init.normal_(new_id_embedding.weight[old_num_news:],
                       mean=0, std=0.02)

    # 替换
    model.news_encoder.id_embedding = new_id_embedding

    # 步骤2: 冻结大部分参数，仅训练新嵌入
    for param in model.parameters():
        param.requires_grad = False

    # 仅解冻新新闻的ID嵌入
    model.news_encoder.id_embedding.weight[old_num_news:].requires_grad = True

    # 步骤3: 快速微调（1-2个epoch）
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4  # 小学习率
    )

    # 仅使用包含新新闻的样本训练
    for epoch in range(2):
        for batch in new_news_dataloader:
            loss = model(...)
            loss.backward()
            optimizer.step()

    return model
```

**性能预期**：
- ✅ ID嵌入质量高
- ✅ 性能接近完整重训（-1~2%）
- ⚠️ 需要定期执行（每天/每周）
- ⚠️ 需要收集新新闻的点击数据

---

### 场景2：新用户加入

#### 问题描述
```
新用户注册
├─ 用户ID: 50001
├─ 无历史点击
└─ 冷启动问题
```

#### ✅ 解决方案

**好消息：无需重新训练！**

原因：当前架构中，用户表示主要依赖**历史点击**，而非用户ID嵌入：

```python
# src/model_llm.py:335-377
def get_user_representation(self, user_idx, history, ...):
    # 1. 用户ID嵌入（占比小）
    user_emb = self.user_embedding(user_idx)  # (B, 128)
    user_repr = self.user_transform(user_emb)  # (B, 256)

    # 2. 历史点击（占比大！）
    hist_repr = self.news_encoder(history, ...)  # (B, max_hist, 256)
    hist_attn_weights = self.history_attention(hist_repr)
    hist_aggregated = torch.sum(hist_repr * hist_attn_weights, dim=1)

    # 3. 融合
    user_repr = user_repr + hist_aggregated  # 历史占主导！
```

**处理策略**：

```python
def predict_for_new_user(new_user_id, candidate_news, history=[]):
    """新用户推理（冷启动）"""

    # 策略1: 使用默认用户嵌入
    if new_user_id >= model.num_users:
        user_idx = torch.tensor([0])  # padding_idx
    else:
        user_idx = torch.tensor([new_user_id])

    # 策略2: 历史为空时的处理
    if len(history) == 0:
        # 方法A: 使用全局流行新闻作为虚拟历史
        history = get_popular_news(top_k=10)

        # 方法B: 使用零向量
        history = torch.zeros(1, 10, dtype=torch.long)

    # 正常推理
    scores = model.predict(user_idx, candidate_news, history, ...)

    return scores
```

**冷启动优化**：

```python
# 基于内容的冷启动推荐
def cold_start_recommendation(new_user_profile):
    """新用户冷启动：依赖LLM语义匹配"""

    # 步骤1: 根据用户画像生成查询向量
    profile_text = f"User interests: {new_user_profile['interests']}"
    profile_emb = get_llm_embedding(profile_text)  # (1536,)

    # 步骤2: 与所有新闻LLM嵌入计算相似度
    news_embeddings = np.load('llm_embeddings.npy')  # (51282, 1536)
    similarities = cosine_similarity(
        profile_emb.reshape(1, -1),
        news_embeddings
    )[0]  # (51282,)

    # 步骤3: 推荐Top-K
    top_k_indices = np.argsort(similarities)[-10:][::-1]

    return top_k_indices
```

**性能预期**：
- ✅ 无需重新训练
- ✅ 依赖LLM语义匹配
- ⚠️ 初期推荐质量一般
- ✅ 随用户点击增加，质量快速提升

---

### 场景3：用户历史点击变化

#### 问题描述
```
用户 #12345 新增点击
├─ 昨天点击: [News A, News B, News C]
├─ 今天点击: [News A, News B, News C, News D, News E]
└─ 历史变化
```

#### ✅ 解决方案

**好消息：完全无需重新训练！**

用户历史是**推理时动态输入**，不存储在模型中：

```python
# 实时推理
def recommend_for_user(user_id, timestamp=None):
    """为用户实时推荐（自动使用最新历史）"""

    # 步骤1: 从数据库查询最新历史
    history = database.get_user_click_history(
        user_id=user_id,
        limit=50,  # 最近50次点击
        before=timestamp
    )

    # 步骤2: 获取候选新闻
    candidates = database.get_candidate_news(
        exclude=history  # 排除已点击
    )

    # 步骤3: 推理（自动使用最新历史）
    scores = model.predict(
        user_idx=torch.tensor([user_id]),
        candidate_news_indices=torch.tensor(candidates),
        history=torch.tensor([history]),  # 最新历史！
        llm_embeddings=llm_embeddings,
        gnn_embeddings=gnn_embeddings
    )

    # 步骤4: 排序推荐
    top_k = torch.topk(scores, k=10)

    return candidates[top_k.indices]
```

**关键优势**：
- ✅ 历史自动实时更新
- ✅ 无缓存问题
- ✅ 反映用户最新兴趣
- ✅ 完全无需重新训练

**优化建议**：

```python
# 历史长度自适应
def get_adaptive_history(user_id, max_length=50):
    """根据用户活跃度自适应调整历史长度"""

    total_clicks = database.count_user_clicks(user_id)

    if total_clicks < 10:
        # 新用户：使用全部历史
        history_length = total_clicks
    elif total_clicks < 100:
        # 中等用户：最近30次
        history_length = 30
    else:
        # 活跃用户：最近50次（避免过长）
        history_length = 50

    return database.get_user_click_history(
        user_id, limit=history_length
    )
```

---

## 三、完整更新策略总结

### 3.1 决策树

```
数据变化
├─ 新新闻加入？
│   ├─ 数量少（<100）
│   │   └─ 方案A: 零样本推理（无需训练）
│   └─ 数量多（≥100）
│       └─ 方案B: 增量微调（每周/每月）
│
├─ 新用户加入？
│   └─ 无需训练（使用冷启动策略）
│
└─ 用户历史变化？
    └─ 无需训练（推理时动态输入）
```

### 3.2 训练频率建议

| 场景 | 更新方式 | 频率 | 成本 |
|------|---------|------|------|
| **新新闻 (<100条/天)** | 零样本推理 | 实时 | 低（仅API调用） |
| **新新闻 (≥100条/天)** | 增量微调 | 每周 | 中（2-3小时训练） |
| **新新闻 (大量批量)** | 完整重训 | 每月 | 高（3小时训练） |
| **新用户** | 无需训练 | N/A | 无 |
| **用户历史** | 无需训练 | N/A | 无 |
| **模型参数更新** | 完整重训 | 每季度 | 高 |

### 3.3 推荐的生产架构

```
┌─────────────────────────────────────────────────┐
│              在线推理服务 (FastAPI)               │
│  - 加载固定模型 (best_model.pth)                │
│  - 加载LLM嵌入 (实时追加新新闻)                  │
│  - 查询用户历史 (Redis缓存)                      │
│  - 实时推理                                      │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│            离线更新服务 (每周/每月)               │
│  1. 收集新新闻点击数据                           │
│  2. 生成新新闻LLM嵌入                            │
│  3. 更新知识图谱                                 │
│  4. 增量微调ID嵌入（可选）                       │
│  5. 更新模型文件                                 │
└─────────────────────────────────────────────────┘
```

---

## 四、代码实现示例

### 4.1 新新闻零样本推理

```python
# new_news_handler.py
import numpy as np
import torch
from openai import OpenAI

class NewNewsHandler:
    """处理新新闻的零样本推理"""

    def __init__(self, model, llm_emb_path, api_key):
        self.model = model
        self.llm_embeddings = np.load(llm_emb_path)  # (N, 1536)
        self.client = OpenAI(api_key=api_key)
        self.num_news = len(self.llm_embeddings)

    def add_new_news(self, news_id, news_text, entities):
        """添加新新闻"""

        # 1. 生成LLM嵌入
        print(f"Generating LLM embedding for news {news_id}...")
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=news_text
        )
        new_llm_emb = np.array(response.data[0].embedding)

        # 2. 追加到嵌入矩阵
        self.llm_embeddings = np.vstack([
            self.llm_embeddings,
            new_llm_emb
        ])

        # 3. 更新知识图谱（如果有实体）
        if entities:
            self.update_knowledge_graph(news_id, entities)

        # 4. 保存更新后的嵌入
        np.save('llm_embeddings_updated.npy', self.llm_embeddings)

        print(f"✓ Added news {news_id} (total: {len(self.llm_embeddings)})")

    def predict_for_new_news(self, user_id, new_news_id, history):
        """为新新闻推理"""

        if new_news_id >= self.num_news:
            # 新新闻：使用零样本
            print(f"Zero-shot inference for news {new_news_id}")

            # 使用默认ID嵌入（零向量或padding）
            id_emb = torch.zeros(1, 128)
            llm_emb = torch.from_numpy(
                self.llm_embeddings[new_news_id]
            ).float().unsqueeze(0)

            # GNN嵌入（如果图已更新）
            gnn_emb = self.get_gnn_embedding(new_news_id)

            # 手动融合
            with torch.no_grad():
                id_repr = self.model.news_encoder.id_proj(id_emb)
                llm_repr = self.model.news_encoder.llm_proj(llm_emb)
                gnn_repr = self.model.news_encoder.gnn_proj(gnn_emb)

                # 注意力融合
                modalities = torch.stack([id_repr, llm_repr, gnn_repr], dim=1)
                query = self.model.news_encoder.fusion_query.expand(1, -1).unsqueeze(1)
                keys = self.model.news_encoder.fusion_key(modalities)
                scores = torch.bmm(query, keys.transpose(1, 2))
                attn = torch.softmax(scores / (256 ** 0.5), dim=-1)
                news_repr = torch.bmm(attn, modalities).squeeze(1)

            return news_repr
        else:
            # 旧新闻：正常推理
            return self.model.news_encoder(
                torch.tensor([new_news_id]),
                llm_embeddings=torch.from_numpy(self.llm_embeddings).float(),
                gnn_embeddings=None
            )

# 使用示例
handler = NewNewsHandler(model, 'llm_embeddings.npy', api_key="...")
handler.add_new_news(
    news_id=51283,
    news_text="Title: GPT-5 Released | Abstract: OpenAI announces...",
    entities=['OpenAI', 'GPT-5']
)

score = handler.predict_for_new_news(
    user_id=12345,
    new_news_id=51283,
    history=[100, 200, 300]
)
```

### 4.2 增量微调

```python
# incremental_trainer.py
import torch
import torch.nn as nn

class IncrementalTrainer:
    """增量微调新新闻ID嵌入"""

    def __init__(self, model):
        self.model = model

    def expand_embeddings(self, num_new_news, num_new_users=0):
        """扩展嵌入层"""

        # 扩展新闻ID嵌入
        old_num_news = self.model.news_encoder.id_embedding.num_embeddings
        new_num_news = old_num_news + num_new_news

        new_news_emb = nn.Embedding(new_num_news, 128, padding_idx=0)
        with torch.no_grad():
            # 复制旧权重
            new_news_emb.weight[:old_num_news] = \
                self.model.news_encoder.id_embedding.weight
            # 新权重小方差初始化
            nn.init.normal_(new_news_emb.weight[old_num_news:], 0, 0.02)

        self.model.news_encoder.id_embedding = new_news_emb

        # 扩展用户ID嵌入（如需要）
        if num_new_users > 0:
            old_num_users = self.model.user_embedding.num_embeddings
            new_num_users_total = old_num_users + num_new_users

            new_user_emb = nn.Embedding(new_num_users_total, 128)
            with torch.no_grad():
                new_user_emb.weight[:old_num_users] = \
                    self.model.user_embedding.weight
                nn.init.normal_(new_user_emb.weight[old_num_users:], 0, 0.02)

            self.model.user_embedding = new_user_emb

        print(f"✓ Expanded embeddings: {old_num_news} → {new_num_news} news")

    def incremental_finetune(self, new_news_dataloader, epochs=2):
        """增量微调"""

        # 冻结大部分参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 仅解冻新嵌入
        old_num_news = self.model.news_encoder.id_embedding.num_embeddings - \
                      len(new_news_dataloader.dataset)
        self.model.news_encoder.id_embedding.weight[old_num_news:].requires_grad = True

        # 优化器（小学习率）
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4
        )

        # 训练
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in new_news_dataloader:
                optimizer.zero_grad()

                scores = self.model(
                    batch['user_idx'],
                    batch['news_idx'],
                    batch['history'],
                    batch['llm_embeddings'],
                    batch['gnn_embeddings']
                )

                loss = nn.BCELoss()(scores, batch['labels'])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Loss = {total_loss/len(new_news_dataloader):.4f}")

        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        print("✓ Incremental fine-tuning completed")

# 使用示例
trainer = IncrementalTrainer(model)
trainer.expand_embeddings(num_new_news=1000)
trainer.incremental_finetune(new_news_dataloader, epochs=2)
```

---

## 五、最佳实践建议

### 5.1 生产部署建议

```yaml
# 推荐的更新策略
daily:
  - 新新闻LLM嵌入生成（API调用）
  - 知识图谱追加新节点
  - 零样本推理（无需训练）

weekly:
  - 增量微调新新闻ID嵌入（如果新新闻>100条）
  - 更新模型checkpoint

monthly:
  - 完整重训模型（基于全量数据）
  - 更新所有超参数
  - A/B测试新模型

quarterly:
  - 架构升级（如：更换LLM、GNN）
  - 特征工程优化
```

### 5.2 性能监控

```python
# 监控新新闻的推荐质量
def monitor_new_news_performance():
    """监控新新闻的零样本推理质量"""

    metrics = {
        'new_news_ctr': [],  # 新新闻点击率
        'old_news_ctr': [],  # 旧新闻点击率
        'zero_shot_coverage': []  # 零样本覆盖率
    }

    # 如果新新闻CTR < 旧新闻CTR * 0.7
    # → 触发增量微调

    if metrics['new_news_ctr'] < metrics['old_news_ctr'] * 0.7:
        print("⚠️ New news quality degraded, triggering incremental training")
        trigger_incremental_training()
```

### 5.3 数据库设计建议

```sql
-- 新闻表
CREATE TABLE news (
    news_id INT PRIMARY KEY,
    title VARCHAR(500),
    abstract TEXT,
    created_at TIMESTAMP,
    has_id_embedding BOOLEAN DEFAULT FALSE,  -- 是否训练过ID嵌入
    llm_embedding_index INT  -- LLM嵌入在.npy文件中的索引
);

-- 用户点击历史表（时序数据库）
CREATE TABLE user_clicks (
    user_id INT,
    news_id INT,
    clicked_at TIMESTAMP,
    INDEX(user_id, clicked_at)  -- 快速查询最新历史
);
```

---

## 六、总结

### ✅ 无需重新训练的场景（95%情况）

1. **新新闻加入（少量）**
   - 使用零样本推理
   - 仅调用LLM API生成嵌入
   - 性能下降5-10%（可接受）

2. **新用户加入**
   - 使用冷启动策略
   - 依赖LLM语义匹配
   - 随用户点击快速改善

3. **用户历史变化**
   - 推理时动态输入
   - 自动反映最新兴趣
   - 完全无需训练

### ⚠️ 需要增量微调的场景（5%情况）

1. **新新闻大量批量加入**
   - 每周/每月微调
   - 仅训练新ID嵌入
   - 2-3小时完成

### ❌ 需要完整重训的场景（罕见）

1. **模型架构升级**
2. **全量数据重新标注**
3. **超参数大规模调整**

**关键结论**：
> 在当前架构下，**95%的数据变化无需重新训练模型**，
> 依靠LLM嵌入的零样本能力和用户历史的动态输入即可应对。
> 这是多模态架构的核心优势！
