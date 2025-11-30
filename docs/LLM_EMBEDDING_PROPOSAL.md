# LLM Embedding 增强新闻推荐系统 - 技术方案

## 1. 方案概述

在现有的 **GNN + KG** 推荐系统基础上，接入 **LLM Embedding API**，利用大模型的文本理解能力生成高质量新闻向量，提升推荐效果。

### 1.1 核心思想

```
原架构: 新闻表示 = ID嵌入 + GNN实体嵌入

升级后: 新闻表示 = ID嵌入 + GNN实体嵌入 + LLM文本嵌入
                         ↓
                     多模态融合层
                         ↓
                  增强的新闻表示
```

### 1.2 预期收益

✅ **更好的语义理解**: LLM 能捕捉标题/摘要的深层语义
✅ **冷启动能力**: 新新闻即使没有 ID 嵌入，也能有高质量表示
✅ **跨新闻泛化**: 相似主题的新闻在向量空间中更接近
✅ **多语言支持**: LLM 支持多语言文本（如需要）

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    数据准备阶段 (离线)                        │
└─────────────────────────────────────────────────────────────┘

新闻数据 (news.tsv)
  ├─ 标题 (title)
  ├─ 摘要 (abstract)
  └─ 实体 (entities)
      ↓
  [文本拼接]
      ↓
  "Title: {title}\nAbstract: {abstract}"
      ↓
  [批量调用 LLM Embedding API]
      ↓
  LLM Embeddings (51K × 1536维)
      ↓
  [保存到文件]
      ↓
  news_llm_embeddings.npy


┌─────────────────────────────────────────────────────────────┐
│                    训练/推理阶段 (在线)                       │
└─────────────────────────────────────────────────────────────┘

输入: 新闻 ID
  ↓
┌───────────────┬───────────────┬───────────────┐
│   ID Embedding│ LLM Embedding │  GNN Embedding│
│   (128维)     │  (1536维)     │   (128维)     │
└───────────────┴───────────────┴───────────────┘
         ↓              ↓              ↓
         └──────────────┼──────────────┘
                        ↓
            [多模态融合层 - Attention Gate]
                        ↓
            投影到统一维度 (256维)
                        ↓
            最终新闻表示 (256维)
                        ↓
            与用户表示计算相似度
                        ↓
              点击概率预测
```

### 2.2 模型架构细节

```python
class MultiModalNewsEncoder(nn.Module):
    """
    多模态新闻编码器
    融合: ID嵌入 + LLM文本嵌入 + GNN实体嵌入
    """

    def __init__(
        self,
        num_news,
        id_emb_dim=128,      # ID 嵌入维度
        llm_emb_dim=1536,    # LLM 嵌入维度 (OpenAI)
        gnn_emb_dim=128,     # GNN 嵌入维度
        output_dim=256,      # 输出维度
        use_llm=True,        # 是否使用 LLM
        use_gnn=True,        # 是否使用 GNN
        fusion_method='attention'  # 融合方法
    ):
        super().__init__()

        # 1. ID 嵌入
        self.id_embedding = nn.Embedding(num_news, id_emb_dim)

        # 2. LLM 嵌入投影层
        if use_llm:
            self.llm_proj = nn.Sequential(
                nn.Linear(llm_emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

        # 3. GNN (已有)
        if use_gnn:
            self.gnn = NewsEntityGNN(...)
            self.gnn_proj = nn.Sequential(
                nn.Linear(gnn_emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

        # 4. ID 嵌入投影层
        self.id_proj = nn.Sequential(
            nn.Linear(id_emb_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 5. 融合层
        if fusion_method == 'attention':
            # 注意力门控融合
            self.fusion_gate = nn.Sequential(
                nn.Linear(output_dim * 3, 3),
                nn.Softmax(dim=-1)
            )
        elif fusion_method == 'concat':
            # 简单拼接 + MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(output_dim * 2, output_dim)
            )

    def forward(self, news_ids, llm_embeddings=None, gnn_embeddings=None):
        """
        Args:
            news_ids: (batch_size,)
            llm_embeddings: (num_news, llm_emb_dim) 预加载的 LLM 嵌入
            gnn_embeddings: (num_news, gnn_emb_dim) 预计算的 GNN 嵌入

        Returns:
            news_repr: (batch_size, output_dim)
        """
        # 1. 获取三种表示
        id_emb = self.id_embedding(news_ids)  # (B, id_emb_dim)
        id_repr = self.id_proj(id_emb)        # (B, output_dim)

        # 2. LLM 表示
        if llm_embeddings is not None:
            llm_emb = llm_embeddings[news_ids]  # (B, llm_emb_dim)
            llm_repr = self.llm_proj(llm_emb)   # (B, output_dim)
        else:
            llm_repr = torch.zeros_like(id_repr)

        # 3. GNN 表示
        if gnn_embeddings is not None:
            gnn_emb = gnn_embeddings[news_ids]  # (B, gnn_emb_dim)
            gnn_repr = self.gnn_proj(gnn_emb)   # (B, output_dim)
        else:
            gnn_repr = torch.zeros_like(id_repr)

        # 4. 融合
        if self.fusion_method == 'attention':
            # 注意力门控
            concat = torch.cat([id_repr, llm_repr, gnn_repr], dim=-1)
            gate = self.fusion_gate(concat)  # (B, 3)

            news_repr = (
                gate[:, 0:1] * id_repr +
                gate[:, 1:2] * llm_repr +
                gate[:, 2:3] * gnn_repr
            )
        else:
            # 简单拼接
            concat = torch.cat([id_repr, llm_repr, gnn_repr], dim=-1)
            news_repr = self.fusion_mlp(concat)

        return news_repr
```

---

## 3. LLM Embedding API 选择

### 3.1 主流 API 对比

| API 服务 | 模型名称 | 维度 | 价格 (USD/1M tokens) | 性能 | 推荐度 |
|---------|---------|------|---------------------|------|--------|
| **OpenAI** | text-embedding-3-small | 1536 | $0.02 | ⭐⭐⭐⭐ | ✅ 推荐 |
| **OpenAI** | text-embedding-3-large | 3072 | $0.13 | ⭐⭐⭐⭐⭐ | ⚠️ 贵 |
| **OpenAI** | text-embedding-ada-002 | 1536 | $0.10 | ⭐⭐⭐ | 被淘汰 |
| **智谱AI** | embedding-2 | 1024 | ¥0.0005/千tokens | ⭐⭐⭐⭐ | ✅ 国内首选 |
| **百度文心** | embedding-v1 | 384 | ¥0.002/千tokens | ⭐⭐⭐ | 可用 |
| **阿里通义** | text-embedding-v2 | 1536 | ¥0.0007/千tokens | ⭐⭐⭐⭐ | ✅ 推荐 |
| **Cohere** | embed-english-v3.0 | 1024 | $0.10 | ⭐⭐⭐⭐ | 英文优 |

### 3.2 推荐配置

**方案 1: OpenAI (国际)**
```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["News title and abstract here"],
    encoding_format="float"
)
embedding = response.data[0].embedding  # 1536维
```

**方案 2: 智谱AI (国内)**
```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="...")

response = client.embeddings.create(
    model="embedding-2",
    input="News title and abstract here"
)
embedding = response.data[0].embedding  # 1024维
```

**方案 3: 阿里通义 (国内，推荐)**
```python
import dashscope

response = dashscope.TextEmbedding.call(
    model=dashscope.TextEmbedding.Models.text_embedding_v2,
    input="News title and abstract here"
)
embedding = response.output['embeddings'][0]['embedding']  # 1536维
```

---

## 4. 实施方案

### 4.1 阶段 1: 预计算 LLM Embeddings (离线)

```python
# 脚本: src/precompute_llm_embeddings.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def precompute_llm_embeddings(
    news_path,
    output_path,
    api_key,
    model="text-embedding-3-small",
    batch_size=100
):
    """
    预计算所有新闻的 LLM embeddings

    Args:
        news_path: news.tsv 路径
        output_path: 输出 .npy 文件路径
        api_key: LLM API key
        batch_size: 批量调用大小
    """
    # 1. 加载新闻数据
    news_df = pd.read_csv(news_path, sep='\t', ...)

    # 2. 构建文本
    texts = []
    for _, row in news_df.iterrows():
        title = row['title'] if pd.notna(row['title']) else ""
        abstract = row['abstract'] if pd.notna(row['abstract']) else ""

        # 拼接策略
        text = f"Title: {title}\nAbstract: {abstract}"
        texts.append(text)

    # 3. 批量调用 API
    client = OpenAI(api_key=api_key)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        try:
            response = client.embeddings.create(
                model=model,
                input=batch
            )

            batch_embeddings = [
                data.embedding for data in response.data
            ]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            print(f"Error at batch {i}: {e}")
            # 降级策略：使用零向量
            embeddings.extend([
                [0.0] * 1536 for _ in batch
            ])

    # 4. 保存
    embeddings = np.array(embeddings, dtype=np.float32)

    # 创建映射: news_id -> embedding index
    news_id_to_idx = {
        news_id: idx
        for idx, news_id in enumerate(news_df['news_id'])
    }

    np.save(output_path, embeddings)

    # 保存映射
    import pickle
    with open(output_path.replace('.npy', '_id_mapping.pkl'), 'wb') as f:
        pickle.dump(news_id_to_idx, f)

    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Shape: {embeddings.shape}")

    return embeddings, news_id_to_idx
```

### 4.2 阶段 2: 数据加载器改造

```python
# 修改: src/data_loader.py

class MINDDatasetWithLLM(MINDDataset):
    """扩展数据集，支持加载 LLM embeddings"""

    def __init__(self, behaviors_path, news_path,
                 llm_embedding_path=None, mode='train'):
        super().__init__(behaviors_path, news_path, mode)

        # 加载 LLM embeddings
        if llm_embedding_path and os.path.exists(llm_embedding_path):
            self.llm_embeddings = np.load(llm_embedding_path)
            print(f"Loaded LLM embeddings: {self.llm_embeddings.shape}")
        else:
            self.llm_embeddings = None

    def get_llm_embeddings_tensor(self):
        """返回 PyTorch tensor"""
        if self.llm_embeddings is not None:
            return torch.from_numpy(self.llm_embeddings).float()
        return None
```

### 4.3 阶段 3: 模型改造

```python
# 新文件: src/model_llm.py

class LLMEnhancedRecommender(nn.Module):
    """
    LLM + GNN + ID 三模态融合推荐模型
    """

    def __init__(
        self,
        num_users,
        num_news,
        embedding_dim=128,
        llm_emb_dim=1536,
        gnn_emb_dim=128,
        output_dim=256,
        use_llm=True,
        use_gnn=True,
        dropout=0.2
    ):
        super().__init__()

        # 用户编码器 (复用原有)
        self.user_encoder = UserEncoder(...)

        # 新闻编码器 (多模态)
        self.news_encoder = MultiModalNewsEncoder(
            num_news=num_news,
            id_emb_dim=embedding_dim,
            llm_emb_dim=llm_emb_dim,
            gnn_emb_dim=gnn_emb_dim,
            output_dim=output_dim,
            use_llm=use_llm,
            use_gnn=use_gnn
        )

    def forward(self, user_idx, news_idx, history,
                llm_embeddings=None, gnn_embeddings=None):
        """
        Args:
            llm_embeddings: (num_news, llm_emb_dim) 全局 LLM 嵌入
            gnn_embeddings: (num_news, gnn_emb_dim) 全局 GNN 嵌入
        """
        # 用户表示
        user_repr = self.user_encoder(user_idx, history, ...)

        # 新闻表示 (多模态融合)
        news_repr = self.news_encoder(
            news_idx,
            llm_embeddings=llm_embeddings,
            gnn_embeddings=gnn_embeddings
        )

        # 相似度
        scores = torch.sum(user_repr * news_repr, dim=1)
        scores = torch.sigmoid(scores)

        return scores
```

### 4.4 阶段 4: 训练脚本改造

```python
# 新文件: src/train_llm.py

def main(args):
    # 1. 加载数据
    train_dataset = MINDDatasetWithLLM(
        behaviors_path=...,
        news_path=...,
        llm_embedding_path=args.llm_embedding_path
    )

    # 2. 获取 LLM embeddings (一次性加载到内存/GPU)
    llm_embeddings = train_dataset.get_llm_embeddings_tensor()
    if llm_embeddings is not None:
        llm_embeddings = llm_embeddings.to(device)

    # 3. 创建模型
    model = LLMEnhancedRecommender(
        num_users=num_users,
        num_news=num_news,
        use_llm=args.use_llm,
        use_gnn=args.use_gnn
    ).to(device)

    # 4. 训练循环
    for epoch in range(args.epochs):
        # 预计算 GNN embeddings (如果需要)
        if args.use_gnn:
            gnn_embeddings = model.get_gnn_embeddings()
        else:
            gnn_embeddings = None

        # 训练
        for batch in train_loader:
            scores = model(
                user_idx, news_idx, history,
                llm_embeddings=llm_embeddings,  # 全局共享
                gnn_embeddings=gnn_embeddings   # 全局共享
            )
            ...
```

---

## 5. 成本估算

### 5.1 MIND-small 数据集

| 项目 | 数量 | 详情 |
|------|------|------|
| 训练集新闻 | 51,282 | |
| 验证集新闻 | 42,416 | |
| **总新闻数** | **~51K** | 去重后 |
| 平均文本长度 | ~100 tokens | title + abstract |
| **总 tokens** | **~5M** | |

### 5.2 API 成本

**OpenAI text-embedding-3-small**
- 价格: $0.02 / 1M tokens
- 总成本: 5M × $0.02 = **$0.10** (约 ¥0.7)

**智谱AI embedding-2**
- 价格: ¥0.0005 / 1K tokens
- 总成本: 5M × ¥0.0005 = **¥2.5**

**阿里通义 text-embedding-v2**
- 价格: ¥0.0007 / 1K tokens
- 总成本: 5M × ¥0.0007 = **¥3.5**

**结论**: 成本极低，一次性投入可忽略不计。

---

## 6. 优化策略

### 6.1 批量调用优化

```python
# 批量大小建议
batch_size = 100  # OpenAI 支持最多 2048 个输入

# 并发调用 (谨慎使用，避免触发限流)
import asyncio
from openai import AsyncOpenAI

async def batch_embed(texts, batch_size=100):
    client = AsyncOpenAI(api_key=...)

    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        task = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    return responses
```

### 6.2 缓存策略

```python
import hashlib
import pickle
from pathlib import Path

class EmbeddingCache:
    """LLM Embedding 缓存"""

    def __init__(self, cache_dir="./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_key(self, text, model):
        """生成缓存键"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text, model):
        """获取缓存"""
        key = self._get_key(text, model)
        cache_file = self.cache_dir / f"{key}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, text, model, embedding):
        """设置缓存"""
        key = self._get_key(text, model)
        cache_file = self.cache_dir / f"{key}.pkl"

        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
```

### 6.3 错误处理

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_embedding_api(texts, model="text-embedding-3-small"):
    """带重试的 API 调用"""
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [data.embedding for data in response.data]

    except Exception as e:
        print(f"API Error: {e}")
        raise
```

---

## 7. 实验对比方案

### 7.1 对比实验设计

| 模型 | ID嵌入 | GNN | LLM | 预期效果 |
|------|--------|-----|-----|---------|
| Baseline | ✅ | ❌ | ❌ | 基准 |
| +GNN | ✅ | ✅ | ❌ | +5% AUC |
| +LLM | ✅ | ❌ | ✅ | +10% AUC |
| **+GNN+LLM** | ✅ | ✅ | ✅ | **+15% AUC** |

### 7.2 评估指标

```python
# 主要指标
- AUC (Area Under Curve)
- MRR (Mean Reciprocal Rank)
- nDCG@5, nDCG@10
- Hit Rate@10

# 额外分析
- 冷启动性能 (新新闻 Top-K)
- 长尾新闻覆盖率
- 不同类别新闻的性能
```

---

## 8. 实施时间线

| 阶段 | 任务 | 时间 | 负责人 |
|------|------|------|--------|
| **Week 1** | 预计算 LLM embeddings | 1天 | 数据工程师 |
| | 数据加载器改造 | 1天 | |
| | 模型架构设计 | 2天 | 算法工程师 |
| **Week 2** | 训练脚本改造 | 2天 | |
| | 基础实验 (Baseline vs +LLM) | 2天 | |
| **Week 3** | 完整实验 (+GNN+LLM) | 2天 | |
| | 超参数调优 | 2天 | |
| **Week 4** | 效果评估和报告 | 2天 | |
| | 部署和上线 | 2天 | |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| API 限流 | 预计算失败 | 批量调用 + 重试 + 缓存 |
| 成本超预算 | 项目中止 | 先在 tiny 数据集测试 |
| 效果不佳 | 浪费时间 | 快速实验，设定早停条件 |
| 维度不匹配 | 模型训练失败 | 投影层对齐维度 |
| 内存不足 | 训练崩溃 | 使用 float16 + 梯度累积 |

---

## 10. 下一步行动

### 10.1 立即开始

1. ✅ 选择 LLM API (推荐: 阿里通义 / OpenAI)
2. ✅ 在 tiny 数据集上预计算 embeddings (500 条新闻)
3. ✅ 实现简单融合模型 (Concat 方法)
4. ✅ 运行基础实验，验证可行性

### 10.2 快速验证脚本

```bash
# Step 1: 预计算 (tiny 数据集)
python src/precompute_llm_embeddings.py \
    --news_path data/mind_tiny/news.tsv \
    --output_path data/mind_tiny/llm_embeddings.npy \
    --api_key YOUR_API_KEY \
    --model text-embedding-3-small

# Step 2: 训练
python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy

# Step 3: 对比
python compare_results.py \
    --baseline output/baseline/eval/metrics.json \
    --gnn output/gnn/eval/metrics.json \
    --llm output/llm/eval/metrics.json \
    --gnn_llm output/gnn_llm/eval/metrics.json
```

---

## 11. 参考文献

1. **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
2. **智谱AI**: https://open.bigmodel.cn/dev/api#text_embedding
3. **阿里通义**: https://help.aliyun.com/document_detail/2587498.html
4. **Multi-Modal Fusion**: "Multimodal Learning with Transformers" (ACL 2022)
5. **News Recommendation**: "Neural News Recommendation with Multi-Head Self-Attention" (EMNLP 2019)

---

**方案制定日期**: 2025-11-29
**版本**: v1.0
**状态**: 📋 待实施
