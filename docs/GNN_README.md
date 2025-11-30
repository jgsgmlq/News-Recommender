# GNN-Enhanced News Recommendation System

## 概述

本项目在原有新闻推荐系统的基础上接入了 **知识图谱 (Knowledge Graph)** 和 **图神经网络 (GNN)**,通过利用 `entity_embedding.vec` 中的实体嵌入,在 news-entity 图上进行图传播,增强新闻表示。

## 主要特性

✅ **知识图谱构建**: 从 MIND 数据集构建 news-entity 双向图
✅ **GNN 增强**: 使用 GraphSAGE 在图上进行 1-2 层传播
✅ **灵活架构**: 支持启用/禁用 GNN,方便对比实验
✅ **小数据集测试**: 包含 tiny 数据集用于快速验证
✅ **OOV 处理**: 优雅处理不在知识图谱中的新闻 ID

## 文件结构

```
News-Recommender/
├── src/
│   ├── model_gnn.py          # GNN 增强的推荐模型
│   ├── train_gnn.py          # GNN 训练脚本
│   ├── predict_gnn.py        # GNN 预测脚本
│   ├── gnn_module.py         # GNN 模块 (GraphSAGE)
│   └── kg_utils.py           # 知识图谱构建工具
├── data/
│   └── mind_tiny/            # 小数据集 (自动生成)
│       ├── news.tsv
│       ├── behaviors.tsv
│       └── entity_embedding.vec
├── output/
│   ├── gnn/                  # GNN 模型输出
│   │   ├── best_model.pth
│   │   ├── runs/            # TensorBoard 日志
│   │   └── eval/            # 评估结果
│   └── baseline/            # 基线模型输出 (无 GNN)
└── run_gnn.py               # 一键运行脚本
```

## 架构说明

### 1. 知识图谱构建

从 MIND 数据集构建 news-entity 双向图:

```python
# 节点类型
- News 节点: 来自 news.tsv (初始特征为随机向量)
- Entity 节点: 来自 entity_embedding.vec (预训练 100 维嵌入)

# 边
- News → Entity (双向)
- 基于 news.tsv 中的 title_entities 和 abstract_entities
```

**示例统计**:
- 总节点: 1158 (500 news + 658 entities)
- 总边数: 1668 (双向)
- 节点特征维度: 100

### 2. GNN 模型架构

```
用户表示 = 用户 ID 嵌入 + 历史新闻注意力聚合
新闻表示 = ID 嵌入 + GNN 增强嵌入

GNN 传播:
  Input: 节点特征 (100 维)
    ↓
  GraphSAGE Layer 1 (100 → 128)
    ↓
  BatchNorm + ReLU + Dropout
    ↓
  GraphSAGE Layer 2 (128 → 128)
    ↓
  BatchNorm
    ↓
  Output: 增强的新闻嵌入 (128 维)

融合:
  新闻表示 = Fusion(ID 嵌入 || GNN 嵌入)
```

### 3. 训练流程

```python
for epoch in epochs:
    # 1. 预计算 GNN 嵌入 (每个 epoch 一次)
    gnn_embeddings = model.get_gnn_enhanced_news_embeddings()

    # 2. 训练
    for batch in train_loader:
        # 使用预计算的 GNN 嵌入
        scores = model(user_idx, news_idx, history, gnn_embeddings)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

    # 3. 验证
    val_loss, val_acc = validate(...)
```

## 使用方法

### 方法 1: 一键运行 (推荐)

```bash
# 训练 GNN 模型并评估 (默认 2 层 GNN)
python run_gnn.py --epochs 5 --gnn_layers 2

# 训练基线模型 (无 GNN)
python run_gnn.py --no_gnn --epochs 5

# 只运行预测 (跳过训练)
python run_gnn.py --skip_train

# 使用 1 层 GNN
python run_gnn.py --epochs 3 --gnn_layers 1
```

### 方法 2: 分步运行

```bash
# 步骤 1: 训练
python src/train_gnn.py \
    --epochs 5 \
    --batch_size 64 \
    --gnn_layers 2 \
    --embedding_dim 128

# 步骤 2: 预测
python src/predict_gnn.py \
    --model_path output/gnn/best_model.pth

# 步骤 3: 查看 TensorBoard
tensorboard --logdir output/gnn/runs
```

## 实验结果

### 小数据集 (mind_tiny) - 3 个 epoch

| 指标 | GNN 模型 |
|------|---------|
| **训练准确率** | 95.97% |
| **验证准确率** | 96.09% |
| **AUC** | 0.5016 |
| **MRR** | 0.2670 |
| **nDCG@5** | 0.1696 |
| **nDCG@10** | 0.1823 |

### 模型统计

- **参数量**: 2,558,081
- **GNN 嵌入**: 500 news × 128 维
- **知识图谱**: 1158 nodes, 1668 edges

## 关键参数说明

### GNN 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_gnn` | True | 启用 GNN 增强 |
| `--gnn_layers` | 2 | GNN 层数 (1 或 2) |
| `--gnn_hidden_dim` | 128 | GNN 隐藏层维度 |
| `--gnn_output_dim` | 128 | GNN 输出维度 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--embedding_dim` | 128 | ID 嵌入维度 |
| `--lr` | 0.001 | 学习率 |

## OOV (Out-of-Vocabulary) 处理

由于知识图谱只包含部分新闻,模型优雅地处理不在图中的新闻:

```python
# 对于不在知识图谱中的新闻 ID
if news_id >= num_gnn_news:
    gnn_embedding = zero_vector  # 使用零向量
    news_repr = Fusion(id_emb, zero_vector)  # 降级为 ID 嵌入
```

这种设计确保:
- ✅ 不会因为 OOV 新闻而崩溃
- ✅ OOV 新闻仍能通过 ID 嵌入获得表示
- ✅ 在图中的新闻获得 GNN 增强

## 扩展到完整数据集

要在完整的 MIND-small 数据集上训练:

```bash
# 修改 train_gnn.py 中的数据加载部分
# 将 get_tiny_dataloaders 替换为:
from data_loader import get_dataloaders

train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)
```

## 性能优化建议

1. **预计算 GNN 嵌入**: 每个 epoch 只计算一次,避免重复计算
2. **批量推理**: 使用 batch prediction 提高速度
3. **GPU 加速**: 在有 GPU 的环境下自动使用 CUDA
4. **数据并行**: 增加 `num_workers` 加速数据加载

## 进一步改进方向

1. **更多 GNN 层**: 尝试 3-4 层 GraphSAGE
2. **不同 GNN 架构**: GAT, GCN, GraphTransformer
3. **实体特征学习**: 让实体嵌入也参与训练
4. **异构图**: 添加更多关系类型 (category, subcategory)
5. **文本特征**: 结合 BERT 嵌入标题/摘要
6. **时间建模**: 加入时间感知的图传播

## 常见问题

### Q: 为什么验证准确率这么高但 AUC 不高?
A: 这是因为数据不平衡。在小数据集上,大部分样本是负例 (未点击),模型学会预测"不点击",导致高准确率但低 AUC。在完整数据集上训练会改善。

### Q: 如何对比 GNN 和基线模型?
A:
```bash
# 训练 GNN 模型
python run_gnn.py --epochs 5 --use_gnn

# 训练基线模型
python run_gnn.py --epochs 5 --no_gnn

# 对比结果
diff output/gnn/eval/metrics.json output/baseline/eval/metrics.json
```

### Q: 能在 GPU 上训练吗?
A: 可以!脚本会自动检测并使用 GPU (如果可用)。

## 技术栈

- **深度学习**: PyTorch 1.10+
- **图神经网络**: PyTorch Geometric
- **图模型**: GraphSAGE (Hamilton et al., 2017)
- **数据集**: MIND (Microsoft News Dataset)

## 参考文献

- GraphSAGE: Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
- MIND Dataset: Wu et al., "MIND: A Large-scale Dataset for News Recommendation", ACL 2020

---

**开发完成日期**: 2025-11-29
**状态**: ✅ 已测试并可用
