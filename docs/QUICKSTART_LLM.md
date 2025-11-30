# LLM Embedding 快速开始指南

## 📋 准备工作

### 1. 安装依赖

```bash
pip install openai tenacity
```

### 2. 获取 OpenAI API Key

1. 访问 https://platform.openai.com/api-keys
2. 创建新的 API key
3. 复制并保存 (格式: `sk-...`)

---

## 🚀 快速验证 (Tiny 数据集)

### Step 1: 预计算 LLM Embeddings

在 tiny 数据集 (500 条新闻) 上预计算 embeddings:

```bash
python src/precompute_llm_embeddings.py \
    --api_key sk-YOUR_API_KEY_HERE \
    --news_path data/mind_tiny/news.tsv \
    --output_path data/mind_tiny/llm_embeddings.npy \
    --model text-embedding-3-small \
    --batch_size 100
```

**预期:**
- 运行时间: ~2-3 分钟
- API 调用: 5 次 (500 / 100)
- 成本: **$0.001** (约 ¥0.007, 不到 1 分钱)
- 输出文件:
  - `llm_embeddings.npy` (500 × 1536 维)
  - `llm_embeddings_mapping.pkl` (ID 映射)
  - `llm_embeddings_metadata.txt` (元数据)

### Step 2: 训练多模态模型

#### 实验 1: 仅 LLM (对比基线)

```bash
python src/train_llm.py \
    --epochs 3 \
    --batch_size 64 \
    --use_llm \
    --no_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy \
    --fusion_method attention
```

#### 实验 2: LLM + GNN (完整模型)

```bash
python src/train_llm.py \
    --epochs 3 \
    --batch_size 64 \
    --use_llm \
    --use_gnn \
    --gnn_layers 2 \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy \
    --fusion_method attention
```

**预期训练时间**: ~5 分钟 (CPU) / ~2 分钟 (GPU)

### Step 3: 预测和评估

```bash
python src/predict_llm.py \
    --model_path output/llm_gnn/best_model.pth \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

### Step 4: 查看结果

```bash
# 查看评估指标
cat output/llm_gnn/eval/metrics.json

# 查看 TensorBoard
tensorboard --logdir output/
```

---

## 📊 预期效果提升

### Tiny 数据集 (500 news, 3 epochs)

| 模型 | AUC | MRR | nDCG@10 |
|------|-----|-----|---------|
| Baseline (ID only) | 0.50 | 0.27 | 0.18 |
| +GNN | 0.50 | 0.27 | 0.18 |
| **+LLM** | **0.55** | **0.32** | **0.23** |
| **+LLM+GNN** | **0.58** | **0.35** | **0.25** |

**预期提升**:
- AUC: +0.08 (+16%)
- MRR: +0.08 (+30%)
- nDCG@10: +0.07 (+40%)

---

## 🔧 参数调优建议

### 融合方法对比

```bash
# 注意力融合 (推荐)
--fusion_method attention

# 门控融合
--fusion_method gate

# 拼接融合 (最简单)
--fusion_method concat
```

### 输出维度调整

```bash
# 默认: 256 维
--output_dim 256

# 更大容量 (如果数据足够)
--output_dim 512

# 更小容量 (防止过拟合)
--output_dim 128
```

---

## 💰 成本估算

### Tiny 数据集 (500 news)

| 模型 | Tokens | API 调用 | 成本 (USD) | 成本 (CNY) |
|------|--------|---------|-----------|-----------|
| text-embedding-3-small | 50K | 5 | $0.001 | ¥0.007 |
| text-embedding-3-large | 50K | 5 | $0.0065 | ¥0.045 |

### 完整 MIND-small (51K news)

| 模型 | Tokens | API 调用 | 成本 (USD) | 成本 (CNY) |
|------|--------|---------|-----------|-----------|
| text-embedding-3-small | 5M | 510 | $0.10 | ¥0.70 |
| text-embedding-3-large | 5M | 510 | $0.65 | ¥4.50 |

**结论**: 成本极低，完全可以快速迭代实验

---

## 🐛 常见问题

### Q1: API 调用失败

```python
# 错误信息
RateLimitError: Rate limit reached

# 解决方案
--batch_size 50  # 减小批次
# 或等待几分钟后重试
```

### Q2: 内存不足

```python
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案 1: 使用 CPU
# (会自动检测，无需设置)

# 解决方案 2: 减小批次
--batch_size 32
```

### Q3: LLM embeddings 未加载

```bash
# 确认文件存在
ls data/mind_tiny/llm_embeddings.npy

# 确认路径正确
--llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

### Q4: 效果提升不明显

**可能原因**:
1. 数据集太小 (试试完整数据集)
2. 训练不充分 (增加 epochs)
3. 融合方法不当 (试试 `attention`)
4. 过拟合 (检查 train vs val 曲线)

---

## 🔬 实验对比脚本

创建一个对比脚本,自动运行所有实验:

```bash
# compare_all.sh

#!/bin/bash

echo "===== Experiment 1: Baseline (ID only) ====="
python src/train_llm.py \
    --epochs 3 \
    --no_llm \
    --no_gnn

echo "===== Experiment 2: +GNN ====="
python src/train_llm.py \
    --epochs 3 \
    --no_llm \
    --use_gnn

echo "===== Experiment 3: +LLM ====="
python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --no_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy

echo "===== Experiment 4: +LLM+GNN ====="
python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy

echo "===== Comparing Results ====="
python compare_results.py
```

---

## 📈 TensorBoard 实时监控

```bash
# 启动 TensorBoard
tensorboard --logdir output/

# 访问
http://localhost:6006
```

**可视化内容**:
- 训练/验证损失
- 训练/验证准确率
- 学习率变化
- 不同模型的对比

---

## 🎯 下一步

### 1. 扩展到完整数据集

```bash
# 预计算完整数据集 (51K news)
python src/precompute_llm_embeddings.py \
    --api_key sk-... \
    --news_path data/mind_small/train/news.tsv \
    --output_path data/mind_small/llm_embeddings.npy

# 训练 (修改数据加载为完整数据集)
# 需要修改 train_llm.py 中的 get_tiny_dataloaders
```

### 2. 尝试其他融合方法

```bash
# Gate fusion
--fusion_method gate

# Concat fusion
--fusion_method concat
```

### 3. 超参数调优

```bash
# 更大的输出维度
--output_dim 512

# 更深的 GNN
--gnn_layers 3

# 更低的学习率
--lr 0.0005
```

### 4. 分析注意力权重

查看模型学习到的融合权重:
- ID embedding 权重
- LLM embedding 权重
- GNN embedding 权重

---

## 💡 Tips

1. **先在 tiny 上验证**: 快速迭代,成本低
2. **使用 TensorBoard**: 实时监控训练过程
3. **保存所有实验结果**: 方便对比
4. **注意过拟合**: 监控 train/val gap
5. **GPU 加速**: 如有 GPU,可大幅缩短训练时间

---

## 📞 支持

遇到问题? 查看:
- 详细文档: `LLM_EMBEDDING_PROPOSAL.md`
- 技术方案: `GNN_README.md`
- OpenAI 文档: https://platform.openai.com/docs/guides/embeddings

---

**准备好了吗? 让我们开始吧!** 🚀

```bash
# 一键运行 (需要先设置 API key)
export OPENAI_API_KEY=sk-...

python src/precompute_llm_embeddings.py \
    --api_key $OPENAI_API_KEY \
    --news_path data/mind_tiny/news.tsv \
    --output_path data/mind_tiny/llm_embeddings.npy

python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```
