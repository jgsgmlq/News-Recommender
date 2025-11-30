# 🚀 一键运行指南

## API 已配置完成！无需手动输入

你的第三方 OpenAI API 配置已经写入代码：
- **API Key**: `sk-f2BTSMHiHgfs2fj4JgyjszLS5HhfHznJnzx688ZVctR09TR0`
- **Base URL**: `https://api.f2gpt.com/v1`

---

## 快速开始 (推荐)

### 方式 1: 一键测试脚本

```bash
python quick_test.py
```

这个脚本会自动完成：
1. ✅ 检查数据文件
2. ✅ 调用 API 预计算 LLM embeddings (500条新闻)
3. ✅ 训练 LLM+GNN 模型 (3 epochs)
4. ✅ 评估并显示结果

**预计时间**: 5-8 分钟
**预计成本**: < $0.001 (不到 1 分钱)

---

## 方式 2: 分步运行

### Step 1: 预计算 LLM Embeddings

```bash
python src/precompute_llm_embeddings.py \
    --news_path data/mind_tiny/news.tsv \
    --output_path data/mind_tiny/llm_embeddings.npy
```

**说明**:
- API key 和 base_url 已经硬编码在脚本中
- 无需手动指定 `--api_key` 和 `--base_url`
- 如需覆盖，可以手动传参

**输出**:
- `data/mind_tiny/llm_embeddings.npy` (500 × 1536 维向量)
- `data/mind_tiny/llm_embeddings_mapping.pkl` (ID 映射)
- `data/mind_tiny/llm_embeddings_metadata.txt` (元数据)

### Step 2: 训练模型

```bash
python src/train_llm.py \
    --epochs 3 \
    --batch_size 64 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

**输出**:
- `output/llm_gnn/best_model.pth` (训练好的模型)
- `output/llm_gnn/runs/` (TensorBoard 日志)

### Step 3: 评估

```bash
python src/predict_llm.py \
    --model_path output/llm_gnn/best_model.pth \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

**输出**:
- `output/llm_gnn/eval/metrics.json` (评估指标)
- `output/llm_gnn/eval/prediction.txt`
- `output/llm_gnn/eval/truth.txt`

### Step 4: 查看结果

```bash
# 查看评估指标
cat output/llm_gnn/eval/metrics.json

# 启动 TensorBoard
tensorboard --logdir output/
```

---

## 预期效果

### Tiny 数据集 (500 news, 3 epochs)

| 指标 | 预期值 |
|------|-------|
| AUC | ~0.55-0.58 |
| MRR | ~0.30-0.35 |
| nDCG@5 | ~0.20-0.25 |
| nDCG@10 | ~0.23-0.28 |

相比基线模型 (ID only):
- **AUC 提升**: +15-20%
- **MRR 提升**: +25-35%
- **nDCG 提升**: +30-40%

---

## 对比不同配置

### 实验 1: 仅 LLM (无 GNN)

```bash
python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --no_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

### 实验 2: LLM + GNN (完整)

```bash
python src/train_llm.py \
    --epochs 3 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_tiny/llm_embeddings.npy
```

### 实验 3: 不同融合方法

```bash
# 注意力融合 (推荐)
--fusion_method attention

# 门控融合
--fusion_method gate

# 简单拼接
--fusion_method concat
```

---

## 自定义配置

如需使用其他 API 配置，可以覆盖默认值：

```bash
python src/precompute_llm_embeddings.py \
    --api_key YOUR_CUSTOM_KEY \
    --base_url YOUR_CUSTOM_URL \
    --news_path data/mind_tiny/news.tsv \
    --output_path data/mind_tiny/llm_embeddings.npy
```

---

## 常见问题

### Q: API 调用失败？

**错误信息**:
```
Error: Connection timeout / Rate limit
```

**解决**:
1. 检查网络连接
2. 确认 API key 有效
3. 减小 batch_size: `--batch_size 50`
4. 等待几分钟后重试

### Q: 内存不足？

**解决**:
```bash
# 减小批次大小
--batch_size 32

# 使用 CPU (会自动检测)
```

### Q: 想查看实时训练过程？

```bash
# 在另一个终端运行
tensorboard --logdir output/

# 访问
http://localhost:6006
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `quick_test.py` | 一键测试脚本 (推荐) |
| `src/precompute_llm_embeddings.py` | 预计算 LLM embeddings |
| `src/model_llm.py` | 多模态融合模型 |
| `src/train_llm.py` | 训练脚本 |
| `src/predict_llm.py` | 预测评估脚本 |
| `QUICKSTART_LLM.md` | 详细文档 |
| `LLM_EMBEDDING_PROPOSAL.md` | 技术方案 |

---

## 下一步

### 1. 扩展到完整数据集

修改脚本使用完整 MIND-small (51K 新闻):
```bash
python src/precompute_llm_embeddings.py \
    --news_path data/mind_small/train/news.tsv \
    --output_path data/mind_small/llm_embeddings.npy
```

**成本**: ~$0.10 (约 ¥0.7)

### 2. 超参数调优

```bash
# 更大的输出维度
--output_dim 512

# 更多训练轮数
--epochs 10

# 更低的学习率
--lr 0.0005
```

### 3. 对比所有配置

创建对比实验:
- Baseline (ID only)
- +GNN
- +LLM
- +LLM+GNN

查看哪个组合效果最好！

---

## 支持

遇到问题？查看:
- 快速开始: `QUICKSTART_LLM.md`
- 技术方案: `LLM_EMBEDDING_PROPOSAL.md`
- GNN 文档: `GNN_README.md`

---

**准备好了吗？立即运行！** 🚀

```bash
python quick_test.py
```
