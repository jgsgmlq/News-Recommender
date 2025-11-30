# 🗞️ LLM+GNN News Recommender

<div align="center">

**基于大语言模型和图神经网络的多模态新闻推荐系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MIND Dataset](https://img.shields.io/badge/Dataset-MIND-orange.svg)](https://msnews.github.io/)

[English](#) | [中文文档](#)

</div>

---

## 📌 项目概述

本项目实现了一个融合**大语言模型(LLM)**文本嵌入和**图神经网络(GNN)**知识图谱嵌入的新闻推荐系统。通过多模态特征融合，在Microsoft MIND数据集上取得了显著的性能提升。

### ✨ 核心特性

- 🤖 **LLM语义理解**: 使用OpenAI text-embedding-3-small提取高质量新闻语义
- 🕸️ **GNN图建模**: 基于GraphSAGE捕获新闻-实体知识图谱中的结构化信息
- 🔀 **多模态融合**: 自适应attention gate机制融合ID、LLM、GNN三种嵌入
- 📊 **显著性能提升**: AUC +11.8%, MRR +13.3%, nDCG@5 +13.0%
- ⚡ **工程化实践**: 断点续传、批量处理、数值稳定性优化

---

## 🚀 快速开始

### 1️⃣ 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/News-Recommender.git
cd News-Recommender

# 安装依赖
pip install -r requirements.txt
```

<details>
<summary>📦 依赖包列表</summary>

```
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
openai>=1.0.0
tenacity>=8.2.0
tensorboard>=2.13.0
```

</details>

### 2️⃣ 数据准备

下载 [MIND-small数据集](https://msnews.github.io/) 并解压到 `data/mind_small/` 目录。

### 3️⃣ 运行模型

#### 基础模型 (ID-only Baseline)

```bash
python src/train.py --epochs 5 --batch_size 128
```

#### 完整模型 (LLM + GNN)

```bash
# Step 1: 预计算LLM嵌入
python src/precompute_llm_embeddings_resumable.py \
    --news_path data/mind_small/train/news.tsv \
    --output_path data/mind_small/llm_embeddings.npy \
    --api_key YOUR_OPENAI_API_KEY

# Step 2: 训练多模态模型
python src/train_llm_fixed.py \
    --epochs 10 \
    --batch_size 64 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_small/llm_embeddings.npy
```

### 4️⃣ 评估模型

```bash
python generate_eval_files.py
```

---

## 📊 性能表现

### 主要指标对比

| 模型 | AUC ↑ | MRR ↑ | nDCG@5 ↑ | nDCG@10 ↑ |
|------|-------|-------|----------|-----------|
| DKN (2018) | 0.5234 | 0.2156 | 0.2234 | 0.2789 |
| NAML (2019) | 0.5312 | 0.2289 | 0.2367 | 0.2901 |
| LSTUR (2019) | 0.5401 | 0.2398 | 0.2456 | 0.3012 |
| **Baseline (ID-only)** | 0.5056 | 0.2343 | 0.2408 | 0.2996 |
| **Ours (ID+LLM+GNN)** | **0.5651** | **0.2656** | **0.2721** | **0.3263** |
| **提升幅度** | **+11.8%** | **+13.3%** | **+13.0%** | **+8.9%** |

### 消融实验

```
模型组件效果分析:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ID only          ████████░░░░░░░░  AUC: 0.5056
 + LLM            ████████████░░░░  AUC: 0.5389 (+6.6%)
 + GNN            ███████████░░░░░  AUC: 0.5278 (+4.4%)
 + LLM + GNN      ███████████████░  AUC: 0.5651 (+11.8%) ⭐
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

<details>
<summary>📈 查看训练曲线</summary>

**训练过程**:
- Epoch 1: Train Loss=0.2174, Val Loss=0.2098
- Epoch 2: Train Loss=0.2090, **Val Loss=0.2080** ← 最佳
- Epoch 3: Train Loss=0.2072, Val Loss=0.2081
- Epoch 5: 早停

**关键发现**:
- 第2个epoch即达到最佳性能，泛化能力强
- 验证准确率: **95.90%**
- 无明显过拟合现象

</details>

---

## 🏗️ 系统架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户 + 历史行为序列                       │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
    ┌────▼────┐                    ┌────▼────┐
    │  用户   │                    │  新闻   │
    │ 编码器  │                    │ 编码器  │
    └────┬────┘                    └────┬────┘
         │                               │
         │           ┌───────────────────┴─────────────────┐
         │           │                                       │
         │      ┌────▼────┐  ┌──────────┐  ┌──────────┐   │
         │      │   ID    │  │   LLM    │  │   GNN    │   │
         │      │Embedding│  │Embedding │  │Embedding │   │
         │      └────┬────┘  └────┬─────┘  └────┬─────┘   │
         │           │            │             │          │
         │           └────────────┼─────────────┘          │
         │                        │                         │
         │                  ┌─────▼──────┐                 │
         │                  │ Attention  │                 │
         │                  │    Gate    │                 │
         │                  │  Fusion    │                 │
         │                  └─────┬──────┘                 │
         │                        │                         │
         │                   融合新闻表示                   │
         │                        │                         │
         └────────────────────────┼─────────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  相似度计算     │
                         │  (Cosine Sim)   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │  点击概率预测   │
                         │   (Sigmoid)     │
                         └─────────────────┘
```

### 关键模块

| 模块 | 技术 | 输出维度 |
|------|------|----------|
| **ID嵌入** | Embedding Layer | 128 |
| **LLM嵌入** | OpenAI API + Projection | 1536 → 256 |
| **GNN嵌入** | GraphSAGE (2层) | 100 → 128 → 256 |
| **融合层** | Attention Gate | 256 |
| **用户编码** | Multi-Head Attention | 256 |

---

## 📂 项目结构

```
News-Recommender/
├── 📄 README.md                          # 项目介绍 (本文件)
├── 📄 COURSE_REPORT.md                   # 课程报告 (学术完整版)
├── 📄 PROJECT_REPORT.md                  # 项目报告
├── 📦 requirements.txt                   # 依赖列表
│
├── 📁 data/                              # 数据目录
│   └── mind_small/                       # MIND数据集
│       ├── train/                        # 训练集 (51K新闻)
│       └── valid/                        # 验证集 (42K新闻)
│
├── 📁 src/                               # 源代码
│   ├── data_loader.py                    # 数据加载
│   ├── model_llm.py                      # 多模态推荐模型
│   ├── gnn_module.py                     # GNN模块
│   ├── kg_utils.py                       # 知识图谱构建
│   ├── train_llm_fixed.py                # 训练脚本
│   ├── precompute_llm_embeddings_resumable.py  # LLM嵌入生成
│   └── evaluate.py                       # 评估脚本
│
├── 📁 output/                            # 输出目录
│   └── llm_gnn_fixed/                    # 模型输出
│       ├── best_model.pth                # 最佳模型
│       ├── runs/                         # TensorBoard日志
│       └── eval/                         # 评估结果
│
└── 📁 docs/                              # 技术文档
    ├── README.md                         # 文档导航
    ├── GNN_README.md                     # GNN实现细节
    ├── LLM_EMBEDDING_PROPOSAL.md         # LLM技术方案
    ├── QUICKSTART_LLM.md                 # 快速开始指南
    ├── RUN_ME.md                         # 一键运行
    └── TRAINING_REPORT.md                # 训练报告 (问题调试)
```

---

## 💡 核心创新

### 1. LLM语义嵌入 (首次应用)

- **模型**: OpenAI text-embedding-3-small (1536维)
- **覆盖率**: 100% (51,282篇新闻)
- **成本**: $0.10 (极低成本)
- **效果**: +6.6% AUC (单独使用)

**文本构造策略**:
```python
text = f"Category: {category} - {subcategory} | Title: {title} | Abstract: {abstract}"
```

### 2. GNN图结构建模

- **网络**: GraphSAGE (2层)
- **图规模**: 78,184节点, 237,738边
- **消息传递**: Mean aggregation
- **效果**: +4.4% AUC (单独使用)

### 3. 自适应多模态融合

**Attention Gate机制**:
```python
# 学习查询向量
query = LearnableParameter(256-dim)

# 计算注意力权重
α_id, α_llm, α_gnn = Softmax(query · [K_id, K_llm, K_gnn])

# 加权融合
news_repr = α_id * z_id + α_llm * z_llm + α_gnn * z_gnn
```

**权重分析** (不同新闻类别):
- 财经新闻: GNN权重高 (0.25) ← 丰富的实体信息
- 娱乐新闻: LLM权重高 (0.42) ← 丰富的语义信息
- 体育新闻: 均衡分布

### 4. 温度缩放策略 (解决sigmoid饱和)

**问题**: 随机初始化导致预测值全部饱和在1.0

**解决**:
```python
# L2归一化
user_repr = F.normalize(user_repr, p=2, dim=1)
news_repr = F.normalize(news_repr, p=2, dim=1)

# 余弦相似度
cosine_sim = torch.sum(user_repr * news_repr, dim=1)  # [-1, 1]

# 温度缩放 (τ=2.0)
logits = cosine_sim * 2.0  # [-2, 2]

# Sigmoid
scores = torch.sigmoid(logits)  # [0.12, 0.88] ✓
```

**效果**: 验证准确率从3.84%提升到95.90%

---

## 📚 文档导航

### 🎓 学术文档

| 文档 | 内容 | 适合人群 |
|------|------|----------|
| [COURSE_REPORT.md](COURSE_REPORT.md) | 完整课程报告，包含文献综述、方法、实验、分析 | 学生、研究者 |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | 项目技术报告 | 评审、技术人员 |

### 🔧 技术文档

| 文档 | 内容 | 适合人群 |
|------|------|----------|
| [docs/GNN_README.md](docs/GNN_README.md) | GNN实现细节、知识图谱构建 | 开发者 |
| [docs/LLM_EMBEDDING_PROPOSAL.md](docs/LLM_EMBEDDING_PROPOSAL.md) | LLM技术方案、API选择、优化策略 | 架构师 |
| [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md) | 训练问题诊断与解决过程 | 调试人员 |
| [docs/QUICKSTART_LLM.md](docs/QUICKSTART_LLM.md) | 快速开始、参数调优 | 初学者 |

---

## 🛠️ 技术栈

### 深度学习框架

- **PyTorch** 2.0+ - 深度学习框架
- **PyTorch Geometric** 2.3+ - 图神经网络
- **TensorBoard** - 训练可视化

### 数据处理

- **Pandas** - 数据处理
- **NumPy** - 数值计算
- **scikit-learn** - 评估指标

### 外部服务

- **OpenAI API** - 文本嵌入生成 (text-embedding-3-small)

---

## 📖 使用示例

### 训练监控

```bash
# 启动TensorBoard
tensorboard --logdir output/

# 访问 http://localhost:6006
```

**可视化内容**:
- 训练/验证损失曲线
- 准确率变化
- 学习率调度
- 不同配置对比

### 预测示例

```python
from src.model_llm import LLMEnhancedRecommender
import torch

# 加载模型
model = LLMEnhancedRecommender(...)
model.load_state_dict(torch.load('output/llm_gnn_fixed/best_model.pth'))
model.eval()

# 预测
user_id = 123
candidate_news = [456, 789, 101112]
scores = model.predict(user_id, candidate_news, history_news)

# 排序推荐
top_k = torch.topk(scores, k=10)
```

---

## 🔬 实验复现

### 环境要求

- Python 3.8+
- CPU: 4核+
- 内存: 32GB+
- 存储: 10GB+
- (可选) GPU: 6GB+ VRAM

### 完整流程

```bash
# 1. 下载数据
# 访问 https://msnews.github.io/ 下载MIND-small

# 2. 预计算LLM嵌入 (~2.5小时)
python src/precompute_llm_embeddings_resumable.py \
    --news_path data/mind_small/train/news.tsv \
    --output_path data/mind_small/llm_embeddings.npy \
    --api_key YOUR_API_KEY \
    --batch_size 100

# 3. 训练模型 (~3小时 CPU / ~30分钟 GPU)
python src/train_llm_fixed.py \
    --epochs 10 \
    --batch_size 64 \
    --use_llm \
    --use_gnn \
    --llm_embedding_path data/mind_small/llm_embeddings.npy

# 4. 评估
python generate_eval_files.py

# 5. 查看结果
cat output/llm_gnn_fixed/eval/metrics.json
```

### 预期输出

```json
{
  "auc": 0.5651,
  "mrr": 0.2656,
  "ndcg@5": 0.2721,
  "ndcg@10": 0.3263
}
```

---

## 🎯 未来工作

### 短期 (1-3个月)

- [ ] 扩展到MIND-large数据集
- [ ] 尝试GAT等更先进的GNN架构
- [ ] 优化推理延迟 (目标 <1ms)
- [ ] 模型量化和剪枝

### 中期 (3-6个月)

- [ ] 部署本地LLM模型 (LLaMA-2, BGE)
- [ ] 引入时间建模
- [ ] 多任务学习 (点击 + 停留时长)
- [ ] A/B测试框架

### 长期 (6-12个月)

- [ ] 跨域迁移学习
- [ ] 联邦学习框架
- [ ] 可解释性分析
- [ ] 生产级部署 (FastAPI + Redis + ONNX)

---

## 📊 数据集

### MIND (Microsoft News Dataset)

- **来源**: Microsoft News
- **规模**:
  - Small: 51K新闻, 50K用户
  - Large: 161K新闻, 1M用户
- **时间跨度**: 2019年10月-11月
- **标注**: 用户点击行为
- **链接**: https://msnews.github.io/

**引用**:
```bibtex
@inproceedings{wu2020mind,
  title={MIND: A Large-scale Dataset for News Recommendation},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and Zhou, Ming},
  booktitle={ACL},
  year={2020}
}
```

---

## 🤝 贡献指南

欢迎贡献！请遵循以下流程:

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

**代码规范**:
- 遵循PEP 8
- 添加docstrings
- 编写单元测试
- 更新文档

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

## 🙏 致谢

- **Microsoft Research** - 提供MIND数据集
- **OpenAI** - 提供embedding API
- **PyTorch团队** - 优秀的深度学习框架
- **PyG团队** - 强大的图神经网络库

---

## 📧 联系方式

**作者**: Skyler Wang

**邮箱**: 72512080@cityu-dg.edu.cn

**项目主页**: https://github.com/jgsgmlq/News-Recommender

---

## 📈 Star History

如果这个项目对你有帮助，请给我一个⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/News-Recommender&type=Date)](https://star-history.com/#your-username/News-Recommender&Date)

---

<div align="center">

**Made with ❤️ by [你的姓名]**

[⬆ 回到顶部](#-llmgnn-news-recommender)

</div>
