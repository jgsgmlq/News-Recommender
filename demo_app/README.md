# 新闻推荐系统 Web Demo

> AI Guide 课程作业 - 多模态新闻推荐系统可视化展示

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 项目简介

这是一个基于 **多模态融合** 的新闻推荐系统 Web 演示应用，集成了：
- **ID 嵌入**：协同过滤信号
- **LLM 嵌入**：OpenAI text-embedding-3-small (1536维语义表示)
- **GNN 嵌入**：GraphSAGE知识图谱嵌入

通过 **自适应注意力门控机制** 动态融合三种模态，实现智能新闻推荐。

## ✨ 功能特点

### 1. 📊 系统统计展示
- 新闻总数、用户总数实时展示
- 模型状态监控
- 性能指标展示（AUC提升11.8%）

### 2. 👤 用户历史可视化
- 展示用户历史点击新闻
- 新闻分类标签
- 摘要预览

### 3. 🎯 智能推荐生成
- Top-K 新闻推荐
- 实时推理（<2秒）
- 匹配度评分

### 4. 🎨 注意力权重可视化
- 三种模态权重分布柱状图
- 注意力机制解释
- 推荐原因生成

### 5. 📰 推荐详情展示
- 新闻完整信息
- 多模态权重分析
- 个性化推荐理由

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
Flask >= 3.0
```

### 安装依赖

```bash
# 在 demo_app 目录下
pip install -r requirements.txt
```

### 启动应用

```bash
# 方法1: 直接运行
python app.py

# 方法2: Flask命令
flask run --host=0.0.0.0 --port=5000
```

### 访问地址

打开浏览器访问：
```
http://localhost:5000
```

## 📁 项目结构

```
demo_app/
├── app.py                      # Flask后端主程序
├── requirements.txt            # Python依赖
├── README.md                   # 本文档
│
├── templates/                  # HTML模板
│   └── index.html             # 主页面
│
└── static/                    # 静态资源
    ├── css/
    │   └── style.css          # 样式文件
    └── js/
        └── main.js            # JavaScript交互逻辑
```

## 🎮 使用说明

### 步骤1: 选择用户
1. 在下拉菜单中选择一个用户
2. 系统自动加载该用户的历史点击记录

### 步骤2: 查看历史
- 左侧面板展示用户历史点击的新闻
- 包含新闻类别、标题、摘要

### 步骤3: 生成推荐
1. 点击"生成推荐"按钮
2. 系统调用训练好的模型进行推理
3. 右侧面板展示推荐结果（Top-10）

### 步骤4: 查看详情
1. 点击任意推荐新闻卡片
2. 弹出详情模态框，包含：
   - 新闻完整信息
   - 注意力权重可视化
   - 个性化推荐原因

## 🔧 配置说明

### 模型路径配置

在 `app.py` 中修改以下路径：

```python
# 模型文件
model_path = 'output/llm_gnn_fixed/best_model.pth'

# LLM嵌入文件
llm_emb_path = 'data/mind_small/train/llm_embeddings.npy'

# 新闻数据
news_path = 'data/mind_small/train/news.tsv'

# 用户行为数据
behaviors_path = 'data/mind_small/train/behaviors.tsv'
```

### 端口配置

```python
# 修改 app.py 最后一行
app.run(debug=True, host='0.0.0.0', port=5000)  # 改为其他端口
```

## 🎨 界面预览

### 主界面
- **顶部**：系统统计卡片（新闻数、用户数、模型状态、性能指标）
- **中部**：用户选择器 + 推荐按钮
- **下部**：左右分栏（用户历史 | 推荐结果）

### 推荐详情弹窗
- **左侧**：新闻信息（类别、匹配度、摘要）
- **右侧**：注意力权重柱状图 + 解释
- **底部**：推荐原因（基于权重生成）

## 📊 技术架构

```
┌─────────────────────────────────────────────┐
│              前端 (Browser)                  │
│  HTML + CSS + JavaScript + Chart.js         │
└─────────────────┬───────────────────────────┘
                  │ AJAX/Fetch API
┌─────────────────▼───────────────────────────┐
│          后端 (Flask Server)                 │
│  - 路由管理                                  │
│  - 模型推理                                  │
│  - 数据处理                                  │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│        深度学习模型 (PyTorch)                │
│  - LLMEnhancedRecommender                   │
│  - MultiModalNewsEncoder                    │
│  - Attention Gate Fusion                    │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              数据层                          │
│  - news.tsv (新闻数据)                      │
│  - behaviors.tsv (用户行为)                 │
│  - llm_embeddings.npy (LLM嵌入)            │
│  - best_model.pth (模型权重)               │
└─────────────────────────────────────────────┘
```

## 🔍 API文档

### GET /api/stats
获取系统统计信息

**响应示例：**
```json
{
  "total_news": 51282,
  "total_users": 50,
  "categories": {
    "sports": 12000,
    "entertainment": 10000,
    ...
  },
  "model_loaded": true
}
```

### GET /api/users
获取用户列表

**响应示例：**
```json
{
  "users": [
    {"user_id": "U1234", "history_count": 15},
    ...
  ]
}
```

### GET /api/user/<user_id>/history
获取用户历史点击

**响应示例：**
```json
{
  "user_id": "U1234",
  "history": [
    {
      "news_id": "N12345",
      "title": "...",
      "category": "sports",
      "abstract": "..."
    },
    ...
  ]
}
```

### POST /api/recommend
生成推荐

**请求体：**
```json
{
  "user_id": "U1234",
  "top_k": 10
}
```

**响应示例：**
```json
{
  "user_id": "U1234",
  "recommendations": [
    {
      "rank": 1,
      "news_id": "N12345",
      "title": "...",
      "category": "sports",
      "score": 0.856,
      "attention_weights": {
        "id": 0.42,
        "llm": 0.38,
        "gnn": 0.20
      }
    },
    ...
  ]
}
```

## 🎓 课程作业说明

### 项目亮点

1. **完整的端到端实现**
   - 从数据处理、模型训练到Web部署
   - 生产级代码质量

2. **创新的多模态架构**
   - 自适应注意力门控机制
   - 三种模态动态融合

3. **优秀的可视化展示**
   - 注意力权重可视化
   - 推荐原因解释
   - 交互式用户体验

4. **良好的工程实践**
   - 模块化设计
   - 错误处理
   - 性能优化

### 演示建议

1. **准备阶段**
   - 提前启动应用，确保模型加载成功
   - 测试几个典型用户的推荐效果
   - 准备演示场景（不同类型新闻的推荐）

2. **演示流程**
   - 介绍系统架构（5分钟）
   - 展示推荐流程（5分钟）
   - 解释注意力机制（5分钟）
   - Q&A（5分钟）

3. **重点展示**
   - 注意力权重的自适应性（不同新闻权重不同）
   - 推荐质量（匹配度高、排序合理）
   - 系统性能（推理速度快）

## 🐛 常见问题

### Q1: 模型加载失败
**A:** 检查模型文件路径是否正确，确保 `best_model.pth` 存在。如果不存在，应用会使用随机初始化的模型（仅用于界面测试）。

### Q2: LLM嵌入文件过大
**A:** `llm_embeddings.npy` 约300MB，首次加载需要几秒钟。可以考虑使用 `mmap_mode='r'` 进行内存映射。

### Q3: 推荐速度慢
**A:**
- 减少候选新闻数量（修改 `app.py` 中的 `candidate_indices`）
- 使用GPU加速（修改 `map_location='cuda'`）
- 批量推理优化

### Q4: 端口被占用
**A:** 修改 `app.py` 中的端口号，或杀死占用5000端口的进程：
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :5000
kill -9 <PID>
```

## 📝 待改进功能

- [ ] 新闻搜索功能
- [ ] 用户反馈收集（点赞/点踩）
- [ ] 在线学习（根据反馈更新推荐）
- [ ] 新闻详情页跳转
- [ ] 推荐结果导出
- [ ] 用户画像可视化
- [ ] A/B测试对比

## 📚 参考文献

1. **Transformer**: Vaswani et al. "Attention is All You Need" (NeurIPS 2017)
2. **GraphSAGE**: Hamilton et al. "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
3. **MIND Dataset**: Wu et al. "MIND: A Large-scale Dataset for News Recommendation" (ACL 2020)
4. **Multi-Modal Fusion**: Baltrušaitis et al. "Multimodal Machine Learning: A Survey and Taxonomy" (TPAMI 2019)

## 📧 联系方式

如有问题，请联系：
- 课程：AI Guide
- 项目：多模态新闻推荐系统

## 📄 许可证

MIT License

---

**🎉 祝您演示顺利！**
