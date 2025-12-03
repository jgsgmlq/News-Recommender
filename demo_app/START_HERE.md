# 🚀 立即开始

## ✅ 问题已修复

已修复的问题：
1. ✅ 模块导入错误（`gnn_module` 导入问题）
2. ✅ 模型大小不匹配（自动适配checkpoint大小）
3. ✅ 依赖已安装（Flask, PyTorch等）

---

## 🎯 启动步骤

### 方法1：使用启动脚本（推荐）

**Windows:**
```bash
# 双击运行或在命令行执行
D:\Desktop\News-Recommender\demo_app\run.bat
```

**Linux/Mac:**
```bash
cd D:\Desktop\News-Recommender\demo_app
chmod +x run.sh
./run.sh
```

### 方法2：直接运行Python

```bash
cd D:\Desktop\News-Recommender\demo_app
python app.py
```

### 方法3：使用Anaconda环境

```bash
# 激活你的环境
conda activate KidMagic

# 进入目录
cd D:\Desktop\News-Recommender\demo_app

# 启动应用
python app.py
```

---

## 🌐 访问地址

启动成功后，你会看到：

```
============================================================
  新闻推荐系统 Web Demo
  Course: AI Guide
============================================================

访问地址: http://localhost:5000
按 Ctrl+C 退出

 * Running on http://127.0.0.1:5000
 * Running on http://10.34.27.26:5000
```

然后在浏览器中打开：
- **本地访问**: http://localhost:5000
- **或**: http://127.0.0.1:5000

---

## ✨ 启动成功标志

如果看到以下输出，说明启动成功：

```
Loading model...
Loading news data...
Loaded 51282 news articles
Loading user behavior data...
Loaded 49 users
Loading LLM embeddings...
Loading model weights...
Model news embedding size: 51283
✓ Model loaded successfully!
Setup complete!

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

## 🎮 使用流程

1. **打开浏览器** → 访问 http://localhost:5000
2. **选择用户** → 在下拉菜单中选择一个用户
3. **查看历史** → 左侧自动显示用户历史点击
4. **生成推荐** → 点击"生成推荐"按钮
5. **查看结果** → 右侧显示Top-10推荐新闻
6. **详细分析** → 点击任意推荐卡片查看详情

---

## 📊 界面功能说明

### 顶部统计卡片
- 📚 新闻总数：51,282
- 👥 用户总数：49
- 🤖 模型状态：✅ 已加载
- 🎯 AUC提升：11.8%

### 左侧：用户历史
- 显示用户最近10条点击记录
- 包含新闻类别、标题、摘要

### 右侧：推荐结果
- Top-10推荐新闻
- 每条新闻显示：
  - 排名（右上角数字）
  - 匹配度评分
  - 新闻类别
  - 标题和摘要
  - 注意力权重（ID/LLM/GNN）

### 详情弹窗
点击推荐卡片后显示：
- 📰 新闻完整信息
- 📊 注意力权重柱状图
- 💡 个性化推荐原因

---

## ⚠️ 常见问题

### Q1: 端口5000被占用
**错误信息**: `Address already in use`

**解决方法**:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <进程ID> /F

# 或修改 app.py 最后一行的端口
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Q2: 模型加载警告
**警告信息**: `LLM embeddings not found, using random embeddings`

**说明**: 这是正常的，应用会使用随机嵌入继续运行，不影响界面演示。

如需使用真实LLM嵌入，确保文件存在：
```
D:\Desktop\News-Recommender\data\mind_small\train\llm_embeddings.npy
```

### Q3: 推荐速度慢
**原因**: 候选新闻太多

**解决**: 修改 `app.py` 第205行：
```python
candidate_indices = list(all_indices - set(history_indices))[:100]  # 改为50
```

### Q4: 浏览器无法访问
**检查清单**:
- ✅ 服务器已启动（终端显示 "Running on..."）
- ✅ 防火墙未阻止端口5000
- ✅ 浏览器地址正确（http://localhost:5000）
- ✅ 使用现代浏览器（Chrome/Firefox/Edge）

---

## 🎓 演示建议

### 准备工作
1. ✅ 提前启动应用，确保正常运行
2. ✅ 测试几个用户的推荐效果
3. ✅ 准备演示脚本（见 `DEMO_GUIDE.md`）

### 演示要点
1. **系统介绍**（2分钟）
   - 多模态架构
   - 自适应注意力机制

2. **功能演示**（8分钟）
   - 用户选择 → 历史加载
   - 推荐生成 → 结果展示
   - 注意力权重可视化
   - 推荐原因生成

3. **技术亮点**（5分钟）
   - 性能提升（AUC +11.8%）
   - 可解释性（推荐理由）
   - 工程实践（增量更新）

### 互动环节
- 邀请听众选择用户
- 对比不同新闻的权重分布
- 解释为什么推荐某条新闻

---

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| `QUICKSTART.md` | 5分钟快速开始 |
| `README.md` | 完整技术文档 |
| `DEMO_GUIDE.md` | 详细演示脚本（20分钟） |

---

## 🔍 调试模式

如果遇到问题，可以开启详细日志：

```bash
# 在 app.py 开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📞 获取帮助

如果遇到其他问题：
1. 查看终端输出的完整错误信息
2. 检查 `app.py` 中的错误处理代码
3. 阅读 `README.md` 故障排除部分
4. 随时咨询！

---

## 🎉 开始使用

现在你可以：
```bash
cd D:\Desktop\News-Recommender\demo_app
python app.py
```

然后打开浏览器访问 http://localhost:5000

**祝你演示顺利！** 🚀
