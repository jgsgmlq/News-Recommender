# 快速开始 - 5分钟启动Demo

## 🚀 三步启动

### 步骤1: 进入目录
```bash
cd D:\Desktop\News-Recommender\demo_app
```

### 步骤2: 安装依赖（首次运行）
```bash
pip install -r requirements.txt
```

### 步骤3: 启动应用
```bash
# 方法1: 直接运行
python app.py

# 方法2: 使用启动脚本（Windows）
run.bat

# 方法3: 使用启动脚本（Linux/Mac）
chmod +x run.sh
./run.sh
```

### 步骤4: 打开浏览器
访问: http://localhost:5000

---

## ✅ 检查清单

运行前确保以下文件存在（如果不存在，应用会使用模拟数据）：

```
D:\Desktop\News-Recommender\
├── output\llm_gnn_fixed\best_model.pth      ✓ 训练好的模型
├── data\mind_small\train\
│   ├── news.tsv                             ✓ 新闻数据
│   ├── behaviors.tsv                        ✓ 用户行为数据
│   └── llm_embeddings.npy                   ✓ LLM嵌入
```

**提示**：如果文件不存在，应用会自动生成模拟数据用于演示界面。

---

## 🎮 使用流程

1. **选择用户** - 在下拉菜单中选择用户
2. **查看历史** - 左侧显示用户历史点击
3. **生成推荐** - 点击"生成推荐"按钮
4. **查看结果** - 右侧显示Top-10推荐新闻
5. **详细分析** - 点击任意推荐卡片查看详情

---

## ❓ 常见问题

### Q: 端口5000被占用怎么办？
A: 修改 `app.py` 最后一行：
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # 改为8080或其他端口
```

### Q: 模型加载失败怎么办？
A: 应用会自动使用模拟数据，可以正常演示界面功能。

### Q: 推荐速度很慢怎么办？
A: 减少候选新闻数量，修改 `app.py` 第180行：
```python
candidate_indices = list(all_indices - set(history_indices))[:100]  # 改为50
```

---

## 📞 获取帮助

如遇到问题：
1. 查看终端输出的错误信息
2. 阅读 `README.md` 完整文档
3. 查看 `DEMO_GUIDE.md` 演示指南

---

**🎉 祝您使用愉快！**
