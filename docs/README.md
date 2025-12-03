# 技术文档目录

本文件夹包含项目的详细技术文档。

## 文档说明

### 📘 [GNN_README.md](GNN_README.md)
**GNN增强新闻推荐系统**

内容:
- 知识图谱构建方法
- GraphSAGE架构说明
- GNN训练流程
- 实验结果(MIND Tiny)
- 使用示例

适合: 想了解GNN实现细节的读者

---

### 📗 [LLM_EMBEDDING_PROPOSAL.md](LLM_EMBEDDING_PROPOSAL.md)
**LLM嵌入技术方案(最详细)**

内容:
- 整体架构设计
- LLM API选择对比
- 多模态融合机制详解
- 实施方案(4个阶段)
- 成本估算
- 优化策略
- 风险分析

适合: 需要复现或扩展项目的开发者

---

### 📙 [QUICKSTART_LLM.md](QUICKSTART_LLM.md)
**LLM快速开始指南**

内容:
- 环境配置
- 快速验证流程(Tiny数据集)
- 参数调优建议
- 成本说明
- 常见问题FAQ

适合: 想快速运行实验的用户

---

### 📕 [RUN_ME.md](RUN_ME.md)
**一键运行指南**

内容:
- API配置说明
- 一键测试脚本
- 预期效果
- 故障排除

适合: 首次使用的用户

---

### 📔 [TRAINING_REPORT.md](TRAINING_REPORT.md)
**详细训练报告(问题诊断与解决)**

内容:
- 初始问题表现(训练失败)
- 诊断过程(3个脚本)
- 根本原因分析(sigmoid饱和)
- 解决方案(温度缩放)
- 实验结果(96.99%准确率)
- 技术细节
- 文件修改清单
- 未来改进建议

适合: 想深入理解模型训练过程和调试方法的读者

---

### 📗[ATTENTION_MECHANISM_EXPLAINED.md](ATTENTION_MECHANISM_EXPLAINED.md)
**自适应注意力门控机制详解**

内容:
- 什么是自适应注意力门控？
- 数学原理
- 代码实现详解
- 与其他融合方法的对比
- 自适应能力体现
- 为什么Attention方法效果最好
- 核心代码位置索引

适合：想了解本项目所采用自适应注意力门控机制基本原理的读者

---

### 📕[fusion_comparison.md](fusion_comparison.md)
**多模态融合方法对比**

内容：
- 简单拼接（Concat）
- 门控融合（Gate）
- 自适应注意力门控（Attention）⭐

适合：想了解本项目所采用自适应注意力门控机制基本原理的读者

---

### 📙[INCREMENTAL_UPDATE_STRATEGY.md](INCREMENTAL_UPDATE_STRATEGY.md)
**增量更新策略：新数据场景分析**

内容：
- 当前架构分析
- 三种场景详细分析
- 完整更新策略总结
- 代码实现示例
- 最佳实践建议

  适合：想后期进行模型更新的用户

## 文档使用建议

### 如果你是...

**学生/研究者**:
1. 先读 `../COURSE_REPORT.md` (课程报告,全面了解项目)
2. 再读 `TRAINING_REPORT.md` (学习调试方法)
3. 最后读 `LLM_EMBEDDING_PROPOSAL.md` (技术细节)

**开发者**:
1. 先读 `RUN_ME.md` (快速运行)
2. 再读 `QUICKSTART_LLM.md` (参数调优)
3. 需要时参考 `GNN_README.md` 和 `LLM_EMBEDDING_PROPOSAL.md`

**课程评审**:
1. 主要阅读 `../COURSE_REPORT.md`
2. 技术细节参考 `TRAINING_REPORT.md`

---

## 文档层次关系

```
News-Recommender/
├── README.md                          # 项目总览
├── COURSE_REPORT.md                   # 课程报告(学术风格)⭐
├── PROJECT_REPORT.md                  # 项目报告(工程风格)
└── docs/                              # 技术文档
    ├── README.md                      # 本文件
    ├── GNN_README.md                  # GNN技术说明
    ├── LLM_EMBEDDING_PROPOSAL.md      # LLM方案(最详细)
    ├── QUICKSTART_LLM.md              # 快速开始
    ├── RUN_ME.md                      # 一键运行
    └── TRAINING_REPORT.md             # 训练报告(调试过程)
```

---

## 更新日志

- **2025-11-28**: 整理文档结构,将技术文档移至docs/
- **2025-11-29**: 完成TRAINING_REPORT.md,记录sigmoid饱和问题解决过程
- **2025-11-29**: 完成LLM_EMBEDDING_PROPOSAL.md技术方案
- **2025-11-29**: 完成GNN_README.md,验证GNN模块
- **2025-11-29**: 初始版本,基础文档创建

---

## 联系方式

有问题?
- 查看对应文档的FAQ部分
- 提交GitHub Issue
- 联系项目作者

---

**文档维护**: Skyler Wang
**最后更新**: 2025-12-03
