"""快速测试修复后的模型"""
import sys
import torch
import numpy as np

sys.path.insert(0, 'src')
from model_llm import LLMEnhancedRecommender

# 创建模型
model = LLMEnhancedRecommender(
    num_users=1000,
    num_news=500,
    embedding_dim=128,
    llm_emb_dim=1536,
    gnn_input_dim=100,
    gnn_hidden_dim=128,
    gnn_output_dim=128,
    output_dim=256,
    gnn_layers=2,
    use_llm=False,  # 暂时不用LLM，只测试架构
    use_gnn=False,
    fusion_method='concat',
    dropout=0.3
)

model.eval()

# 创建假数据
batch_size = 32
user_idx = torch.randint(0, 1000, (batch_size,))
news_idx = torch.randint(0, 500, (batch_size,))
history = torch.randint(0, 500, (batch_size, 10))

# 前向传播
with torch.no_grad():
    scores = model(user_idx, news_idx, history, None, None)

print("="*60)
print("修复后的模型测试")
print("="*60)
print(f"预测分数（前10个）: {scores[:10]}")
print(f"预测分数统计:")
print(f"  最小值: {scores.min().item():.6f}")
print(f"  最大值: {scores.max().item():.6f}")
print(f"  平均值: {scores.mean().item():.6f}")
print(f"  标准差: {scores.std().item():.6f}")
print(f"\n预测分数范围是否合理: {0.1 < scores.min() < 0.9 and 0.1 < scores.max() < 0.9}")
print("="*60)
