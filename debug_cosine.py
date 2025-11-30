"""调试余弦相似度"""
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
    use_llm=False,
    use_gnn=False,
    fusion_method='concat',
    dropout=0.3
)

model.eval()

# 创建假数据
batch_size = 100
user_idx = torch.randint(0, 1000, (batch_size,))
news_idx = torch.randint(0, 500, (batch_size,))
history = torch.randint(0, 500, (batch_size, 10))

# 手动计算来查看余弦相似度
with torch.no_grad():
    user_repr = model.get_user_representation(user_idx, history, None, None)
    news_repr = model.news_encoder(news_idx, None, None)

    print("="*60)
    print("调试余弦相似度")
    print("="*60)

    print(f"\n归一化前:")
    print(f"  user_repr范围: [{user_repr.min():.4f}, {user_repr.max():.4f}]")
    print(f"  news_repr范围: [{news_repr.min():.4f}, {news_repr.max():.4f}]")
    print(f"  user_repr均值: {user_repr.mean():.4f}, 标准差: {user_repr.std():.4f}")
    print(f"  news_repr均值: {news_repr.mean():.4f}, 标准差: {news_repr.std():.4f}")

    # L2归一化
    user_repr_norm = torch.nn.functional.normalize(user_repr, p=2, dim=1)
    news_repr_norm = torch.nn.functional.normalize(news_repr, p=2, dim=1)

    print(f"\n归一化后:")
    print(f"  user_repr_norm范围: [{user_repr_norm.min():.4f}, {user_repr_norm.max():.4f}]")
    print(f"  news_repr_norm范围: [{news_repr_norm.min():.4f}, {news_repr_norm.max():.4f}]")
    print(f"  user_repr_norm模长: {torch.norm(user_repr_norm[0]):.4f} (应该=1)")
    print(f"  news_repr_norm模长: {torch.norm(news_repr_norm[0]):.4f} (应该=1)")

    # 余弦相似度
    cosine_sim = torch.sum(user_repr_norm * news_repr_norm, dim=1)

    print(f"\n余弦相似度:")
    print(f"  范围: [{cosine_sim.min():.6f}, {cosine_sim.max():.6f}]")
    print(f"  均值: {cosine_sim.mean():.6f}")
    print(f"  标准差: {cosine_sim.std():.6f}")
    print(f"  前10个值: {cosine_sim[:10]}")

    # 缩放后的logits
    logits = cosine_sim * 10.0

    print(f"\nLogits (cosine_sim * 10):")
    print(f"  范围: [{logits.min():.6f}, {logits.max():.6f}]")
    print(f"  均值: {logits.mean():.6f}")

    # Sigmoid
    scores = torch.sigmoid(logits)

    print(f"\nSigmoid后的预测:")
    print(f"  范围: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  均值: {scores.mean():.6f}")

    print("\n"+"="*60)
    print("建议：如果余弦相似度都是正值且>0.5，需要降低温度系数")
    print("="*60)
