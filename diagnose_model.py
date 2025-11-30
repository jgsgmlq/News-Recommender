"""
诊断模型输出 - 检查模型预测值
"""
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from data_loader import MINDDataset, collate_fn
from model_llm import LLMEnhancedRecommender

def diagnose_model():
    """诊断模型输出"""
    print("=" * 60)
    print("模型输出诊断")
    print("=" * 60)

    device = torch.device('cpu')

    # 加载数据
    train_behaviors = 'data/mind_tiny/behaviors.tsv'
    train_news = 'data/mind_tiny/news.tsv'
    dataset = MINDDataset(train_behaviors, train_news, mode='train')

    # 只取100个样本用于快速测试
    from torch.utils.data import Subset
    test_indices = list(range(100))
    test_dataset = Subset(dataset, test_indices)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    num_users = len(dataset.user2idx)
    num_news = len(dataset.news2idx)

    model = LLMEnhancedRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=128,
        llm_emb_dim=1536,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        output_dim=256,
        gnn_layers=2,
        use_llm=True,
        use_gnn=False,  # 暂时不用GNN
        fusion_method='concat',
        dropout=0.3
    ).to(device)

    # 加载LLM嵌入
    llm_embeddings = None
    try:
        llm_emb_array = np.load('data/mind_tiny/llm_embeddings.npy')
        llm_embeddings = torch.from_numpy(llm_emb_array).float().to(device)
        print(f"LLM嵌入已加载: {llm_embeddings.shape}\n")
    except:
        print("未找到LLM嵌入\n")

    # 测试一个批次
    print("=" * 60)
    print("检查模型输出（未训练的随机初始化模型）")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            user_idx = batch['user_idx'].to(device)
            news_idx = batch['news_idx'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)

            # 前向传播 - 使用模型的forward方法
            scores = model(user_idx, news_idx, history, llm_embeddings, None)

            # 也获取中间值用于分析
            user_repr = model.get_user_representation(user_idx, history, llm_embeddings, None)
            news_repr = model.news_encoder(news_idx, llm_embeddings, None)

            # 计算点积（with temperature scaling）
            logits_raw = torch.sum(user_repr * news_repr, dim=1)
            logits_scaled = logits_raw / (256 ** 0.5)

            print(f"\n批次 {batch_idx}:")
            print(f"  样本数: {len(labels)}")
            print(f"  真实标签: {labels}")
            print(f"\n  中间值分析:")
            print(f"    user_repr范围: [{user_repr.min().item():.4f}, {user_repr.max().item():.4f}]")
            print(f"    user_repr均值: {user_repr.mean().item():.4f}, 标准差: {user_repr.std().item():.4f}")
            print(f"    news_repr范围: [{news_repr.min().item():.4f}, {news_repr.max().item():.4f}]")
            print(f"    news_repr均值: {news_repr.mean().item():.4f}, 标准差: {news_repr.std().item():.4f}")
            print(f"    logits_raw（未缩放）: {logits_raw[:5]}")  # 显示前5个
            print(f"    logits_scaled（缩放后）: {logits_scaled[:5]}")  # 显示前5个
            print(f"    logits_scaled范围: [{logits_scaled.min().item():.4f}, {logits_scaled.max().item():.4f}]")
            print(f"    logits_scaled均值: {logits_scaled.mean().item():.4f}")
            print(f"\n  预测分数（sigmoid后）: {scores[:10]}")  # 显示前10个
            print(f"\n  预测分数统计:")
            print(f"    最小值: {scores.min().item():.6f}")
            print(f"    最大值: {scores.max().item():.6f}")
            print(f"    平均值: {scores.mean().item():.6f}")
            print(f"    标准差: {scores.std().item():.6f}")

            # 计算损失
            criterion = torch.nn.BCELoss()
            loss = criterion(scores, labels)
            print(f"\n  BCE损失: {loss.item():.4f}")

            # 计算准确率
            predicted = (scores > 0.5).float()
            accuracy = (predicted == labels).float().mean().item()
            print(f"  准确率: {100.0 * accuracy:.2f}%")

            # 详细分析
            print(f"\n  详细分析:")
            print(f"    预测为1的数量: {(predicted == 1).sum().item()}")
            print(f"    预测为0的数量: {(predicted == 0).sum().item()}")
            print(f"    真实为1的数量: {(labels == 1).sum().item()}")
            print(f"    真实为0的数量: {(labels == 0).sum().item()}")

            if batch_idx == 0:  # 只检查第一个批次
                break

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    diagnose_model()
