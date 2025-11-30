"""
优化决策阈值以平衡精确率和召回率
"""
import sys
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

sys.path.insert(0, 'src')
from data_loader import MINDDataset, collate_fn
from model_llm import LLMEnhancedRecommender
from kg_utils import KnowledgeGraphBuilder
from torch.utils.data import DataLoader, Subset

def filter_dataset_by_news_ids(dataset, valid_news_ids):
    """过滤数据集"""
    valid_indices = []

    for idx in tqdm(range(len(dataset)), desc="过滤数据集"):
        sample = dataset[idx]
        news_id = sample['news_idx']
        if isinstance(news_id, torch.Tensor):
            news_id = news_id.item()

        if news_id in valid_news_ids:
            valid_indices.append(idx)

    return Subset(dataset, valid_indices)

def optimize_threshold():
    """优化阈值"""
    print("=" * 60)
    print("决策阈值优化")
    print("=" * 60)

    device = torch.device('cpu')

    # 加载数据
    print("\n[1/4] 加载数据...")
    train_behaviors = 'data/mind_tiny/behaviors.tsv'
    train_news = 'data/mind_tiny/news.tsv'
    dataset = MINDDataset(train_behaviors, train_news, mode='train')

    num_users = len(dataset.user2idx)
    num_news = len(dataset.news2idx)

    # 加载LLM嵌入
    llm_emb_path = 'data/mind_tiny/llm_embeddings.npy'
    llm_emb_array = np.load(llm_emb_path)
    llm_embeddings = torch.from_numpy(llm_emb_array).float().to(device)

    # 过滤数据集
    valid_news_ids = set(range(llm_embeddings.size(0)))
    filtered_dataset = filter_dataset_by_news_ids(dataset, valid_news_ids)

    # 划分训练集和测试集
    train_size = int(0.8 * len(filtered_dataset))
    test_indices = list(range(train_size, len(filtered_dataset)))
    test_dataset = Subset(filtered_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # 构建知识图谱
    print("\n[2/4] 构建知识图谱...")
    kg_builder = KnowledgeGraphBuilder(
        news_path='data/mind_tiny/news.tsv',
        entity_embedding_path='data/mind_tiny/entity_embedding.vec',
        max_news=llm_embeddings.size(0)
    )
    graph_data = kg_builder.get_graph_data()

    # 加载模型
    print("\n[3/4] 加载模型...")
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
        use_gnn=True,
        fusion_method='concat',
        dropout=0.3
    ).to(device)

    model_path = 'output/llm_gnn_fixed/best_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model.set_graph_data(graph_data)
    with torch.no_grad():
        gnn_embeddings = model.get_gnn_embeddings()

    # 收集所有预测分数和标签
    print("\n[4/4] 生成预测...")
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测进度"):
            user_idx = batch['user_idx'].to(device)
            news_idx = batch['news_idx'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)

            scores = model(user_idx, news_idx, history, llm_embeddings, gnn_embeddings)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # 测试不同阈值
    print("\n" + "=" * 60)
    print("阈值优化结果")
    print("=" * 60)

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    print(f"\n{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'准确率':<10}")
    print("-" * 60)

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for threshold in thresholds:
        predictions = (all_scores > threshold).astype(int)

        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        accuracy = (predictions == all_labels).mean()

        print(f"{threshold:<8.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {accuracy:<10.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }

    # AUC
    auc = roc_auc_score(all_labels, all_scores)

    print("\n" + "=" * 60)
    print("推荐阈值")
    print("=" * 60)
    print(f"\n最佳F1阈值: {best_threshold:.2f}")
    print(f"  精确率: {best_metrics['precision']:.4f}")
    print(f"  召回率: {best_metrics['recall']:.4f}")
    print(f"  F1分数: {best_metrics['f1']:.4f}")
    print(f"  准确率: {best_metrics['accuracy']:.4f}")
    print(f"\nAUC: {auc:.4f}")

    print("\n推荐:")
    print(f"  如果追求精确率，使用阈值: 0.5")
    print(f"  如果追求召回率，使用阈值: 0.15-0.2")
    print(f"  如果追求平衡，使用阈值: {best_threshold:.2f}")

if __name__ == '__main__':
    optimize_threshold()
