"""
模型性能评估脚本
评估LLM+GNN模型的各项指标
"""
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import os

sys.path.insert(0, 'src')
from data_loader import MINDDataset, collate_fn
from model_llm import LLMEnhancedRecommender
from kg_utils import KnowledgeGraphBuilder

def filter_dataset_by_news_ids(dataset, valid_news_ids):
    """过滤数据集，只保留有LLM嵌入的新闻"""
    valid_indices = []

    for idx in tqdm(range(len(dataset)), desc="过滤数据集"):
        sample = dataset[idx]
        news_id = sample['news_idx']
        if isinstance(news_id, torch.Tensor):
            news_id = news_id.item()

        if news_id in valid_news_ids:
            valid_indices.append(idx)

    return Subset(dataset, valid_indices)

def evaluate_model():
    """评估模型性能"""
    print("=" * 60)
    print("LLM+GNN模型性能评估")
    print("=" * 60)

    device = torch.device('cpu')

    # ==================== 1. 加载数据 ====================
    print("\n[1/5] 加载数据集...")

    train_behaviors = 'data/mind_tiny/behaviors.tsv'
    train_news = 'data/mind_tiny/news.tsv'
    dataset = MINDDataset(train_behaviors, train_news, mode='train')

    num_users = len(dataset.user2idx)
    num_news = len(dataset.news2idx)

    print(f"  总用户数: {num_users}")
    print(f"  总新闻数: {num_news}")
    print(f"  原始样本数: {len(dataset)}")

    # 加载LLM嵌入
    llm_emb_path = 'data/mind_tiny/llm_embeddings.npy'
    llm_emb_array = np.load(llm_emb_path)
    llm_embeddings = torch.from_numpy(llm_emb_array).float().to(device)
    print(f"  LLM嵌入形状: {llm_embeddings.shape}")

    # 过滤数据集
    valid_news_ids = set(range(llm_embeddings.size(0)))
    filtered_dataset = filter_dataset_by_news_ids(dataset, valid_news_ids)
    print(f"  过滤后样本数: {len(filtered_dataset)}")

    # 划分训练集和测试集
    train_size = int(0.8 * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size

    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(filtered_dataset)))

    train_dataset = Subset(filtered_dataset, train_indices)
    test_dataset = Subset(filtered_dataset, test_indices)

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")

    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ==================== 2. 构建知识图谱 ====================
    print("\n[2/5] 构建知识图谱...")

    kg_builder = KnowledgeGraphBuilder(
        news_path='data/mind_tiny/news.tsv',
        entity_embedding_path='data/mind_tiny/entity_embedding.vec',
        max_news=llm_embeddings.size(0)
    )

    graph_data = kg_builder.get_graph_data()
    print(f"  图节点数: {graph_data['node_features'].size(0)}")
    print(f"  图边数: {graph_data['edge_index'].size(1)}")

    # ==================== 3. 加载模型 ====================
    print("\n[3/5] 加载模型...")

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

    # 加载最佳模型权重
    model_path = 'output/llm_gnn_fixed/best_model.pth'
    if not os.path.exists(model_path):
        print(f"  错误: 模型文件不存在: {model_path}")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"  模型已加载: {model_path}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 设置图数据
    model.set_graph_data(graph_data)

    # 预计算GNN嵌入
    print("  预计算GNN嵌入...")
    with torch.no_grad():
        gnn_embeddings = model.get_gnn_embeddings()
    print(f"  GNN嵌入形状: {gnn_embeddings.shape}")

    # ==================== 4. 评估模型 ====================
    print("\n[4/5] 在测试集上评估...")

    all_predictions = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估进度"):
            user_idx = batch['user_idx'].to(device)
            news_idx = batch['news_idx'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            scores = model(user_idx, news_idx, history, llm_embeddings, gnn_embeddings)

            # 收集预测和标签
            predictions = (scores > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # ==================== 5. 计算评估指标 ====================
    print("\n[5/5] 计算评估指标...")

    # 基本指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    # AUC
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = 0.0

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ==================== 输出结果 ====================
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    print("\n【分类性能指标】")
    print(f"  准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"  召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1分数 (F1-Score):  {f1:.4f}")
    print(f"  AUC-ROC:            {auc:.4f}")

    print("\n【混淆矩阵】")
    print(f"  真负例 (TN): {tn:6d}  |  假正例 (FP): {fp:6d}")
    print(f"  假负例 (FN): {fn:6d}  |  真正例 (TP): {tp:6d}")

    print("\n【预测分布】")
    print(f"  预测为正例的数量: {int(all_predictions.sum()):6d} / {len(all_predictions)} ({all_predictions.mean()*100:.2f}%)")
    print(f"  实际正例的数量:   {int(all_labels.sum()):6d} / {len(all_labels)} ({all_labels.mean()*100:.2f}%)")

    print("\n【预测分数统计】")
    print(f"  最小值: {all_scores.min():.4f}")
    print(f"  最大值: {all_scores.max():.4f}")
    print(f"  平均值: {all_scores.mean():.4f}")
    print(f"  中位数: {np.median(all_scores):.4f}")
    print(f"  标准差: {all_scores.std():.4f}")

    # 按真实标签分组的预测分数
    pos_scores = all_scores[all_labels == 1]
    neg_scores = all_scores[all_labels == 0]

    print("\n【正样本预测分数】")
    if len(pos_scores) > 0:
        print(f"  数量:   {len(pos_scores)}")
        print(f"  平均值: {pos_scores.mean():.4f}")
        print(f"  中位数: {np.median(pos_scores):.4f}")
        print(f"  标准差: {pos_scores.std():.4f}")
    else:
        print("  无正样本")

    print("\n【负样本预测分数】")
    if len(neg_scores) > 0:
        print(f"  数量:   {len(neg_scores)}")
        print(f"  平均值: {neg_scores.mean():.4f}")
        print(f"  中位数: {np.median(neg_scores):.4f}")
        print(f"  标准差: {neg_scores.std():.4f}")
    else:
        print("  无负样本")

    # 分数区间分布
    print("\n【预测分数分布】")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(all_scores, bins=bins)

    for i in range(len(hist)):
        count = hist[i]
        percentage = count / len(all_scores) * 100
        bar = '█' * int(percentage / 2)
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {count:5d} ({percentage:5.2f}%) {bar}")

    # 详细分类报告
    print("\n【详细分类报告】")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=['负例 (不点击)', '正例 (点击)'],
        digits=4,
        zero_division=0
    ))

    # ==================== 性能分析 ====================
    print("\n" + "=" * 60)
    print("性能分析")
    print("=" * 60)

    print("\n【优势】")
    if accuracy > 0.9:
        print(f"  ✅ 准确率优秀: {accuracy*100:.2f}%")
    if auc > 0.9:
        print(f"  ✅ AUC优秀: {auc:.4f}")
    if f1 > 0.7:
        print(f"  ✅ F1分数良好: {f1:.4f}")

    print("\n【潜在问题】")
    if precision < 0.5:
        print(f"  ⚠️ 精确率较低: {precision:.4f} - 可能有较多假正例")
    if recall < 0.5:
        print(f"  ⚠️ 召回率较低: {recall:.4f} - 可能有较多假负例")

    pos_ratio = all_labels.mean()
    pred_ratio = all_predictions.mean()
    if abs(pos_ratio - pred_ratio) > 0.1:
        print(f"  ⚠️ 预测分布偏差: 实际{pos_ratio*100:.1f}% vs 预测{pred_ratio*100:.1f}%")

    if len(pos_scores) > 0 and len(neg_scores) > 0:
        score_separation = pos_scores.mean() - neg_scores.mean()
        print(f"\n【区分度】")
        print(f"  正负样本分数差: {score_separation:.4f}")
        if score_separation > 0.2:
            print(f"  ✅ 模型能较好地区分正负样本")
        elif score_separation > 0.1:
            print(f"  ⚠️ 模型区分度一般")
        else:
            print(f"  ❌ 模型区分度较差")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)

if __name__ == '__main__':
    evaluate_model()
