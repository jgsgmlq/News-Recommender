"""
为LLM+GNN Fixed模型生成评估文件
生成metrics.json, prediction.txt, truth.txt
"""
import sys
import torch
import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'src')
from data_loader import MINDDataset
from model_llm import LLMEnhancedRecommender
from kg_utils import KnowledgeGraphBuilder


def dcg_score(y_true, y_score, k=10):
    """计算DCG@k"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """计算nDCG@k"""
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best if best > 0 else 0.0


def mrr_score(y_true, y_score):
    """计算MRR (Mean Reciprocal Rank)"""
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true) if np.sum(y_true) > 0 else 0.0


def group_labels(behaviors_file):
    """
    将behaviors.tsv中的数据按印象ID分组
    返回: {impression_id: {'user': user_id, 'history': [news_ids], 'candidates': [(news_id, label)]}}
    """
    impressions = {}

    with open(behaviors_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            impression_id = int(parts[0])
            user_id = parts[1]
            history_str = parts[3]
            impressions_str = parts[4]

            # 解析历史
            history = []
            if history_str:
                history = history_str.split()

            # 解析候选新闻
            candidates = []
            if impressions_str:
                for item in impressions_str.split():
                    if '-' in item:
                        news_id, label = item.split('-')
                        label = int(label)
                        candidates.append((news_id, label))

            impressions[impression_id] = {
                'user': user_id,
                'history': history,
                'candidates': candidates
            }

    return impressions


def generate_eval_files():
    """生成评估文件"""
    print("=" * 60)
    print("生成LLM+GNN Fixed模型评估文件")
    print("=" * 60)

    device = torch.device('cpu')
    output_dir = 'output/llm_gnn_fixed/eval'
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. 加载数据 ====================
    print("\n[1/6] 加载数据集...")

    train_behaviors = 'data/mind_small/train/behaviors.tsv'
    train_news = 'data/mind_small/train/news.tsv'
    dataset = MINDDataset(train_behaviors, train_news, mode='train')

    num_users = len(dataset.user2idx)
    num_news = len(dataset.news2idx)

    # 加载LLM嵌入
    llm_emb_path = 'data/mind_small/llm_embeddings.npy'
    llm_emb_array = np.load(llm_emb_path)
    llm_embeddings = torch.from_numpy(llm_emb_array).float().to(device)

    print(f"  用户数: {num_users}")
    print(f"  新闻数: {num_news}")
    print(f"  LLM嵌入: {llm_embeddings.shape}")

    # ==================== 2. 构建知识图谱 ====================
    print("\n[2/6] 构建知识图谱...")

    kg_builder = KnowledgeGraphBuilder(
        news_path='data/mind_small/train/news.tsv',
        entity_embedding_path='data/mind_small/train/entity_embedding.vec',
        max_news=llm_embeddings.size(0)
    )

    graph_data = kg_builder.get_graph_data()

    # ==================== 3. 加载模型 ====================
    print("\n[3/6] 加载模型...")

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
    print(f"  模型已加载: {model_path}")

    # 设置图数据并预计算GNN嵌入
    model.set_graph_data(graph_data)
    with torch.no_grad():
        gnn_embeddings = model.get_gnn_embeddings()
    print(f"  GNN嵌入: {gnn_embeddings.shape}")

    # ==================== 4. 加载印象数据 ====================
    print("\n[4/6] 加载印象数据...")

    impressions = group_labels(train_behaviors)
    print(f"  印象数: {len(impressions)}")

    # 使用所有有效印象（至少有一个候选新闻）
    filtered_impressions = {}

    for imp_id, imp_data in impressions.items():
        # 检查是否有有效的候选新闻
        valid_candidates = []
        for news_id_str, label in imp_data['candidates']:
            if news_id_str in dataset.news2idx:
                valid_candidates.append((news_id_str, label))

        if len(valid_candidates) > 0:
            imp_data['candidates'] = valid_candidates
            filtered_impressions[imp_id] = imp_data

    print(f"  有效印象数: {len(filtered_impressions)}")

    # ==================== 5. 生成预测 ====================
    print("\n[5/6] 生成预测...")

    all_predictions = []
    all_truths = []
    all_scores_list = []

    predictions_file = []
    truths_file = []

    with torch.no_grad():
        for imp_id in tqdm(sorted(filtered_impressions.keys()), desc="预测进度"):
            imp_data = filtered_impressions[imp_id]

            user_id_str = imp_data['user']
            if user_id_str not in dataset.user2idx:
                continue

            user_idx = dataset.user2idx[user_id_str]

            # 处理历史
            history_ids = []
            for news_id_str in imp_data['history']:
                if news_id_str in dataset.news2idx:
                    history_ids.append(dataset.news2idx[news_id_str])

            # 填充或截断历史到固定长度
            max_hist_len = 20
            if len(history_ids) < max_hist_len:
                history_ids = history_ids + [0] * (max_hist_len - len(history_ids))
            else:
                history_ids = history_ids[:max_hist_len]

            # 处理候选新闻
            candidate_indices = []
            labels = []
            for news_id_str, label in imp_data['candidates']:
                candidate_indices.append(dataset.news2idx[news_id_str])
                labels.append(label)

            if len(candidate_indices) == 0:
                continue

            # 转换为tensor
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
            history_tensor = torch.tensor([history_ids], dtype=torch.long, device=device)
            candidate_tensor = torch.tensor(candidate_indices, dtype=torch.long, device=device)

            # 预测
            scores = model.predict(
                user_idx_tensor,
                candidate_tensor,
                history_tensor,
                llm_embeddings,
                gnn_embeddings
            )

            scores_np = scores.cpu().numpy()
            labels_np = np.array(labels)

            # 按分数排序（降序）
            sorted_indices = np.argsort(scores_np)[::-1]
            sorted_indices_1based = (sorted_indices + 1).tolist()  # 转换为1-based索引

            # 保存用于计算指标
            all_scores_list.append(scores_np)
            all_truths.append(labels_np)

            # 保存用于文件输出
            predictions_file.append(f"{imp_id} {sorted_indices_1based}")
            truths_file.append(f"{imp_id} {labels_np.tolist()}")

    # ==================== 6. 计算指标并保存 ====================
    print("\n[6/6] 计算指标并保存文件...")

    # 计算AUC
    all_labels_flat = np.concatenate([t for t in all_truths])
    all_scores_flat = np.concatenate([s for s in all_scores_list])

    try:
        auc = roc_auc_score(all_labels_flat, all_scores_flat)
    except:
        auc = 0.5

    # 计算MRR、nDCG@5、nDCG@10
    mrr_list = []
    ndcg5_list = []
    ndcg10_list = []

    for y_true, y_score in zip(all_truths, all_scores_list):
        mrr_list.append(mrr_score(y_true, y_score))
        ndcg5_list.append(ndcg_score(y_true, y_score, k=5))
        ndcg10_list.append(ndcg_score(y_true, y_score, k=10))

    mrr = np.mean(mrr_list)
    ndcg5 = np.mean(ndcg5_list)
    ndcg10 = np.mean(ndcg10_list)

    # 保存metrics.json
    metrics = {
        "AUC": float(auc),
        "MRR": float(mrr),
        "nDCG@5": float(ndcg5),
        "nDCG@10": float(ndcg10)
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  已保存: {metrics_path}")

    # 保存prediction.txt
    predictions_path = os.path.join(output_dir, 'prediction.txt')
    with open(predictions_path, 'w') as f:
        for line in predictions_file:
            f.write(line + '\n')
    print(f"  已保存: {predictions_path}")

    # 保存truth.txt
    truths_path = os.path.join(output_dir, 'truth.txt')
    with open(truths_path, 'w') as f:
        for line in truths_file:
            f.write(line + '\n')
    print(f"  已保存: {truths_path}")

    # 输出指标
    print("\n" + "=" * 60)
    print("评估指标")
    print("=" * 60)
    print(f"  AUC:      {auc:.6f}")
    print(f"  MRR:      {mrr:.6f}")
    print(f"  nDCG@5:   {ndcg5:.6f}")
    print(f"  nDCG@10:  {ndcg10:.6f}")
    print("=" * 60)
    print(f"  印象总数: {len(all_truths)}")
    print(f"  样本总数: {len(all_labels_flat)}")
    print("=" * 60)


if __name__ == '__main__':
    generate_eval_files()
