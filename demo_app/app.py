"""
新闻推荐系统 Web Demo
Course: AI Guide
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, render_template, jsonify, request

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_llm import LLMEnhancedRecommender
from src.data_loader import MINDDataset

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文


# ==================== 全局变量 ====================
MODEL = None
LLM_EMBEDDINGS = None
GNN_EMBEDDINGS = None
NEWS_DATA = None
USER_HISTORY = {}
NEWS_CATEGORIES = {}


# ==================== 模型加载 ====================
def load_model():
    """加载训练好的模型"""
    global MODEL, LLM_EMBEDDINGS, GNN_EMBEDDINGS, NEWS_DATA, USER_HISTORY, NEWS_CATEGORIES

    print("Loading model...")

    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'output', 'llm_gnn_fixed', 'best_model.pth')
    llm_emb_path = os.path.join(base_dir, 'data', 'mind_small', 'train', 'llm_embeddings.npy')
    news_path = os.path.join(base_dir, 'data', 'mind_small', 'train', 'news.tsv')
    behaviors_path = os.path.join(base_dir, 'data', 'mind_small', 'train', 'behaviors.tsv')

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Using random model for demo...")

    # 加载新闻数据
    print("Loading news data...")
    try:
        NEWS_DATA = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract',
                   'url', 'title_entities', 'abstract_entities']
        )
        print(f"Loaded {len(NEWS_DATA)} news articles")
    except Exception as e:
        print(f"Error loading news data: {e}")
        # 创建示例数据
        NEWS_DATA = pd.DataFrame({
            'news_id': [f'N{i}' for i in range(100)],
            'category': np.random.choice(['sports', 'entertainment', 'finance', 'tech'], 100),
            'subcategory': ['general'] * 100,
            'title': [f'Sample News Title {i}' for i in range(100)],
            'abstract': [f'This is sample news abstract {i}...' for i in range(100)]
        })

    # 创建新闻ID映射
    news2idx = {news_id: idx for idx, news_id in enumerate(NEWS_DATA['news_id'])}

    # 加载用户历史数据（采样）
    print("Loading user behavior data...")
    try:
        behaviors_df = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )

        # 加载更多用户（前200个）
        all_users = behaviors_df['user_id'].unique()
        sampled_users = all_users[:min(200, len(all_users))]

        for user_id in sampled_users:
            user_data = behaviors_df[behaviors_df['user_id'] == user_id].iloc[0]
            if pd.notna(user_data['history']):
                history_ids = user_data['history'].split()[:20]  # 最多20条
                history_indices = [news2idx.get(nid, 0) for nid in history_ids if nid in news2idx]
                if len(history_indices) > 0:  # 确保有有效历史
                    USER_HISTORY[user_id] = {
                        'news_ids': history_ids[:10],
                        'indices': history_indices[:10]
                    }
        print(f"Loaded {len(USER_HISTORY)} users")
    except Exception as e:
        print(f"Error loading behaviors: {e}")
        # 创建示例用户历史
        for i in range(10):
            user_id = f'U{i+1}'
            history_indices = random.sample(range(len(NEWS_DATA)), 10)
            USER_HISTORY[user_id] = {
                'news_ids': NEWS_DATA.iloc[history_indices]['news_id'].tolist(),
                'indices': history_indices
            }

    # 统计新闻类别
    NEWS_CATEGORIES = NEWS_DATA['category'].value_counts().to_dict()

    # 加载LLM嵌入
    print("Loading LLM embeddings...")
    try:
        if os.path.exists(llm_emb_path):
            LLM_EMBEDDINGS = torch.from_numpy(np.load(llm_emb_path)).float()
            print(f"Loaded LLM embeddings: {LLM_EMBEDDINGS.shape}")
        else:
            print("LLM embeddings not found, using random embeddings")
            LLM_EMBEDDINGS = torch.randn(len(NEWS_DATA), 1536)
    except Exception as e:
        print(f"Error loading LLM embeddings: {e}")
        LLM_EMBEDDINGS = torch.randn(len(NEWS_DATA), 1536)

    # 加载模型
    print("Loading model weights...")
    try:
        num_users = len(USER_HISTORY)
        num_news = len(NEWS_DATA)

        # 如果模型文件存在，先检查其大小
        model_num_news = num_news
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                # 检查模型中的新闻嵌入大小
                if 'news_encoder.id_embedding.weight' in checkpoint:
                    model_num_news = checkpoint['news_encoder.id_embedding.weight'].shape[0]
                    print(f"Model news embedding size: {model_num_news}")
            except Exception as e:
                print(f"Error checking model size: {e}")

        MODEL = LLMEnhancedRecommender(
            num_users=max(num_users, 50000),
            num_news=max(model_num_news, num_news, 51282),  # 使用模型中的大小
            embedding_dim=128,
            llm_emb_dim=1536,
            gnn_output_dim=128,
            output_dim=256,
            use_llm=True,
            use_gnn=True,  # 启用GNN
            fusion_method='attention',
            dropout=0.2
        )

        if os.path.exists(model_path):
            try:
                MODEL.load_state_dict(checkpoint, strict=False)
                print("✓ Model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load all model weights: {e}")
                print("Using partially loaded model...")
        else:
            print("Model weights not found, using random initialization")

        MODEL.eval()

        # 生成GNN嵌入（使用随机初始化模拟）
        print("Generating GNN embeddings...")
        if MODEL.use_gnn:
            with torch.no_grad():
                # 为demo生成简化的GNN嵌入
                GNN_EMBEDDINGS = torch.randn(MODEL.news_encoder.num_news, 128) * 0.1
                print(f"Generated GNN embeddings: {GNN_EMBEDDINGS.shape}")
        else:
            GNN_EMBEDDINGS = None

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

    print("Setup complete!")


# ==================== API路由 ====================
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """获取系统统计信息"""
    return jsonify({
        'total_news': len(NEWS_DATA),
        'total_users': len(USER_HISTORY),
        'categories': NEWS_CATEGORIES,
        'model_loaded': MODEL is not None
    })


@app.route('/api/users')
def get_users():
    """获取用户列表"""
    users = []
    for user_id, data in list(USER_HISTORY.items())[:20]:
        users.append({
            'user_id': user_id,
            'history_count': len(data['indices'])
        })
    return jsonify({'users': users})


@app.route('/api/user/<user_id>/history')
def get_user_history(user_id):
    """获取用户历史"""
    if user_id not in USER_HISTORY:
        return jsonify({'error': 'User not found'}), 404

    history_ids = USER_HISTORY[user_id]['news_ids']
    history_news = []

    for news_id in history_ids:
        news_row = NEWS_DATA[NEWS_DATA['news_id'] == news_id]
        if len(news_row) > 0:
            news = news_row.iloc[0]
            history_news.append({
                'news_id': news['news_id'],
                'title': news['title'],
                'category': news['category'],
                'abstract': news['abstract'][:100] + '...' if len(str(news['abstract'])) > 100 else news['abstract']
            })

    return jsonify({
        'user_id': user_id,
        'history': history_news
    })


@app.route('/api/recommend', methods=['POST'])
def recommend():
    """推荐新闻"""
    data = request.json
    user_id = data.get('user_id')
    top_k = data.get('top_k', 10)

    if user_id not in USER_HISTORY:
        return jsonify({'error': 'User not found'}), 404

    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 获取用户历史
        history_indices = USER_HISTORY[user_id]['indices']

        # 映射用户ID
        user_idx_map = {uid: idx for idx, uid in enumerate(USER_HISTORY.keys())}
        user_idx = user_idx_map.get(user_id, 0)

        # 准备输入
        user_idx_tensor = torch.tensor([user_idx])
        history_tensor = torch.tensor([history_indices + [0] * (50 - len(history_indices))])

        # 获取候选新闻（排除历史）
        all_indices = set(range(min(len(NEWS_DATA), 1000)))  # 限制候选集大小
        candidate_indices = list(all_indices - set(history_indices))[:200]  # 最多200个候选

        # 推理
        with torch.no_grad():
            scores = []
            attention_weights_list = []

            for candidate_idx in candidate_indices:
                # 获取用户表示
                user_repr = MODEL.get_user_representation(
                    user_idx_tensor,
                    history_tensor,
                    llm_embeddings=LLM_EMBEDDINGS,
                    gnn_embeddings=GNN_EMBEDDINGS
                )

                # 获取新闻表示和注意力权重
                news_idx_tensor = torch.tensor([candidate_idx])

                # 手动获取新闻表示和注意力权重
                news_encoder = MODEL.news_encoder

                # ID嵌入
                id_emb = news_encoder.id_embedding(news_idx_tensor)
                id_repr = news_encoder.id_proj(id_emb)

                # LLM嵌入
                if candidate_idx < len(LLM_EMBEDDINGS):
                    llm_emb = LLM_EMBEDDINGS[candidate_idx].unsqueeze(0)
                    llm_repr = news_encoder.llm_proj(llm_emb)
                else:
                    llm_repr = torch.zeros_like(id_repr)

                # GNN嵌入
                if news_encoder.use_gnn and GNN_EMBEDDINGS is not None and candidate_idx < len(GNN_EMBEDDINGS):
                    gnn_emb = GNN_EMBEDDINGS[candidate_idx].unsqueeze(0)
                    gnn_repr = news_encoder.gnn_proj(gnn_emb)
                else:
                    gnn_repr = torch.zeros_like(id_repr)

                # 融合（获取注意力权重）
                modalities = [id_repr, llm_repr, gnn_repr] if news_encoder.use_gnn else [id_repr, llm_repr]

                if news_encoder.fusion_method == 'attention':
                    stacked = torch.stack(modalities, dim=1)
                    query = news_encoder.fusion_query.expand(1, -1).unsqueeze(1)
                    keys = news_encoder.fusion_key(stacked)
                    attn_scores = torch.bmm(query, keys.transpose(1, 2))
                    attn_weights = F.softmax(attn_scores / (news_encoder.output_dim ** 0.5), dim=-1)
                    news_repr = torch.bmm(attn_weights, stacked).squeeze(1)

                    # 保存注意力权重
                    weights = attn_weights[0, 0].tolist()
                    # 确保有3个权重值
                    if len(weights) == 2:
                        weights.append(0.0)
                    attention_weights_list.append(weights)
                else:
                    news_repr = id_repr
                    attention_weights_list.append([1.0, 0.0, 0.0])

                # 计算得分
                user_repr_norm = F.normalize(user_repr, p=2, dim=1)
                news_repr_norm = F.normalize(news_repr, p=2, dim=1)
                cosine_sim = torch.sum(user_repr_norm * news_repr_norm, dim=1)
                score = torch.sigmoid(cosine_sim * 2.0)

                scores.append(score.item())

        # 排序
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        # 构建推荐结果
        recommendations = []
        for rank, idx in enumerate(sorted_indices):
            candidate_idx = candidate_indices[idx]
            if candidate_idx < len(NEWS_DATA):
                news = NEWS_DATA.iloc[candidate_idx]
                attn_weights = attention_weights_list[idx]

                recommendations.append({
                    'rank': rank + 1,
                    'news_id': news['news_id'],
                    'title': news['title'],
                    'category': news['category'],
                    'subcategory': news['subcategory'],
                    'abstract': str(news['abstract'])[:200] + '...' if len(str(news['abstract'])) > 200 else str(news['abstract']),
                    'score': float(scores[idx]),
                    'attention_weights': {
                        'id': float(attn_weights[0]) if len(attn_weights) > 0 else 0.0,
                        'llm': float(attn_weights[1]) if len(attn_weights) > 1 else 0.0,
                        'gnn': float(attn_weights[2]) if len(attn_weights) > 2 else 0.0
                    }
                })

        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/news/random')
def get_random_news():
    """获取随机新闻（用于展示）"""
    sample_size = min(20, len(NEWS_DATA))
    sample_news = NEWS_DATA.sample(n=sample_size)

    news_list = []
    for _, news in sample_news.iterrows():
        news_list.append({
            'news_id': news['news_id'],
            'title': news['title'],
            'category': news['category'],
            'abstract': str(news['abstract'])[:150] + '...' if len(str(news['abstract'])) > 150 else str(news['abstract'])
        })

    return jsonify({'news': news_list})


# ==================== 启动应用 ====================
if __name__ == '__main__':
    load_model()

    print("\n" + "="*60)
    print("  新闻推荐系统 Web Demo")
    print("  Course: AI Guide")
    print("="*60)
    print("\n访问地址: http://localhost:5000")
    print("按 Ctrl+C 退出\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
