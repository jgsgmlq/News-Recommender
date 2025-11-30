"""
Prediction and Evaluation script for LLM-Enhanced News Recommendation Model
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from model_llm import LLMEnhancedRecommender
from data_loader import MINDDataset


def load_model(checkpoint_path, device, llm_embedding_path=None):
    """Load trained LLM model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model args
    model_args = checkpoint.get('args', {})

    # Load LLM embeddings if available
    llm_embeddings = None
    if llm_embedding_path and os.path.exists(llm_embedding_path):
        print(f"Loading LLM embeddings from {llm_embedding_path}")
        llm_embeddings = np.load(llm_embedding_path)
        llm_embeddings = torch.from_numpy(llm_embeddings).float().to(device)
        print(f"LLM embeddings loaded: {llm_embeddings.shape}")
        llm_emb_dim = llm_embeddings.size(1)
    else:
        llm_emb_dim = 1536  # Default

    # Create model
    model = LLMEnhancedRecommender(
        num_users=checkpoint['num_users'],
        num_news=checkpoint['num_news'],
        embedding_dim=model_args.get('embedding_dim', 128),
        llm_emb_dim=llm_emb_dim,
        gnn_input_dim=100,
        gnn_hidden_dim=model_args.get('gnn_hidden_dim', 128),
        gnn_output_dim=model_args.get('gnn_output_dim', 128),
        output_dim=model_args.get('output_dim', 256),
        gnn_layers=model_args.get('gnn_layers', 2),
        use_llm=model_args.get('use_llm', True) and llm_embeddings is not None,
        use_gnn=model_args.get('use_gnn', True),
        fusion_method=model_args.get('fusion_method', 'attention'),
        dropout=model_args.get('dropout', 0.2)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")
    print(f"Using LLM: {model_args.get('use_llm', True) and llm_embeddings is not None}")
    print(f"Using GNN: {model_args.get('use_gnn', True)}")

    # Load graph data if available
    graph_data = checkpoint.get('graph_data', None)
    if graph_data is not None and model_args.get('use_gnn', True):
        model.set_graph_data(graph_data)
        print("Knowledge graph loaded into model")

    return model, checkpoint, llm_embeddings


def predict_impressions(model, val_dataset, device, llm_embeddings=None, use_gnn=True):
    """
    Generate predictions for all impressions in validation set

    Returns:
        predictions: dict {impression_id: [(news_idx, score, label), ...]}
    """
    print("\nGenerating predictions...")

    # Pre-compute GNN embeddings if using GNN
    gnn_embeddings = None
    if use_gnn and model.use_gnn:
        print("Pre-computing GNN embeddings...")
        with torch.no_grad():
            gnn_embeddings = model.get_gnn_embeddings()
        if gnn_embeddings is not None:
            print(f"GNN embeddings computed: {gnn_embeddings.shape}")

    # Group samples by impression_id
    impression_data = defaultdict(list)

    for sample in val_dataset.samples:
        impression_data[sample['impression_id']].append({
            'user_idx': sample['user_idx'],
            'news_idx': sample['news_idx'],
            'label': sample['label'],
            'history': sample['history']
        })

    predictions = {}

    model.eval()
    with torch.no_grad():
        for imp_id, samples in tqdm(impression_data.items(), desc='Predicting'):
            if len(samples) == 0:
                continue

            # Get user and history (same for all samples in this impression)
            user_idx = samples[0]['user_idx']
            history = samples[0]['history'][-50:] if len(samples[0]['history']) > 50 else samples[0]['history']

            if len(history) == 0:
                history = [0]

            # Prepare tensors
            user_tensor = torch.tensor([user_idx]).to(device)
            history_tensor = torch.tensor([history]).to(device)

            # Get candidate news indices and labels
            candidate_news = [s['news_idx'] for s in samples]
            labels = [s['label'] for s in samples]

            candidate_tensor = torch.tensor(candidate_news).to(device)

            # Predict scores
            scores = model.predict(user_tensor, candidate_tensor, history_tensor,
                                   llm_embeddings, gnn_embeddings)
            scores = scores.cpu().numpy()

            # Store results
            predictions[imp_id] = [(news_idx, score, label)
                                   for news_idx, score, label in zip(candidate_news, scores, labels)]

    return predictions


def generate_evaluation_files(predictions, output_dir):
    """Generate prediction.txt and truth.txt for evaluation"""
    pred_file = os.path.join(output_dir, 'prediction.txt')
    truth_file = os.path.join(output_dir, 'truth.txt')

    print(f"\nGenerating evaluation files...")
    print(f"  Prediction file: {pred_file}")
    print(f"  Truth file: {truth_file}")

    with open(pred_file, 'w') as pf, open(truth_file, 'w') as tf:
        for imp_id in sorted(predictions.keys()):
            items = predictions[imp_id]

            # Sort by predicted score (descending)
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

            # Get ranks (1-indexed)
            ranks = list(range(1, len(sorted_items) + 1))

            # Get labels in original order
            labels = [item[2] for item in items]

            # Write prediction (impression_id + ranks as JSON)
            pf.write(f"{imp_id} {json.dumps(ranks)}\n")

            # Write truth (impression_id + labels as JSON)
            tf.write(f"{imp_id} {json.dumps(labels)}\n")

    print("Evaluation files generated successfully!")


def compute_simple_metrics(predictions):
    """Compute simple evaluation metrics"""
    from sklearn.metrics import roc_auc_score
    import numpy as np

    all_labels = []
    all_scores = []
    mrr_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for imp_id, items in predictions.items():
        # Extract labels and scores
        labels = [item[2] for item in items]
        scores = [item[1] for item in items]

        all_labels.extend(labels)
        all_scores.extend(scores)

        # MRR: reciprocal rank of first relevant item
        sorted_items = sorted(enumerate(items), key=lambda x: x[1][1], reverse=True)
        for rank, (idx, (_, _, label)) in enumerate(sorted_items, 1):
            if label == 1:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

        # nDCG@5 and nDCG@10
        def dcg_at_k(scores, k):
            scores = np.array(scores)[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))

        # Get relevance scores (sorted by prediction)
        relevance = [item[2] for item in sorted(items, key=lambda x: x[1], reverse=True)]

        # Ideal relevance (sorted by actual labels)
        ideal_relevance = sorted([item[2] for item in items], reverse=True)

        # nDCG@5
        if len(relevance) >= 5:
            dcg5 = dcg_at_k(relevance, 5)
            idcg5 = dcg_at_k(ideal_relevance, 5)
            ndcg5_scores.append(dcg5 / idcg5 if idcg5 > 0 else 0.0)

        # nDCG@10
        if len(relevance) >= 10:
            dcg10 = dcg_at_k(relevance, 10)
            idcg10 = dcg_at_k(ideal_relevance, 10)
            ndcg10_scores.append(dcg10 / idcg10 if idcg10 > 0 else 0.0)

    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except:
        auc = 0.0

    # Compute averages
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    ndcg5 = np.mean(ndcg5_scores) if ndcg5_scores else 0.0
    ndcg10 = np.mean(ndcg10_scores) if ndcg10_scores else 0.0

    metrics = {
        'AUC': auc,
        'MRR': mrr,
        'nDCG@5': ndcg5,
        'nDCG@10': ndcg10
    }

    return metrics


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine LLM embedding path
    if args.llm_embedding_path is None:
        # Try to infer from model path
        model_dir = os.path.dirname(args.model_path)
        inferred_path = os.path.join(model_dir, '..', '..', 'data', 'mind_tiny', 'llm_embeddings.npy')
        if os.path.exists(inferred_path):
            args.llm_embedding_path = inferred_path
            print(f"Inferred LLM embedding path: {inferred_path}")

    # Load model
    model, checkpoint, llm_embeddings = load_model(
        args.model_path, device, args.llm_embedding_path
    )

    # Load validation dataset (using tiny dataset for quick testing)
    print("\n" + "=" * 60)
    print("Loading Validation Data (Tiny)")
    print("=" * 60)

    val_behaviors = os.path.join(args.data_dir, 'mind_tiny', 'behaviors.tsv')
    val_news = os.path.join(args.data_dir, 'mind_tiny', 'news.tsv')

    val_dataset = MINDDataset(val_behaviors, val_news, mode='valid')

    # Update dataset vocabularies with those from training
    val_dataset.user2idx = checkpoint['user2idx']
    val_dataset.news2idx = checkpoint['news2idx']

    print(f"Validation samples: {len(val_dataset)}")

    # Generate predictions
    use_gnn = checkpoint.get('args', {}).get('use_gnn', True)
    predictions = predict_impressions(model, val_dataset, device, llm_embeddings, use_gnn)
    print(f"Generated predictions for {len(predictions)} impressions")

    # Compute simple metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    metrics = compute_simple_metrics(predictions)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Generate evaluation files
    os.makedirs(args.output_dir, exist_ok=True)
    generate_evaluation_files(predictions, args.output_dir)

    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict and Evaluate LLM-Enhanced News Recommendation Model'
    )

    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\News-Recommender\data',
                        help='Path to data directory')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--llm_embedding_path', type=str, default=None,
                        help='Path to LLM embeddings (.npy)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory for predictions')

    args = parser.parse_args()

    # Auto-determine output directory if not provided
    if args.output_dir is None:
        model_dir = os.path.dirname(args.model_path)
        args.output_dir = os.path.join(model_dir, 'eval')

    main(args)
