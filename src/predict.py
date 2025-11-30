"""
Prediction and Evaluation script for News Recommendation Model
"""
import os
import sys
import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from model import NewsRecommender
from data_loader import MINDDataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = NewsRecommender(
        num_users=checkpoint['num_users'],
        num_news=checkpoint['num_news'],
        embedding_dim=128,  # Should match training config
        dropout=0.2
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")

    return model, checkpoint


def predict_impressions(model, val_dataset, device):
    """
    Generate predictions for all impressions in validation set

    Returns:
        predictions: dict {impression_id: [(news_idx, score, label), ...]}
    """
    print("\nGenerating predictions...")

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
            scores = model.predict(user_tensor, candidate_tensor, history_tensor)
            scores = scores.cpu().numpy()

            # Store results
            predictions[imp_id] = [(news_idx, score, label)
                                   for news_idx, score, label in zip(candidate_news, scores, labels)]

    return predictions


def generate_evaluation_files(predictions, output_dir):
    """
    Generate prediction.txt and truth.txt for evaluation

    Format:
    - Each line: impression_id [rank1, rank2, ...]
    - Ranks are sorted by predicted scores
    """
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


def run_evaluation(output_dir, eval_script_path):
    """Run the official evaluation script"""
    import subprocess

    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)

    # Create directories for evaluation
    res_dir = os.path.join(output_dir, 'res')
    ref_dir = os.path.join(output_dir, 'ref')
    scores_dir = os.path.join(output_dir, 'scores')

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    # Copy prediction and truth files
    import shutil
    shutil.copy(os.path.join(output_dir, 'prediction.txt'),
                os.path.join(res_dir, 'prediction.txt'))
    shutil.copy(os.path.join(output_dir, 'truth.txt'),
                os.path.join(ref_dir, 'truth.txt'))

    # Run evaluation
    cmd = [sys.executable, eval_script_path, output_dir, scores_dir]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\nEvaluation completed successfully!")

        # Read and print scores
        scores_file = os.path.join(scores_dir, 'scores.txt')
        if os.path.exists(scores_file):
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            with open(scores_file, 'r') as f:
                print(f.read())
        return True
    else:
        print(f"\nEvaluation failed!")
        print(f"Error: {result.stderr}")
        return False


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, checkpoint = load_model(args.model_path, device)

    # Load validation dataset
    print("\n" + "=" * 60)
    print("Loading Validation Data")
    print("=" * 60)

    val_behaviors = os.path.join(args.data_dir, 'mind_small', 'valid', 'behaviors.tsv')
    val_news = os.path.join(args.data_dir, 'mind_small', 'valid', 'news.tsv')

    val_dataset = MINDDataset(val_behaviors, val_news, mode='valid')

    # Update dataset vocabularies with those from training
    val_dataset.user2idx = checkpoint['user2idx']
    val_dataset.news2idx = checkpoint['news2idx']

    print(f"Validation samples: {len(val_dataset)}")

    # Generate predictions
    predictions = predict_impressions(model, val_dataset, device)
    print(f"Generated predictions for {len(predictions)} impressions")

    # Generate evaluation files
    os.makedirs(args.output_dir, exist_ok=True)
    generate_evaluation_files(predictions, args.output_dir)

    # Run evaluation
    if args.run_eval:
        eval_script = os.path.join(os.path.dirname(__file__), 'evaluate.py')
        if os.path.exists(eval_script):
            run_evaluation(args.output_dir, eval_script)
        else:
            print(f"\nWarning: Evaluation script not found at {eval_script}")
            print("Please run evaluation manually using evaluate.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict and Evaluate News Recommendation Model')

    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\News-Recommender\data',
                        help='Path to data directory')
    parser.add_argument('--model_path', type=str, default=r'D:\Desktop\News-Recommender\output\best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=r'D:\Desktop\News-Recommender\output\eval',
                        help='Path to output directory for predictions')
    parser.add_argument('--run_eval', action='store_true', default=True,
                        help='Run evaluation script after prediction')

    args = parser.parse_args()

    main(args)
