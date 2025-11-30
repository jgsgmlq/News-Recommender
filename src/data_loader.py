"""
MIND Dataset Loader for News Recommendation
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader


class MINDDataset(Dataset):
    """MIND Dataset for news recommendation"""

    def __init__(self, behaviors_path, news_path, mode='train'):
        """
        Args:
            behaviors_path: Path to behaviors.tsv
            news_path: Path to news.tsv
            mode: 'train' or 'valid'
        """
        self.mode = mode

        # Load news data
        print(f"Loading news from {news_path}")
        self.news_df = self._load_news(news_path)

        # Load behaviors data
        print(f"Loading behaviors from {behaviors_path}")
        self.behaviors_df = self._load_behaviors(behaviors_path)

        # Build vocabularies
        self._build_vocabularies()

        # Prepare training samples
        self.samples = self._prepare_samples()
        print(f"Total samples: {len(self.samples)}")

    def _load_news(self, path):
        """Load news.tsv"""
        news_df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        )
        return news_df

    def _load_behaviors(self, path):
        """Load behaviors.tsv"""
        behaviors_df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time', 'history', 'impressions']
        )
        return behaviors_df

    def _build_vocabularies(self):
        """Build news_id and user_id vocabularies"""
        # News vocabulary
        all_news_ids = set(self.news_df['news_id'].values)

        # Add news from behaviors history and impressions
        for _, row in self.behaviors_df.iterrows():
            if pd.notna(row['history']):
                all_news_ids.update(row['history'].split())
            if pd.notna(row['impressions']):
                impression_news = [imp.split('-')[0] for imp in row['impressions'].split()]
                all_news_ids.update(impression_news)

        # Create news_id to index mapping (0 reserved for padding)
        self.news2idx = {news_id: idx + 1 for idx, news_id in enumerate(sorted(all_news_ids))}
        self.news2idx['<PAD>'] = 0
        self.idx2news = {idx: news_id for news_id, idx in self.news2idx.items()}

        # User vocabulary
        all_user_ids = set(self.behaviors_df['user_id'].values)
        self.user2idx = {user_id: idx for idx, user_id in enumerate(sorted(all_user_ids))}
        self.idx2user = {idx: user_id for user_id, idx in self.user2idx.items()}

        print(f"Vocabulary built: {len(self.news2idx)} news, {len(self.user2idx)} users")

    def _prepare_samples(self):
        """Prepare training samples from behaviors"""
        samples = []

        for _, row in self.behaviors_df.iterrows():
            user_id = row['user_id']
            user_idx = self.user2idx[user_id]

            # Parse history
            if pd.notna(row['history']) and row['history']:
                history = [self.news2idx.get(nid, 0) for nid in row['history'].split()]
            else:
                history = []

            # Parse impressions
            if pd.notna(row['impressions']) and row['impressions']:
                impressions = row['impressions'].split()

                for imp in impressions:
                    parts = imp.split('-')
                    if len(parts) == 2:
                        news_id, label = parts
                        news_idx = self.news2idx.get(news_id, 0)
                        label = int(label)

                        samples.append({
                            'user_idx': user_idx,
                            'news_idx': news_idx,
                            'history': history,
                            'label': label,
                            'impression_id': row['impression_id']
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Limit history length to 50 (most recent)
        history = sample['history'][-50:] if len(sample['history']) > 50 else sample['history']

        return {
            'user_idx': sample['user_idx'],
            'news_idx': sample['news_idx'],
            'history': history,
            'label': sample['label'],
            'impression_id': sample['impression_id']
        }


def collate_fn(batch):
    """Custom collate function to handle variable length history"""
    user_indices = torch.LongTensor([item['user_idx'] for item in batch])
    news_indices = torch.LongTensor([item['news_idx'] for item in batch])
    labels = torch.FloatTensor([item['label'] for item in batch])

    # Pad history to same length
    max_hist_len = max(len(item['history']) for item in batch)
    if max_hist_len == 0:
        max_hist_len = 1

    history_padded = []
    for item in batch:
        hist = item['history']
        if len(hist) == 0:
            hist = [0]  # Add padding if empty
        padded = hist + [0] * (max_hist_len - len(hist))
        history_padded.append(padded)

    history_tensor = torch.LongTensor(history_padded)

    return {
        'user_idx': user_indices,
        'news_idx': news_indices,
        'history': history_tensor,
        'label': labels
    }


def get_dataloaders(data_dir, batch_size=64, num_workers=0):
    """
    Create train and validation dataloaders

    Args:
        data_dir: Root directory containing mind_small folder
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    train_behaviors = os.path.join(data_dir, 'mind_small', 'train', 'behaviors.tsv')
    train_news = os.path.join(data_dir, 'mind_small', 'train', 'news.tsv')

    val_behaviors = os.path.join(data_dir, 'mind_small', 'valid', 'behaviors.tsv')
    val_news = os.path.join(data_dir, 'mind_small', 'valid', 'news.tsv')

    print("=" * 60)
    print("Creating Training Dataset")
    print("=" * 60)
    train_dataset = MINDDataset(train_behaviors, train_news, mode='train')

    print("\n" + "=" * 60)
    print("Creating Validation Dataset")
    print("=" * 60)
    val_dataset = MINDDataset(val_behaviors, val_news, mode='valid')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == '__main__':
    # Test data loader
    data_dir = r'D:\Desktop\News-Recommender\data'
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(data_dir, batch_size=32)

    print("\n" + "=" * 60)
    print("Testing Data Loader")
    print("=" * 60)

    # Test one batch
    for batch in train_loader:
        print(f"User indices shape: {batch['user_idx'].shape}")
        print(f"News indices shape: {batch['news_idx'].shape}")
        print(f"History shape: {batch['history'].shape}")
        print(f"Labels shape: {batch['label'].shape}")
        print(f"First user idx: {batch['user_idx'][0]}")
        print(f"First news idx: {batch['news_idx'][0]}")
        print(f"First label: {batch['label'][0]}")
        break
