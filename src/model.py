"""
Simple ID-based News Recommendation Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsRecommender(nn.Module):
    """
    Simple news recommendation model based on user/news ID embeddings

    Architecture:
    - User embedding
    - News embedding
    - User history aggregation (average pooling)
    - Score = similarity(user_repr, news_embedding)
    """

    def __init__(self, num_users, num_news, embedding_dim=128, dropout=0.2):
        """
        Args:
            num_users: Number of unique users
            num_news: Number of unique news articles
            embedding_dim: Dimension of embeddings
            dropout: Dropout rate
        """
        super(NewsRecommender, self).__init__()

        self.embedding_dim = embedding_dim

        # User embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
            padding_idx=None
        )

        # News embedding (with padding_idx=0)
        self.news_embedding = nn.Embedding(
            num_embeddings=num_news,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Transform layers
        self.user_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.news_transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # History attention (simple version)
        self.history_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.news_embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_user_representation(self, user_idx, history):
        """
        Get user representation from user_id and history

        Args:
            user_idx: (batch_size,)
            history: (batch_size, max_hist_len)

        Returns:
            user_repr: (batch_size, embedding_dim)
        """
        batch_size = user_idx.size(0)

        # User embedding
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)

        # History embeddings
        hist_emb = self.news_embedding(history)  # (batch_size, max_hist_len, embedding_dim)

        # Create mask for padding
        mask = (history != 0).float().unsqueeze(-1)  # (batch_size, max_hist_len, 1)

        # Apply attention to history
        hist_attn_scores = self.history_attention(hist_emb)  # (batch_size, max_hist_len, 1)

        # Mask out padding
        hist_attn_scores = hist_attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax attention
        hist_attn_weights = F.softmax(hist_attn_scores, dim=1)  # (batch_size, max_hist_len, 1)

        # Weighted sum of history
        hist_repr = torch.sum(hist_emb * hist_attn_weights, dim=1)  # (batch_size, embedding_dim)

        # Combine user embedding and history representation
        user_repr = user_emb + hist_repr  # Simple addition

        # Transform
        user_repr = self.user_transform(user_repr)

        return user_repr

    def get_news_representation(self, news_idx):
        """
        Get news representation

        Args:
            news_idx: (batch_size,)

        Returns:
            news_repr: (batch_size, embedding_dim)
        """
        news_emb = self.news_embedding(news_idx)
        news_repr = self.news_transform(news_emb)
        return news_repr

    def forward(self, user_idx, news_idx, history):
        """
        Forward pass

        Args:
            user_idx: (batch_size,)
            news_idx: (batch_size,)
            history: (batch_size, max_hist_len)

        Returns:
            scores: (batch_size,) - click probability scores
        """
        # Get representations
        user_repr = self.get_user_representation(user_idx, history)  # (batch_size, embedding_dim)
        news_repr = self.get_news_representation(news_idx)  # (batch_size, embedding_dim)

        # Compute similarity score (dot product)
        scores = torch.sum(user_repr * news_repr, dim=1)  # (batch_size,)

        # Apply sigmoid to get probability
        scores = torch.sigmoid(scores)

        return scores

    def predict(self, user_idx, candidate_news_indices, history):
        """
        Predict scores for multiple candidate news for ranking

        Args:
            user_idx: (1,) - single user
            candidate_news_indices: (num_candidates,)
            history: (1, max_hist_len)

        Returns:
            scores: (num_candidates,)
        """
        num_candidates = candidate_news_indices.size(0)

        # Expand user and history
        user_idx_expanded = user_idx.expand(num_candidates)  # (num_candidates,)
        history_expanded = history.expand(num_candidates, -1)  # (num_candidates, max_hist_len)

        # Get scores
        with torch.no_grad():
            scores = self.forward(user_idx_expanded, candidate_news_indices, history_expanded)

        return scores


if __name__ == '__main__':
    # Test model
    print("Testing NewsRecommender model...")

    num_users = 1000
    num_news = 5000
    batch_size = 32
    max_hist_len = 20

    model = NewsRecommender(num_users=num_users, num_news=num_news, embedding_dim=64)

    # Create dummy input
    user_idx = torch.randint(0, num_users, (batch_size,))
    news_idx = torch.randint(1, num_news, (batch_size,))
    history = torch.randint(0, num_news, (batch_size, max_hist_len))

    # Forward pass
    scores = model(user_idx, news_idx, history)

    print(f"Input shapes:")
    print(f"  user_idx: {user_idx.shape}")
    print(f"  news_idx: {news_idx.shape}")
    print(f"  history: {history.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    # Test prediction mode
    print("\nTesting prediction mode...")
    user_idx_single = torch.tensor([0])
    candidate_news = torch.randint(1, num_news, (10,))
    history_single = torch.randint(0, num_news, (1, max_hist_len))

    pred_scores = model.predict(user_idx_single, candidate_news, history_single)
    print(f"Prediction scores shape: {pred_scores.shape}")
    print(f"Prediction scores: {pred_scores}")

    print("\nModel test passed!")
