"""
GNN-Enhanced News Recommendation Model
Integrates Knowledge Graph with GNN for news representation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_module import NewsEntityGNN


class GNNNewsRecommender(nn.Module):
    """
    GNN-enhanced news recommendation model

    Architecture:
    - User representation: User ID embedding + History aggregation
    - News representation: ID embedding + GNN-enhanced KG embedding
    - Score: similarity(user_repr, news_repr)
    """

    def __init__(
        self,
        num_users,
        num_news,
        embedding_dim=128,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_layers=2,
        dropout=0.2,
        use_gnn=True
    ):
        """
        Args:
            num_users: Number of unique users
            num_news: Number of unique news articles
            embedding_dim: Dimension of ID embeddings
            gnn_input_dim: GNN input dimension (entity embedding dim)
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_layers: Number of GNN layers (1 or 2)
            dropout: Dropout rate
            use_gnn: Whether to use GNN enhancement
        """
        super(GNNNewsRecommender, self).__init__()

        self.embedding_dim = embedding_dim
        self.use_gnn = use_gnn

        # User embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        # News ID embedding (with padding_idx=0)
        self.news_id_embedding = nn.Embedding(
            num_embeddings=num_news,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # GNN for knowledge graph
        if self.use_gnn:
            self.gnn = NewsEntityGNN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_layers=gnn_layers,
                dropout=dropout
            )

            # Fusion layer (combine ID embedding + GNN embedding)
            self.news_fusion = nn.Sequential(
                nn.Linear(embedding_dim + gnn_output_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
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

        # History attention
        self.history_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1)
        )

        # Store graph data
        self.graph_data = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.news_id_embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_graph_data(self, graph_data):
        """
        Set knowledge graph data for GNN

        Args:
            graph_data: Dict containing:
                - node_features: (num_nodes, gnn_input_dim)
                - edge_index: (2, num_edges)
                - news_to_node: mapping from news_id to node_id
        """
        self.graph_data = graph_data

        # Move graph data to same device as model
        if graph_data is not None:
            device = next(self.parameters()).device
            self.graph_data['node_features'] = graph_data['node_features'].to(device)
            self.graph_data['edge_index'] = graph_data['edge_index'].to(device)

    def get_gnn_enhanced_news_embeddings(self):
        """
        Pre-compute GNN-enhanced embeddings for all news

        Returns:
            enhanced_embeddings: (num_news, gnn_output_dim)
        """
        if not self.use_gnn or self.graph_data is None:
            return None

        # Run GNN on full graph
        node_features = self.graph_data['node_features']
        edge_index = self.graph_data['edge_index']
        num_news = self.graph_data['num_news']

        # Forward propagation
        all_embeddings = self.gnn(node_features, edge_index)

        # Extract news node embeddings (first num_news nodes)
        news_embeddings = all_embeddings[:num_news]

        return news_embeddings

    def get_news_representation(self, news_idx, gnn_embeddings=None):
        """
        Get news representation with optional GNN enhancement

        Args:
            news_idx: (batch_size,)
            gnn_embeddings: Pre-computed GNN embeddings for all news (num_news, gnn_output_dim)

        Returns:
            news_repr: (batch_size, embedding_dim)
        """
        # News ID embedding
        id_emb = self.news_id_embedding(news_idx)  # (batch_size, embedding_dim)

        if not self.use_gnn or gnn_embeddings is None:
            # No GNN, use ID embedding only
            news_repr = self.news_transform(id_emb)
            return news_repr

        # Get GNN embeddings for this batch
        # Handle out-of-bounds indices by using zero embeddings
        batch_size = news_idx.size(0)
        gnn_dim = gnn_embeddings.size(1)
        num_gnn_news = gnn_embeddings.size(0)

        gnn_emb = torch.zeros(batch_size, gnn_dim, device=news_idx.device, dtype=gnn_embeddings.dtype)

        # Only get embeddings for valid indices
        valid_mask = news_idx < num_gnn_news
        valid_indices = news_idx[valid_mask]

        if valid_indices.numel() > 0:
            gnn_emb[valid_mask] = gnn_embeddings[valid_indices]

        # Fuse ID embedding and GNN embedding
        combined = torch.cat([id_emb, gnn_emb], dim=-1)
        fused_emb = self.news_fusion(combined)

        # Transform
        news_repr = self.news_transform(fused_emb)

        return news_repr

    def get_user_representation(self, user_idx, history, gnn_embeddings=None):
        """
        Get user representation from user_id and history

        Args:
            user_idx: (batch_size,)
            history: (batch_size, max_hist_len)
            gnn_embeddings: Pre-computed GNN embeddings for all news

        Returns:
            user_repr: (batch_size, embedding_dim)
        """
        batch_size = user_idx.size(0)

        # User embedding
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)

        # History news representations
        # We need to get representations for history news
        if self.use_gnn and gnn_embeddings is not None:
            # ID embeddings
            hist_id_emb = self.news_id_embedding(history)  # (batch_size, max_hist_len, embedding_dim)

            # GNN embeddings (handle out-of-bounds indices)
            batch_size, max_hist_len = history.size()
            gnn_dim = gnn_embeddings.size(1)
            num_gnn_news = gnn_embeddings.size(0)

            hist_gnn_emb = torch.zeros(batch_size, max_hist_len, gnn_dim, device=history.device, dtype=gnn_embeddings.dtype)

            # Only get embeddings for valid indices
            valid_mask = history < num_gnn_news
            valid_indices = history[valid_mask]

            if valid_indices.numel() > 0:
                hist_gnn_emb[valid_mask] = gnn_embeddings[valid_indices]

            # Fuse
            hist_combined = torch.cat([hist_id_emb, hist_gnn_emb], dim=-1)
            hist_emb = self.news_fusion(hist_combined)  # (batch_size, max_hist_len, embedding_dim)
        else:
            hist_emb = self.news_id_embedding(history)  # (batch_size, max_hist_len, embedding_dim)

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
        user_repr = user_emb + hist_repr

        # Transform
        user_repr = self.user_transform(user_repr)

        return user_repr

    def forward(self, user_idx, news_idx, history, gnn_embeddings=None):
        """
        Forward pass

        Args:
            user_idx: (batch_size,)
            news_idx: (batch_size,)
            history: (batch_size, max_hist_len)
            gnn_embeddings: Pre-computed GNN embeddings for all news

        Returns:
            scores: (batch_size,) - click probability scores
        """
        # Get representations
        user_repr = self.get_user_representation(user_idx, history, gnn_embeddings)
        news_repr = self.get_news_representation(news_idx, gnn_embeddings)

        # Compute similarity score (dot product)
        scores = torch.sum(user_repr * news_repr, dim=1)  # (batch_size,)

        # Apply sigmoid to get probability
        scores = torch.sigmoid(scores)

        return scores

    def predict(self, user_idx, candidate_news_indices, history, gnn_embeddings=None):
        """
        Predict scores for multiple candidate news for ranking

        Args:
            user_idx: (1,) - single user
            candidate_news_indices: (num_candidates,)
            history: (1, max_hist_len)
            gnn_embeddings: Pre-computed GNN embeddings for all news

        Returns:
            scores: (num_candidates,)
        """
        num_candidates = candidate_news_indices.size(0)

        # Expand user and history
        user_idx_expanded = user_idx.expand(num_candidates)
        history_expanded = history.expand(num_candidates, -1)

        # Get scores
        with torch.no_grad():
            scores = self.forward(user_idx_expanded, candidate_news_indices, history_expanded, gnn_embeddings)

        return scores


if __name__ == '__main__':
    # Test GNN-enhanced model
    print("Testing GNNNewsRecommender model...")

    num_users = 1000
    num_news = 5000
    batch_size = 32
    max_hist_len = 20

    # Create model
    model = GNNNewsRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=128,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_layers=2,
        use_gnn=True
    )

    # Create dummy graph data
    num_nodes = 6000  # 5000 news + 1000 entities
    num_edges = 15000

    graph_data = {
        'node_features': torch.randn(num_nodes, 100),
        'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
        'num_news': num_news,
        'num_entities': 1000,
        'news_to_node': {i: i for i in range(num_news)}
    }

    model.set_graph_data(graph_data)

    # Pre-compute GNN embeddings
    print("Computing GNN embeddings...")
    gnn_embeddings = model.get_gnn_enhanced_news_embeddings()
    print(f"GNN embeddings shape: {gnn_embeddings.shape}")

    # Create dummy input
    user_idx = torch.randint(0, num_users, (batch_size,))
    news_idx = torch.randint(1, num_news, (batch_size,))
    history = torch.randint(0, num_news, (batch_size, max_hist_len))

    # Forward pass
    print("\nForward pass...")
    scores = model(user_idx, news_idx, history, gnn_embeddings)

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

    pred_scores = model.predict(user_idx_single, candidate_news, history_single, gnn_embeddings)
    print(f"Prediction scores shape: {pred_scores.shape}")
    print(f"Prediction scores: {pred_scores}")

    # Test without GNN
    print("\nTesting without GNN...")
    model_no_gnn = GNNNewsRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=128,
        use_gnn=False
    )

    scores_no_gnn = model_no_gnn(user_idx, news_idx, history, gnn_embeddings=None)
    print(f"Scores without GNN shape: {scores_no_gnn.shape}")

    print("\nGNN model test passed!")
