"""
Multi-Modal News Recommendation Model
Integrates: ID Embeddings + LLM Text Embeddings + GNN Entity Embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_module import NewsEntityGNN


class MultiModalNewsEncoder(nn.Module):
    """
    Multi-modal news encoder
    Fuses: ID embedding + LLM text embedding + GNN entity embedding
    """

    def __init__(
        self,
        num_news,
        id_emb_dim=128,
        llm_emb_dim=1536,
        gnn_emb_dim=128,
        output_dim=256,
        use_llm=True,
        use_gnn=True,
        fusion_method='attention',
        dropout=0.2
    ):
        """
        Args:
            num_news: Number of news articles
            id_emb_dim: ID embedding dimension
            llm_emb_dim: LLM embedding dimension (1536 for OpenAI)
            gnn_emb_dim: GNN embedding dimension
            output_dim: Output dimension
            use_llm: Whether to use LLM embeddings
            use_gnn: Whether to use GNN embeddings
            fusion_method: 'attention', 'concat', or 'gate'
            dropout: Dropout rate
        """
        super().__init__()

        self.num_news = num_news
        self.use_llm = use_llm
        self.use_gnn = use_gnn
        self.fusion_method = fusion_method
        self.output_dim = output_dim

        # 1. ID Embedding (learnable)
        self.id_embedding = nn.Embedding(
            num_embeddings=num_news,
            embedding_dim=id_emb_dim,
            padding_idx=0
        )

        # 2. Projection layers
        self.id_proj = nn.Sequential(
            nn.Linear(id_emb_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if use_llm:
            self.llm_proj = nn.Sequential(
                nn.Linear(llm_emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        if use_gnn:
            self.gnn_proj = nn.Sequential(
                nn.Linear(gnn_emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # 3. Fusion layer
        num_modalities = 1 + int(use_llm) + int(use_gnn)

        if fusion_method == 'attention':
            # Attention-based fusion with learnable gates
            self.fusion_query = nn.Parameter(torch.randn(1, output_dim))
            self.fusion_key = nn.Linear(output_dim, output_dim)

        elif fusion_method == 'gate':
            # Gate-based fusion
            self.fusion_gate = nn.Sequential(
                nn.Linear(output_dim * num_modalities, num_modalities),
                nn.Softmax(dim=-1)
            )

        elif fusion_method == 'concat':
            # Simple concatenation + MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(output_dim * num_modalities, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.id_embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, news_ids, llm_embeddings=None, gnn_embeddings=None):
        """
        Forward pass

        Args:
            news_ids: (batch_size,) news indices
            llm_embeddings: (num_news, llm_emb_dim) pre-loaded LLM embeddings
            gnn_embeddings: (num_news, gnn_emb_dim) pre-computed GNN embeddings

        Returns:
            news_repr: (batch_size, output_dim)
        """
        batch_size = news_ids.size(0)
        device = news_ids.device

        # 1. Get ID representation
        id_emb = self.id_embedding(news_ids)  # (B, id_emb_dim)
        id_repr = self.id_proj(id_emb)        # (B, output_dim)

        # Collect all modalities
        modalities = [id_repr]

        # 2. Get LLM representation
        if self.use_llm and llm_embeddings is not None:
            # Handle out-of-bounds indices
            num_llm_news = llm_embeddings.size(0)
            llm_emb = torch.zeros(batch_size, llm_embeddings.size(1),
                                  device=device, dtype=llm_embeddings.dtype)

            valid_mask = news_ids < num_llm_news
            valid_indices = news_ids[valid_mask]

            if valid_indices.numel() > 0:
                llm_emb[valid_mask] = llm_embeddings[valid_indices]

            llm_repr = self.llm_proj(llm_emb)  # (B, output_dim)
            modalities.append(llm_repr)

        # 3. Get GNN representation
        if self.use_gnn and gnn_embeddings is not None:
            # Handle out-of-bounds indices
            num_gnn_news = gnn_embeddings.size(0)
            gnn_emb = torch.zeros(batch_size, gnn_embeddings.size(1),
                                  device=device, dtype=gnn_embeddings.dtype)

            valid_mask = news_ids < num_gnn_news
            valid_indices = news_ids[valid_mask]

            if valid_indices.numel() > 0:
                gnn_emb[valid_mask] = gnn_embeddings[valid_indices]

            gnn_repr = self.gnn_proj(gnn_emb)  # (B, output_dim)
            modalities.append(gnn_repr)

        # 4. Fuse modalities
        if len(modalities) == 1:
            # Only ID, no fusion needed
            news_repr = modalities[0]

        elif self.fusion_method == 'attention':
            # Attention-based fusion
            stacked = torch.stack(modalities, dim=1)  # (B, num_modalities, output_dim)

            # Compute attention scores
            query = self.fusion_query.expand(batch_size, -1).unsqueeze(1)  # (B, 1, output_dim)
            keys = self.fusion_key(stacked)  # (B, num_modalities, output_dim)

            scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, num_modalities)
            attn_weights = F.softmax(scores / (self.output_dim ** 0.5), dim=-1)  # (B, 1, num_modalities)

            news_repr = torch.bmm(attn_weights, stacked).squeeze(1)  # (B, output_dim)

        elif self.fusion_method == 'gate':
            # Gate-based fusion
            concat = torch.cat(modalities, dim=-1)  # (B, output_dim * num_modalities)
            gates = self.fusion_gate(concat)  # (B, num_modalities)

            news_repr = sum(
                gates[:, i:i+1] * modalities[i]
                for i in range(len(modalities))
            )

        elif self.fusion_method == 'concat':
            # Concatenation + MLP
            concat = torch.cat(modalities, dim=-1)
            news_repr = self.fusion_mlp(concat)

        return news_repr


class LLMEnhancedRecommender(nn.Module):
    """
    LLM-Enhanced News Recommendation Model
    Integrates ID + LLM + GNN embeddings
    """

    def __init__(
        self,
        num_users,
        num_news,
        embedding_dim=128,
        llm_emb_dim=1536,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        output_dim=256,
        gnn_layers=2,
        use_llm=True,
        use_gnn=True,
        fusion_method='attention',
        dropout=0.2
    ):
        """
        Args:
            num_users: Number of users
            num_news: Number of news articles
            embedding_dim: ID embedding dimension
            llm_emb_dim: LLM embedding dimension
            gnn_input_dim: GNN input dimension (entity embedding dim)
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            output_dim: Final representation dimension
            gnn_layers: Number of GNN layers
            use_llm: Whether to use LLM embeddings
            use_gnn: Whether to use GNN embeddings
            fusion_method: Fusion method
            dropout: Dropout rate
        """
        super().__init__()

        self.use_llm = use_llm
        self.use_gnn = use_gnn
        self.output_dim = output_dim

        # User embedding
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )

        # News encoder (multi-modal)
        self.news_encoder = MultiModalNewsEncoder(
            num_news=num_news,
            id_emb_dim=embedding_dim,
            llm_emb_dim=llm_emb_dim,
            gnn_emb_dim=gnn_output_dim,
            output_dim=output_dim,
            use_llm=use_llm,
            use_gnn=use_gnn,
            fusion_method=fusion_method,
            dropout=dropout
        )

        # GNN for knowledge graph
        if use_gnn:
            self.gnn = NewsEntityGNN(
                input_dim=gnn_input_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_layers=gnn_layers,
                dropout=dropout
            )

        # User transform layers
        self.user_transform = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # History attention
        self.history_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1)
        )

        # Store graph data
        self.graph_data = None

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_graph_data(self, graph_data):
        """Set knowledge graph data for GNN"""
        self.graph_data = graph_data

        if graph_data is not None:
            device = next(self.parameters()).device
            self.graph_data['node_features'] = graph_data['node_features'].to(device)
            self.graph_data['edge_index'] = graph_data['edge_index'].to(device)

    def get_gnn_embeddings(self):
        """Pre-compute GNN embeddings for all news"""
        if not self.use_gnn or self.graph_data is None:
            return None

        node_features = self.graph_data['node_features']
        edge_index = self.graph_data['edge_index']
        num_news = self.graph_data['num_news']

        # Forward propagation
        all_embeddings = self.gnn(node_features, edge_index)

        # Extract news node embeddings
        news_embeddings = all_embeddings[:num_news]

        return news_embeddings

    def get_user_representation(self, user_idx, history, llm_embeddings=None, gnn_embeddings=None):
        """
        Get user representation from user_id and history

        Args:
            user_idx: (batch_size,)
            history: (batch_size, max_hist_len)
            llm_embeddings: Pre-loaded LLM embeddings
            gnn_embeddings: Pre-computed GNN embeddings

        Returns:
            user_repr: (batch_size, output_dim)
        """
        # User embedding
        user_emb = self.user_embedding(user_idx)  # (batch_size, embedding_dim)

        # History news representations (use multi-modal encoder)
        hist_repr = self.news_encoder(
            history.reshape(-1),  # Flatten
            llm_embeddings=llm_embeddings,
            gnn_embeddings=gnn_embeddings
        )  # (batch_size * max_hist_len, output_dim)

        hist_repr = hist_repr.reshape(history.size(0), history.size(1), -1)  # (batch_size, max_hist_len, output_dim)

        # Create mask for padding
        mask = (history != 0).float().unsqueeze(-1)  # (batch_size, max_hist_len, 1)

        # Apply attention to history
        hist_attn_scores = self.history_attention(hist_repr)  # (batch_size, max_hist_len, 1)
        hist_attn_scores = hist_attn_scores.masked_fill(mask == 0, -1e9)
        hist_attn_weights = F.softmax(hist_attn_scores, dim=1)  # (batch_size, max_hist_len, 1)

        # Weighted sum of history
        hist_aggregated = torch.sum(hist_repr * hist_attn_weights, dim=1)  # (batch_size, output_dim)

        # Transform user embedding
        user_repr = self.user_transform(user_emb)  # (batch_size, output_dim)

        # Combine user and history
        user_repr = user_repr + hist_aggregated

        return user_repr

    def forward(self, user_idx, news_idx, history, llm_embeddings=None, gnn_embeddings=None):
        """
        Forward pass

        Args:
            user_idx: (batch_size,)
            news_idx: (batch_size,)
            history: (batch_size, max_hist_len)
            llm_embeddings: (num_news, llm_emb_dim) global LLM embeddings
            gnn_embeddings: (num_news, gnn_emb_dim) global GNN embeddings

        Returns:
            scores: (batch_size,) click probabilities
        """
        # Get representations
        user_repr = self.get_user_representation(user_idx, history, llm_embeddings, gnn_embeddings)
        news_repr = self.news_encoder(news_idx, llm_embeddings, gnn_embeddings)

        # L2 normalization for stable cosine similarity
        user_repr = torch.nn.functional.normalize(user_repr, p=2, dim=1)
        news_repr = torch.nn.functional.normalize(news_repr, p=2, dim=1)

        # Compute cosine similarity (range: [-1, 1])
        cosine_sim = torch.sum(user_repr * news_repr, dim=1)

        # Scale to reasonable range for sigmoid (multiply by 2.0)
        # This gives logits in range [-2, 2], with sigmoid(0)=0.5, sigmoid(1)=0.73, sigmoid(-1)=0.27
        logits = cosine_sim * 2.0

        # Sigmoid
        scores = torch.sigmoid(logits)

        return scores

    def predict(self, user_idx, candidate_news_indices, history, llm_embeddings=None, gnn_embeddings=None):
        """
        Predict scores for multiple candidate news

        Args:
            user_idx: (1,) single user
            candidate_news_indices: (num_candidates,)
            history: (1, max_hist_len)
            llm_embeddings: Pre-loaded LLM embeddings
            gnn_embeddings: Pre-computed GNN embeddings

        Returns:
            scores: (num_candidates,)
        """
        num_candidates = candidate_news_indices.size(0)

        # Expand user and history
        user_idx_expanded = user_idx.expand(num_candidates)
        history_expanded = history.expand(num_candidates, -1)

        # Get scores
        with torch.no_grad():
            scores = self.forward(
                user_idx_expanded,
                candidate_news_indices,
                history_expanded,
                llm_embeddings,
                gnn_embeddings
            )

        return scores


if __name__ == '__main__':
    # Test multi-modal model
    print("Testing LLMEnhancedRecommender...")

    num_users = 1000
    num_news = 5000
    batch_size = 32
    max_hist_len = 20

    # Create model
    model = LLMEnhancedRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=128,
        llm_emb_dim=1536,
        gnn_output_dim=128,
        output_dim=256,
        use_llm=True,
        use_gnn=True,
        fusion_method='attention'
    )

    # Create dummy data
    user_idx = torch.randint(0, num_users, (batch_size,))
    news_idx = torch.randint(1, num_news, (batch_size,))
    history = torch.randint(0, num_news, (batch_size, max_hist_len))

    # Create dummy LLM embeddings
    llm_embeddings = torch.randn(num_news, 1536)

    # Create dummy GNN embeddings
    gnn_embeddings = torch.randn(num_news, 128)

    # Forward pass
    print("\nForward pass...")
    scores = model(user_idx, news_idx, history, llm_embeddings, gnn_embeddings)

    print(f"Input shapes:")
    print(f"  user_idx: {user_idx.shape}")
    print(f"  news_idx: {news_idx.shape}")
    print(f"  history: {history.shape}")
    print(f"  llm_embeddings: {llm_embeddings.shape}")
    print(f"  gnn_embeddings: {gnn_embeddings.shape}")
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    # Test prediction
    print("\nTesting prediction mode...")
    user_idx_single = torch.tensor([0])
    candidate_news = torch.randint(1, num_news, (10,))
    history_single = torch.randint(0, num_news, (1, max_hist_len))

    pred_scores = model.predict(user_idx_single, candidate_news, history_single, llm_embeddings, gnn_embeddings)
    print(f"Prediction scores shape: {pred_scores.shape}")
    print(f"Prediction scores: {pred_scores}")

    print("\nLLM model test passed!")
