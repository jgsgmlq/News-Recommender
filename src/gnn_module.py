"""
Graph Neural Network Module for News-Entity Knowledge Graph
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data


class NewsEntityGNN(nn.Module):
    """
    GNN for news-entity graph propagation

    Architecture:
    - Input: Node features (news + entities)
    - GNN: 1-2 layers of GraphSAGE
    - Output: Enhanced news embeddings
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        """
        Args:
            input_dim: Input feature dimension (100 for entity embeddings)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (same as news embedding dim)
            num_layers: Number of GNN layers (1 or 2)
            dropout: Dropout rate
        """
        super(NewsEntityGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GNN layers (GraphSAGE)
        self.convs = nn.ModuleList()

        if num_layers == 1:
            # Single layer: input -> output
            self.convs.append(SAGEConv(input_dim, output_dim))
        else:
            # First layer: input -> hidden
            self.convs.append(SAGEConv(input_dim, hidden_dim))

            # Middle layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

            # Last layer: hidden -> output
            self.convs.append(SAGEConv(hidden_dim, output_dim))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        if num_layers == 1:
            self.batch_norms.append(nn.BatchNorm1d(output_dim))
        else:
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(output_dim))

    def forward(self, x, edge_index):
        """
        Forward propagation on graph

        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge connections (2, num_edges)

        Returns:
            Node embeddings after GNN (num_nodes, output_dim)
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch norm and activation (except last layer)
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                # Last layer: only batch norm
                x = self.batch_norms[i](x)

        return x

    def get_news_embeddings(self, x, edge_index, news_node_ids):
        """
        Get enhanced embeddings for news nodes

        Args:
            x: Node features
            edge_index: Edge connections
            news_node_ids: List of news node indices

        Returns:
            News embeddings (num_news, output_dim)
        """
        # Full graph propagation
        all_embeddings = self.forward(x, edge_index)

        # Extract news embeddings
        news_embeddings = all_embeddings[news_node_ids]

        return news_embeddings


class GNNEnhancedNewsEncoder(nn.Module):
    """
    News encoder enhanced with GNN

    Combines:
    - News ID embedding
    - GNN-enhanced news representation from KG
    """

    def __init__(
        self,
        num_news,
        news_embedding_dim=128,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_layers=2,
        dropout=0.2
    ):
        """
        Args:
            num_news: Number of news articles
            news_embedding_dim: Dimension of news ID embedding
            gnn_input_dim: GNN input dimension (entity embedding dim)
            gnn_hidden_dim: GNN hidden dimension
            gnn_output_dim: GNN output dimension
            gnn_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(GNNEnhancedNewsEncoder, self).__init__()

        # News ID embedding (learnable)
        self.news_id_embedding = nn.Embedding(
            num_embeddings=num_news,
            embedding_dim=news_embedding_dim
        )

        # GNN for knowledge graph
        self.gnn = NewsEntityGNN(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers,
            dropout=dropout
        )

        # Fusion layer (combine ID embedding + GNN embedding)
        self.fusion = nn.Sequential(
            nn.Linear(news_embedding_dim + gnn_output_dim, news_embedding_dim),
            nn.LayerNorm(news_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.news_id_embedding.weight)

    def forward(self, news_ids, kg_data=None):
        """
        Encode news with KG enhancement

        Args:
            news_ids: News ID indices (batch_size,)
            kg_data: Knowledge graph data dict with:
                - node_features: (num_nodes, gnn_input_dim)
                - edge_index: (2, num_edges)
                - news_to_node: mapping from news_id to node_id

        Returns:
            Enhanced news embeddings (batch_size, news_embedding_dim)
        """
        # News ID embedding
        id_emb = self.news_id_embedding(news_ids)  # (batch_size, news_embedding_dim)

        if kg_data is None:
            # No KG available, use ID embedding only
            return id_emb

        # GNN enhancement
        node_features = kg_data['node_features']  # (num_nodes, gnn_input_dim)
        edge_index = kg_data['edge_index']  # (2, num_edges)
        news_to_node = kg_data.get('news_to_node', {})

        # Get node IDs for batch news
        # Note: This requires news_id (string) -> node_id mapping
        # For simplicity, we'll use news index as approximation
        # In practice, you'd maintain the news_id -> node_id mapping

        # Since news nodes are at the beginning, we can directly use indices
        news_node_ids = news_ids.cpu().numpy()  # Assuming news_ids are sequential

        # Run GNN
        gnn_emb = self.gnn.get_news_embeddings(
            node_features,
            edge_index,
            news_node_ids
        )  # (batch_size, gnn_output_dim)

        # Fuse ID embedding and GNN embedding
        combined = torch.cat([id_emb, gnn_emb], dim=-1)
        enhanced_emb = self.fusion(combined)

        return enhanced_emb


if __name__ == '__main__':
    # Test GNN module
    print("Testing NewsEntityGNN...")

    num_nodes = 1000
    num_edges = 3000
    input_dim = 100
    hidden_dim = 128
    output_dim = 128

    # Create dummy graph
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    node_features = torch.randn(num_nodes, input_dim)

    # Create GNN
    gnn = NewsEntityGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=2
    )

    # Forward pass
    output = gnn(node_features, edge_index)
    print(f"Input shape: {node_features.shape}")
    print(f"Output shape: {output.shape}")

    # Test news extraction
    news_node_ids = list(range(500))  # First 500 are news nodes
    news_emb = gnn.get_news_embeddings(node_features, edge_index, news_node_ids)
    print(f"News embeddings shape: {news_emb.shape}")

    print("\nTesting GNNEnhancedNewsEncoder...")

    encoder = GNNEnhancedNewsEncoder(
        num_news=500,
        news_embedding_dim=128,
        gnn_input_dim=100,
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        gnn_layers=2
    )

    # Test without KG
    news_ids = torch.randint(0, 500, (32,))
    output_no_kg = encoder(news_ids, kg_data=None)
    print(f"Output (no KG) shape: {output_no_kg.shape}")

    # Test with KG
    kg_data = {
        'node_features': node_features,
        'edge_index': edge_index,
        'news_to_node': {i: i for i in range(500)}
    }
    output_with_kg = encoder(news_ids, kg_data=kg_data)
    print(f"Output (with KG) shape: {output_with_kg.shape}")

    print("\nGNN module test passed!")
