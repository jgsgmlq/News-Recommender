"""
Knowledge Graph Utilities for MIND Dataset
Builds news-entity graph for GNN
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict


class KnowledgeGraphBuilder:
    """Build news-entity knowledge graph from MIND dataset"""

    def __init__(self, news_path, entity_embedding_path, max_news=None):
        """
        Args:
            news_path: Path to news.tsv
            entity_embedding_path: Path to entity_embedding.vec
            max_news: Maximum number of news to use (for memory efficiency)
        """
        self.max_news = max_news

        # Load entity embeddings
        print(f"Loading entity embeddings from {entity_embedding_path}")
        self.entity_embeddings, self.entity2idx = self._load_entity_embeddings(entity_embedding_path)

        # Load news and extract entities
        print(f"Loading news from {news_path}")
        self.news_df, self.news_entities = self._load_news_entities(news_path)

        # Build graph
        print("Building news-entity graph")
        self.edge_index, self.node_features = self._build_graph()

        print(f"Graph built: {self.edge_index.shape[1]} edges")

    def _load_entity_embeddings(self, path):
        """Load entity embeddings from .vec file"""
        entity_embeddings = {}
        entity2idx = {}

        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue

                entity_id = parts[0]
                embedding = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                entity_embeddings[entity_id] = embedding
                entity2idx[entity_id] = idx

        print(f"Loaded {len(entity_embeddings)} entities, dim={len(list(entity_embeddings.values())[0])}")
        return entity_embeddings, entity2idx

    def _load_news_entities(self, path):
        """Load news and extract entity information"""
        news_df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
            nrows=self.max_news  # Limit for memory
        )

        # Extract entities from each news
        news_entities = {}

        for idx, row in news_df.iterrows():
            entities = set()

            # Parse title entities
            if pd.notna(row['title_entities']) and row['title_entities'] != '[]':
                try:
                    title_ents = json.loads(row['title_entities'])
                    for ent in title_ents:
                        if 'WikidataId' in ent:
                            entities.add(ent['WikidataId'])
                except:
                    pass

            # Parse abstract entities
            if pd.notna(row['abstract_entities']) and row['abstract_entities'] != '[]':
                try:
                    abstract_ents = json.loads(row['abstract_entities'])
                    for ent in abstract_ents:
                        if 'WikidataId' in ent:
                            entities.add(ent['WikidataId'])
                except:
                    pass

            news_entities[row['news_id']] = list(entities)

        # Count entities
        total_entities = sum(len(ents) for ents in news_entities.values())
        news_with_entities = sum(1 for ents in news_entities.values() if len(ents) > 0)

        print(f"News loaded: {len(news_df)}")
        print(f"News with entities: {news_with_entities}/{len(news_df)}")
        print(f"Total news-entity connections: {total_entities}")

        return news_df, news_entities

    def _build_graph(self):
        """
        Build bipartite graph: news <-> entities

        Returns:
            edge_index: (2, num_edges) edge connections
            node_features: node feature matrix
        """
        # Create node mappings
        news_ids = list(self.news_df['news_id'])
        num_news = len(news_ids)

        # Filter entities that appear in news
        active_entities = set()
        for entities in self.news_entities.values():
            active_entities.update(entities)

        # Only keep entities that have embeddings
        active_entities = [e for e in active_entities if e in self.entity2idx]
        num_entities = len(active_entities)

        print(f"Active entities (with embeddings): {num_entities}")

        # Create unified node index
        # Nodes: [news_0, news_1, ..., news_n, entity_0, entity_1, ..., entity_m]
        news_to_node = {news_id: i for i, news_id in enumerate(news_ids)}
        entity_to_node = {entity_id: num_news + i for i, entity_id in enumerate(active_entities)}

        # Build edges: news -> entity (bidirectional)
        edges = []

        for news_id, entities in self.news_entities.items():
            if news_id not in news_to_node:
                continue

            news_node = news_to_node[news_id]

            for entity_id in entities:
                if entity_id not in entity_to_node:
                    continue

                entity_node = entity_to_node[entity_id]

                # Add bidirectional edges
                edges.append([news_node, entity_node])
                edges.append([entity_node, news_node])

        if len(edges) == 0:
            print("Warning: No edges found!")
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Create node features
        # News nodes: random initialization (will be learned)
        # Entity nodes: use pre-trained embeddings
        embedding_dim = len(list(self.entity_embeddings.values())[0])

        news_features = torch.randn(num_news, embedding_dim) * 0.1  # Small random init

        entity_features = []
        for entity_id in active_entities:
            entity_features.append(self.entity_embeddings[entity_id])
        entity_features = torch.tensor(np.array(entity_features), dtype=torch.float32)

        node_features = torch.cat([news_features, entity_features], dim=0)

        # Store mappings for later use
        self.news_to_node = news_to_node
        self.entity_to_node = entity_to_node
        self.num_news = num_news
        self.num_entities = num_entities

        return edge_index, node_features

    def get_news_node_ids(self, news_ids):
        """Convert news_ids to node indices"""
        return [self.news_to_node.get(nid, -1) for nid in news_ids]

    def get_graph_data(self):
        """Get graph data for PyG"""
        return {
            'edge_index': self.edge_index,
            'node_features': self.node_features,
            'num_nodes': self.node_features.size(0),
            'num_news': self.num_news,
            'num_entities': self.num_entities,
            'news_to_node': self.news_to_node,
        }


def create_small_dataset(data_dir, output_dir, num_news=1000, num_behaviors=5000):
    """
    Create a small subset of MIND dataset for quick testing

    Args:
        data_dir: Original data directory
        output_dir: Output directory for small dataset
        num_news: Number of news articles to include
        num_behaviors: Number of behavior samples to include
    """
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    # Copy small subset of news
    train_news_path = os.path.join(data_dir, 'mind_small', 'train', 'news.tsv')
    output_news_path = os.path.join(output_dir, 'news.tsv')

    with open(train_news_path, 'r', encoding='utf-8') as f_in:
        with open(output_news_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= num_news:
                    break
                f_out.write(line)

    print(f"Created small news dataset: {num_news} articles")

    # Copy small subset of behaviors
    train_behaviors_path = os.path.join(data_dir, 'mind_small', 'train', 'behaviors.tsv')
    output_behaviors_path = os.path.join(output_dir, 'behaviors.tsv')

    with open(train_behaviors_path, 'r', encoding='utf-8') as f_in:
        with open(output_behaviors_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= num_behaviors:
                    break
                f_out.write(line)

    print(f"Created small behaviors dataset: {num_behaviors} samples")

    # Copy entity embeddings (full file, but will only use relevant ones)
    shutil.copy(
        os.path.join(data_dir, 'mind_small', 'train', 'entity_embedding.vec'),
        os.path.join(output_dir, 'entity_embedding.vec')
    )

    print(f"Small dataset created at: {output_dir}")
    return output_dir


if __name__ == '__main__':
    # Test knowledge graph builder
    data_dir = r'D:\Desktop\News-Recommender\data'

    # Create small dataset first
    small_data_dir = create_small_dataset(
        data_dir,
        os.path.join(data_dir, 'mind_tiny'),
        num_news=500,
        num_behaviors=2000
    )

    # Build knowledge graph
    kg_builder = KnowledgeGraphBuilder(
        news_path=os.path.join(small_data_dir, 'news.tsv'),
        entity_embedding_path=os.path.join(small_data_dir, 'entity_embedding.vec'),
        max_news=500
    )

    graph_data = kg_builder.get_graph_data()
    print("\nGraph statistics:")
    print(f"  Total nodes: {graph_data['num_nodes']}")
    print(f"  News nodes: {graph_data['num_news']}")
    print(f"  Entity nodes: {graph_data['num_entities']}")
    print(f"  Edges: {graph_data['edge_index'].shape[1]}")
    print(f"  Node features shape: {graph_data['node_features'].shape}")
