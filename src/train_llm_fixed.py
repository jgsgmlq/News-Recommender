"""
Fixed Training script for LLM-Enhanced News Recommendation Model
Key fixes:
1. Filter training data to only use news with LLM embeddings
2. Better fusion mechanism with proper masking
3. Improved optimization strategy
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_loader import MINDDataset, collate_fn
from model_llm import LLMEnhancedRecommender
from kg_utils import KnowledgeGraphBuilder

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Training will continue without logging.")
    SummaryWriter = None


def filter_dataset_by_news_ids(dataset, valid_news_ids):
    """
    Filter dataset to only include samples where all news IDs are valid

    Args:
        dataset: MINDDataset instance
        valid_news_ids: set of valid news IDs (with LLM embeddings)

    Returns:
        Subset of dataset with valid samples
    """
    valid_indices = []

    print(f"\nFiltering dataset to use only news with LLM embeddings...")
    print(f"Valid news IDs: 0-{max(valid_news_ids)}")

    for idx in tqdm(range(len(dataset)), desc="Filtering"):
        sample = dataset[idx]

        # Check if candidate news is valid
        news_id = sample['news_idx']
        if isinstance(news_id, torch.Tensor):
            news_id = news_id.item()
        if news_id not in valid_news_ids:
            continue

        # We only need the candidate news to have LLM embedding
        # History news can be without LLM embeddings (will use zero vectors)
        valid_indices.append(idx)

    print(f"Filtered dataset: {len(valid_indices)} / {len(dataset)} samples")
    print(f"Retention rate: {100.0 * len(valid_indices) / len(dataset):.2f}%")

    return Subset(dataset, valid_indices)


def load_llm_embeddings(llm_embedding_path, device):
    """Load pre-computed LLM embeddings"""
    if llm_embedding_path and os.path.exists(llm_embedding_path):
        print(f"\nLoading LLM embeddings from {llm_embedding_path}")
        llm_embeddings = np.load(llm_embedding_path)
        llm_embeddings = torch.from_numpy(llm_embeddings).float().to(device)
        print(f"LLM embeddings loaded: {llm_embeddings.shape}")
        return llm_embeddings
    else:
        print("\nNo LLM embeddings provided")
        return None


def get_filtered_dataloaders(data_dir, llm_embeddings, batch_size=64, num_workers=0):
    """Create filtered dataloaders for small dataset"""
    train_behaviors = os.path.join(data_dir, 'mind_small', 'train', 'behaviors.tsv')
    train_news = os.path.join(data_dir, 'mind_small', 'train', 'news.tsv')

    print("=" * 60)
    print("Creating Filtered Training Dataset (Small)")
    print("=" * 60)

    # Create full dataset first
    full_dataset = MINDDataset(train_behaviors, train_news, mode='train')

    # Get valid news IDs (those with LLM embeddings)
    num_llm_news = llm_embeddings.size(0) if llm_embeddings is not None else 0
    valid_news_ids = set(range(num_llm_news))

    print(f"\nDataset statistics BEFORE filtering:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  News with LLM embeddings: {num_llm_news}")

    # Filter dataset
    filtered_dataset = filter_dataset_by_news_ids(full_dataset, valid_news_ids)

    # Use 80-20 split for train/val
    train_size = int(0.8 * len(filtered_dataset))
    val_size = len(filtered_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        filtered_dataset, [train_size, val_size]
    )

    print(f"\nDataset statistics AFTER filtering:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch,
                llm_embeddings=None, gnn_embeddings=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        user_idx = batch['user_idx'].to(device)
        news_idx = batch['news_idx'].to(device)
        history = batch['history'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        scores = model(user_idx, news_idx, history, llm_embeddings, gnn_embeddings)

        # Compute loss
        loss = criterion(scores, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predicted = (scores > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, llm_embeddings=None, gnn_embeddings=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move to device
            user_idx = batch['user_idx'].to(device)
            news_idx = batch['news_idx'].to(device)
            history = batch['history'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            scores = model(user_idx, news_idx, history, llm_embeddings, gnn_embeddings)

            # Compute loss
            loss = criterion(scores, labels)

            # Statistics
            total_loss += loss.item()
            predicted = (scores > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Fixed LLM-Enhanced News Recommender')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='output/llm_gnn_fixed', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM embeddings')
    parser.add_argument('--use_gnn', action='store_true', help='Use GNN embeddings')
    parser.add_argument('--llm_embedding_path', type=str, default=None, help='Path to LLM embeddings')
    parser.add_argument('--fusion_method', type=str, default='concat',
                        choices=['attention', 'concat', 'gate'], help='Fusion method')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}\n")

    # Load LLM embeddings
    llm_embeddings = load_llm_embeddings(args.llm_embedding_path, device) if args.use_llm else None

    # Create filtered dataloaders
    print("=" * 60)
    print("Loading Filtered Tiny Dataset")
    print("=" * 60)
    train_loader, val_loader, train_dataset, val_dataset = get_filtered_dataloaders(
        args.data_dir,
        llm_embeddings,
        batch_size=args.batch_size
    )

    # Build knowledge graph
    graph_data = None
    if args.use_gnn:
        print("\n" + "=" * 60)
        print("Building Knowledge Graph")
        print("=" * 60)
        kg_builder = KnowledgeGraphBuilder(
            news_path=os.path.join(args.data_dir, 'mind_small', 'train', 'news.tsv'),
            entity_embedding_path=os.path.join(args.data_dir, 'mind_small', 'train', 'entity_embedding.vec'),
            max_news=None  # Load all news to match vocabulary
        )

        graph_data = kg_builder.get_graph_data()
        print(f"\nKnowledge Graph statistics:")
        print(f"  Total nodes: {graph_data['num_nodes']}")
        print(f"  News nodes: {graph_data['num_news']}")
        print(f"  Entity nodes: {graph_data['num_entities']}")
        print(f"  Edges: {graph_data['edge_index'].shape[1]}")

    # Get dataset info
    news_dataset = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
    if hasattr(news_dataset, 'dataset'):
        news_dataset = news_dataset.dataset
    num_users = len(news_dataset.user2idx)
    num_news = len(news_dataset.news2idx)

    # Create model
    print("\n" + "=" * 60)
    print("Creating LLM-Enhanced Model")
    print("=" * 60)
    model = LLMEnhancedRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=128,  # ID embedding dimension
        llm_emb_dim=1536,   # OpenAI text-embedding-3-small dimension
        gnn_input_dim=100,  # Entity embedding dimension
        gnn_hidden_dim=128,
        gnn_output_dim=128,
        output_dim=256,
        gnn_layers=2,
        use_llm=args.use_llm,
        use_gnn=args.use_gnn,
        fusion_method=args.fusion_method,
        dropout=0.3  # Increased dropout
    ).to(device)

    # Set graph data if using GNN
    if args.use_gnn and graph_data is not None:
        model.set_graph_data(graph_data)
        print("Knowledge graph loaded into model")

    print(f"\nModel configuration:")
    print(f"  Use LLM: {args.use_llm}")
    print(f"  Use GNN: {args.use_gnn}")
    print(f"  Fusion method: {args.fusion_method}")
    print(f"  Dropout: 0.3")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'eval'), exist_ok=True)

    # TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Pre-compute GNN embeddings (once per epoch)
        gnn_embeddings = None
        if args.use_gnn and graph_data is not None:
            print("Pre-computing GNN embeddings...")
            model.eval()
            with torch.no_grad():
                gnn_embeddings = model.get_gnn_embeddings()
            print(f"GNN embeddings computed: {gnn_embeddings.shape}")
            model.train()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            llm_embeddings, gnn_embeddings
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            llm_embeddings, gnn_embeddings
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    if writer:
        writer.close()

    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
