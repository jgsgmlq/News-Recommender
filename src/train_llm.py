"""
Training script for LLM-Enhanced News Recommendation Model
Integrates: ID + LLM Text Embeddings + GNN Entity Embeddings
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def get_tiny_dataloaders(data_dir, batch_size=64, num_workers=0):
    """Create dataloaders for tiny dataset"""
    train_behaviors = os.path.join(data_dir, 'mind_tiny', 'behaviors.tsv')
    train_news = os.path.join(data_dir, 'mind_tiny', 'news.tsv')

    print("=" * 60)
    print("Creating Training Dataset (Tiny)")
    print("=" * 60)
    train_dataset = MINDDataset(train_behaviors, train_news, mode='train')

    # Use 80-20 split for train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

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


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataloaders
    print("\n" + "=" * 60)
    print("Loading Tiny Dataset")
    print("=" * 60)
    train_loader, val_loader, train_split, val_split = get_tiny_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Get the original dataset from the split
    original_dataset = train_split.dataset

    # Get vocabulary sizes
    num_users = len(original_dataset.user2idx)
    num_news = len(original_dataset.news2idx)

    print(f"\nDataset statistics:")
    print(f"  Number of users: {num_users}")
    print(f"  Number of news: {num_news}")
    print(f"  Training samples: {len(train_split)}")
    print(f"  Validation samples: {len(val_split)}")

    # Load LLM embeddings
    llm_embeddings = load_llm_embeddings(args.llm_embedding_path, device)

    # Build knowledge graph (if using GNN)
    graph_data = None
    if args.use_gnn:
        print("\n" + "=" * 60)
        print("Building Knowledge Graph")
        print("=" * 60)
        kg_builder = KnowledgeGraphBuilder(
            news_path=os.path.join(args.data_dir, 'mind_tiny', 'news.tsv'),
            entity_embedding_path=os.path.join(args.data_dir, 'mind_tiny', 'entity_embedding.vec'),
            max_news=None
        )

        graph_data = kg_builder.get_graph_data()
        print(f"\nKnowledge Graph statistics:")
        print(f"  Total nodes: {graph_data['num_nodes']}")
        print(f"  News nodes: {graph_data['num_news']}")
        print(f"  Entity nodes: {graph_data['num_entities']}")
        print(f"  Edges: {graph_data['edge_index'].shape[1]}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating LLM-Enhanced Model")
    print("=" * 60)

    # Determine LLM embedding dimension
    llm_emb_dim = llm_embeddings.size(1) if llm_embeddings is not None else 1536

    model = LLMEnhancedRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=args.embedding_dim,
        llm_emb_dim=llm_emb_dim,
        gnn_input_dim=100,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_output_dim=args.gnn_output_dim,
        output_dim=args.output_dim,
        gnn_layers=args.gnn_layers,
        use_llm=args.use_llm and llm_embeddings is not None,
        use_gnn=args.use_gnn,
        fusion_method=args.fusion_method,
        dropout=args.dropout
    )
    model = model.to(device)

    # Set graph data
    if args.use_gnn and graph_data is not None:
        model.set_graph_data(graph_data)
        print("Knowledge graph loaded into model")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel configuration:")
    print(f"  Use LLM: {args.use_llm and llm_embeddings is not None}")
    print(f"  Use GNN: {args.use_gnn}")
    print(f"  Fusion method: {args.fusion_method}")
    print(f"  Output dim: {args.output_dim}")
    print(f"  Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Tensorboard writer (optional)
    model_suffix = []
    if args.use_llm and llm_embeddings is not None:
        model_suffix.append('llm')
    if args.use_gnn:
        model_suffix.append('gnn')
    model_name = '_'.join(model_suffix) if model_suffix else 'baseline'

    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(output_dir, 'runs'))
    else:
        writer = None

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Pre-compute GNN embeddings (once per epoch)
        gnn_embeddings = None
        if args.use_gnn:
            print("Pre-computing GNN embeddings...")
            model.eval()
            with torch.no_grad():
                gnn_embeddings = model.get_gnn_embeddings()
            if gnn_embeddings is not None:
                print(f"GNN embeddings computed: {gnn_embeddings.shape}")
            model.train()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            llm_embeddings, gnn_embeddings
        )
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device,
                                      llm_embeddings, gnn_embeddings)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Log to tensorboard (if available)
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_users': num_users,
                'num_news': num_news,
                'user2idx': original_dataset.user2idx,
                'news2idx': original_dataset.news2idx,
                'args': vars(args),
                'graph_data': graph_data if args.use_gnn else None,
            }

            save_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")

    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train LLM-Enhanced News Recommendation Model'
    )

    # Data
    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\News-Recommender\data',
                        help='Path to data directory')
    parser.add_argument('--llm_embedding_path', type=str, default=None,
                        help='Path to pre-computed LLM embeddings (.npy)')
    parser.add_argument('--output_dir', type=str, default=r'D:\Desktop\News-Recommender\output',
                        help='Path to output directory')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='ID embedding dimension')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Final output dimension')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN hidden dimension')
    parser.add_argument('--gnn_output_dim', type=int, default=128,
                        help='GNN output dimension')
    parser.add_argument('--gnn_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Features
    parser.add_argument('--use_llm', action='store_true', default=True,
                        help='Use LLM embeddings')
    parser.add_argument('--no_llm', action='store_false', dest='use_llm',
                        help='Disable LLM embeddings')
    parser.add_argument('--use_gnn', action='store_true', default=True,
                        help='Use GNN embeddings')
    parser.add_argument('--no_gnn', action='store_false', dest='use_gnn',
                        help='Disable GNN embeddings')
    parser.add_argument('--fusion_method', type=str, default='attention',
                        choices=['attention', 'gate', 'concat'],
                        help='Fusion method for multi-modal')

    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=2,
                        help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate decay gamma')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Validate LLM embedding path
    if args.use_llm and not args.llm_embedding_path:
        print("Warning: --use_llm is enabled but --llm_embedding_path is not provided")
        print("LLM embeddings will not be used")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    main(args)
