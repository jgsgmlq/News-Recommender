"""
Training script for News Recommendation Model
"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import get_dataloaders
from model import NewsRecommender


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
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
        scores = model(user_idx, news_idx, history)

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


def validate(model, val_loader, criterion, device):
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
            scores = model(user_idx, news_idx, history)

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
    print("Loading Data")
    print("=" * 60)
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Get vocabulary sizes
    num_users = len(train_dataset.user2idx)
    num_news = len(train_dataset.news2idx)

    print(f"\nDataset statistics:")
    print(f"  Number of users: {num_users}")
    print(f"  Number of news: {num_news}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    model = NewsRecommender(
        num_users=num_users,
        num_news=num_news,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout
    )
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Tensorboard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs'))

    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Log to tensorboard
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
                'user2idx': train_dataset.user2idx,
                'news2idx': train_dataset.news2idx,
            }

            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")

    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train News Recommendation Model')

    # Data
    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\News-Recommender\data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=r'D:\Desktop\News-Recommender\output',
                        help='Path to output directory')

    # Model
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=128,
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run training
    main(args)
