"""
One-command pipeline for GNN-Enhanced News Recommendation
Runs training and prediction with knowledge graph enhancement
"""
import os
import sys
import argparse
import subprocess


def main(args):
    print("=" * 80)
    print("GNN-Enhanced News Recommendation Pipeline")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')

    # Step 1: Training (optional)
    if not args.skip_train:
        print("\n" + "=" * 80)
        print("STEP 1: Training GNN-Enhanced Model")
        print("=" * 80)

        train_script = os.path.join(src_dir, 'train_gnn.py')
        train_cmd = [
            sys.executable, train_script,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--gnn_layers', str(args.gnn_layers),
            '--embedding_dim', str(args.embedding_dim),
            '--gnn_hidden_dim', str(args.gnn_hidden_dim),
            '--gnn_output_dim', str(args.gnn_output_dim),
        ]

        if not args.use_gnn:
            train_cmd.append('--no_gnn')

        print(f"\nRunning: {' '.join(train_cmd)}\n")
        result = subprocess.run(train_cmd)

        if result.returncode != 0:
            print("\nError: Training failed!")
            return 1

    # Step 2: Prediction and Evaluation
    print("\n" + "=" * 80)
    print("STEP 2: Prediction and Evaluation")
    print("=" * 80)

    predict_script = os.path.join(src_dir, 'predict_gnn.py')
    output_suffix = 'gnn' if args.use_gnn else 'baseline'
    model_path = os.path.join(base_dir, 'output', output_suffix, 'best_model.pth')

    predict_cmd = [
        sys.executable, predict_script,
        '--model_path', model_path
    ]

    print(f"\nRunning: {' '.join(predict_cmd)}\n")
    result = subprocess.run(predict_cmd)

    if result.returncode != 0:
        print("\nError: Prediction failed!")
        return 1

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel saved at: {model_path}")
    print(f"Evaluation results at: {os.path.join(base_dir, 'output', output_suffix, 'eval')}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {os.path.join(base_dir, 'output', output_suffix, 'runs')}")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GNN-Enhanced News Recommendation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GNN model and evaluate (default)
  python run_gnn.py --epochs 5 --gnn_layers 2

  # Train baseline model without GNN
  python run_gnn.py --no_gnn --epochs 5

  # Only run prediction on existing model
  python run_gnn.py --skip_train

  # Train with 1-layer GNN
  python run_gnn.py --epochs 3 --gnn_layers 1
        """
    )

    # Pipeline control
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and only run prediction')

    # Model configuration
    parser.add_argument('--use_gnn', action='store_true', default=True,
                        help='Use GNN enhancement (default: True)')
    parser.add_argument('--no_gnn', action='store_false', dest='use_gnn',
                        help='Disable GNN enhancement (baseline)')
    parser.add_argument('--gnn_layers', type=int, default=2, choices=[1, 2],
                        help='Number of GNN layers (default: 2)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128,
                        help='GNN hidden dimension (default: 128)')
    parser.add_argument('--gnn_output_dim', type=int, default=128,
                        help='GNN output dimension (default: 128)')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')

    args = parser.parse_args()

    sys.exit(main(args))
