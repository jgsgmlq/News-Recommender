"""
Main script to run the complete news recommendation pipeline
"""
import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print output"""
    print("\n" + "=" * 70)
    print(f"{description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        sys.exit(1)

    return result.returncode == 0


def main(args):
    src_dir = os.path.join(os.path.dirname(__file__), 'src')

    # Step 1: Train model
    if not args.skip_train:
        train_cmd = [
            sys.executable,
            os.path.join(src_dir, 'train.py'),
            '--data_dir', args.data_dir,
            '--output_dir', args.output_dir,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--embedding_dim', str(args.embedding_dim),
        ]
        run_command(train_cmd, "Step 1: Training Model")
    else:
        print("\nSkipping training (using existing model)")

    # Step 2: Predict and evaluate
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    eval_dir = os.path.join(args.output_dir, 'eval')

    predict_cmd = [
        sys.executable,
        os.path.join(src_dir, 'predict.py'),
        '--data_dir', args.data_dir,
        '--model_path', model_path,
        '--output_dir', eval_dir,
        '--run_eval'
    ]
    run_command(predict_cmd, "Step 2: Prediction and Evaluation")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel saved at: {model_path}")
    print(f"Evaluation results saved at: {eval_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run complete news recommendation pipeline')

    parser.add_argument('--data_dir', type=str, default=r'D:\Desktop\News-Recommender\data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=r'D:\Desktop\News-Recommender\output',
                        help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training and use existing model')

    args = parser.parse_args()

    main(args)
