"""
Interactive Test Script for LLM-Enhanced Recommendation
Runs complete pipeline on tiny dataset with user's OpenAI API key
"""
import os
import sys
import subprocess
import time


def print_section(title):
    """Print formatted section title"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_dependencies():
    """Check if required packages are installed"""
    print_section("Checking Dependencies")

    required = ['openai', 'tenacity', 'torch', 'torch_geometric', 'pandas', 'numpy', 'tqdm']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} (missing)")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Please install with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    print("\n✓ All dependencies installed!")
    return True


def get_api_key():
    """Get OpenAI API key from user"""
    print_section("OpenAI API Key")

    print("Please enter your OpenAI API key.")
    print("You can get one from: https://platform.openai.com/api-keys")
    print("\nFormat: sk-...")
    print()

    api_key = input("API Key: ").strip()

    if not api_key.startswith('sk-'):
        print("\n⚠️  Invalid API key format. Should start with 'sk-'")
        return None

    return api_key


def check_data_files():
    """Check if tiny dataset exists"""
    print_section("Checking Data Files")

    data_dir = 'data/mind_tiny'
    required_files = ['news.tsv', 'behaviors.tsv', 'entity_embedding.vec']

    all_exist = True
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ {file} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {file} (missing)")
            all_exist = False

    if not all_exist:
        print("\n⚠️  Some data files are missing!")
        print("Please ensure tiny dataset is generated.")
        return False

    print("\n✓ All data files found!")
    return True


def step_precompute_embeddings(api_key):
    """Step 1: Precompute LLM embeddings"""
    print_section("Step 1: Precompute LLM Embeddings")

    output_path = 'data/mind_tiny/llm_embeddings.npy'

    # Check if already exists
    if os.path.exists(output_path):
        print(f"✓ LLM embeddings already exist: {output_path}")
        response = input("\nRecompute? (y/N): ").strip().lower()
        if response != 'y':
            return True

    print("This will:")
    print("  - Process 500 news articles")
    print("  - Call OpenAI API ~5 times")
    print("  - Cost: ~$0.001 (less than 1 cent)")
    print("  - Take: ~2-3 minutes")
    print()

    response = input("Proceed? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipped.")
        return False

    print("\nStarting API calls...")
    cmd = [
        sys.executable,
        'src/precompute_llm_embeddings.py',
        '--api_key', api_key,
        '--news_path', 'data/mind_tiny/news.tsv',
        '--output_path', output_path,
        '--model', 'text-embedding-3-small',
        '--batch_size', '100'
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ LLM embeddings computed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error computing embeddings: {e}")
        return False


def step_train_model(use_llm, use_gnn):
    """Step 2: Train model"""
    model_name = []
    if use_llm:
        model_name.append("LLM")
    if use_gnn:
        model_name.append("GNN")
    model_str = "+".join(model_name) if model_name else "Baseline"

    print_section(f"Step 2: Train Model ({model_str})")

    print("This will:")
    print(f"  - Train {model_str} model")
    print("  - 3 epochs on tiny dataset")
    print("  - Take: ~3-5 minutes (CPU) / ~1-2 minutes (GPU)")
    print()

    response = input("Proceed? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipped.")
        return False

    cmd = [
        sys.executable,
        'src/train_llm.py',
        '--epochs', '3',
        '--batch_size', '64',
        '--fusion_method', 'attention'
    ]

    if use_llm:
        cmd.extend(['--use_llm', '--llm_embedding_path', 'data/mind_tiny/llm_embeddings.npy'])
    else:
        cmd.append('--no_llm')

    if use_gnn:
        cmd.extend(['--use_gnn', '--gnn_layers', '2'])
    else:
        cmd.append('--no_gnn')

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {model_str} model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error training model: {e}")
        return False


def step_predict_evaluate(model_name):
    """Step 3: Predict and evaluate"""
    print_section("Step 3: Predict and Evaluate")

    model_path = f'output/{model_name}/best_model.pth'

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return False

    print(f"Evaluating model: {model_name}")
    print()

    response = input("Proceed? (Y/n): ").strip().lower()
    if response == 'n':
        print("Skipped.")
        return False

    cmd = [
        sys.executable,
        'src/predict_llm.py',
        '--model_path', model_path,
        '--llm_embedding_path', 'data/mind_tiny/llm_embeddings.npy'
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Evaluation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during evaluation: {e}")
        return False


def show_results():
    """Show final results"""
    print_section("Results Summary")

    import json

    experiments = {
        'baseline': 'output/baseline/eval/metrics.json',
        'gnn': 'output/gnn/eval/metrics.json',
        'llm': 'output/llm/eval/metrics.json',
        'llm_gnn': 'output/llm_gnn/eval/metrics.json'
    }

    results = {}
    for name, path in experiments.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[name] = json.load(f)

    if not results:
        print("No results found yet.")
        return

    # Print table
    print(f"{'Model':<15} {'AUC':<8} {'MRR':<8} {'nDCG@5':<8} {'nDCG@10':<8}")
    print("-" * 55)

    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics.get('AUC', 0):<8.4f} "
              f"{metrics.get('MRR', 0):<8.4f} "
              f"{metrics.get('nDCG@5', 0):<8.4f} "
              f"{metrics.get('nDCG@10', 0):<8.4f}")

    print()


def main():
    """Main function"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         LLM-Enhanced News Recommendation - Quick Test         ║
║                     Tiny Dataset (500 news)                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    # Step 0: Check dependencies
    if not check_dependencies():
        return 1

    # Check data files
    if not check_data_files():
        return 1

    # Get API key
    api_key = get_api_key()
    if not api_key:
        return 1

    # Choose experiments to run
    print_section("Select Experiments")

    print("Which experiments do you want to run?")
    print()
    print("1. Baseline (ID only)")
    print("2. +GNN")
    print("3. +LLM")
    print("4. +LLM+GNN (recommended)")
    print("5. All of the above")
    print()

    choice = input("Enter choice (1-5): ").strip()

    experiments = {
        '1': [(False, False, 'baseline')],
        '2': [(False, True, 'gnn')],
        '3': [(True, False, 'llm')],
        '4': [(True, True, 'llm_gnn')],
        '5': [
            (False, False, 'baseline'),
            (False, True, 'gnn'),
            (True, False, 'llm'),
            (True, True, 'llm_gnn')
        ]
    }

    if choice not in experiments:
        print("Invalid choice.")
        return 1

    exp_list = experiments[choice]

    # Step 1: Precompute embeddings if needed
    need_llm = any(use_llm for use_llm, _, _ in exp_list)
    if need_llm:
        if not step_precompute_embeddings(api_key):
            print("\n✗ Failed to compute LLM embeddings")
            return 1

    # Step 2 & 3: Train and evaluate each experiment
    for use_llm, use_gnn, model_name in exp_list:
        if not step_train_model(use_llm, use_gnn):
            print(f"\n✗ Failed to train {model_name}")
            continue

        if not step_predict_evaluate(model_name):
            print(f"\n✗ Failed to evaluate {model_name}")
            continue

    # Show results
    show_results()

    # Next steps
    print_section("Next Steps")
    print("✓ Experiments completed!")
    print()
    print("View detailed logs:")
    print("  tensorboard --logdir output/")
    print()
    print("View metrics:")
    print("  cat output/llm_gnn/eval/metrics.json")
    print()
    print("For complete dataset, see: QUICKSTART_LLM.md")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
