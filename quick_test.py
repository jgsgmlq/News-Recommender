"""
Quick Test Script - One-Click LLM Enhancement Test
Uses pre-configured third-party OpenAI API
No need to input API key manually!
"""
import os
import sys
import subprocess
import time


# ===== Configuration (Already configured for you!) =====
API_KEY = "sk-f2BTSMHiHgfs2fj4JgyjszLS5HhfHznJnzx688ZVctR09TR0"
BASE_URL = "https://api.f2gpt.com/v1"
DATA_DIR = "data/mind_tiny"
OUTPUT_DIR = "output"


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def check_data():
    """Check if tiny dataset exists"""
    print_header("Step 0: Checking Data")

    news_file = os.path.join(DATA_DIR, "news.tsv")
    if not os.path.exists(news_file):
        print(f"❌ Data file not found: {news_file}")
        print("\nPlease ensure tiny dataset exists:")
        print("  python src/kg_utils.py")
        return False

    print(f"✅ Data found: {news_file}")
    return True


def step1_precompute_embeddings():
    """Step 1: Precompute LLM embeddings"""
    print_header("Step 1: Precompute LLM Embeddings")

    output_path = os.path.join(DATA_DIR, "llm_embeddings.npy")

    # Check if already exists
    if os.path.exists(output_path):
        print(f"✅ LLM embeddings already exist: {output_path}")
        response = input("Recompute? (y/N): ").strip().lower()
        if response != 'y':
            return True

    print("📊 Processing 500 news articles...")
    print("⏱️  Estimated time: 2-3 minutes")
    print("💰 Estimated cost: ~$0.001 (less than 1 cent)")
    print()

    cmd = [
        sys.executable,
        "src/precompute_llm_embeddings.py",
        "--news_path", os.path.join(DATA_DIR, "news.tsv"),
        "--output_path", output_path,
        # API key and base_url are now defaults in the script
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ LLM embeddings computed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        return False


def step2_train_model():
    """Step 2: Train LLM+GNN model"""
    print_header("Step 2: Train LLM+GNN Model")

    print("🤖 Training multi-modal model...")
    print("⏱️  Estimated time: 3-5 minutes (CPU) / 1-2 minutes (GPU)")
    print()

    cmd = [
        sys.executable,
        "src/train_llm.py",
        "--epochs", "3",
        "--batch_size", "64",
        "--use_llm",
        "--use_gnn",
        "--gnn_layers", "2",
        "--fusion_method", "attention",
        "--llm_embedding_path", os.path.join(DATA_DIR, "llm_embeddings.npy")
    ]

    print(f"Running: {' '.join(cmd[:5])} ...\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Model trained successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        return False


def step3_evaluate():
    """Step 3: Evaluate model"""
    print_header("Step 3: Evaluate Model")

    model_path = os.path.join(OUTPUT_DIR, "llm_gnn", "best_model.pth")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False

    print("📈 Generating predictions and computing metrics...")
    print()

    cmd = [
        sys.executable,
        "src/predict_llm.py",
        "--model_path", model_path,
        "--llm_embedding_path", os.path.join(DATA_DIR, "llm_embeddings.npy")
    ]

    print(f"Running: {' '.join(cmd[:3])} ...\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Evaluation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {e}")
        return False


def show_results():
    """Show results"""
    print_header("Results")

    metrics_file = os.path.join(OUTPUT_DIR, "llm_gnn", "eval", "metrics.json")

    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        print("📊 Performance Metrics:")
        print(f"  AUC:      {metrics.get('AUC', 0):.4f}")
        print(f"  MRR:      {metrics.get('MRR', 0):.4f}")
        print(f"  nDCG@5:   {metrics.get('nDCG@5', 0):.4f}")
        print(f"  nDCG@10:  {metrics.get('nDCG@10', 0):.4f}")
        print()
    else:
        print("❌ Metrics file not found")


def main():
    """Main function"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              LLM-Enhanced News Recommendation                 ║
║                  Quick Test (Pre-configured)                  ║
╚═══════════════════════════════════════════════════════════════╝

API Configuration:
  Endpoint: https://api.f2gpt.com/v1
  API Key:  sk-f2BTSMHi... (pre-configured)
  Dataset:  Tiny (500 news articles)

This will run:
  1. Precompute LLM embeddings (~2-3 min, <$0.001)
  2. Train LLM+GNN model (~3-5 min)
  3. Evaluate and show results
    """)

    # Step 0: Check data
    if not check_data():
        return 1

    input("\nPress Enter to start...")

    # Step 1: Precompute embeddings
    if not step1_precompute_embeddings():
        print("\n❌ Failed to compute embeddings")
        return 1

    # Step 2: Train model
    if not step2_train_model():
        print("\n❌ Failed to train model")
        return 1

    # Step 3: Evaluate
    if not step3_evaluate():
        print("\n❌ Failed to evaluate")
        return 1

    # Show results
    show_results()

    print_header("Next Steps")
    print("✅ Quick test completed!")
    print()
    print("View detailed logs:")
    print("  tensorboard --logdir output/")
    print()
    print("View all results:")
    print("  cat output/llm_gnn/eval/metrics.json")
    print()
    print("Compare with baseline:")
    print("  python compare_results.py")

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
