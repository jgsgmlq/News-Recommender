"""
Precompute LLM Embeddings for News Articles
Uses OpenAI API to generate high-quality text embeddings
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMEmbeddingGenerator:
    """Generate and cache LLM embeddings for news articles"""

    def __init__(
        self,
        api_key,
        base_url=None,
        model="text-embedding-3-small",
        batch_size=100,
        cache_dir="./cache/llm_embeddings"
    ):
        """
        Args:
            api_key: OpenAI API key
            base_url: Optional custom API base URL
            model: Embedding model name
            batch_size: Batch size for API calls
            cache_dir: Directory to cache embeddings
        """
        # Initialize OpenAI client with custom base_url if provided
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = model
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Model dimension mapping
        self.model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }

        self.embedding_dim = self.model_dims.get(model, 1536)

        print(f"Initialized LLM Embedding Generator")
        print(f"  Model: {model}")
        print(f"  Dimension: {self.embedding_dim}")
        print(f"  Batch size: {batch_size}")

    def prepare_news_text(self, row):
        """
        Prepare text from news article

        Args:
            row: DataFrame row with news data

        Returns:
            Formatted text string
        """
        title = row['title'] if pd.notna(row['title']) else ""
        abstract = row['abstract'] if pd.notna(row['abstract']) else ""
        category = row['category'] if pd.notna(row['category']) else ""
        subcategory = row['subcategory'] if pd.notna(row['subcategory']) else ""

        # Format: "Category: {cat} | {title}. {abstract}"
        text_parts = []

        if category:
            if subcategory:
                text_parts.append(f"Category: {category} - {subcategory}")
            else:
                text_parts.append(f"Category: {category}")

        if title:
            text_parts.append(f"Title: {title}")

        if abstract:
            text_parts.append(f"Abstract: {abstract}")

        text = " | ".join(text_parts)

        # Fallback for empty text
        if not text.strip():
            text = "No content available"

        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_embedding_api(self, texts):
        """
        Call OpenAI Embedding API with retry logic

        Args:
            texts: List of text strings

        Returns:
            List of embeddings
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            print(f"\nAPI Error: {e}")
            raise

    def generate_embeddings(self, news_path, verbose=True):
        """
        Generate embeddings for all news articles

        Args:
            news_path: Path to news.tsv file
            verbose: Show progress bar

        Returns:
            embeddings: numpy array (num_news, embedding_dim)
            news_id_to_idx: dict mapping news_id to index
        """
        # Load news data
        print(f"\nLoading news from {news_path}")
        news_df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract',
                   'url', 'title_entities', 'abstract_entities']
        )

        print(f"Loaded {len(news_df)} news articles")

        # Prepare texts
        print("\nPreparing news texts...")
        texts = []
        news_ids = []

        for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Preparing"):
            text = self.prepare_news_text(row)
            texts.append(text)
            news_ids.append(row['news_id'])

        # Calculate cost estimate
        avg_tokens = 100  # Rough estimate
        total_tokens = len(texts) * avg_tokens
        cost_usd = total_tokens / 1_000_000 * 0.02  # $0.02 per 1M tokens
        print(f"\nEstimated cost: ${cost_usd:.4f} (based on {total_tokens:,} tokens)")

        # Generate embeddings in batches
        print(f"\nGenerating embeddings (batch_size={self.batch_size})...")
        all_embeddings = []
        failed_indices = []

        pbar = tqdm(range(0, len(texts), self.batch_size), desc="API Calls")

        for i in pbar:
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Call API
                batch_embeddings = self.call_embedding_api(batch_texts)
                all_embeddings.extend(batch_embeddings)

                # Update progress
                pbar.set_postfix({
                    'completed': len(all_embeddings),
                    'total': len(texts)
                })

                # Rate limiting: sleep briefly between batches
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)

            except Exception as e:
                print(f"\nFailed to process batch {i}-{i+len(batch_texts)}: {e}")

                # Use zero vectors as fallback
                zero_embedding = [0.0] * self.embedding_dim
                all_embeddings.extend([zero_embedding] * len(batch_texts))
                failed_indices.extend(range(i, i + len(batch_texts)))

        # Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Create news_id to index mapping
        news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news_ids)}

        # Report
        print(f"\n{'='*60}")
        print(f"Embedding Generation Complete")
        print(f"{'='*60}")
        print(f"  Total news: {len(embeddings)}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Failed: {len(failed_indices)}")
        print(f"  Success rate: {100 * (1 - len(failed_indices) / len(texts)):.2f}%")

        return embeddings, news_id_to_idx, news_ids

    def save_embeddings(self, embeddings, news_id_to_idx, news_ids, output_path):
        """
        Save embeddings and metadata

        Args:
            embeddings: numpy array
            news_id_to_idx: news_id to index mapping
            news_ids: list of news IDs
            output_path: output file path
        """
        # Save embeddings
        np.save(output_path, embeddings)
        print(f"\nSaved embeddings to: {output_path}")

        # Save mapping
        mapping_path = output_path.replace('.npy', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'news_id_to_idx': news_id_to_idx,
                'news_ids': news_ids,
                'model': self.model,
                'embedding_dim': self.embedding_dim
            }, f)
        print(f"Saved mapping to: {mapping_path}")

        # Save metadata
        metadata_path = output_path.replace('.npy', '_metadata.txt')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {self.model}\n")
            f.write(f"Embedding dimension: {self.embedding_dim}\n")
            f.write(f"Number of news: {len(embeddings)}\n")
            f.write(f"Shape: {embeddings.shape}\n")
        print(f"Saved metadata to: {metadata_path}")


def main(args):
    # Initialize generator
    generator = LLMEmbeddingGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir
    )

    # Generate embeddings
    embeddings, news_id_to_idx, news_ids = generator.generate_embeddings(
        news_path=args.news_path,
        verbose=True
    )

    # Save
    generator.save_embeddings(
        embeddings=embeddings,
        news_id_to_idx=news_id_to_idx,
        news_ids=news_ids,
        output_path=args.output_path
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"\nYou can now use these embeddings in training:")
    print(f"  python src/train_llm.py \\")
    print(f"    --llm_embedding_path {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute LLM Embeddings for News Articles'
    )

    # API Configuration (with defaults for third-party platform)
    parser.add_argument('--api_key', type=str,
                        default='sk-f2BTSMHiHgfs2fj4JgyjszLS5HhfHznJnzx688ZVctR09TR0',
                        help='OpenAI API key')
    parser.add_argument('--base_url', type=str,
                        default='https://api.f2gpt.com/v1',
                        help='API base URL (for third-party platforms)')

    # Required arguments
    parser.add_argument('--news_path', type=str, required=True,
                        help='Path to news.tsv file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path for embeddings (.npy)')

    # Optional arguments
    parser.add_argument('--model', type=str,
                        default='text-embedding-3-small',
                        help='OpenAI embedding model')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for API calls')
    parser.add_argument('--cache_dir', type=str,
                        default='./cache/llm_embeddings',
                        help='Cache directory')

    args = parser.parse_args()

    print(f"Using API endpoint: {args.base_url}")
    print(f"API key: {args.api_key[:15]}...")

    main(args)
