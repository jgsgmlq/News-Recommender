"""
支持断点续传的LLM嵌入生成脚本
Resumable LLM Embedding Generation with Checkpointing
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class ResumableLLMEmbeddingGenerator:
    """支持断点续传的LLM嵌入生成器"""

    def __init__(
        self,
        api_key,
        base_url=None,
        model="text-embedding-3-small",
        batch_size=100,
        chunk_size=5000,  # 每5000条新闻保存一次
        cache_dir="./cache/llm_embeddings"
    ):
        """
        Args:
            api_key: OpenAI API key
            base_url: Optional custom API base URL
            model: Embedding model name
            batch_size: Batch size for API calls (每次调用API处理的新闻数)
            chunk_size: Chunk size for saving checkpoints (每处理多少条新闻保存一次)
            cache_dir: Directory to cache embeddings
        """
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

        self.model = model
        self.batch_size = batch_size
        self.chunk_size = chunk_size
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

        print(f"初始化可恢复LLM嵌入生成器")
        print(f"  模型: {model}")
        print(f"  嵌入维度: {self.embedding_dim}")
        print(f"  API批次大小: {batch_size}")
        print(f"  检查点间隔: {chunk_size}条新闻")

    def prepare_news_text(self, row):
        """准备新闻文本"""
        title = row['title'] if pd.notna(row['title']) else ""
        abstract = row['abstract'] if pd.notna(row['abstract']) else ""
        category = row['category'] if pd.notna(row['category']) else ""
        subcategory = row['subcategory'] if pd.notna(row['subcategory']) else ""

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

        if not text.strip():
            text = "No content available"

        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_embedding_api(self, texts):
        """调用OpenAI嵌入API (带重试机制)"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            print(f"\nAPI错误: {e}")
            raise

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            print(f"\n[CHECKPOINT] Found checkpoint file: {checkpoint_path}")
            print(f"  Processed: {checkpoint['processed_count']}/{checkpoint['total_count']}")
            print(f"  Progress: {checkpoint['processed_count']/checkpoint['total_count']*100:.2f}%")
            return checkpoint
        else:
            return None

    def save_checkpoint(self, checkpoint_path, checkpoint_data):
        """保存检查点"""
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)

    def load_partial_embeddings(self, output_path):
        """加载已有的部分嵌入"""
        if os.path.exists(output_path):
            embeddings = np.load(output_path)
            print(f"  加载已有嵌入: {embeddings.shape}")
            return embeddings
        else:
            return None

    def generate_embeddings_resumable(self, news_path, output_path, verbose=True):
        """
        支持断点续传的嵌入生成

        Args:
            news_path: Path to news.tsv file
            output_path: Output path for embeddings
            verbose: Show progress bar

        Returns:
            embeddings: numpy array
            news_id_to_idx: dict mapping
            news_ids: list of news IDs
        """
        # 检查点文件路径
        checkpoint_path = output_path.replace('.npy', '_checkpoint.json')

        # 加载新闻数据
        print(f"\n加载新闻数据: {news_path}")
        news_df = pd.read_csv(
            news_path,
            sep='\t',
            header=None,
            names=['news_id', 'category', 'subcategory', 'title', 'abstract',
                   'url', 'title_entities', 'abstract_entities']
        )
        print(f"  总新闻数: {len(news_df)}")

        # 准备所有文本
        print("\n准备新闻文本...")
        texts = []
        news_ids = []

        for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="准备文本"):
            text = self.prepare_news_text(row)
            texts.append(text)
            news_ids.append(row['news_id'])

        total_count = len(texts)

        # 尝试加载检查点
        checkpoint = self.load_checkpoint(checkpoint_path)

        if checkpoint:
            # 从检查点恢复
            processed_indices = set(checkpoint['processed_indices'])
            all_embeddings = self.load_partial_embeddings(output_path)

            if all_embeddings is None:
                print("  [WARNING] Checkpoint exists but embeddings file is missing, starting from scratch")
                all_embeddings = []
                processed_indices = set()
            else:
                all_embeddings = all_embeddings.tolist()
                print(f"  [RESUME] Progress restored: {len(processed_indices)}/{total_count}")
        else:
            # 从头开始
            print("\n开始新的嵌入生成任务")
            all_embeddings = []
            processed_indices = set()

        # 计算剩余工作量
        remaining_indices = [i for i in range(total_count) if i not in processed_indices]
        print(f"\n剩余待处理: {len(remaining_indices)}/{total_count}")

        if len(remaining_indices) == 0:
            print("[DONE] All news processed!")
            embeddings = np.array(all_embeddings, dtype=np.float32)
            news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news_ids)}
            return embeddings, news_id_to_idx, news_ids

        # 估算成本
        avg_tokens = 100
        remaining_tokens = len(remaining_indices) * avg_tokens
        cost_usd = remaining_tokens / 1_000_000 * 0.02
        print(f"预计剩余费用: ${cost_usd:.4f}")

        # 分批处理
        print(f"\n开始生成嵌入 (batch_size={self.batch_size}, chunk_size={self.chunk_size})...")

        failed_indices = []
        chunk_start_time = time.time()

        # 创建进度条
        pbar = tqdm(
            range(0, len(remaining_indices), self.batch_size),
            desc="API调用",
            initial=0
        )

        for batch_idx, i in enumerate(pbar):
            # 获取当前批次的索引
            batch_indices = remaining_indices[i:i + self.batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]

            try:
                # 调用API
                batch_embeddings = self.call_embedding_api(batch_texts)

                # 将嵌入放到正确的位置
                for idx, embedding in zip(batch_indices, batch_embeddings):
                    if idx < len(all_embeddings):
                        all_embeddings[idx] = embedding
                    else:
                        # 如果索引超出范围，需要填充
                        while len(all_embeddings) < idx:
                            all_embeddings.append([0.0] * self.embedding_dim)
                        all_embeddings.append(embedding)

                    processed_indices.add(idx)

                # 更新进度
                pbar.set_postfix({
                    '已完成': len(processed_indices),
                    '总数': total_count,
                    '完成率': f"{len(processed_indices)/total_count*100:.1f}%"
                })

                # 每处理一个chunk或每隔一定时间保存一次
                if (len(processed_indices) % self.chunk_size == 0) or \
                   (time.time() - chunk_start_time > 300):  # 每5分钟保存一次

                    # 保存当前进度
                    self._save_progress(
                        all_embeddings,
                        processed_indices,
                        total_count,
                        output_path,
                        checkpoint_path
                    )

                    chunk_start_time = time.time()
                    print(f"\n  [CHECKPOINT] Saved: {len(processed_indices)}/{total_count}")

                # Rate limiting
                if i + self.batch_size < len(remaining_indices):
                    time.sleep(0.1)

            except Exception as e:
                print(f"\n[ERROR] Batch failed {i}-{i+len(batch_texts)}: {e}")

                # 使用零向量作为后备
                zero_embedding = [0.0] * self.embedding_dim
                for idx in batch_indices:
                    if idx < len(all_embeddings):
                        all_embeddings[idx] = zero_embedding
                    else:
                        while len(all_embeddings) < idx:
                            all_embeddings.append(zero_embedding)
                        all_embeddings.append(zero_embedding)

                    processed_indices.add(idx)
                    failed_indices.append(idx)

        # 最终保存
        print("\n保存最终结果...")
        self._save_progress(
            all_embeddings,
            processed_indices,
            total_count,
            output_path,
            checkpoint_path,
            is_final=True
        )

        # 转换为numpy数组
        embeddings = np.array(all_embeddings, dtype=np.float32)
        news_id_to_idx = {news_id: idx for idx, news_id in enumerate(news_ids)}

        # 报告
        print(f"\n{'='*60}")
        print(f"嵌入生成完成")
        print(f"{'='*60}")
        print(f"  总新闻数: {len(embeddings)}")
        print(f"  嵌入形状: {embeddings.shape}")
        print(f"  失败数: {len(failed_indices)}")
        print(f"  成功率: {100 * (1 - len(failed_indices) / total_count):.2f}%")

        return embeddings, news_id_to_idx, news_ids

    def _save_progress(self, all_embeddings, processed_indices, total_count,
                      output_path, checkpoint_path, is_final=False):
        """保存进度"""
        # 保存嵌入
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        np.save(output_path, embeddings_array)

        if not is_final:
            # 保存检查点
            checkpoint_data = {
                'processed_count': len(processed_indices),
                'total_count': total_count,
                'processed_indices': list(processed_indices),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            self.save_checkpoint(checkpoint_path, checkpoint_data)

    def save_embeddings(self, embeddings, news_id_to_idx, news_ids, output_path):
        """保存嵌入和元数据"""
        # 保存嵌入
        np.save(output_path, embeddings)
        print(f"\n[SAVED] Embeddings: {output_path}")

        # 保存映射
        mapping_path = output_path.replace('.npy', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'news_id_to_idx': news_id_to_idx,
                'news_ids': news_ids,
                'model': self.model,
                'embedding_dim': self.embedding_dim
            }, f)
        print(f"[SAVED] Mapping: {mapping_path}")

        # 保存元数据
        metadata_path = output_path.replace('.npy', '_metadata.txt')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {self.model}\n")
            f.write(f"Embedding dimension: {self.embedding_dim}\n")
            f.write(f"Number of news: {len(embeddings)}\n")
            f.write(f"Shape: {embeddings.shape}\n")
        print(f"[SAVED] Metadata: {metadata_path}")

        # 删除检查点文件 (已完成)
        checkpoint_path = output_path.replace('.npy', '_checkpoint.json')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"[CLEANED] Removed checkpoint file")


def main(args):
    # 初始化生成器
    generator = ResumableLLMEmbeddingGenerator(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        cache_dir=args.cache_dir
    )

    # 生成嵌入 (支持断点续传)
    embeddings, news_id_to_idx, news_ids = generator.generate_embeddings_resumable(
        news_path=args.news_path,
        output_path=args.output_path,
        verbose=True
    )

    # 保存
    generator.save_embeddings(
        embeddings=embeddings,
        news_id_to_idx=news_id_to_idx,
        news_ids=news_ids,
        output_path=args.output_path
    )

    print(f"\n{'='*60}")
    print("全部完成!")
    print(f"{'='*60}")
    print(f"\n现在可以使用这些嵌入进行训练:")
    print(f"  python src/train_llm_fixed.py \\")
    print(f"    --llm_embedding_path {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='支持断点续传的LLM嵌入生成'
    )

    # API配置
    parser.add_argument('--api_key', type=str,
                        default='sk-f2BTSMHiHgfs2fj4JgyjszLS5HhfHznJnzx688ZVctR09TR0',
                        help='OpenAI API key')
    parser.add_argument('--base_url', type=str,
                        default='https://api.f2gpt.com/v1',
                        help='API base URL')

    # 必需参数
    parser.add_argument('--news_path', type=str, required=True,
                        help='新闻文件路径 (news.tsv)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出路径 (.npy)')

    # 可选参数
    parser.add_argument('--model', type=str,
                        default='text-embedding-3-small',
                        help='OpenAI嵌入模型')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='API批次大小')
    parser.add_argument('--chunk_size', type=int, default=5000,
                        help='检查点间隔 (每处理多少条新闻保存一次)')
    parser.add_argument('--cache_dir', type=str,
                        default='./cache/llm_embeddings',
                        help='缓存目录')

    args = parser.parse_args()

    print(f"使用API端点: {args.base_url}")
    print(f"API密钥: {args.api_key[:15]}...")

    main(args)
