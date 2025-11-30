"""
诊断数据加载器 - 检查标签和输入数据格式
"""
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from data_loader import MINDDataset, collate_fn

def diagnose_dataset():
    """诊断数据集"""
    print("=" * 60)
    print("数据加载器诊断")
    print("=" * 60)

    # 加载数据集
    train_behaviors = 'data/mind_tiny/behaviors.tsv'
    train_news = 'data/mind_tiny/news.tsv'

    dataset = MINDDataset(train_behaviors, train_news, mode='train')

    # 检查几个样本
    print("\n" + "=" * 60)
    print("检查原始样本（前5个）")
    print("=" * 60)
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i}:")
        print(f"  user_idx: {sample['user_idx']} (type: {type(sample['user_idx'])})")
        print(f"  news_idx: {sample['news_idx']} (type: {type(sample['news_idx'])})")
        print(f"  history length: {len(sample['history'])}")
        print(f"  label: {sample['label']} (type: {type(sample['label'])})")

    # 检查标签分布
    print("\n" + "=" * 60)
    print("检查标签分布")
    print("=" * 60)
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    unique_labels = set(labels)
    print(f"唯一标签值: {unique_labels}")
    print(f"标签0的数量: {labels.count(0)}")
    print(f"标签1的数量: {labels.count(1)}")
    print(f"正样本比例: {100.0 * labels.count(1) / len(labels):.2f}%")

    # 检查DataLoader
    print("\n" + "=" * 60)
    print("检查DataLoader批次")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:  # 只检查第一个批次
            print(f"\n批次 {batch_idx}:")
            print(f"  user_idx shape: {batch['user_idx'].shape}, dtype: {batch['user_idx'].dtype}")
            print(f"  news_idx shape: {batch['news_idx'].shape}, dtype: {batch['news_idx'].dtype}")
            print(f"  history shape: {batch['history'].shape}, dtype: {batch['history'].dtype}")
            print(f"  label shape: {batch['label'].shape}, dtype: {batch['label'].dtype}")

            print(f"\n  user_idx 范围: [{batch['user_idx'].min().item()}, {batch['user_idx'].max().item()}]")
            print(f"  news_idx 范围: [{batch['news_idx'].min().item()}, {batch['news_idx'].max().item()}]")
            print(f"  label 值: {batch['label'].unique()}")
            print(f"  label 统计:")
            print(f"    最小值: {batch['label'].min().item()}")
            print(f"    最大值: {batch['label'].max().item()}")
            print(f"    平均值: {batch['label'].mean().item():.4f}")

            # 检查是否有异常值
            print(f"\n  异常值检查:")
            print(f"    user_idx 是否有NaN: {torch.isnan(batch['user_idx'].float()).any().item()}")
            print(f"    news_idx 是否有NaN: {torch.isnan(batch['news_idx'].float()).any().item()}")
            print(f"    label 是否有NaN: {torch.isnan(batch['label']).any().item()}")
            print(f"    label 是否有Inf: {torch.isinf(batch['label']).any().item()}")

            break

    # 加载LLM嵌入并检查
    print("\n" + "=" * 60)
    print("检查LLM嵌入")
    print("=" * 60)

    llm_emb_path = 'data/mind_tiny/llm_embeddings.npy'
    try:
        llm_embeddings = np.load(llm_emb_path)
        llm_embeddings_tensor = torch.from_numpy(llm_embeddings).float()

        print(f"LLM嵌入形状: {llm_embeddings_tensor.shape}")
        print(f"LLM嵌入dtype: {llm_embeddings_tensor.dtype}")
        print(f"LLM嵌入统计:")
        print(f"  最小值: {llm_embeddings_tensor.min().item():.4f}")
        print(f"  最大值: {llm_embeddings_tensor.max().item():.4f}")
        print(f"  平均值: {llm_embeddings_tensor.mean().item():.4f}")
        print(f"  标准差: {llm_embeddings_tensor.std().item():.4f}")
        print(f"  是否有NaN: {torch.isnan(llm_embeddings_tensor).any().item()}")
        print(f"  是否有Inf: {torch.isinf(llm_embeddings_tensor).any().item()}")

        # 检查一个样本的LLM嵌入
        print(f"\n第一条新闻的LLM嵌入（前10个值）:")
        print(llm_embeddings_tensor[0][:10])

    except Exception as e:
        print(f"加载LLM嵌入时出错: {e}")

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    diagnose_dataset()
