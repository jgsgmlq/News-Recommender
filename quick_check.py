# -*- coding: utf-8 -*-
"""快速检查API进度"""
import os
import sys
from datetime import datetime

def quick_check():
    """快速检查当前进度"""
    print("=" * 60)
    print("  LLM Embedding 进度快速检查")
    print("=" * 60)
    print()

    # 检查文件
    embedding_file = "data/mind_small/llm_embeddings.npy"

    if os.path.exists(embedding_file):
        size_mb = os.path.getsize(embedding_file) / (1024 * 1024)

        # 预估进度（总文件约300MB）
        expected_total_mb = 300
        progress_pct = (size_mb / expected_total_mb) * 100

        # 预估完成条数（51282条，每条约6KB）
        estimated_count = int((size_mb * 1024) / 6)

        print(f"[时间] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[文件] {embedding_file}")
        print(f"[大小] {size_mb:.2f} MB / ~{expected_total_mb} MB")
        print(f"[进度] {progress_pct:.1f}%")
        print(f"[预估] 约 {estimated_count:,} / 51,282 条新闻")
        print()

        # 速度判断
        if progress_pct < 5:
            print("[状态] 刚开始，请等待...")
        elif progress_pct < 50:
            print("[状态] 进行中...")
        elif progress_pct < 90:
            print("[状态] 过半，即将完成")
        else:
            print("[状态] 接近完成！")

    else:
        print("[状态] 文件尚未创建")
        print("[提示] 预计算可能还未开始或刚启动")

    print()
    print("=" * 60)
    print()
    print("提示：运行多次此脚本，对比文件大小变化可判断速度")
    print("正常速度：每分钟增长 1-3 MB")
    print()

if __name__ == '__main__':
    quick_check()
